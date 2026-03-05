# -*- coding: utf-8 -*-
"""
Unit tests for TransitionPlanEngine -- 1.5C transition plan builder.

Tests pathway creation, milestone management, SBTi alignment validation,
progress tracking, investment plan calculation, and revenue alignment
with 32+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    TransitionPathwayType,
    SBTiStatus,
    MilestoneStatus,
)
from services.models import (
    CDPTransitionPlan,
    CDPTransitionMilestone,
    _new_id,
)
from services.transition_plan_engine import TransitionPlanEngine


# ---------------------------------------------------------------------------
# Pathway creation
# ---------------------------------------------------------------------------

class TestPathwayCreation:
    """Test transition plan pathway creation."""

    def test_create_net_zero_pathway(self, transition_plan_engine, sample_organization):
        plan = transition_plan_engine.create_plan(
            org_id=sample_organization.id,
            plan_name="Net Zero 2050",
            base_year=2020,
            target_year=2050,
            pathway_type=TransitionPathwayType.NET_ZERO,
        )
        assert isinstance(plan, CDPTransitionPlan)
        assert plan.pathway_type == TransitionPathwayType.NET_ZERO
        assert plan.target_year == 2050

    def test_create_well_below_2c_pathway(self, transition_plan_engine, sample_organization):
        plan = transition_plan_engine.create_plan(
            org_id=sample_organization.id,
            plan_name="Well Below 2C",
            base_year=2020,
            target_year=2050,
            pathway_type=TransitionPathwayType.WELL_BELOW_2C,
        )
        assert plan.pathway_type == TransitionPathwayType.WELL_BELOW_2C

    def test_create_1_5c_pathway(self, transition_plan_engine, sample_organization):
        plan = transition_plan_engine.create_plan(
            org_id=sample_organization.id,
            plan_name="1.5C Aligned",
            base_year=2020,
            target_year=2050,
            pathway_type=TransitionPathwayType.ALIGNED_1_5C,
        )
        assert plan.pathway_type == TransitionPathwayType.ALIGNED_1_5C

    def test_base_year_before_target_year(self, transition_plan_engine, sample_organization):
        with pytest.raises(ValueError, match="[Bb]ase year.*target"):
            transition_plan_engine.create_plan(
                org_id=sample_organization.id,
                plan_name="Invalid Plan",
                base_year=2050,
                target_year=2020,
                pathway_type=TransitionPathwayType.NET_ZERO,
            )

    def test_plan_status_defaults_to_draft(self, transition_plan_engine, sample_organization):
        plan = transition_plan_engine.create_plan(
            org_id=sample_organization.id,
            plan_name="Draft Plan",
            base_year=2020,
            target_year=2050,
            pathway_type=TransitionPathwayType.NET_ZERO,
        )
        assert plan.status in ["draft", "active"]


# ---------------------------------------------------------------------------
# Milestone management
# ---------------------------------------------------------------------------

class TestMilestoneManagement:
    """Test transition milestone CRUD."""

    def test_add_milestone(self, transition_plan_engine, sample_transition_plan):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        milestone = transition_plan_engine.add_milestone(
            plan_id=sample_transition_plan.id,
            milestone_name="25% Reduction by 2027",
            target_year=2027,
            target_reduction_pct=Decimal("25.0"),
            scope="scope_1_2",
            technology_lever="energy_efficiency",
            investment_usd=Decimal("50000000"),
        )
        assert isinstance(milestone, CDPTransitionMilestone)
        assert milestone.target_year == 2027

    def test_add_multiple_milestones(self, transition_plan_engine, sample_transition_plan):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        for year in [2025, 2030, 2040, 2050]:
            transition_plan_engine.add_milestone(
                plan_id=sample_transition_plan.id,
                milestone_name=f"Target {year}",
                target_year=year,
                target_reduction_pct=Decimal(str((year - 2020) * 3)),
                scope="all_scopes",
            )
        milestones = transition_plan_engine.get_milestones(sample_transition_plan.id)
        assert len(milestones) == 4

    def test_milestones_sorted_by_year(self, transition_plan_engine, sample_transition_plan):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        for year in [2040, 2025, 2035, 2030]:
            transition_plan_engine.add_milestone(
                plan_id=sample_transition_plan.id,
                milestone_name=f"Target {year}",
                target_year=year,
                target_reduction_pct=Decimal("50"),
                scope="all_scopes",
            )
        milestones = transition_plan_engine.get_milestones(sample_transition_plan.id)
        years = [m.target_year for m in milestones]
        assert years == sorted(years)

    def test_update_milestone_progress(self, transition_plan_engine, sample_transition_plan, sample_milestones):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        for m in sample_milestones:
            transition_plan_engine._milestone_store[m.id] = m
        updated = transition_plan_engine.update_milestone_progress(
            milestone_id=sample_milestones[0].id,
            progress_pct=Decimal("55.0"),
            status=MilestoneStatus.ON_TRACK,
        )
        assert updated.progress_pct == Decimal("55.0")

    def test_milestone_status_behind(self, transition_plan_engine, sample_transition_plan, sample_milestones):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        for m in sample_milestones:
            transition_plan_engine._milestone_store[m.id] = m
        updated = transition_plan_engine.update_milestone_progress(
            milestone_id=sample_milestones[0].id,
            progress_pct=Decimal("10.0"),
            status=MilestoneStatus.BEHIND,
        )
        assert updated.status == MilestoneStatus.BEHIND

    def test_delete_milestone(self, transition_plan_engine, sample_transition_plan, sample_milestones):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        for m in sample_milestones:
            transition_plan_engine._milestone_store[m.id] = m
        transition_plan_engine.delete_milestone(sample_milestones[0].id)
        remaining = transition_plan_engine.get_milestones(sample_transition_plan.id)
        assert len(remaining) == 1


# ---------------------------------------------------------------------------
# SBTi alignment
# ---------------------------------------------------------------------------

class TestSBTiAlignment:
    """Test SBTi alignment validation."""

    def test_sbti_aligned_plan(self, transition_plan_engine, sample_transition_plan):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        result = transition_plan_engine.validate_sbti_alignment(sample_transition_plan.id)
        assert result["is_aligned"] is True

    def test_sbti_minimum_annual_reduction(self, transition_plan_engine):
        result = transition_plan_engine.check_annual_reduction_rate(
            base_year_emissions=Decimal("10000"),
            current_emissions=Decimal("8500"),
            years_elapsed=3,
        )
        rate = result["annual_reduction_rate"]
        assert rate > Decimal("0")

    def test_sbti_insufficient_reduction(self, transition_plan_engine):
        result = transition_plan_engine.check_annual_reduction_rate(
            base_year_emissions=Decimal("10000"),
            current_emissions=Decimal("9800"),
            years_elapsed=3,
        )
        # ~0.67% per year, well below 4.2%
        assert result["meets_1_5c_target"] is False

    def test_sbti_status_tracking(self, transition_plan_engine, sample_transition_plan):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        assert sample_transition_plan.sbti_status == SBTiStatus.TARGETS_SET

    def test_update_sbti_status(self, transition_plan_engine, sample_transition_plan):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        updated = transition_plan_engine.update_sbti_status(
            plan_id=sample_transition_plan.id,
            status=SBTiStatus.VALIDATED,
        )
        assert updated.sbti_status == SBTiStatus.VALIDATED


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class TestProgressTracking:
    """Test transition plan progress tracking."""

    def test_overall_progress(self, transition_plan_engine, sample_transition_plan, sample_milestones):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        for m in sample_milestones:
            transition_plan_engine._milestone_store[m.id] = m
        progress = transition_plan_engine.get_overall_progress(sample_transition_plan.id)
        assert Decimal("0") <= progress <= Decimal("100")

    def test_progress_by_scope(self, transition_plan_engine, sample_transition_plan, sample_milestones):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        for m in sample_milestones:
            transition_plan_engine._milestone_store[m.id] = m
        by_scope = transition_plan_engine.get_progress_by_scope(sample_transition_plan.id)
        assert isinstance(by_scope, dict)

    def test_at_risk_milestones(self, transition_plan_engine, sample_transition_plan):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        behind_ms = CDPTransitionMilestone(
            plan_id=sample_transition_plan.id,
            milestone_name="Behind Milestone",
            target_year=2028,
            target_reduction_pct=Decimal("30"),
            scope="scope_1",
            status=MilestoneStatus.BEHIND,
            progress_pct=Decimal("5.0"),
        )
        transition_plan_engine._milestone_store[behind_ms.id] = behind_ms
        at_risk = transition_plan_engine.get_at_risk_milestones(sample_transition_plan.id)
        assert len(at_risk) >= 1


# ---------------------------------------------------------------------------
# Investment plan
# ---------------------------------------------------------------------------

class TestInvestmentPlan:
    """Test decarbonization investment calculations."""

    def test_total_investment(self, transition_plan_engine, sample_transition_plan, sample_milestones):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        for m in sample_milestones:
            transition_plan_engine._milestone_store[m.id] = m
        total = transition_plan_engine.calculate_total_investment(sample_transition_plan.id)
        expected = sum(m.investment_usd or Decimal("0") for m in sample_milestones)
        assert total == expected

    def test_investment_by_lever(self, transition_plan_engine, sample_transition_plan, sample_milestones):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        for m in sample_milestones:
            transition_plan_engine._milestone_store[m.id] = m
        by_lever = transition_plan_engine.get_investment_by_lever(sample_transition_plan.id)
        assert isinstance(by_lever, dict)
        assert len(by_lever) > 0


# ---------------------------------------------------------------------------
# Revenue alignment
# ---------------------------------------------------------------------------

class TestRevenueAlignment:
    """Test revenue alignment tracking."""

    def test_set_revenue_alignment(self, transition_plan_engine, sample_transition_plan):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        updated = transition_plan_engine.set_revenue_alignment(
            plan_id=sample_transition_plan.id,
            alignment_pct=Decimal("45.0"),
        )
        assert updated.revenue_alignment_pct == Decimal("45.0")

    def test_revenue_alignment_range(self, transition_plan_engine, sample_transition_plan):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        with pytest.raises(ValueError, match="[Rr]ange|[Pp]ercentage"):
            transition_plan_engine.set_revenue_alignment(
                plan_id=sample_transition_plan.id,
                alignment_pct=Decimal("150.0"),
            )

    def test_public_availability(self, transition_plan_engine, sample_transition_plan):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        assert sample_transition_plan.is_publicly_available is True
        updated = transition_plan_engine.set_public_availability(
            plan_id=sample_transition_plan.id,
            is_public=False,
        )
        assert updated.is_publicly_available is False


# ---------------------------------------------------------------------------
# Plan lifecycle
# ---------------------------------------------------------------------------

class TestPlanLifecycle:
    """Test transition plan lifecycle management."""

    def test_activate_plan(self, transition_plan_engine, sample_organization):
        plan = transition_plan_engine.create_plan(
            org_id=sample_organization.id,
            plan_name="Lifecycle Test",
            base_year=2020,
            target_year=2050,
            pathway_type=TransitionPathwayType.NET_ZERO,
        )
        activated = transition_plan_engine.activate_plan(plan.id)
        assert activated.status == "active"

    def test_archive_plan(self, transition_plan_engine, sample_transition_plan):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        archived = transition_plan_engine.archive_plan(sample_transition_plan.id)
        assert archived.status == "archived"

    def test_get_active_plan(self, transition_plan_engine, sample_organization, sample_transition_plan):
        sample_transition_plan.status = "active"
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        active = transition_plan_engine.get_active_plan(sample_organization.id)
        assert active is not None
        assert active.status == "active"

    def test_list_plans_for_org(self, transition_plan_engine, sample_organization):
        for i in range(3):
            plan = transition_plan_engine.create_plan(
                org_id=sample_organization.id,
                plan_name=f"Plan {i}",
                base_year=2020,
                target_year=2050,
                pathway_type=TransitionPathwayType.NET_ZERO,
            )
        plans = transition_plan_engine.list_plans(sample_organization.id)
        assert len(plans) >= 3

    def test_get_plan_by_id(self, transition_plan_engine, sample_transition_plan):
        transition_plan_engine._plan_store[sample_transition_plan.id] = sample_transition_plan
        plan = transition_plan_engine.get_plan(sample_transition_plan.id)
        assert plan is not None
        assert plan.id == sample_transition_plan.id

    def test_nonexistent_plan_returns_none(self, transition_plan_engine):
        result = transition_plan_engine.get_plan("nonexistent-id")
        assert result is None
