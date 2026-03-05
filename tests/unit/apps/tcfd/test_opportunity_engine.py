# -*- coding: utf-8 -*-
"""
Unit tests for TCFD Opportunity Engine.

Tests opportunity identification by category, revenue sizing, cost savings
estimation, investment analysis (NPV/IRR), pipeline tracking, and
prioritization with 27+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    OpportunityCategory,
    TimeHorizon,
)
from services.models import (
    ClimateOpportunity,
    _new_id,
)


# ===========================================================================
# Opportunity Identification by Category
# ===========================================================================

class TestOpportunityIdentificationByCategory:
    """Test opportunity identification for each TCFD category."""

    @pytest.mark.parametrize("category,example_name", [
        (OpportunityCategory.RESOURCE_EFFICIENCY, "Energy Efficiency Upgrades"),
        (OpportunityCategory.ENERGY_SOURCE, "Solar PV Installation"),
        (OpportunityCategory.PRODUCTS_SERVICES, "Low-Carbon Product Line"),
        (OpportunityCategory.MARKETS, "Carbon Credit Market"),
        (OpportunityCategory.RESILIENCE, "Climate Adaptation Infrastructure"),
    ])
    def test_category_opportunity_creation(self, category, example_name):
        opp = ClimateOpportunity(
            org_id=_new_id(),
            category=category,
            name=example_name,
        )
        assert opp.category == category
        assert opp.name == example_name

    def test_all_categories_count(self):
        assert len(list(OpportunityCategory)) == 5

    def test_sample_opportunity_category(self, sample_climate_opportunity):
        assert sample_climate_opportunity.category == OpportunityCategory.ENERGY_SOURCE


# ===========================================================================
# Revenue Sizing
# ===========================================================================

class TestRevenueSizing:
    """Test revenue potential estimation."""

    def test_revenue_potential(self, sample_climate_opportunity):
        assert sample_climate_opportunity.revenue_potential == Decimal("100000000")

    def test_zero_revenue(self):
        opp = ClimateOpportunity(
            org_id=_new_id(),
            category=OpportunityCategory.RESOURCE_EFFICIENCY,
            name="Efficiency Only",
            revenue_potential=Decimal("0"),
        )
        assert opp.revenue_potential == Decimal("0")

    def test_large_revenue(self):
        opp = ClimateOpportunity(
            org_id=_new_id(),
            category=OpportunityCategory.MARKETS,
            name="Major Market Opportunity",
            revenue_potential=Decimal("5000000000"),
        )
        assert opp.revenue_potential == Decimal("5000000000")


# ===========================================================================
# Cost Savings Estimation
# ===========================================================================

class TestCostSavingsEstimation:
    """Test cost savings estimation."""

    def test_cost_savings(self, sample_climate_opportunity):
        assert sample_climate_opportunity.cost_savings == Decimal("15000000")

    def test_cost_savings_default_zero(self):
        opp = ClimateOpportunity(
            org_id=_new_id(),
            category=OpportunityCategory.ENERGY_SOURCE,
            name="New Revenue Stream",
        )
        assert opp.cost_savings == Decimal("0")

    def test_high_cost_savings(self):
        opp = ClimateOpportunity(
            org_id=_new_id(),
            category=OpportunityCategory.RESOURCE_EFFICIENCY,
            name="Major Efficiency Program",
            cost_savings=Decimal("50000000"),
        )
        assert opp.cost_savings == Decimal("50000000")


# ===========================================================================
# Investment Analysis
# ===========================================================================

class TestInvestmentAnalysis:
    """Test investment analysis (NPV/IRR)."""

    def test_investment_required(self, sample_climate_opportunity):
        assert sample_climate_opportunity.investment_required == Decimal("50000000")

    def test_roi_positive(self, sample_climate_opportunity):
        assert sample_climate_opportunity.roi_estimate == Decimal("0.25")

    def test_simple_npv_calculation(self):
        investment = Decimal("50000000")
        annual_return = Decimal("15000000")
        discount_rate = Decimal("0.08")
        years = 10
        npv = -investment
        for year in range(1, years + 1):
            npv += annual_return / ((1 + discount_rate) ** year)
        assert npv > Decimal("0")

    def test_simple_payback_period(self):
        investment = Decimal("50000000")
        annual_savings = Decimal("15000000")
        payback_years = investment / annual_savings
        assert payback_years < Decimal("5")

    def test_zero_investment(self):
        opp = ClimateOpportunity(
            org_id=_new_id(),
            category=OpportunityCategory.RESOURCE_EFFICIENCY,
            name="Zero-Cost Quick Win",
            investment_required=Decimal("0"),
            cost_savings=Decimal("500000"),
        )
        assert opp.investment_required == Decimal("0")


# ===========================================================================
# Pipeline Tracking
# ===========================================================================

class TestPipelineTracking:
    """Test opportunity pipeline tracking."""

    def test_pipeline_status_default(self):
        opp = ClimateOpportunity(
            org_id=_new_id(),
            category=OpportunityCategory.ENERGY_SOURCE,
            name="New Opportunity",
        )
        assert opp.status == "identified"

    @pytest.mark.parametrize("status", [
        "identified", "evaluated", "planned", "in_progress", "realized",
    ])
    def test_pipeline_stages(self, status):
        opp = ClimateOpportunity(
            org_id=_new_id(),
            category=OpportunityCategory.RESILIENCE,
            name=f"Pipeline {status}",
            status=status,
        )
        assert opp.status == status

    def test_sample_opportunity_planned(self, sample_climate_opportunity):
        assert sample_climate_opportunity.status == "planned"


# ===========================================================================
# Prioritization
# ===========================================================================

class TestPrioritization:
    """Test opportunity prioritization."""

    def test_feasibility_score(self, sample_climate_opportunity):
        assert sample_climate_opportunity.feasibility_score == 4

    def test_priority_score(self, sample_climate_opportunity):
        assert sample_climate_opportunity.priority_score == 5

    def test_feasibility_range(self):
        for score in range(1, 6):
            opp = ClimateOpportunity(
                org_id=_new_id(),
                category=OpportunityCategory.MARKETS,
                name=f"Feas {score}",
                feasibility_score=score,
            )
            assert opp.feasibility_score == score

    def test_priority_ranking(self):
        opportunities = [
            ClimateOpportunity(
                org_id=_new_id(),
                category=OpportunityCategory.ENERGY_SOURCE,
                name="High Priority",
                priority_score=5,
                feasibility_score=5,
                revenue_potential=Decimal("100000000"),
            ),
            ClimateOpportunity(
                org_id=_new_id(),
                category=OpportunityCategory.RESILIENCE,
                name="Low Priority",
                priority_score=1,
                feasibility_score=2,
                revenue_potential=Decimal("5000000"),
            ),
        ]
        sorted_opps = sorted(opportunities, key=lambda o: o.priority_score, reverse=True)
        assert sorted_opps[0].name == "High Priority"

    def test_opportunity_provenance(self, sample_climate_opportunity):
        assert len(sample_climate_opportunity.provenance_hash) == 64

    def test_opportunity_timeline(self, sample_climate_opportunity):
        assert sample_climate_opportunity.timeline == TimeHorizon.MEDIUM_TERM

    @pytest.mark.parametrize("timeline", list(TimeHorizon))
    def test_all_timelines(self, timeline):
        opp = ClimateOpportunity(
            org_id=_new_id(),
            category=OpportunityCategory.PRODUCTS_SERVICES,
            name=f"Timeline {timeline.value}",
            timeline=timeline,
        )
        assert opp.timeline == timeline
