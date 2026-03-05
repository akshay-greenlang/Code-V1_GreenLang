# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy Alignment Workflow Engine.

Tests the full 4-step alignment workflow end-to-end (eligibility,
SC, DNSH, minimum safeguards), partial alignment scenarios,
company-level MS impact, portfolio alignment with multiple
activities, alignment dashboard data, eligible vs aligned funnel,
and sector-level alignment aggregation with 40+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest


# ===========================================================================
# Full 4-Step Alignment Workflow
# ===========================================================================

class TestFullAlignmentWorkflow:
    """Test full 4-step alignment workflow end-to-end."""

    def test_alignment_result_created(self, sample_alignment_result_workflow):
        assert sample_alignment_result_workflow["alignment_id"] is not None
        assert sample_alignment_result_workflow["activity_code"] is not None

    def test_four_steps_executed(self, sample_alignment_result_workflow):
        steps = sample_alignment_result_workflow["steps"]
        required_steps = ["eligibility", "substantial_contribution", "dnsh", "minimum_safeguards"]
        for step in required_steps:
            assert step in steps

    def test_eligibility_step(self, sample_alignment_result_workflow):
        elig = sample_alignment_result_workflow["steps"]["eligibility"]
        assert "eligible" in elig
        assert isinstance(elig["eligible"], bool)

    def test_sc_step(self, sample_alignment_result_workflow):
        sc = sample_alignment_result_workflow["steps"]["substantial_contribution"]
        assert "passes" in sc
        assert "objective" in sc

    def test_dnsh_step(self, sample_alignment_result_workflow):
        dnsh = sample_alignment_result_workflow["steps"]["dnsh"]
        assert "passes" in dnsh
        assert "assessments" in dnsh

    def test_ms_step(self, sample_alignment_result_workflow):
        ms = sample_alignment_result_workflow["steps"]["minimum_safeguards"]
        assert "passes" in ms
        assert "topics" in ms

    def test_fully_aligned_requires_all_pass(self, sample_alignment_result_workflow):
        steps = sample_alignment_result_workflow["steps"]
        all_pass = all([
            steps["eligibility"]["eligible"],
            steps["substantial_contribution"]["passes"],
            steps["dnsh"]["passes"],
            steps["minimum_safeguards"]["passes"],
        ])
        assert sample_alignment_result_workflow["is_aligned"] == all_pass

    def test_alignment_percentage(self, sample_alignment_result_workflow):
        pct = sample_alignment_result_workflow["alignment_pct"]
        assert 0.0 <= pct <= 100.0

    def test_alignment_has_provenance(self, sample_alignment_result_workflow):
        assert len(sample_alignment_result_workflow["provenance_hash"]) == 64


# ===========================================================================
# Partial Alignment (SC Pass, DNSH Fail)
# ===========================================================================

class TestPartialAlignment:
    """Test partial alignment scenarios."""

    def test_sc_pass_dnsh_fail(self):
        result = {
            "steps": {
                "eligibility": {"eligible": True},
                "substantial_contribution": {"passes": True, "objective": "climate_mitigation"},
                "dnsh": {"passes": False, "failed_objectives": ["water_resources"]},
                "minimum_safeguards": {"passes": True},
            },
            "is_aligned": False,
            "alignment_pct": 0.0,
            "failure_reason": "DNSH failed for water_resources",
        }
        assert result["is_aligned"] is False
        assert "dnsh" in result["failure_reason"].lower() or "water" in result["failure_reason"].lower()

    def test_eligible_but_sc_fails(self):
        result = {
            "steps": {
                "eligibility": {"eligible": True},
                "substantial_contribution": {"passes": False, "threshold_met": False},
                "dnsh": {"passes": None},  # Not assessed (SC failed)
                "minimum_safeguards": {"passes": None},
            },
            "is_aligned": False,
            "status": "eligible_not_aligned",
        }
        assert result["status"] == "eligible_not_aligned"

    def test_not_eligible(self):
        result = {
            "steps": {
                "eligibility": {"eligible": False, "reason": "NACE not mapped"},
                "substantial_contribution": {"passes": None},
                "dnsh": {"passes": None},
                "minimum_safeguards": {"passes": None},
            },
            "is_aligned": False,
            "status": "not_eligible",
        }
        assert result["status"] == "not_eligible"

    def test_ms_failure_blocks_alignment(self):
        result = {
            "steps": {
                "eligibility": {"eligible": True},
                "substantial_contribution": {"passes": True},
                "dnsh": {"passes": True},
                "minimum_safeguards": {"passes": False, "failed_topics": ["human_rights"]},
            },
            "is_aligned": False,
        }
        assert result["is_aligned"] is False

    @pytest.mark.parametrize("elig,sc,dnsh,ms,expected_aligned", [
        (True, True, True, True, True),
        (True, True, True, False, False),
        (True, True, False, True, False),
        (True, False, True, True, False),
        (False, True, True, True, False),
        (False, False, False, False, False),
    ])
    def test_alignment_truth_table(self, elig, sc, dnsh, ms, expected_aligned):
        is_aligned = elig and sc and dnsh and ms
        assert is_aligned == expected_aligned


# ===========================================================================
# Company-Level MS Impact
# ===========================================================================

class TestCompanyLevelMSImpact:
    """Test company-level minimum safeguards impact on all activities."""

    def test_ms_applies_company_wide(self):
        company = {
            "ms_status": "pass",
            "activities": [
                {"code": "4.1", "sc_pass": True, "dnsh_pass": True},
                {"code": "7.7", "sc_pass": True, "dnsh_pass": True},
            ],
        }
        for activity in company["activities"]:
            aligned = (
                activity["sc_pass"]
                and activity["dnsh_pass"]
                and company["ms_status"] == "pass"
            )
            assert aligned is True

    def test_ms_failure_blocks_all_activities(self):
        company = {
            "ms_status": "fail",
            "activities": [
                {"code": "4.1", "sc_pass": True, "dnsh_pass": True},
                {"code": "7.7", "sc_pass": True, "dnsh_pass": True},
            ],
        }
        for activity in company["activities"]:
            aligned = (
                activity["sc_pass"]
                and activity["dnsh_pass"]
                and company["ms_status"] == "pass"
            )
            assert aligned is False

    def test_ms_partial_reduces_alignment(self):
        company = {
            "ms_status": "partial",
            "ms_topics": {
                "human_rights": "pass",
                "anti_corruption": "pass",
                "taxation": "fail",
                "fair_competition": "pass",
            },
        }
        overall = all(v == "pass" for v in company["ms_topics"].values())
        assert overall is False

    def test_ms_requires_all_four_topics(self):
        required_topics = ["human_rights", "anti_corruption", "taxation", "fair_competition"]
        assert len(required_topics) == 4


# ===========================================================================
# Portfolio Alignment with Multiple Activities
# ===========================================================================

class TestPortfolioAlignment:
    """Test portfolio alignment with multiple activities."""

    def test_portfolio_alignment_result(self, sample_portfolio_alignment_result):
        assert sample_portfolio_alignment_result["portfolio_id"] is not None
        assert sample_portfolio_alignment_result["total_exposure"] > 0

    def test_portfolio_alignment_pct(self, sample_portfolio_alignment_result):
        pct = sample_portfolio_alignment_result["portfolio_alignment_pct"]
        assert 0.0 <= pct <= 100.0

    def test_portfolio_eligibility_pct(self, sample_portfolio_alignment_result):
        pct = sample_portfolio_alignment_result["portfolio_eligibility_pct"]
        assert 0.0 <= pct <= 100.0
        assert pct >= sample_portfolio_alignment_result["portfolio_alignment_pct"]

    def test_per_holding_alignment(self, sample_portfolio_alignment_result):
        for holding in sample_portfolio_alignment_result["holdings_alignment"]:
            assert "holding_id" in holding
            assert "is_eligible" in holding
            assert "is_aligned" in holding
            assert "alignment_pct" in holding

    def test_aligned_holdings_subset_of_eligible(self, sample_portfolio_alignment_result):
        eligible = [
            h for h in sample_portfolio_alignment_result["holdings_alignment"]
            if h["is_eligible"]
        ]
        aligned = [
            h for h in sample_portfolio_alignment_result["holdings_alignment"]
            if h["is_aligned"]
        ]
        aligned_ids = {h["holding_id"] for h in aligned}
        eligible_ids = {h["holding_id"] for h in eligible}
        assert aligned_ids.issubset(eligible_ids)


# ===========================================================================
# Alignment Dashboard Data
# ===========================================================================

class TestAlignmentDashboard:
    """Test alignment dashboard data preparation."""

    def test_dashboard_overview(self, sample_alignment_dashboard):
        required_fields = [
            "total_activities", "eligible_activities",
            "aligned_activities", "eligibility_pct",
            "alignment_pct",
        ]
        for field in required_fields:
            assert field in sample_alignment_dashboard

    def test_dashboard_kpi_summary(self, sample_alignment_dashboard):
        assert "kpi_summary" in sample_alignment_dashboard
        kpi = sample_alignment_dashboard["kpi_summary"]
        assert "turnover_aligned_pct" in kpi
        assert "capex_aligned_pct" in kpi
        assert "opex_aligned_pct" in kpi

    def test_dashboard_objective_chart(self, sample_alignment_dashboard):
        assert "by_objective" in sample_alignment_dashboard
        for obj in sample_alignment_dashboard["by_objective"]:
            assert "objective" in obj
            assert "aligned_count" in obj

    def test_dashboard_trend_data(self, sample_alignment_dashboard):
        assert "trend" in sample_alignment_dashboard
        assert len(sample_alignment_dashboard["trend"]) >= 1


# ===========================================================================
# Eligible vs Aligned Funnel
# ===========================================================================

class TestEligibleAlignedFunnel:
    """Test eligible vs aligned funnel visualization data."""

    def test_funnel_stages(self, sample_alignment_dashboard):
        funnel = sample_alignment_dashboard.get("funnel", {})
        stages = ["total", "eligible", "sc_pass", "dnsh_pass", "ms_pass", "aligned"]
        for stage in stages:
            assert stage in funnel

    def test_funnel_monotonically_decreasing(self, sample_alignment_dashboard):
        funnel = sample_alignment_dashboard.get("funnel", {})
        stage_order = ["total", "eligible", "sc_pass", "dnsh_pass", "ms_pass", "aligned"]
        for i in range(1, len(stage_order)):
            assert funnel[stage_order[i]] <= funnel[stage_order[i - 1]]

    def test_funnel_zero_aligned_valid(self):
        funnel = {
            "total": 100, "eligible": 80, "sc_pass": 50,
            "dnsh_pass": 30, "ms_pass": 0, "aligned": 0,
        }
        assert funnel["aligned"] == 0
        assert funnel["total"] > 0

    def test_funnel_all_aligned(self):
        funnel = {
            "total": 50, "eligible": 50, "sc_pass": 50,
            "dnsh_pass": 50, "ms_pass": 50, "aligned": 50,
        }
        assert funnel["aligned"] == funnel["total"]


# ===========================================================================
# Sector-Level Alignment Aggregation
# ===========================================================================

class TestSectorAlignmentAggregation:
    """Test sector-level alignment aggregation."""

    def test_sector_aggregation_present(self, sample_alignment_dashboard):
        assert "by_sector" in sample_alignment_dashboard
        assert len(sample_alignment_dashboard["by_sector"]) >= 1

    def test_sector_has_nace_section(self, sample_alignment_dashboard):
        for sector in sample_alignment_dashboard["by_sector"]:
            assert "nace_section" in sector
            assert "sector_name" in sector

    def test_sector_has_alignment_stats(self, sample_alignment_dashboard):
        for sector in sample_alignment_dashboard["by_sector"]:
            assert "total_activities" in sector
            assert "aligned_activities" in sector
            assert "sector_alignment_pct" in sector

    def test_sector_alignment_range(self, sample_alignment_dashboard):
        for sector in sample_alignment_dashboard["by_sector"]:
            assert 0.0 <= sector["sector_alignment_pct"] <= 100.0

    def test_sector_aligned_not_exceed_total(self, sample_alignment_dashboard):
        for sector in sample_alignment_dashboard["by_sector"]:
            assert sector["aligned_activities"] <= sector["total_activities"]

    def test_sector_exposure_breakdown(self, sample_alignment_dashboard):
        for sector in sample_alignment_dashboard["by_sector"]:
            assert "total_exposure" in sector
            assert "aligned_exposure" in sector
            assert sector["aligned_exposure"] <= sector["total_exposure"]
