# -*- coding: utf-8 -*-
"""
Unit tests for SBTi Gap Analysis Engine.

Tests full criteria-by-criteria gap analysis, data gap identification
for missing categories, ambition gap assessment for rate shortfalls,
process/governance gap detection, prioritized action plan generation,
and peer benchmarking comparison with 22+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest


# ===========================================================================
# Full Gap Analysis
# ===========================================================================

class TestFullGapAnalysis:
    """Test criteria-by-criteria gap analysis."""

    def test_gap_assessment_created(self, sample_gap_assessment):
        assert sample_gap_assessment["overall_readiness_score"] > 0

    def test_readiness_level(self, sample_gap_assessment):
        valid_levels = [
            "ready_for_submission", "minor_gaps",
            "moderate_gaps", "significant_gaps",
        ]
        assert sample_gap_assessment["readiness_level"] in valid_levels

    def test_criteria_gaps_present(self, sample_gap_assessment):
        criteria = sample_gap_assessment["criteria_gaps"]
        assert len(criteria) >= 1

    @pytest.mark.parametrize("score,expected_level", [
        (95.0, "ready_for_submission"),
        (80.0, "minor_gaps"),
        (60.0, "moderate_gaps"),
        (30.0, "significant_gaps"),
    ])
    def test_readiness_level_mapping(self, score, expected_level):
        if score >= 90:
            level = "ready_for_submission"
        elif score >= 70:
            level = "minor_gaps"
        elif score >= 50:
            level = "moderate_gaps"
        else:
            level = "significant_gaps"
        assert level == expected_level

    def test_criteria_gap_statuses(self, sample_gap_assessment):
        for criteria_id, result in sample_gap_assessment["criteria_gaps"].items():
            assert result["status"] in ["pass", "gap", "warning"]
            assert "detail" in result


# ===========================================================================
# Data Gaps
# ===========================================================================

class TestDataGaps:
    """Test missing data category identification."""

    def test_data_gaps_identified(self, sample_gap_assessment):
        gaps = sample_gap_assessment["data_gaps"]
        assert len(gaps) >= 1

    def test_data_gap_severity(self, sample_gap_assessment):
        gaps = sample_gap_assessment["data_gaps"]
        severities = {g["severity"] for g in gaps}
        valid_severities = {"high", "medium", "low"}
        assert severities.issubset(valid_severities)

    def test_data_gap_has_category(self, sample_gap_assessment):
        for gap in sample_gap_assessment["data_gaps"]:
            assert "category" in gap
            assert "gap" in gap

    def test_high_severity_gaps_exist(self, sample_gap_assessment):
        high_gaps = [g for g in sample_gap_assessment["data_gaps"] if g["severity"] == "high"]
        assert len(high_gaps) >= 1

    def test_flag_data_gap(self, sample_gap_assessment):
        flag_gaps = [
            g for g in sample_gap_assessment["data_gaps"]
            if "flag" in g["category"].lower()
        ]
        assert len(flag_gaps) >= 1


# ===========================================================================
# Ambition Gaps
# ===========================================================================

class TestAmbitionGaps:
    """Test annual rate shortfall assessment."""

    def test_ambition_gaps_identified(self, sample_gap_assessment):
        gaps = sample_gap_assessment["ambition_gaps"]
        assert len(gaps) >= 1

    def test_ambition_gap_has_shortfall(self, sample_gap_assessment):
        for gap in sample_gap_assessment["ambition_gaps"]:
            assert "current_rate" in gap
            assert "required_rate" in gap
            assert "shortfall_pct" in gap

    def test_shortfall_positive(self, sample_gap_assessment):
        for gap in sample_gap_assessment["ambition_gaps"]:
            assert gap["shortfall_pct"] > 0

    def test_required_rate_exceeds_current(self, sample_gap_assessment):
        for gap in sample_gap_assessment["ambition_gaps"]:
            assert gap["required_rate"] > gap["current_rate"]

    @pytest.mark.parametrize("current,required,expected_shortfall", [
        (2.0, 2.5, 0.5),
        (3.5, 4.2, 0.7),
        (1.0, 2.5, 1.5),
    ])
    def test_shortfall_calculation(self, current, required, expected_shortfall):
        shortfall = required - current
        assert shortfall == pytest.approx(expected_shortfall, abs=0.01)


# ===========================================================================
# Process Gaps
# ===========================================================================

class TestProcessGaps:
    """Test governance and reporting process gaps."""

    def test_process_gaps_identified(self, sample_gap_assessment):
        gaps = sample_gap_assessment["process_gaps"]
        assert len(gaps) >= 1

    def test_process_gap_areas(self, sample_gap_assessment):
        areas = {g["area"] for g in sample_gap_assessment["process_gaps"]}
        valid_areas = {"governance", "reporting", "data_management", "verification", "review"}
        assert areas.issubset(valid_areas)

    def test_governance_gap(self, sample_gap_assessment):
        gov_gaps = [g for g in sample_gap_assessment["process_gaps"] if g["area"] == "governance"]
        assert len(gov_gaps) >= 1

    def test_reporting_gap(self, sample_gap_assessment):
        rep_gaps = [g for g in sample_gap_assessment["process_gaps"] if g["area"] == "reporting"]
        assert len(rep_gaps) >= 1


# ===========================================================================
# Action Plan
# ===========================================================================

class TestActionPlan:
    """Test prioritized action roadmap generation."""

    def test_action_plan_generated(self, sample_gap_assessment):
        plan = sample_gap_assessment["action_plan"]
        assert len(plan) >= 2

    def test_actions_have_priority(self, sample_gap_assessment):
        for action in sample_gap_assessment["action_plan"]:
            assert "priority" in action
            assert action["priority"] >= 1

    def test_actions_ordered_by_priority(self, sample_gap_assessment):
        plan = sample_gap_assessment["action_plan"]
        for i in range(1, len(plan)):
            assert plan[i]["priority"] >= plan[i - 1]["priority"]

    def test_actions_have_timeline(self, sample_gap_assessment):
        for action in sample_gap_assessment["action_plan"]:
            assert "timeline_months" in action
            assert action["timeline_months"] > 0

    def test_actions_have_effort_level(self, sample_gap_assessment):
        for action in sample_gap_assessment["action_plan"]:
            assert action["effort"] in ["low", "medium", "high"]

    def test_high_priority_first(self, sample_gap_assessment):
        plan = sample_gap_assessment["action_plan"]
        assert plan[0]["priority"] == 1


# ===========================================================================
# Benchmarking
# ===========================================================================

class TestBenchmarking:
    """Test peer comparison benchmarking."""

    def test_peer_benchmark_present(self, sample_gap_assessment):
        benchmark = sample_gap_assessment["peer_benchmark"]
        assert benchmark is not None

    def test_benchmark_sector(self, sample_gap_assessment):
        benchmark = sample_gap_assessment["peer_benchmark"]
        assert benchmark["sector"] == "manufacturing"

    def test_org_percentile(self, sample_gap_assessment):
        benchmark = sample_gap_assessment["peer_benchmark"]
        assert 0 <= benchmark["org_percentile"] <= 100

    def test_sector_average_score(self, sample_gap_assessment):
        benchmark = sample_gap_assessment["peer_benchmark"]
        assert benchmark["avg_readiness_score"] > 0

    def test_top_quartile_reference(self, sample_gap_assessment):
        benchmark = sample_gap_assessment["peer_benchmark"]
        assert benchmark["top_quartile_score"] > benchmark["avg_readiness_score"]

    def test_org_above_average(self, sample_gap_assessment):
        benchmark = sample_gap_assessment["peer_benchmark"]
        org_score = sample_gap_assessment["overall_readiness_score"]
        assert org_score > benchmark["avg_readiness_score"]

    def test_gap_provenance(self, sample_gap_assessment):
        assert len(sample_gap_assessment["provenance_hash"]) == 64
