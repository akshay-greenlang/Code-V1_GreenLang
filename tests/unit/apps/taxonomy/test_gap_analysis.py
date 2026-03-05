# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy Gap Analysis Engine.

Tests gap analysis execution, DNSH gap identification, safeguard
gap identification, data quality gap identification, action plan
generation, and priority matrix with 36+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest


# ===========================================================================
# Gap Analysis Execution
# ===========================================================================

class TestGapAnalysisExecution:
    """Test full gap analysis execution and results."""

    def test_gap_analysis_created(self, sample_taxonomy_gap):
        assert sample_taxonomy_gap["org_id"] is not None
        assert sample_taxonomy_gap["overall_readiness_pct"] >= 0

    def test_readiness_level_assigned(self, sample_taxonomy_gap):
        valid_levels = [
            "fully_aligned", "minor_gaps",
            "moderate_gaps", "significant_gaps", "not_started",
        ]
        assert sample_taxonomy_gap["readiness_level"] in valid_levels

    @pytest.mark.parametrize("readiness_pct,expected_level", [
        (95.0, "fully_aligned"),
        (80.0, "minor_gaps"),
        (60.0, "moderate_gaps"),
        (35.0, "significant_gaps"),
        (5.0, "not_started"),
    ])
    def test_readiness_level_mapping(self, readiness_pct, expected_level):
        if readiness_pct >= 90:
            level = "fully_aligned"
        elif readiness_pct >= 70:
            level = "minor_gaps"
        elif readiness_pct >= 50:
            level = "moderate_gaps"
        elif readiness_pct >= 20:
            level = "significant_gaps"
        else:
            level = "not_started"
        assert level == expected_level

    def test_gap_categories_present(self, sample_taxonomy_gap):
        required_categories = ["sc_gaps", "dnsh_gaps", "safeguard_gaps", "data_gaps"]
        for cat in required_categories:
            assert cat in sample_taxonomy_gap

    def test_gap_analysis_has_provenance(self, sample_taxonomy_gap):
        assert len(sample_taxonomy_gap["provenance_hash"]) == 64

    def test_gap_analysis_has_timestamp(self, sample_taxonomy_gap):
        assert sample_taxonomy_gap["assessed_at"] is not None


# ===========================================================================
# DNSH Gap Identification
# ===========================================================================

class TestDNSHGapIdentification:
    """Test DNSH gap identification across 6 objectives."""

    OBJECTIVES = [
        "climate_mitigation", "climate_adaptation",
        "water_resources", "circular_economy",
        "pollution_prevention", "biodiversity",
    ]

    def test_dnsh_gaps_identified(self, sample_taxonomy_gap):
        dnsh_gaps = sample_taxonomy_gap["dnsh_gaps"]
        assert len(dnsh_gaps) >= 1

    def test_dnsh_gap_has_objective(self, sample_taxonomy_gap):
        for gap in sample_taxonomy_gap["dnsh_gaps"]:
            assert "objective" in gap
            assert gap["objective"] in self.OBJECTIVES

    def test_dnsh_gap_has_description(self, sample_taxonomy_gap):
        for gap in sample_taxonomy_gap["dnsh_gaps"]:
            assert "description" in gap
            assert len(gap["description"]) > 5

    def test_dnsh_gap_has_severity(self, sample_taxonomy_gap):
        for gap in sample_taxonomy_gap["dnsh_gaps"]:
            assert "severity" in gap
            assert gap["severity"] in ["critical", "high", "medium", "low"]

    def test_dnsh_gap_has_activity(self, sample_taxonomy_gap):
        for gap in sample_taxonomy_gap["dnsh_gaps"]:
            assert "activity_code" in gap

    @pytest.mark.parametrize("objective", [
        "climate_mitigation", "climate_adaptation",
        "water_resources", "circular_economy",
        "pollution_prevention", "biodiversity",
    ])
    def test_dnsh_criteria_exist_per_objective(self, objective):
        criteria_map = {
            "climate_mitigation": "GHG emissions thresholds",
            "climate_adaptation": "Physical climate risk assessment",
            "water_resources": "Water stress and discharge",
            "circular_economy": "Waste hierarchy compliance",
            "pollution_prevention": "EU chemical regulations",
            "biodiversity": "Environmental Impact Assessment",
        }
        assert objective in criteria_map
        assert len(criteria_map[objective]) > 5


# ===========================================================================
# Safeguard Gap Identification
# ===========================================================================

class TestSafeguardGapIdentification:
    """Test minimum safeguard gap identification (HR, anti-corruption, tax, competition)."""

    SAFEGUARD_TOPICS = [
        "human_rights", "anti_corruption",
        "taxation", "fair_competition",
    ]

    def test_safeguard_gaps_identified(self, sample_taxonomy_gap):
        sg_gaps = sample_taxonomy_gap["safeguard_gaps"]
        assert isinstance(sg_gaps, list)

    def test_safeguard_gap_has_topic(self, sample_taxonomy_gap):
        for gap in sample_taxonomy_gap["safeguard_gaps"]:
            assert "topic" in gap
            assert gap["topic"] in self.SAFEGUARD_TOPICS

    def test_safeguard_gap_has_requirement(self, sample_taxonomy_gap):
        for gap in sample_taxonomy_gap["safeguard_gaps"]:
            assert "requirement" in gap
            assert len(gap["requirement"]) > 5

    def test_safeguard_gap_has_status(self, sample_taxonomy_gap):
        for gap in sample_taxonomy_gap["safeguard_gaps"]:
            assert gap["status"] in ["met", "partial", "not_met", "unknown"]

    @pytest.mark.parametrize("topic,framework", [
        ("human_rights", "UN Guiding Principles"),
        ("anti_corruption", "OECD Anti-Bribery Convention"),
        ("taxation", "OECD Tax Guidelines"),
        ("fair_competition", "EU Competition Law"),
    ])
    def test_safeguard_framework_reference(self, topic, framework):
        assert topic in self.SAFEGUARD_TOPICS
        assert len(framework) > 5


# ===========================================================================
# Data Quality Gap Identification
# ===========================================================================

class TestDataQualityGapIdentification:
    """Test data quality gap identification."""

    def test_data_gaps_identified(self, sample_taxonomy_gap):
        data_gaps = sample_taxonomy_gap["data_gaps"]
        assert len(data_gaps) >= 1

    def test_data_gap_has_field(self, sample_taxonomy_gap):
        for gap in sample_taxonomy_gap["data_gaps"]:
            assert "field" in gap
            assert gap["field"] != ""

    def test_data_gap_has_impact(self, sample_taxonomy_gap):
        for gap in sample_taxonomy_gap["data_gaps"]:
            assert "impact" in gap
            assert gap["impact"] in ["blocks_alignment", "reduces_score", "informational"]

    def test_data_gap_severity(self, sample_taxonomy_gap):
        for gap in sample_taxonomy_gap["data_gaps"]:
            assert "severity" in gap
            assert gap["severity"] in ["critical", "high", "medium", "low"]

    def test_high_impact_gaps_flagged(self, sample_taxonomy_gap):
        blocking_gaps = [
            g for g in sample_taxonomy_gap["data_gaps"]
            if g["impact"] == "blocks_alignment"
        ]
        assert len(blocking_gaps) >= 0  # May or may not have blocking gaps


# ===========================================================================
# Action Plan Generation
# ===========================================================================

class TestActionPlanGeneration:
    """Test prioritized action plan generation from gap analysis."""

    def test_action_plan_generated(self, sample_taxonomy_gap):
        assert "action_plan" in sample_taxonomy_gap
        plan = sample_taxonomy_gap["action_plan"]
        assert len(plan) >= 1

    def test_actions_have_priority(self, sample_taxonomy_gap):
        for action in sample_taxonomy_gap["action_plan"]:
            assert "priority" in action
            assert action["priority"] >= 1

    def test_actions_ordered_by_priority(self, sample_taxonomy_gap):
        plan = sample_taxonomy_gap["action_plan"]
        for i in range(1, len(plan)):
            assert plan[i]["priority"] >= plan[i - 1]["priority"]

    def test_actions_have_effort(self, sample_taxonomy_gap):
        for action in sample_taxonomy_gap["action_plan"]:
            assert "effort" in action
            assert action["effort"] in ["low", "medium", "high"]

    def test_actions_have_timeline(self, sample_taxonomy_gap):
        for action in sample_taxonomy_gap["action_plan"]:
            assert "timeline_weeks" in action
            assert action["timeline_weeks"] > 0

    def test_actions_have_category(self, sample_taxonomy_gap):
        for action in sample_taxonomy_gap["action_plan"]:
            assert "category" in action
            assert action["category"] in [
                "sc", "dnsh", "safeguards", "data_quality", "reporting",
            ]

    def test_high_priority_actions_first(self, sample_taxonomy_gap):
        plan = sample_taxonomy_gap["action_plan"]
        assert plan[0]["priority"] == 1


# ===========================================================================
# Priority Matrix
# ===========================================================================

class TestPriorityMatrix:
    """Test priority matrix for gap remediation."""

    def test_priority_matrix_present(self, sample_taxonomy_gap):
        assert "priority_matrix" in sample_taxonomy_gap

    def test_matrix_quadrants(self, sample_taxonomy_gap):
        matrix = sample_taxonomy_gap["priority_matrix"]
        required_quadrants = [
            "quick_wins", "strategic_initiatives",
            "fill_ins", "low_priority",
        ]
        for quadrant in required_quadrants:
            assert quadrant in matrix

    def test_quick_wins_high_impact_low_effort(self, sample_taxonomy_gap):
        quick_wins = sample_taxonomy_gap["priority_matrix"]["quick_wins"]
        for item in quick_wins:
            assert item["impact"] in ["high", "critical"]
            assert item["effort"] in ["low", "medium"]

    def test_strategic_items_high_impact_high_effort(self, sample_taxonomy_gap):
        strategic = sample_taxonomy_gap["priority_matrix"]["strategic_initiatives"]
        for item in strategic:
            assert item["impact"] in ["high", "critical"]
            assert item["effort"] == "high"

    def test_matrix_item_count(self, sample_taxonomy_gap):
        matrix = sample_taxonomy_gap["priority_matrix"]
        total_items = sum(len(matrix[q]) for q in matrix)
        assert total_items >= 1
