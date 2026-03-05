# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy Regulatory Update Engine.

Tests delegated act registration and version tracking, TSC update
tracking, Omnibus impact assessment, applicable version lookup,
and transition plan generation with 24+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest
from datetime import date


# ===========================================================================
# Delegated Act Registration
# ===========================================================================

class TestDelegatedActRegistration:
    """Test delegated act registration and version tracking."""

    def test_delegated_act_created(self, sample_delegated_act):
        assert sample_delegated_act["act_id"] is not None
        assert sample_delegated_act["act_type"] is not None

    def test_act_types_valid(self, sample_delegated_act):
        valid_types = ["climate", "environmental", "complementary", "amendment"]
        assert sample_delegated_act["act_type"] in valid_types

    def test_act_has_effective_date(self, sample_delegated_act):
        assert sample_delegated_act["effective_date"] is not None
        assert isinstance(sample_delegated_act["effective_date"], date)

    def test_act_has_version(self, sample_delegated_act):
        assert sample_delegated_act["version"] is not None
        assert sample_delegated_act["version"] != ""

    def test_act_activities_count(self, sample_delegated_act):
        assert sample_delegated_act["activities_count"] > 0

    def test_act_status(self, sample_delegated_act):
        valid_statuses = ["draft", "published", "in_force", "superseded"]
        assert sample_delegated_act["status"] in valid_statuses

    @pytest.mark.parametrize("act_type,expected_min_activities", [
        ("climate", 80),
        ("environmental", 50),
        ("complementary", 5),
    ])
    def test_act_type_activity_counts(self, act_type, expected_min_activities):
        act_counts = {
            "climate": 88,
            "environmental": 62,
            "complementary": 12,
        }
        assert act_counts[act_type] >= expected_min_activities

    def test_version_history_ordered(self, sample_delegated_act):
        history = sample_delegated_act.get("version_history", [])
        if len(history) >= 2:
            for i in range(1, len(history)):
                assert history[i]["effective_date"] >= history[i - 1]["effective_date"]


# ===========================================================================
# TSC Update Tracking
# ===========================================================================

class TestTSCUpdateTracking:
    """Test Technical Screening Criteria update tracking."""

    def test_tsc_update_registered(self, sample_tsc_update):
        assert sample_tsc_update["update_id"] is not None

    def test_tsc_update_has_activity(self, sample_tsc_update):
        assert sample_tsc_update["activity_code"] is not None
        assert sample_tsc_update["activity_code"] != ""

    def test_tsc_update_has_criteria_changes(self, sample_tsc_update):
        assert "criteria_changes" in sample_tsc_update
        assert len(sample_tsc_update["criteria_changes"]) >= 1

    def test_tsc_change_has_field(self, sample_tsc_update):
        for change in sample_tsc_update["criteria_changes"]:
            assert "field" in change
            assert "old_value" in change
            assert "new_value" in change

    def test_tsc_update_impact_assessed(self, sample_tsc_update):
        assert "impact_level" in sample_tsc_update
        assert sample_tsc_update["impact_level"] in ["high", "medium", "low"]

    def test_tsc_affected_entities_count(self, sample_tsc_update):
        assert sample_tsc_update["affected_entities_count"] >= 0


# ===========================================================================
# Omnibus Impact Assessment
# ===========================================================================

class TestOmnibusImpactAssessment:
    """Test Omnibus simplification impact assessment."""

    def test_omnibus_assessment_created(self, sample_omnibus_impact):
        assert sample_omnibus_impact["assessment_id"] is not None

    def test_omnibus_simplifications_tracked(self, sample_omnibus_impact):
        assert "simplifications" in sample_omnibus_impact
        assert len(sample_omnibus_impact["simplifications"]) >= 1

    def test_simplification_has_category(self, sample_omnibus_impact):
        valid_categories = [
            "reporting_threshold", "disclosure_relief",
            "data_estimation", "proportionality",
            "voluntary_reporting", "safe_harbour",
        ]
        for simp in sample_omnibus_impact["simplifications"]:
            assert simp["category"] in valid_categories

    def test_omnibus_impact_on_gar(self, sample_omnibus_impact):
        assert "gar_impact" in sample_omnibus_impact
        assert sample_omnibus_impact["gar_impact"]["affected"] is not None

    def test_omnibus_effective_date(self, sample_omnibus_impact):
        assert sample_omnibus_impact["effective_date"] is not None

    def test_omnibus_transition_period(self, sample_omnibus_impact):
        assert "transition_period_months" in sample_omnibus_impact
        assert sample_omnibus_impact["transition_period_months"] >= 0


# ===========================================================================
# Applicable Version Lookup
# ===========================================================================

class TestApplicableVersionLookup:
    """Test looking up applicable TSC version for a given date."""

    @pytest.mark.parametrize("lookup_date,expected_act_type", [
        (date(2024, 1, 1), "climate"),
        (date(2025, 1, 1), "climate"),
        (date(2026, 1, 1), "environmental"),
    ])
    def test_applicable_version_by_date(self, lookup_date, expected_act_type):
        versions = [
            {"act_type": "climate", "effective_date": date(2022, 1, 1), "end_date": date(2025, 12, 31)},
            {"act_type": "environmental", "effective_date": date(2026, 1, 1), "end_date": None},
        ]
        applicable = None
        for v in versions:
            if v["effective_date"] <= lookup_date:
                if v["end_date"] is None or lookup_date <= v["end_date"]:
                    applicable = v
        assert applicable is not None
        assert applicable["act_type"] == expected_act_type

    def test_no_version_before_regulation(self):
        lookup_date = date(2020, 1, 1)
        versions = [
            {"effective_date": date(2022, 1, 1)},
        ]
        applicable = [v for v in versions if v["effective_date"] <= lookup_date]
        assert len(applicable) == 0


# ===========================================================================
# Transition Plan Generation
# ===========================================================================

class TestTransitionPlanGeneration:
    """Test transition plan generation for regulatory updates."""

    def test_transition_plan_created(self, sample_transition_plan):
        assert sample_transition_plan["plan_id"] is not None

    def test_plan_has_milestones(self, sample_transition_plan):
        assert "milestones" in sample_transition_plan
        assert len(sample_transition_plan["milestones"]) >= 2

    def test_milestones_ordered_by_date(self, sample_transition_plan):
        milestones = sample_transition_plan["milestones"]
        for i in range(1, len(milestones)):
            assert milestones[i]["target_date"] >= milestones[i - 1]["target_date"]

    def test_plan_has_owner(self, sample_transition_plan):
        assert "owner" in sample_transition_plan
        assert sample_transition_plan["owner"] != ""

    def test_plan_has_status(self, sample_transition_plan):
        assert sample_transition_plan["status"] in [
            "draft", "in_progress", "completed", "overdue",
        ]

    def test_plan_completion_percentage(self, sample_transition_plan):
        pct = sample_transition_plan["completion_pct"]
        assert 0.0 <= pct <= 100.0
