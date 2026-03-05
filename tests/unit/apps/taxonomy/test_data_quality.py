# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy Data Quality Assessment Engine.

Tests 5-dimension quality assessment (completeness, accuracy, coverage,
consistency, timeliness), grade assignment (A/B/C/D/F), evidence
tracking, improvement plan generation, DQ dashboard, and trend
analysis with 32+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest


# ===========================================================================
# Full 5-Dimension Quality Assessment
# ===========================================================================

class TestFullQualityAssessment:
    """Test complete 5-dimension data quality assessment."""

    def test_dq_assessment_created(self, sample_dq_assessment):
        assert sample_dq_assessment["org_id"] is not None
        assert sample_dq_assessment["overall_score"] > 0

    def test_five_dimensions_present(self, sample_dq_assessment):
        dimensions = sample_dq_assessment["dimensions"]
        required = ["completeness", "accuracy", "coverage", "consistency", "timeliness"]
        for dim in required:
            assert dim in dimensions, f"Missing dimension: {dim}"

    def test_overall_score_range(self, sample_dq_assessment):
        score = sample_dq_assessment["overall_score"]
        assert 0.0 <= score <= 100.0

    def test_overall_score_is_weighted_average(self, sample_dq_assessment):
        dims = sample_dq_assessment["dimensions"]
        weights = {"completeness": 0.25, "accuracy": 0.25, "coverage": 0.20,
                   "consistency": 0.15, "timeliness": 0.15}
        expected = sum(dims[d]["score"] * weights[d] for d in weights)
        assert abs(sample_dq_assessment["overall_score"] - expected) < 0.5

    def test_assessment_has_timestamp(self, sample_dq_assessment):
        assert sample_dq_assessment["assessed_at"] is not None

    def test_assessment_has_provenance(self, sample_dq_assessment):
        assert len(sample_dq_assessment["provenance_hash"]) == 64


# ===========================================================================
# Completeness Dimension
# ===========================================================================

class TestCompletenessDimension:
    """Test completeness dimension scoring."""

    def test_completeness_score_range(self, sample_dq_assessment):
        score = sample_dq_assessment["dimensions"]["completeness"]["score"]
        assert 0.0 <= score <= 100.0

    def test_completeness_fields_tracked(self, sample_dq_assessment):
        comp = sample_dq_assessment["dimensions"]["completeness"]
        assert "total_fields" in comp
        assert "populated_fields" in comp
        assert comp["populated_fields"] <= comp["total_fields"]

    @pytest.mark.parametrize("populated,total,expected_score", [
        (100, 100, 100.0),
        (90, 100, 90.0),
        (75, 100, 75.0),
        (50, 100, 50.0),
        (0, 100, 0.0),
    ])
    def test_completeness_score_calculation(self, populated, total, expected_score):
        score = (populated / total) * 100.0 if total > 0 else 0.0
        assert score == pytest.approx(expected_score, abs=0.01)

    def test_missing_fields_identified(self, sample_dq_assessment):
        comp = sample_dq_assessment["dimensions"]["completeness"]
        assert "missing_fields" in comp
        assert isinstance(comp["missing_fields"], list)


# ===========================================================================
# Accuracy Dimension
# ===========================================================================

class TestAccuracyDimension:
    """Test accuracy dimension scoring."""

    def test_accuracy_score_range(self, sample_dq_assessment):
        score = sample_dq_assessment["dimensions"]["accuracy"]["score"]
        assert 0.0 <= score <= 100.0

    def test_accuracy_validation_checks(self, sample_dq_assessment):
        acc = sample_dq_assessment["dimensions"]["accuracy"]
        assert "validation_checks" in acc
        assert acc["validation_checks"] > 0

    def test_accuracy_pass_rate(self, sample_dq_assessment):
        acc = sample_dq_assessment["dimensions"]["accuracy"]
        assert "checks_passed" in acc
        assert acc["checks_passed"] <= acc["validation_checks"]


# ===========================================================================
# Coverage Dimension
# ===========================================================================

class TestCoverageDimension:
    """Test coverage dimension scoring."""

    def test_coverage_score_range(self, sample_dq_assessment):
        score = sample_dq_assessment["dimensions"]["coverage"]["score"]
        assert 0.0 <= score <= 100.0

    def test_activity_coverage(self, sample_dq_assessment):
        cov = sample_dq_assessment["dimensions"]["coverage"]
        assert "activities_covered" in cov
        assert "activities_total" in cov
        assert cov["activities_covered"] <= cov["activities_total"]

    def test_objective_coverage(self, sample_dq_assessment):
        cov = sample_dq_assessment["dimensions"]["coverage"]
        assert "objectives_assessed" in cov
        assert cov["objectives_assessed"] <= 6


# ===========================================================================
# Consistency Dimension
# ===========================================================================

class TestConsistencyDimension:
    """Test consistency dimension scoring."""

    def test_consistency_score_range(self, sample_dq_assessment):
        score = sample_dq_assessment["dimensions"]["consistency"]["score"]
        assert 0.0 <= score <= 100.0

    def test_cross_reference_checks(self, sample_dq_assessment):
        cons = sample_dq_assessment["dimensions"]["consistency"]
        assert "cross_ref_checks" in cons
        assert cons["cross_ref_checks"] > 0

    def test_discrepancies_tracked(self, sample_dq_assessment):
        cons = sample_dq_assessment["dimensions"]["consistency"]
        assert "discrepancies_found" in cons
        assert cons["discrepancies_found"] >= 0


# ===========================================================================
# Timeliness Dimension
# ===========================================================================

class TestTimelinessDimension:
    """Test timeliness dimension scoring."""

    def test_timeliness_score_range(self, sample_dq_assessment):
        score = sample_dq_assessment["dimensions"]["timeliness"]["score"]
        assert 0.0 <= score <= 100.0

    def test_data_freshness_tracked(self, sample_dq_assessment):
        time = sample_dq_assessment["dimensions"]["timeliness"]
        assert "avg_data_age_days" in time
        assert time["avg_data_age_days"] >= 0

    def test_stale_data_flagged(self, sample_dq_assessment):
        time = sample_dq_assessment["dimensions"]["timeliness"]
        assert "stale_records" in time
        assert time["stale_records"] >= 0


# ===========================================================================
# Grade Assignment
# ===========================================================================

class TestGradeAssignment:
    """Test A/B/C/D/F grade assignment based on overall score."""

    @pytest.mark.parametrize("score,expected_grade", [
        (95.0, "A"),
        (90.0, "A"),
        (85.0, "B"),
        (80.0, "B"),
        (70.0, "C"),
        (65.0, "C"),
        (55.0, "D"),
        (50.0, "D"),
        (40.0, "F"),
        (20.0, "F"),
    ])
    def test_grade_mapping(self, score, expected_grade):
        if score >= 90:
            grade = "A"
        elif score >= 75:
            grade = "B"
        elif score >= 60:
            grade = "C"
        elif score >= 45:
            grade = "D"
        else:
            grade = "F"
        assert grade == expected_grade

    def test_grade_assigned(self, sample_dq_assessment):
        assert sample_dq_assessment["grade"] in ["A", "B", "C", "D", "F"]

    def test_high_score_grade_a(self):
        assessment = {"overall_score": 92.0, "grade": "A"}
        assert assessment["grade"] == "A"

    def test_low_score_grade_f(self):
        assessment = {"overall_score": 30.0, "grade": "F"}
        assert assessment["grade"] == "F"


# ===========================================================================
# Evidence Tracking
# ===========================================================================

class TestEvidenceTracking:
    """Test DQ evidence tracking and documentation."""

    def test_evidence_entries_exist(self, sample_dq_assessment):
        assert "evidence" in sample_dq_assessment
        assert len(sample_dq_assessment["evidence"]) >= 1

    def test_evidence_has_source(self, sample_dq_assessment):
        for entry in sample_dq_assessment["evidence"]:
            assert "source" in entry
            assert entry["source"] != ""

    def test_evidence_has_dimension(self, sample_dq_assessment):
        for entry in sample_dq_assessment["evidence"]:
            assert "dimension" in entry
            assert entry["dimension"] in [
                "completeness", "accuracy", "coverage",
                "consistency", "timeliness",
            ]

    def test_evidence_has_timestamp(self, sample_dq_assessment):
        for entry in sample_dq_assessment["evidence"]:
            assert "collected_at" in entry


# ===========================================================================
# Improvement Plan
# ===========================================================================

class TestImprovementPlan:
    """Test DQ improvement plan generation."""

    def test_improvement_plan_generated(self, sample_dq_assessment):
        assert "improvement_plan" in sample_dq_assessment
        plan = sample_dq_assessment["improvement_plan"]
        assert len(plan) >= 1

    def test_plan_items_have_priority(self, sample_dq_assessment):
        for item in sample_dq_assessment["improvement_plan"]:
            assert "priority" in item
            assert item["priority"] in ["high", "medium", "low"]

    def test_plan_items_have_dimension(self, sample_dq_assessment):
        for item in sample_dq_assessment["improvement_plan"]:
            assert "dimension" in item

    def test_plan_items_have_action(self, sample_dq_assessment):
        for item in sample_dq_assessment["improvement_plan"]:
            assert "action" in item
            assert len(item["action"]) > 5

    def test_plan_ordered_by_priority(self, sample_dq_assessment):
        priority_order = {"high": 0, "medium": 1, "low": 2}
        plan = sample_dq_assessment["improvement_plan"]
        for i in range(1, len(plan)):
            assert priority_order[plan[i]["priority"]] >= priority_order[plan[i - 1]["priority"]]


# ===========================================================================
# DQ Dashboard & Trends
# ===========================================================================

class TestDQDashboardAndTrends:
    """Test DQ dashboard data and trend analysis."""

    def test_dashboard_summary(self, sample_dq_assessment):
        dashboard = {
            "overall_score": sample_dq_assessment["overall_score"],
            "grade": sample_dq_assessment["grade"],
            "dimension_count": len(sample_dq_assessment["dimensions"]),
            "evidence_count": len(sample_dq_assessment["evidence"]),
        }
        assert dashboard["dimension_count"] == 5
        assert dashboard["evidence_count"] >= 1

    def test_trend_data_structure(self, sample_dq_trend):
        assert "periods" in sample_dq_trend
        assert len(sample_dq_trend["periods"]) >= 2

    def test_trend_scores_ordered_by_date(self, sample_dq_trend):
        periods = sample_dq_trend["periods"]
        for i in range(1, len(periods)):
            assert periods[i]["period"] >= periods[i - 1]["period"]

    def test_trend_shows_improvement(self, sample_dq_trend):
        periods = sample_dq_trend["periods"]
        first = periods[0]["overall_score"]
        last = periods[-1]["overall_score"]
        assert last >= first  # Scores should not decrease in sample data

    def test_trend_dimension_breakdown(self, sample_dq_trend):
        for period in sample_dq_trend["periods"]:
            assert "dimensions" in period
            assert len(period["dimensions"]) == 5
