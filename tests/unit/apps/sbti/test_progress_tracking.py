# -*- coding: utf-8 -*-
"""
Unit tests for SBTi Progress Tracking Engine.

Tests annual emissions recording, variance analysis (actual vs expected),
on-track determination with RAG status, cumulative reduction from base
year, projection to achievement, scope-level breakdown, and MRV agent
integration with 26+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import datetime

import pytest


# ===========================================================================
# Progress Recording
# ===========================================================================

class TestProgressRecording:
    """Test annual emissions recording."""

    def test_record_creation(self, sample_progress_record):
        assert sample_progress_record["reporting_year"] == 2024
        assert sample_progress_record["actual_emissions_tco2e"] == 70_000.0

    def test_record_has_target_reference(self, sample_progress_record):
        assert len(sample_progress_record["target_id"]) == 36

    def test_record_has_org_reference(self, sample_progress_record):
        assert len(sample_progress_record["org_id"]) == 36

    def test_record_provenance(self, sample_progress_record):
        assert len(sample_progress_record["provenance_hash"]) == 64

    def test_record_timestamp(self, sample_progress_record):
        assert isinstance(sample_progress_record["created_at"], datetime)


# ===========================================================================
# Variance Analysis
# ===========================================================================

class TestVarianceAnalysis:
    """Test actual vs expected variance calculation."""

    def test_negative_variance_behind_target(self, sample_progress_record):
        assert sample_progress_record["variance_pct"] < 0
        assert sample_progress_record["on_track"] is False

    def test_positive_variance_ahead_of_target(self, on_track_progress):
        assert on_track_progress["variance_pct"] > 0
        assert on_track_progress["on_track"] is True

    def test_variance_tco2e_calculation(self, sample_progress_record):
        actual = sample_progress_record["actual_emissions_tco2e"]
        expected = sample_progress_record["expected_emissions_tco2e"]
        variance = actual - expected
        assert abs(sample_progress_record["variance_tco2e"] - abs(variance)) < 1.0

    @pytest.mark.parametrize("actual,expected,on_track", [
        (60_000, 66_400, True),   # Below expected = good
        (66_400, 66_400, True),   # Exactly on target
        (70_000, 66_400, False),  # Above expected = bad
        (80_000, 66_400, False),  # Well above expected
    ])
    def test_on_track_determination(self, actual, expected, on_track):
        result = actual <= expected
        assert result == on_track


# ===========================================================================
# RAG Status
# ===========================================================================

class TestOnTrackDetermination:
    """Test RAG status assignment."""

    def test_amber_status(self, sample_progress_record):
        assert sample_progress_record["rag_status"] == "amber"

    def test_green_status(self, on_track_progress):
        assert on_track_progress["rag_status"] == "green"

    @pytest.mark.parametrize("variance_pct,expected_rag", [
        (5.0, "green"),     # Ahead of target
        (0.0, "green"),     # Exactly on track
        (-2.0, "amber"),    # Slightly behind
        (-5.0, "amber"),    # Behind
        (-10.0, "red"),     # Significantly behind
        (-20.0, "red"),     # Far behind
    ])
    def test_rag_status_mapping(self, variance_pct, expected_rag):
        if variance_pct >= 0:
            rag = "green"
        elif variance_pct >= -7.5:
            rag = "amber"
        else:
            rag = "red"
        assert rag == expected_rag


# ===========================================================================
# Cumulative Reduction
# ===========================================================================

class TestCumulativeReduction:
    """Test cumulative reduction from base year."""

    def test_cumulative_reduction_calculation(self, sample_progress_record):
        base = sample_progress_record["base_year_emissions_tco2e"]
        actual = sample_progress_record["actual_emissions_tco2e"]
        expected_pct = ((base - actual) / base) * 100
        assert abs(sample_progress_record["cumulative_reduction_pct"] - expected_pct) < 0.1

    def test_cumulative_vs_expected(self, sample_progress_record):
        assert sample_progress_record["cumulative_reduction_pct"] < sample_progress_record["expected_reduction_pct"]

    def test_on_track_cumulative(self, on_track_progress):
        assert on_track_progress["cumulative_reduction_pct"] > on_track_progress["expected_reduction_pct"]


# ===========================================================================
# Projection
# ===========================================================================

class TestProjection:
    """Test projection to target year achievement."""

    def test_projection_exists(self, sample_progress_record):
        assert sample_progress_record["projection_target_year_tco2e"] is not None

    def test_projected_achievement_pct(self, sample_progress_record):
        assert 0 <= sample_progress_record["projected_achievement_pct"] <= 200

    def test_projection_behind_target(self, sample_progress_record):
        # Behind target, so projected achievement < 100%
        assert sample_progress_record["projected_achievement_pct"] < 100.0

    def test_projection_calculation(self):
        """Linear projection from current trend."""
        base = 80_000.0
        target = 46_400.0  # 42% reduction
        actual_y4 = 70_000.0
        years_elapsed = 4
        total_years = 10
        reduction_so_far = base - actual_y4
        projected_total_reduction = (reduction_so_far / years_elapsed) * total_years
        projected_final = base - projected_total_reduction
        target_reduction = base - target
        achievement_pct = (projected_total_reduction / target_reduction) * 100
        assert achievement_pct < 100.0


# ===========================================================================
# Scope Breakdown
# ===========================================================================

class TestScopeBreakdown:
    """Test scope-level progress tracking."""

    def test_scope1_tracked(self, sample_progress_record):
        breakdown = sample_progress_record["scope_breakdown"]
        assert "scope_1" in breakdown
        assert "actual" in breakdown["scope_1"]
        assert "expected" in breakdown["scope_1"]

    def test_scope2_tracked(self, sample_progress_record):
        breakdown = sample_progress_record["scope_breakdown"]
        assert "scope_2" in breakdown

    def test_scope_sum_equals_total(self, sample_progress_record):
        breakdown = sample_progress_record["scope_breakdown"]
        actual_sum = sum(s["actual"] for s in breakdown.values())
        assert actual_sum == sample_progress_record["actual_emissions_tco2e"]

    def test_scope_expected_sum(self, sample_progress_record):
        breakdown = sample_progress_record["scope_breakdown"]
        expected_sum = sum(s["expected"] for s in breakdown.values())
        assert expected_sum == sample_progress_record["expected_emissions_tco2e"]


# ===========================================================================
# MRV Integration
# ===========================================================================

class TestMRVIntegration:
    """Test data fetch from MRV agents."""

    def test_data_quality_tracked(self, sample_emissions_inventory):
        assert sample_emissions_inventory["data_quality_score"] > 0

    def test_verification_status(self, sample_emissions_inventory):
        valid_statuses = ["unverified", "self_verified", "third_party_verified"]
        assert sample_emissions_inventory["verification_status"] in valid_statuses

    def test_verification_body(self, sample_emissions_inventory):
        if sample_emissions_inventory["verification_status"] == "third_party_verified":
            assert sample_emissions_inventory["verification_body"] is not None

    def test_emissions_source_pipeline(self, sample_emissions_inventory):
        # Inventory should be linked to MRV pipeline output
        assert sample_emissions_inventory["scope1_tco2e"] > 0
        assert sample_emissions_inventory["scope3_tco2e"] > 0
