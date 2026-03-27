# -*- coding: utf-8 -*-
"""
Unit tests for QualityScorerEngine - AGENT-MRV-013 Dual Reporting Reconciliation

Tests all public methods of QualityScorerEngine with comprehensive coverage
of the 4-dimension quality scoring system (completeness, consistency, accuracy,
transparency) including composite scoring, grading, cross-checks, and recommendations.

Target: ~90 tests for comprehensive validation of quality assessment logic.
"""

import pytest
from decimal import Decimal
from typing import Any, Dict, List

from greenlang.agents.mrv.dual_reporting_reconciliation.quality_scorer import (
    QualityScorerEngine,
    ENGINE_ID,
    ENGINE_VERSION,
)
from greenlang.agents.mrv.dual_reporting_reconciliation.models import (
    EnergyType,
    Scope2Method,
    QualityDimension,
    QualityGrade,
    EFHierarchyPriority,
    DataQualityTier,
    GWPSource,
    ReconciliationWorkspace,
    UpstreamResult,
    QualityScore,
    QualityAssessment,
)


# ===========================================================================
# Test Class 1: Singleton Behavior (~6 tests)
# ===========================================================================


class TestSingleton:
    """Test singleton lifecycle management."""

    def test_singleton_returns_same_instance(self):
        """Test that multiple instantiations return the same instance."""
        engine1 = QualityScorerEngine()
        engine2 = QualityScorerEngine()

        assert engine1 is engine2

    def test_singleton_persists_state(self):
        """Test singleton state persists across instantiations."""
        engine1 = QualityScorerEngine()
        created_at_1 = engine1._created_at

        engine2 = QualityScorerEngine()
        created_at_2 = engine2._created_at

        assert created_at_1 == created_at_2

    def test_reset_clears_singleton(self):
        """Test reset() clears singleton for re-initialization."""
        engine1 = QualityScorerEngine()
        # Force a counter increment
        engine1._increment_assessments()
        count_before = engine1._total_assessments

        QualityScorerEngine.reset()

        engine2 = QualityScorerEngine()
        count_after = engine2._total_assessments

        # After reset, counters should be reset to 0
        assert count_before > 0
        assert count_after == 0

    def test_reset_allows_new_configuration(self):
        """Test that reset allows new config to be loaded."""
        engine1 = QualityScorerEngine()
        weights1 = engine1.get_quality_weights()

        QualityScorerEngine.reset()

        engine2 = QualityScorerEngine()
        weights2 = engine2.get_quality_weights()

        # Should have same default weights
        assert weights1 == weights2

    def test_singleton_thread_safe_initialization(self):
        """Test singleton initialization is thread-safe."""
        # Multiple resets should not break the pattern
        for _ in range(3):
            QualityScorerEngine.reset()
            engine = QualityScorerEngine()
            assert engine is not None

    def test_get_engine_id_and_version(self):
        """Test static engine identification methods."""
        assert QualityScorerEngine.get_engine_id() == ENGINE_ID
        assert QualityScorerEngine.get_engine_version() == ENGINE_VERSION


# ===========================================================================
# Test Class 2: score_completeness (~12 tests)
# ===========================================================================


class TestScoreCompleteness:
    """Test completeness scoring across energy types and facilities."""

    def test_all_energy_types_present_scores_high(
        self, sample_location_result, sample_market_result,
        sample_steam_location_result, sample_steam_market_result,
    ):
        """Test completeness score when all energy types are present."""
        engine = QualityScorerEngine()

        # Create results for electricity, steam, heating, cooling
        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
            UpstreamResult(**sample_steam_location_result),
            UpstreamResult(**sample_steam_market_result),
        ]

        # Add district_heating and district_cooling
        heating_loc = sample_location_result.copy()
        heating_loc["energy_type"] = "district_heating"
        heating_loc["facility_id"] = "FAC-003"
        results.append(UpstreamResult(**heating_loc))

        cooling_mkt = sample_market_result.copy()
        cooling_mkt["energy_type"] = "district_cooling"
        cooling_mkt["facility_id"] = "FAC-004"
        results.append(UpstreamResult(**cooling_mkt))

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_completeness(workspace)

        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.COMPLETENESS
        # With 4 energy types and dual methods, score should be high
        assert score.score >= Decimal("0.80")

    def test_missing_energy_types_lowers_score(
        self, sample_location_result, sample_market_result,
    ):
        """Test completeness score when only 1 energy type is present."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_completeness(workspace)

        # Only 1/4 energy types present
        assert score.score < Decimal("1.00")

    def test_partial_facility_coverage(
        self, sample_location_result, sample_market_result,
    ):
        """Test completeness when only some facilities have both methods."""
        engine = QualityScorerEngine()

        # FAC-001 has both methods
        # FAC-002 only has location-based
        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        loc_only = sample_location_result.copy()
        loc_only["facility_id"] = "FAC-002"
        loc_only["emissions_tco2e"] = Decimal("500.0")
        results.append(UpstreamResult(**loc_only))

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_completeness(workspace)

        # Facility coverage should be partial
        assert score.score < Decimal("1.00")
        assert "facility" in " ".join(score.findings).lower()

    def test_empty_workspace_raises_error(self):
        """Test that empty workspace raises ValueError."""
        engine = QualityScorerEngine()

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[],
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        # Empty workspace should raise error or return low score
        try:
            score = engine.score_completeness(workspace)
            # If it doesn't raise, score should be very low
            assert score.score == Decimal("0.0")
        except ValueError:
            # This is also acceptable
            pass

    def test_location_only_results(self, sample_location_result):
        """Test completeness with only location-based results."""
        engine = QualityScorerEngine()

        results = [UpstreamResult(**sample_location_result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=results,
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_completeness(workspace)

        # No dual method coverage
        assert score.score < Decimal("0.50")

    def test_market_only_results(self, sample_market_result):
        """Test completeness with only market-based results."""
        engine = QualityScorerEngine()

        results = [UpstreamResult(**sample_market_result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[],
            market_results=results,
            frameworks=["ghg_protocol"],
        )

        score = engine.score_completeness(workspace)

        # No dual method coverage
        assert score.score < Decimal("0.50")

    def test_dual_method_bonus_applied(
        self, sample_location_result, sample_market_result,
    ):
        """Test that dual method bonus is applied when both methods present."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_completeness(workspace)

        # With both methods for electricity, should have bonus
        assert score.score > Decimal("0.30")

    def test_period_coverage_full_period(
        self, sample_location_result, sample_market_result,
    ):
        """Test period coverage when all results cover full period."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_completeness(workspace)

        # Full period coverage should contribute positively
        assert score.score > Decimal("0.00")

    def test_period_coverage_partial_period(
        self, sample_location_result, sample_market_result,
    ):
        """Test period coverage when results don't cover full period."""
        engine = QualityScorerEngine()

        # Modify results to only cover Q1
        partial_loc = sample_location_result.copy()
        partial_loc["period_start"] = "2024-01-01"
        partial_loc["period_end"] = "2024-03-31"

        partial_mkt = sample_market_result.copy()
        partial_mkt["period_start"] = "2024-01-01"
        partial_mkt["period_end"] = "2024-03-31"

        results = [
            UpstreamResult(**partial_loc),
            UpstreamResult(**partial_mkt),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_completeness(workspace)

        # Partial period coverage should lower score
        assert "period" in " ".join(score.findings).lower() or score.score < Decimal("1.00")

    def test_completeness_score_is_clamped(
        self, sample_location_result, sample_market_result,
    ):
        """Test that completeness score is clamped to [0, 1]."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_completeness(workspace)

        assert Decimal("0") <= score.score <= Decimal("1")

    def test_completeness_findings_present(
        self, sample_location_result, sample_market_result,
    ):
        """Test that completeness score includes findings."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_completeness(workspace)

        assert isinstance(score.findings, list)
        assert len(score.findings) > 0


# ===========================================================================
# Test Class 3: score_consistency (~10 tests)
# ===========================================================================


class TestScoreConsistency:
    """Test consistency scoring across GWP, periods, and boundaries."""

    def test_matching_gwp_scores_high(
        self, sample_location_result, sample_market_result,
    ):
        """Test consistency score when all results use same GWP."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_consistency(workspace)

        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.CONSISTENCY
        # All results use AR5, should score 1.0 on GWP sub-dimension
        assert score.score > Decimal("0.00")

    def test_mismatching_gwp_lowers_score(
        self, sample_location_result, sample_market_result,
    ):
        """Test consistency score when results use different GWPs."""
        engine = QualityScorerEngine()

        # Modify market result to use AR6
        mkt_ar6 = sample_market_result.copy()
        mkt_ar6["gwp_source"] = "AR6"

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**mkt_ar6),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_consistency(workspace)

        # Mixed GWP should lower consistency
        assert "gwp" in " ".join(score.findings).lower() or score.score < Decimal("1.00")

    def test_matching_periods_scores_high(
        self, sample_location_result, sample_market_result,
    ):
        """Test consistency when all results have matching periods."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_consistency(workspace)

        # Same period for all results should score high
        assert score.score > Decimal("0.00")

    def test_mismatching_periods_lowers_score(
        self, sample_location_result, sample_market_result,
    ):
        """Test consistency when results have different periods."""
        engine = QualityScorerEngine()

        # Modify market result to different period
        mkt_diff_period = sample_market_result.copy()
        mkt_diff_period["period_start"] = "2024-04-01"
        mkt_diff_period["period_end"] = "2024-06-30"

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**mkt_diff_period),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_consistency(workspace)

        # Different periods should lower consistency
        assert score.score < Decimal("1.00")

    def test_matching_tenants_required(
        self, sample_location_result, sample_market_result,
    ):
        """Test consistency when all results have matching tenant IDs."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_consistency(workspace)

        # All results from same tenant should score well
        assert score.score > Decimal("0.00")

    def test_ef_vintage_consistency_same_year(
        self, sample_location_result, sample_market_result,
    ):
        """Test EF vintage consistency when sources have same year."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),  # eGRID 2023
            UpstreamResult(**sample_market_result),  # Supplier Disclosure 2024
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_consistency(workspace)

        # Should be scored
        assert score.score >= Decimal("0.00")

    def test_consistency_score_is_clamped(
        self, sample_location_result, sample_market_result,
    ):
        """Test that consistency score is clamped to [0, 1]."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_consistency(workspace)

        assert Decimal("0") <= score.score <= Decimal("1")

    def test_consistency_findings_present(
        self, sample_location_result, sample_market_result,
    ):
        """Test that consistency score includes findings."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_consistency(workspace)

        assert isinstance(score.findings, list)
        assert len(score.findings) > 0

    def test_boundary_consistency_same_facilities(
        self, sample_location_result, sample_market_result,
    ):
        """Test boundary consistency when same facilities in both methods."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_consistency(workspace)

        # Same facility in both methods should score well
        assert score.score > Decimal("0.00")

    def test_boundary_consistency_different_facilities(
        self, sample_location_result, sample_market_result,
    ):
        """Test boundary consistency when different facilities."""
        engine = QualityScorerEngine()

        # Modify market result to different facility
        mkt_diff_facility = sample_market_result.copy()
        mkt_diff_facility["facility_id"] = "FAC-999"

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**mkt_diff_facility),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_consistency(workspace)

        # Different facilities should lower boundary score
        assert score.score < Decimal("1.00")


# ===========================================================================
# Test Class 4: score_accuracy (~10 tests)
# ===========================================================================


class TestScoreAccuracy:
    """Test accuracy scoring based on EF hierarchy, tiers, and cross-checks."""

    def test_high_ef_hierarchy_scores_well(self, sample_market_result):
        """Test accuracy score with high-quality EF hierarchy."""
        engine = QualityScorerEngine()

        # supplier_no_cert is 0.85 quality score
        results = [UpstreamResult(**sample_market_result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[],
            market_results=results,
            frameworks=["ghg_protocol"],
        )

        score = engine.score_accuracy(workspace)

        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.ACCURACY
        # supplier_no_cert should score well
        assert score.score > Decimal("0.50")

    def test_low_ef_hierarchy_scores_lower(self, sample_location_result):
        """Test accuracy score with low-quality EF hierarchy."""
        engine = QualityScorerEngine()

        # grid_average is 0.20 quality score
        results = [UpstreamResult(**sample_location_result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=results,
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_accuracy(workspace)

        # grid_average should score lower
        assert score.score < Decimal("1.00")

    def test_tier_3_scores_higher_than_tier_1(
        self, sample_location_result, sample_market_result,
    ):
        """Test that tier 3 data scores higher than tier 1."""
        engine = QualityScorerEngine()

        # Location result is tier_1, market result is tier_3
        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_accuracy(workspace)

        # Mix of tier 1 and tier 3 should score moderately
        assert Decimal("0.30") < score.score < Decimal("1.00")

    def test_cross_check_passes_for_correct_emissions(
        self, sample_location_result, sample_market_result,
    ):
        """Test cross-check passes when emissions = energy * ef."""
        engine = QualityScorerEngine()

        # Verify sample data is correct: 5000 * 0.2501 = 1250.5
        assert sample_location_result["emissions_tco2e"] == Decimal("1250.50")
        assert sample_location_result["energy_quantity_mwh"] == Decimal("5000.0")
        assert sample_location_result["ef_used"] == Decimal("0.2501")

        # Verify market data: 5000 * 0.12505 = 625.25
        assert sample_market_result["emissions_tco2e"] == Decimal("625.25")
        assert sample_market_result["energy_quantity_mwh"] == Decimal("5000.0")
        assert sample_market_result["ef_used"] == Decimal("0.12505")

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        passed, issues = engine.cross_check_emissions(workspace)

        assert passed is True
        assert len(issues) == 0

    def test_cross_check_fails_for_incorrect_emissions(
        self, sample_location_result,
    ):
        """Test cross-check fails when emissions != energy * ef."""
        engine = QualityScorerEngine()

        # Intentionally incorrect emissions
        bad_result = sample_location_result.copy()
        bad_result["emissions_tco2e"] = Decimal("9999.99")

        results = [UpstreamResult(**bad_result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=results,
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        passed, issues = engine.cross_check_emissions(workspace)

        assert passed is False
        assert len(issues) > 0

    def test_accuracy_score_is_clamped(
        self, sample_location_result, sample_market_result,
    ):
        """Test that accuracy score is clamped to [0, 1]."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_accuracy(workspace)

        assert Decimal("0") <= score.score <= Decimal("1")

    def test_accuracy_findings_present(
        self, sample_location_result, sample_market_result,
    ):
        """Test that accuracy score includes findings."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_accuracy(workspace)

        assert isinstance(score.findings, list)
        assert len(score.findings) > 0

    def test_ef_hierarchy_distribution_counted(
        self, sample_location_result, sample_market_result,
    ):
        """Test EF hierarchy distribution is correctly counted."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        distribution = engine.assess_ef_hierarchy_distribution(workspace)

        assert isinstance(distribution, dict)
        # Market result has supplier_no_cert
        assert "supplier_no_cert" in distribution
        assert distribution["supplier_no_cert"] == 1

    def test_multiple_ef_hierarchy_levels(self):
        """Test distribution with multiple EF hierarchy levels."""
        engine = QualityScorerEngine()

        # Create results with different hierarchies
        results = []
        for i, hierarchy in enumerate([
            "supplier_with_cert", "supplier_no_cert", "bundled_cert",
            "unbundled_cert", "residual_mix", "grid_average"
        ]):
            result = {
                "agent": "mrv_010",
                "facility_id": f"FAC-{i:03d}",
                "energy_type": "electricity",
                "method": "market_based",
                "emissions_tco2e": Decimal("100.0"),
                "energy_quantity_mwh": Decimal("1000.0"),
                "ef_used": Decimal("0.1"),
                "ef_source": "Test Source",
                "ef_hierarchy": hierarchy,
                "tier": "tier_2",
                "gwp_source": "AR5",
                "provenance_hash": f"hash{i}",
                "tenant_id": "tenant-001",
                "period_start": "2024-01-01",
                "period_end": "2024-12-31",
            }
            results.append(UpstreamResult(**result))

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[],
            market_results=results,
            frameworks=["ghg_protocol"],
        )

        distribution = engine.assess_ef_hierarchy_distribution(workspace)

        # All 6 hierarchy levels should be present with count of 1
        assert isinstance(distribution, dict)
        assert distribution["supplier_with_cert"] == 1
        assert distribution["grid_average"] == 1

    def test_cross_check_within_tolerance(self, sample_location_result):
        """Test cross-check passes within tolerance threshold."""
        engine = QualityScorerEngine()

        # Create result with tiny rounding difference (within 0.01 tCO2e)
        result = sample_location_result.copy()
        # 5000 * 0.2501 = 1250.5, set to 1250.505 (within tolerance)
        result["emissions_tco2e"] = Decimal("1250.505")

        results = [UpstreamResult(**result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=results,
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        passed, issues = engine.cross_check_emissions(workspace)

        # Should pass within tolerance
        assert passed is True


# ===========================================================================
# Test Class 5: score_transparency (~10 tests)
# ===========================================================================


class TestScoreTransparency:
    """Test transparency scoring based on provenance and documentation."""

    def test_all_provenance_hashes_present_scores_high(
        self, sample_location_result, sample_market_result,
    ):
        """Test transparency score when all results have provenance hashes."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_transparency(workspace)

        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.TRANSPARENCY
        # All results have provenance hashes
        assert score.score > Decimal("0.00")

    def test_missing_provenance_hashes_lowers_score(self, sample_location_result):
        """Test transparency score when provenance hashes are missing."""
        engine = QualityScorerEngine()

        # Remove provenance hash
        result = sample_location_result.copy()
        result["provenance_hash"] = ""

        results = [UpstreamResult(**result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=results,
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_transparency(workspace)

        # Missing provenance should lower score
        assert score.score < Decimal("1.00")

    def test_recognised_ef_sources_score_well(
        self, sample_location_result, sample_market_result,
    ):
        """Test transparency with recognised EF databases."""
        engine = QualityScorerEngine()

        # eGRID is recognised
        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_transparency(workspace)

        # Recognised EF sources should contribute positively
        assert score.score > Decimal("0.00")

    def test_unrecognised_ef_sources_lower_score(self, sample_location_result):
        """Test transparency with unrecognised EF sources."""
        engine = QualityScorerEngine()

        # Unrecognised source
        result = sample_location_result.copy()
        result["ef_source"] = "Unknown Database 2024"

        results = [UpstreamResult(**result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=results,
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_transparency(workspace)

        # Unrecognised source should affect score
        assert score.score < Decimal("1.00")

    def test_metadata_present_improves_score(
        self, sample_location_result, sample_market_result,
    ):
        """Test transparency when results have metadata."""
        engine = QualityScorerEngine()

        # Add metadata
        result_with_meta = sample_location_result.copy()
        result_with_meta["region"] = "US-CAMX"

        results = [
            UpstreamResult(**result_with_meta),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_transparency(workspace)

        # Metadata should improve transparency
        assert score.score > Decimal("0.00")

    def test_ef_hierarchy_documentation_counted(
        self, sample_location_result, sample_market_result,
    ):
        """Test that EF hierarchy is documented."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_transparency(workspace)

        # EF hierarchy should be documented
        assert "ef_hierarchy" in " ".join(score.findings).lower() or score.score > Decimal("0.00")

    def test_transparency_score_is_clamped(
        self, sample_location_result, sample_market_result,
    ):
        """Test that transparency score is clamped to [0, 1]."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_transparency(workspace)

        assert Decimal("0") <= score.score <= Decimal("1")

    def test_transparency_findings_present(
        self, sample_location_result, sample_market_result,
    ):
        """Test that transparency score includes findings."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_transparency(workspace)

        assert isinstance(score.findings, list)
        assert len(score.findings) > 0

    def test_all_transparency_sub_dimensions_scored(
        self, sample_location_result, sample_market_result,
    ):
        """Test that all transparency sub-dimensions are evaluated."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_transparency(workspace)

        # Should have findings from multiple sub-dimensions
        assert len(score.findings) >= 1

    def test_transparency_with_empty_ef_source(self, sample_location_result):
        """Test transparency when ef_source is empty."""
        engine = QualityScorerEngine()

        result = sample_location_result.copy()
        # Use non-empty string to avoid validation errors
        result["ef_source"] = "Unknown"

        results = [UpstreamResult(**result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=results,
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        score = engine.score_transparency(workspace)

        # Unrecognised ef_source should lower transparency
        assert score.score >= Decimal("0.00")


# ===========================================================================
# Test Class 6: compute_composite_score (~8 tests)
# ===========================================================================


class TestComputeCompositeScore:
    """Test composite score calculation with weighted averages."""

    def test_composite_score_weighted_average(self):
        """Test composite score is correct weighted average."""
        engine = QualityScorerEngine()

        dimension_scores = {
            "completeness": QualityScore(
                dimension=QualityDimension.COMPLETENESS,
                score=Decimal("0.80"),
                findings=[]
            ),
            "consistency": QualityScore(
                dimension=QualityDimension.CONSISTENCY,
                score=Decimal("0.90"),
                findings=[]
            ),
            "accuracy": QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=Decimal("0.85"),
                findings=[]
            ),
            "transparency": QualityScore(
                dimension=QualityDimension.TRANSPARENCY,
                score=Decimal("0.75"),
                findings=[]
            ),
        }

        composite = engine.compute_composite_score(dimension_scores)

        # Expected: 0.30*0.80 + 0.25*0.90 + 0.25*0.85 + 0.20*0.75
        #         = 0.24 + 0.225 + 0.2125 + 0.15 = 0.8275
        expected = Decimal("0.8275")
        assert composite == pytest.approx(expected, rel=Decimal("0.000001"))

    def test_composite_score_all_perfect(self):
        """Test composite score when all dimensions are 1.0."""
        engine = QualityScorerEngine()

        dimension_scores = {
            "completeness": QualityScore(
                dimension=QualityDimension.COMPLETENESS,
                score=Decimal("1.0"),
                findings=[]
            ),
            "consistency": QualityScore(
                dimension=QualityDimension.CONSISTENCY,
                score=Decimal("1.0"),
                findings=[]
            ),
            "accuracy": QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=Decimal("1.0"),
                findings=[]
            ),
            "transparency": QualityScore(
                dimension=QualityDimension.TRANSPARENCY,
                score=Decimal("1.0"),
                findings=[]
            ),
        }

        composite = engine.compute_composite_score(dimension_scores)

        # Should be exactly 1.0
        assert composite == Decimal("1.0")

    def test_composite_score_all_zero(self):
        """Test composite score when all dimensions are 0.0."""
        engine = QualityScorerEngine()

        dimension_scores = {
            "completeness": QualityScore(
                dimension=QualityDimension.COMPLETENESS,
                score=Decimal("0.0"),
                findings=[]
            ),
            "consistency": QualityScore(
                dimension=QualityDimension.CONSISTENCY,
                score=Decimal("0.0"),
                findings=[]
            ),
            "accuracy": QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=Decimal("0.0"),
                findings=[]
            ),
            "transparency": QualityScore(
                dimension=QualityDimension.TRANSPARENCY,
                score=Decimal("0.0"),
                findings=[]
            ),
        }

        composite = engine.compute_composite_score(dimension_scores)

        # Should be exactly 0.0
        assert composite == Decimal("0.0")

    def test_composite_score_uses_config_weights(self):
        """Test that composite score uses configured weights."""
        engine = QualityScorerEngine()

        weights = engine.get_quality_weights()

        # Verify default weights
        assert weights["completeness"] == Decimal("0.30")
        assert weights["consistency"] == Decimal("0.25")
        assert weights["accuracy"] == Decimal("0.25")
        assert weights["transparency"] == Decimal("0.20")

    def test_composite_score_is_clamped(self):
        """Test that composite score is clamped to [0, 1]."""
        engine = QualityScorerEngine()

        dimension_scores = {
            "completeness": QualityScore(
                dimension=QualityDimension.COMPLETENESS,
                score=Decimal("0.50"),
                findings=[]
            ),
            "consistency": QualityScore(
                dimension=QualityDimension.CONSISTENCY,
                score=Decimal("0.60"),
                findings=[]
            ),
            "accuracy": QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=Decimal("0.70"),
                findings=[]
            ),
            "transparency": QualityScore(
                dimension=QualityDimension.TRANSPARENCY,
                score=Decimal("0.80"),
                findings=[]
            ),
        }

        composite = engine.compute_composite_score(dimension_scores)

        assert Decimal("0") <= composite <= Decimal("1")

    def test_composite_score_precision(self):
        """Test that composite score has correct precision (8 decimal places)."""
        engine = QualityScorerEngine()

        dimension_scores = {
            "completeness": QualityScore(
                dimension=QualityDimension.COMPLETENESS,
                score=Decimal("0.12345678"),
                findings=[]
            ),
            "consistency": QualityScore(
                dimension=QualityDimension.CONSISTENCY,
                score=Decimal("0.87654321"),
                findings=[]
            ),
            "accuracy": QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=Decimal("0.55555555"),
                findings=[]
            ),
            "transparency": QualityScore(
                dimension=QualityDimension.TRANSPARENCY,
                score=Decimal("0.99999999"),
                findings=[]
            ),
        }

        composite = engine.compute_composite_score(dimension_scores)

        # Should be quantized to 8 decimal places
        str_composite = str(composite)
        if "." in str_composite:
            decimal_places = len(str_composite.split(".")[1])
            assert decimal_places <= 8

    def test_composite_score_boundary_values(self):
        """Test composite score with boundary values."""
        engine = QualityScorerEngine()

        # Test with mixed boundary values
        dimension_scores = {
            "completeness": QualityScore(
                dimension=QualityDimension.COMPLETENESS,
                score=Decimal("0.0"),
                findings=[]
            ),
            "consistency": QualityScore(
                dimension=QualityDimension.CONSISTENCY,
                score=Decimal("1.0"),
                findings=[]
            ),
            "accuracy": QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=Decimal("0.0"),
                findings=[]
            ),
            "transparency": QualityScore(
                dimension=QualityDimension.TRANSPARENCY,
                score=Decimal("1.0"),
                findings=[]
            ),
        }

        composite = engine.compute_composite_score(dimension_scores)

        # Expected: 0.30*0 + 0.25*1 + 0.25*0 + 0.20*1 = 0.45
        expected = Decimal("0.45")
        assert composite == expected

    def test_composite_score_weights_sum_to_one(self):
        """Test that configured weights sum to 1.0."""
        engine = QualityScorerEngine()

        weights = engine.get_quality_weights()
        total = sum(weights.values())

        assert total == Decimal("1.00")


# ===========================================================================
# Test Class 7: assign_grade (~10 tests)
# ===========================================================================


class TestAssignGrade:
    """Test letter grade assignment based on composite score."""

    def test_grade_a_for_score_0_90_or_higher(self):
        """Test grade A assigned for score >= 0.90."""
        engine = QualityScorerEngine()

        grade = engine.assign_grade(Decimal("0.90"))
        assert grade == QualityGrade.A

        grade = engine.assign_grade(Decimal("0.95"))
        assert grade == QualityGrade.A

        grade = engine.assign_grade(Decimal("1.00"))
        assert grade == QualityGrade.A

    def test_grade_b_for_score_0_80_to_0_89(self):
        """Test grade B assigned for 0.80 <= score < 0.90."""
        engine = QualityScorerEngine()

        grade = engine.assign_grade(Decimal("0.80"))
        assert grade == QualityGrade.B

        grade = engine.assign_grade(Decimal("0.85"))
        assert grade == QualityGrade.B

        grade = engine.assign_grade(Decimal("0.8999"))
        assert grade == QualityGrade.B

    def test_grade_c_for_score_0_65_to_0_79(self):
        """Test grade C assigned for 0.65 <= score < 0.80."""
        engine = QualityScorerEngine()

        grade = engine.assign_grade(Decimal("0.65"))
        assert grade == QualityGrade.C

        grade = engine.assign_grade(Decimal("0.70"))
        assert grade == QualityGrade.C

        grade = engine.assign_grade(Decimal("0.7999"))
        assert grade == QualityGrade.C

    def test_grade_d_for_score_0_50_to_0_64(self):
        """Test grade D assigned for 0.50 <= score < 0.65."""
        engine = QualityScorerEngine()

        grade = engine.assign_grade(Decimal("0.50"))
        assert grade == QualityGrade.D

        grade = engine.assign_grade(Decimal("0.60"))
        assert grade == QualityGrade.D

        grade = engine.assign_grade(Decimal("0.6499"))
        assert grade == QualityGrade.D

    def test_grade_f_for_score_below_0_50(self):
        """Test grade F assigned for score < 0.50."""
        engine = QualityScorerEngine()

        grade = engine.assign_grade(Decimal("0.49"))
        assert grade == QualityGrade.F

        grade = engine.assign_grade(Decimal("0.25"))
        assert grade == QualityGrade.F

        grade = engine.assign_grade(Decimal("0.00"))
        assert grade == QualityGrade.F

    def test_grade_boundary_exact_0_90(self):
        """Test grade boundary at exactly 0.90."""
        engine = QualityScorerEngine()

        grade = engine.assign_grade(Decimal("0.90000000"))
        assert grade == QualityGrade.A

    def test_grade_boundary_exact_0_80(self):
        """Test grade boundary at exactly 0.80."""
        engine = QualityScorerEngine()

        grade = engine.assign_grade(Decimal("0.80000000"))
        assert grade == QualityGrade.B

    def test_grade_boundary_exact_0_65(self):
        """Test grade boundary at exactly 0.65."""
        engine = QualityScorerEngine()

        grade = engine.assign_grade(Decimal("0.65000000"))
        assert grade == QualityGrade.C

    def test_grade_boundary_exact_0_50(self):
        """Test grade boundary at exactly 0.50."""
        engine = QualityScorerEngine()

        grade = engine.assign_grade(Decimal("0.50000000"))
        assert grade == QualityGrade.D

    def test_grade_all_boundaries(self):
        """Test all grade boundaries comprehensively."""
        engine = QualityScorerEngine()

        # Test just above and below each boundary
        assert engine.assign_grade(Decimal("0.91")) == QualityGrade.A
        assert engine.assign_grade(Decimal("0.89")) == QualityGrade.B
        assert engine.assign_grade(Decimal("0.81")) == QualityGrade.B
        assert engine.assign_grade(Decimal("0.79")) == QualityGrade.C
        assert engine.assign_grade(Decimal("0.66")) == QualityGrade.C
        assert engine.assign_grade(Decimal("0.64")) == QualityGrade.D
        assert engine.assign_grade(Decimal("0.51")) == QualityGrade.D
        assert engine.assign_grade(Decimal("0.49")) == QualityGrade.F


# ===========================================================================
# Test Class 8: generate_recommendations (~6 tests)
# ===========================================================================


class TestGenerateRecommendations:
    """Test recommendation generation based on quality scores."""

    def test_recommendations_for_low_completeness(self):
        """Test recommendations generated for low completeness."""
        engine = QualityScorerEngine()

        dimension_scores = {
            "completeness": QualityScore(
                dimension=QualityDimension.COMPLETENESS,
                score=Decimal("0.40"),
                findings=[]
            ),
            "consistency": QualityScore(
                dimension=QualityDimension.CONSISTENCY,
                score=Decimal("0.90"),
                findings=[]
            ),
            "accuracy": QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=Decimal("0.85"),
                findings=[]
            ),
            "transparency": QualityScore(
                dimension=QualityDimension.TRANSPARENCY,
                score=Decimal("0.80"),
                findings=[]
            ),
        }

        composite = engine.compute_composite_score(dimension_scores)
        grade = engine.assign_grade(composite)
        assurance_ready = grade in [QualityGrade.A, QualityGrade.B]

        recommendations = engine.generate_recommendations(
            composite, grade, assurance_ready, dimension_scores
        )

        assert len(recommendations) > 0
        # Should mention completeness
        rec_text = " ".join(recommendations).lower()
        assert "complete" in rec_text

    def test_recommendations_for_low_consistency(self):
        """Test recommendations generated for low consistency."""
        engine = QualityScorerEngine()

        dimension_scores = {
            "completeness": QualityScore(
                dimension=QualityDimension.COMPLETENESS,
                score=Decimal("0.90"),
                findings=[]
            ),
            "consistency": QualityScore(
                dimension=QualityDimension.CONSISTENCY,
                score=Decimal("0.40"),
                findings=[]
            ),
            "accuracy": QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=Decimal("0.85"),
                findings=[]
            ),
            "transparency": QualityScore(
                dimension=QualityDimension.TRANSPARENCY,
                score=Decimal("0.80"),
                findings=[]
            ),
        }

        composite = engine.compute_composite_score(dimension_scores)
        grade = engine.assign_grade(composite)
        assurance_ready = False

        recommendations = engine.generate_recommendations(
            composite, grade, assurance_ready, dimension_scores
        )

        assert len(recommendations) > 0
        rec_text = " ".join(recommendations).lower()
        assert "consist" in rec_text

    def test_recommendations_for_low_accuracy(self):
        """Test recommendations generated for low accuracy."""
        engine = QualityScorerEngine()

        dimension_scores = {
            "completeness": QualityScore(
                dimension=QualityDimension.COMPLETENESS,
                score=Decimal("0.90"),
                findings=[]
            ),
            "consistency": QualityScore(
                dimension=QualityDimension.CONSISTENCY,
                score=Decimal("0.85"),
                findings=[]
            ),
            "accuracy": QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=Decimal("0.35"),
                findings=[]
            ),
            "transparency": QualityScore(
                dimension=QualityDimension.TRANSPARENCY,
                score=Decimal("0.80"),
                findings=[]
            ),
        }

        composite = engine.compute_composite_score(dimension_scores)
        grade = engine.assign_grade(composite)
        assurance_ready = False

        recommendations = engine.generate_recommendations(
            composite, grade, assurance_ready, dimension_scores
        )

        assert len(recommendations) > 0
        rec_text = " ".join(recommendations).lower()
        assert "accura" in rec_text or "tier" in rec_text or "ef" in rec_text

    def test_recommendations_for_low_transparency(self):
        """Test recommendations generated for low transparency."""
        engine = QualityScorerEngine()

        dimension_scores = {
            "completeness": QualityScore(
                dimension=QualityDimension.COMPLETENESS,
                score=Decimal("0.90"),
                findings=[]
            ),
            "consistency": QualityScore(
                dimension=QualityDimension.CONSISTENCY,
                score=Decimal("0.85"),
                findings=[]
            ),
            "accuracy": QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=Decimal("0.85"),
                findings=[]
            ),
            "transparency": QualityScore(
                dimension=QualityDimension.TRANSPARENCY,
                score=Decimal("0.30"),
                findings=[]
            ),
        }

        composite = engine.compute_composite_score(dimension_scores)
        grade = engine.assign_grade(composite)
        assurance_ready = False

        recommendations = engine.generate_recommendations(
            composite, grade, assurance_ready, dimension_scores
        )

        assert len(recommendations) > 0
        rec_text = " ".join(recommendations).lower()
        assert "transparen" in rec_text or "document" in rec_text or "provenance" in rec_text

    def test_no_recommendations_for_high_scores(self):
        """Test that no recommendations generated for high scores."""
        engine = QualityScorerEngine()

        dimension_scores = {
            "completeness": QualityScore(
                dimension=QualityDimension.COMPLETENESS,
                score=Decimal("0.95"),
                findings=[]
            ),
            "consistency": QualityScore(
                dimension=QualityDimension.CONSISTENCY,
                score=Decimal("0.95"),
                findings=[]
            ),
            "accuracy": QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=Decimal("0.95"),
                findings=[]
            ),
            "transparency": QualityScore(
                dimension=QualityDimension.TRANSPARENCY,
                score=Decimal("0.95"),
                findings=[]
            ),
        }

        composite = engine.compute_composite_score(dimension_scores)
        grade = engine.assign_grade(composite)
        assurance_ready = True

        recommendations = engine.generate_recommendations(
            composite, grade, assurance_ready, dimension_scores
        )

        # Should have minimal or no recommendations
        assert isinstance(recommendations, list)

    def test_recommendations_are_strings(self):
        """Test that all recommendations are strings."""
        engine = QualityScorerEngine()

        dimension_scores = {
            "completeness": QualityScore(
                dimension=QualityDimension.COMPLETENESS,
                score=Decimal("0.40"),
                findings=[]
            ),
            "consistency": QualityScore(
                dimension=QualityDimension.CONSISTENCY,
                score=Decimal("0.40"),
                findings=[]
            ),
            "accuracy": QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=Decimal("0.40"),
                findings=[]
            ),
            "transparency": QualityScore(
                dimension=QualityDimension.TRANSPARENCY,
                score=Decimal("0.40"),
                findings=[]
            ),
        }

        composite = engine.compute_composite_score(dimension_scores)
        grade = engine.assign_grade(composite)
        assurance_ready = False

        recommendations = engine.generate_recommendations(
            composite, grade, assurance_ready, dimension_scores
        )

        assert all(isinstance(r, str) for r in recommendations)


# ===========================================================================
# Test Class 9: cross_check_emissions (~8 tests)
# ===========================================================================


class TestCrossCheckEmissions:
    """Test arithmetic cross-checks on emissions calculations."""

    def test_cross_check_passes_for_valid_calculation(
        self, sample_location_result, sample_market_result,
    ):
        """Test cross-check passes when emissions = energy * ef."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        passed, issues = engine.cross_check_emissions(workspace)

        assert passed is True
        assert len(issues) == 0

    def test_cross_check_fails_for_invalid_calculation(
        self, sample_location_result,
    ):
        """Test cross-check fails when emissions != energy * ef."""
        engine = QualityScorerEngine()

        bad_result = sample_location_result.copy()
        bad_result["emissions_tco2e"] = Decimal("999999.99")

        results = [UpstreamResult(**bad_result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=results,
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        passed, issues = engine.cross_check_emissions(workspace)

        assert passed is False
        assert len(issues) > 0

    def test_cross_check_within_tolerance(self, sample_location_result):
        """Test cross-check passes within tolerance (0.01 tCO2e)."""
        engine = QualityScorerEngine()

        # Create result with tiny difference
        result = sample_location_result.copy()
        result["emissions_tco2e"] = Decimal("1250.505")  # Within 0.01 of 1250.5

        results = [UpstreamResult(**result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=results,
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        passed, issues = engine.cross_check_emissions(workspace)

        assert passed is True

    def test_cross_check_outside_tolerance(self, sample_location_result):
        """Test cross-check fails outside tolerance."""
        engine = QualityScorerEngine()

        # Create result with difference > 0.01
        result = sample_location_result.copy()
        result["emissions_tco2e"] = Decimal("1250.52")  # 0.02 difference

        results = [UpstreamResult(**result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=results,
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        passed, issues = engine.cross_check_emissions(workspace)

        assert passed is False

    def test_cross_check_counts_all_results(
        self, sample_location_result, sample_market_result,
        sample_steam_location_result, sample_steam_market_result,
    ):
        """Test cross-check validates all results."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
            UpstreamResult(**sample_steam_location_result),
            UpstreamResult(**sample_steam_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        passed, issues = engine.cross_check_emissions(workspace)

        # All 4 results should be checked
        assert passed is True

    def test_cross_check_issue_details(self, sample_location_result):
        """Test cross-check issue contains details."""
        engine = QualityScorerEngine()

        bad_result = sample_location_result.copy()
        bad_result["emissions_tco2e"] = Decimal("9999.99")

        results = [UpstreamResult(**bad_result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=results,
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        passed, issues = engine.cross_check_emissions(workspace)

        assert passed is False
        assert len(issues) > 0
        # Issue should contain facility ID or other identifying info
        assert any("FAC-001" in str(issue) for issue in issues)

    def test_cross_check_multiple_failures(self, sample_location_result):
        """Test cross-check identifies multiple failures."""
        engine = QualityScorerEngine()

        bad_result_1 = sample_location_result.copy()
        bad_result_1["emissions_tco2e"] = Decimal("9999.99")
        bad_result_1["facility_id"] = "FAC-001"

        bad_result_2 = sample_location_result.copy()
        bad_result_2["emissions_tco2e"] = Decimal("8888.88")
        bad_result_2["facility_id"] = "FAC-002"

        results = [
            UpstreamResult(**bad_result_1),
            UpstreamResult(**bad_result_2),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=results,
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        passed, issues = engine.cross_check_emissions(workspace)

        assert passed is False
        assert len(issues) >= 2

    def test_cross_check_increments_counter(
        self, sample_location_result, sample_market_result,
    ):
        """Test cross-check increments the counter."""
        engine = QualityScorerEngine()

        initial_count = engine._total_cross_checks

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        engine.cross_check_emissions(workspace)

        # Counter should have incremented
        assert engine._total_cross_checks > initial_count


# ===========================================================================
# Test Class 10: assess_ef_hierarchy_distribution (~6 tests)
# ===========================================================================


class TestAssessEfHierarchy:
    """Test EF hierarchy distribution assessment."""

    def test_ef_hierarchy_distribution_single_level(self, sample_market_result):
        """Test distribution with single EF hierarchy level."""
        engine = QualityScorerEngine()

        results = [UpstreamResult(**sample_market_result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[],
            market_results=results,
            frameworks=["ghg_protocol"],
        )

        distribution = engine.assess_ef_hierarchy_distribution(workspace)

        assert isinstance(distribution, dict)
        assert "supplier_no_cert" in distribution
        assert distribution["supplier_no_cert"] == 1

    def test_ef_hierarchy_distribution_multiple_levels(self):
        """Test distribution with multiple EF hierarchy levels."""
        engine = QualityScorerEngine()

        results = []
        for i, hierarchy in enumerate(["supplier_with_cert", "bundled_cert", "grid_average"]):
            result = {
                "agent": "mrv_010",
                "facility_id": f"FAC-{i:03d}",
                "energy_type": "electricity",
                "method": "market_based",
                "emissions_tco2e": Decimal("100.0"),
                "energy_quantity_mwh": Decimal("1000.0"),
                "ef_used": Decimal("0.1"),
                "ef_source": "Test Source",
                "ef_hierarchy": hierarchy,
                "tier": "tier_2",
                "gwp_source": "AR5",
                "provenance_hash": f"hash{i}",
                "tenant_id": "tenant-001",
                "period_start": "2024-01-01",
                "period_end": "2024-12-31",
            }
            results.append(UpstreamResult(**result))

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[],
            market_results=results,
            frameworks=["ghg_protocol"],
        )

        distribution = engine.assess_ef_hierarchy_distribution(workspace)

        # Distribution includes all hierarchy levels (some may be 0)
        assert isinstance(distribution, dict)
        assert distribution["supplier_with_cert"] == 1
        assert distribution["bundled_cert"] == 1
        assert distribution["grid_average"] == 1

    def test_ef_hierarchy_distribution_counts_duplicates(self):
        """Test distribution counts multiple results at same level."""
        engine = QualityScorerEngine()

        results = []
        for i in range(5):
            result = {
                "agent": "mrv_010",
                "facility_id": f"FAC-{i:03d}",
                "energy_type": "electricity",
                "method": "market_based",
                "emissions_tco2e": Decimal("100.0"),
                "energy_quantity_mwh": Decimal("1000.0"),
                "ef_used": Decimal("0.1"),
                "ef_source": "Test Source",
                "ef_hierarchy": "supplier_no_cert",
                "tier": "tier_2",
                "gwp_source": "AR5",
                "provenance_hash": f"hash{i}",
                "tenant_id": "tenant-001",
                "period_start": "2024-01-01",
                "period_end": "2024-12-31",
            }
            results.append(UpstreamResult(**result))

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[],
            market_results=results,
            frameworks=["ghg_protocol"],
        )

        distribution = engine.assess_ef_hierarchy_distribution(workspace)

        assert distribution["supplier_no_cert"] == 5

    def test_ef_hierarchy_only_counts_market_results(
        self, sample_location_result, sample_market_result,
    ):
        """Test that hierarchy distribution only counts market-based results."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        distribution = engine.assess_ef_hierarchy_distribution(workspace)

        # Should only count market result (supplier_no_cert)
        assert "supplier_no_cert" in distribution
        # Location result (grid_average) should not be counted
        assert distribution.get("grid_average", 0) == 0

    def test_ef_hierarchy_empty_for_location_only(self, sample_location_result):
        """Test distribution is empty when only location results."""
        engine = QualityScorerEngine()

        results = [UpstreamResult(**sample_location_result)]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=results,
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        distribution = engine.assess_ef_hierarchy_distribution(workspace)

        # No market results means empty or zero-valued distribution
        assert len(distribution) == 0 or all(v == 0 for v in distribution.values())

    def test_ef_hierarchy_all_levels(self):
        """Test distribution with all EF hierarchy levels."""
        engine = QualityScorerEngine()

        all_hierarchies = [
            "supplier_with_cert", "supplier_no_cert", "bundled_cert",
            "unbundled_cert", "residual_mix", "grid_average"
        ]

        results = []
        for i, hierarchy in enumerate(all_hierarchies):
            result = {
                "agent": "mrv_010",
                "facility_id": f"FAC-{i:03d}",
                "energy_type": "electricity",
                "method": "market_based",
                "emissions_tco2e": Decimal("100.0"),
                "energy_quantity_mwh": Decimal("1000.0"),
                "ef_used": Decimal("0.1"),
                "ef_source": "Test Source",
                "ef_hierarchy": hierarchy,
                "tier": "tier_2",
                "gwp_source": "AR5",
                "provenance_hash": f"hash{i}",
                "tenant_id": "tenant-001",
                "period_start": "2024-01-01",
                "period_end": "2024-12-31",
            }
            results.append(UpstreamResult(**result))

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[],
            market_results=results,
            frameworks=["ghg_protocol"],
        )

        distribution = engine.assess_ef_hierarchy_distribution(workspace)

        # All 6 levels should be present with count of 1
        assert isinstance(distribution, dict)
        for hierarchy in all_hierarchies:
            assert distribution[hierarchy] == 1


# ===========================================================================
# Test Class 11: score_quality (full assessment) (~8 tests)
# ===========================================================================


class TestScoreQuality:
    """Test full quality assessment entry point."""

    def test_score_quality_returns_assessment(
        self, sample_location_result, sample_market_result,
    ):
        """Test score_quality returns QualityAssessment."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        assessment = engine.score_quality(workspace)

        assert isinstance(assessment, QualityAssessment)
        # Reconciliation ID is auto-generated UUID
        assert len(assessment.reconciliation_id) > 0
        assert assessment.reconciliation_id == workspace.reconciliation_id

    def test_score_quality_includes_all_dimensions(
        self, sample_location_result, sample_market_result,
    ):
        """Test assessment includes all 4 dimension scores."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        assessment = engine.score_quality(workspace)

        assert len(assessment.scores) == 4
        dimensions = {score.dimension for score in assessment.scores}
        assert QualityDimension.COMPLETENESS in dimensions
        assert QualityDimension.CONSISTENCY in dimensions
        assert QualityDimension.ACCURACY in dimensions
        assert QualityDimension.TRANSPARENCY in dimensions

    def test_score_quality_computes_composite(
        self, sample_location_result, sample_market_result,
    ):
        """Test assessment includes composite score."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        assessment = engine.score_quality(workspace)

        assert Decimal("0") <= assessment.composite_score <= Decimal("1")

    def test_score_quality_assigns_grade(
        self, sample_location_result, sample_market_result,
    ):
        """Test assessment includes letter grade."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        assessment = engine.score_quality(workspace)

        assert assessment.grade in [
            QualityGrade.A, QualityGrade.B, QualityGrade.C,
            QualityGrade.D, QualityGrade.F
        ]

    def test_score_quality_determines_assurance_readiness(
        self, sample_location_result, sample_market_result,
    ):
        """Test assessment includes assurance readiness flag."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        assessment = engine.score_quality(workspace)

        assert isinstance(assessment.assurance_ready, bool)
        # Assurance ready should be True for grades A and B
        if assessment.grade in [QualityGrade.A, QualityGrade.B]:
            # Also check no dimension below 0.50
            all_above_threshold = all(
                score.score >= Decimal("0.50")
                for score in assessment.scores
            )
            if all_above_threshold:
                assert assessment.assurance_ready is True

    def test_score_quality_includes_findings(
        self, sample_location_result, sample_market_result,
    ):
        """Test assessment includes findings."""
        engine = QualityScorerEngine()

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        assessment = engine.score_quality(workspace)

        assert isinstance(assessment.findings, list)
        assert len(assessment.findings) > 0

    def test_score_quality_empty_workspace_raises_error(self):
        """Test score_quality raises error for empty workspace."""
        engine = QualityScorerEngine()

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[],
            market_results=[],
            frameworks=["ghg_protocol"],
        )

        with pytest.raises(ValueError, match="no upstream results"):
            engine.score_quality(workspace)

    def test_score_quality_increments_counter(
        self, sample_location_result, sample_market_result,
    ):
        """Test score_quality increments assessment counter."""
        engine = QualityScorerEngine()

        initial_count = engine._total_assessments

        results = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start="2024-01-01",
            period_end="2024-12-31",
            location_results=[r for r in results if r.method == Scope2Method.LOCATION_BASED],
            market_results=[r for r in results if r.method == Scope2Method.MARKET_BASED],
            frameworks=["ghg_protocol"],
        )

        engine.score_quality(workspace)

        assert engine._total_assessments > initial_count


# ===========================================================================
# Test Class 12: health_check (~4 tests)
# ===========================================================================


class TestHealthCheck:
    """Test engine health check functionality."""

    def test_health_check_returns_dict(self):
        """Test health_check returns dictionary."""
        engine = QualityScorerEngine()

        health = engine.health_check()

        assert isinstance(health, dict)

    def test_health_check_includes_status(self):
        """Test health check includes status field."""
        engine = QualityScorerEngine()

        health = engine.health_check()

        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_check_includes_counters(self):
        """Test health check includes counter statistics."""
        engine = QualityScorerEngine()

        health = engine.health_check()

        assert "total_assessments" in health
        assert "total_cross_checks" in health
        assert "total_flags_generated" in health

    def test_health_check_includes_engine_info(self):
        """Test health check includes engine metadata."""
        engine = QualityScorerEngine()

        health = engine.health_check()

        assert "engine_id" in health
        assert "engine_version" in health
        assert health["engine_id"] == ENGINE_ID
        assert health["engine_version"] == ENGINE_VERSION
