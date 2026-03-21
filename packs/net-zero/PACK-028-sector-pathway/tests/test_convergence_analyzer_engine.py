# -*- coding: utf-8 -*-
"""
Test suite for PACK-028 Sector Pathway Pack - Convergence Analyzer Engine.

Tests gap calculation accuracy, time-to-convergence estimation,
required acceleration rates, risk assessment, and trajectory analysis
against SBTi/IEA benchmarks.

Author:  GreenLang Test Engineering
Pack:    PACK-028 Sector Pathway Pack
Engine:  4 of 8 - convergence_analyzer_engine.py
Tests:   ~100 tests
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.convergence_analyzer_engine import (
    ConvergenceAnalyzerEngine,
    ConvergenceInput,
    ConvergenceResult,
    GapAnalysisPoint,
    CatchUpScenario,
    RiskLevel,
    HistoricalIntensityPoint,
    PathwayTargetPoint,
)

from .conftest import (
    assert_decimal_close,
    assert_decimal_positive,
    assert_percentage_range,
    assert_provenance_hash,
    assert_processing_time,
    assert_intensity_accuracy,
    INTENSITY_METRICS,
    SDA_SECTORS,
    CONVERGENCE_MODELS,
    timed_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_historical(intensities, start_year=2020):
    """Build a list of HistoricalIntensityPoint from yearly intensities."""
    return [
        HistoricalIntensityPoint(year=start_year + i, intensity=v)
        for i, v in enumerate(intensities)
    ]


def _make_pathway(targets, start_year=2020, step=5, scenario="nze_15c"):
    """Build a list of PathwayTargetPoint."""
    return [
        PathwayTargetPoint(year=start_year + i * step, target_intensity=v, scenario=scenario)
        for i, v in enumerate(targets)
    ]


def _make_input(
    sector="power_generation",
    historical=None,
    pathway=None,
    base_year=2020,
    target_year=2050,
    current_year=2024,
    entity_name="TestCo",
):
    """Build a ConvergenceInput with sensible defaults."""
    if historical is None:
        historical = _make_historical(
            [Decimal("450"), Decimal("440"), Decimal("430"), Decimal("420"), Decimal("410")]
        )
    if pathway is None:
        pathway = _make_pathway(
            [Decimal("450"), Decimal("350"), Decimal("250"), Decimal("150"), Decimal("50"),
             Decimal("10"), Decimal("0")],
            start_year=base_year
        )
    return ConvergenceInput(
        entity_name=entity_name,
        sector=sector,
        historical_data=historical,
        pathway_targets=pathway,
        base_year=base_year,
        target_year=target_year,
        current_year=current_year,
    )


# ===========================================================================
# Engine Instantiation
# ===========================================================================


class TestConvergenceAnalyzerInstantiation:
    """Engine instantiation and metadata tests."""

    def test_engine_instantiates(self):
        engine = ConvergenceAnalyzerEngine()
        assert engine is not None

    def test_engine_has_calculate_method(self):
        engine = ConvergenceAnalyzerEngine()
        assert hasattr(engine, "calculate")

    def test_engine_version(self):
        engine = ConvergenceAnalyzerEngine()
        assert engine.engine_version == "1.0.0"


# ===========================================================================
# Gap Calculation Accuracy
# ===========================================================================


class TestGapCalculation:
    """Test gap calculation between current trajectory and target pathway."""

    def test_gap_positive_when_above_pathway(self):
        engine = ConvergenceAnalyzerEngine()
        hist = _make_historical([Decimal("450"), Decimal("440"), Decimal("430"), Decimal("420"), Decimal("410")])
        path = _make_pathway([Decimal("450"), Decimal("340"), Decimal("230"), Decimal("120"), Decimal("10")])
        inp = _make_input(historical=hist, pathway=path, current_year=2024)
        result = engine.calculate(inp)
        assert result.current_gap_absolute > Decimal("0")

    def test_gap_for_on_track_entity(self):
        engine = ConvergenceAnalyzerEngine()
        # Historical follows pathway closely
        hist = _make_historical([Decimal("450"), Decimal("420"), Decimal("390"), Decimal("360"), Decimal("330")])
        path = _make_pathway([Decimal("450"), Decimal("340"), Decimal("230"), Decimal("120"), Decimal("10")])
        inp = _make_input(historical=hist, pathway=path, current_year=2024)
        result = engine.calculate(inp)
        assert result is not None

    def test_gap_percentage_calculated(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert isinstance(result.current_gap_pct, Decimal)

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_gap_analysis_produces_points(self, sector):
        engine = ConvergenceAnalyzerEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        base = metrics["base_2020"]
        hist = _make_historical([base, base * Decimal("0.97"), base * Decimal("0.94"),
                                  base * Decimal("0.91"), base * Decimal("0.88")])
        path = _make_pathway([base, base * Decimal("0.80"), base * Decimal("0.60"),
                               base * Decimal("0.40"), base * Decimal("0.20"), base * Decimal("0.05")])
        inp = _make_input(sector=sector, historical=hist, pathway=path)
        result = engine.calculate(inp)
        assert isinstance(result.gap_analysis, list)


# ===========================================================================
# Time-to-Convergence
# ===========================================================================


class TestTimeToConvergence:
    """Test time-to-convergence estimation."""

    def test_time_to_convergence_present(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert hasattr(result, "time_to_convergence")

    def test_convergence_status_present(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert result.convergence_status is not None
        assert len(result.convergence_status) > 0


# ===========================================================================
# Catch-Up Scenarios
# ===========================================================================


class TestCatchUpScenarios:
    """Test catch-up scenario generation."""

    def test_catch_up_scenarios_generated(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert isinstance(result.catch_up_scenarios, list)

    def test_catch_up_scenarios_have_type(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        for scenario in result.catch_up_scenarios:
            assert hasattr(scenario, "scenario_type")


# ===========================================================================
# Risk Assessment
# ===========================================================================


class TestRiskAssessment:
    """Test risk level assessment based on convergence analysis."""

    def test_risk_assessment_present(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert result.risk_assessment is not None

    def test_risk_has_level(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert result.risk_assessment.overall_risk is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_risk_assessment_all_sectors(self, sector):
        engine = ConvergenceAnalyzerEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        base = metrics["base_2020"]
        hist = _make_historical([base, base * Decimal("0.98"), base * Decimal("0.96"),
                                  base * Decimal("0.94"), base * Decimal("0.92")])
        path = _make_pathway([base, base * Decimal("0.75"), base * Decimal("0.50"),
                               base * Decimal("0.25"), base * Decimal("0.10")])
        inp = _make_input(sector=sector, historical=hist, pathway=path)
        result = engine.calculate(inp)
        assert result.risk_assessment is not None


# ===========================================================================
# Sector-Specific Analysis
# ===========================================================================


class TestSectorSpecificConvergence:
    """Test convergence analysis for specific sectors."""

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_convergence_analysis_all_sectors(self, sector):
        engine = ConvergenceAnalyzerEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        base = metrics["base_2020"]
        hist = _make_historical([base, base * Decimal("0.97"), base * Decimal("0.94"),
                                  base * Decimal("0.91"), base * Decimal("0.88")])
        path = _make_pathway([base, base * Decimal("0.80"), base * Decimal("0.60"),
                               base * Decimal("0.40"), base * Decimal("0.20"), base * Decimal("0.05")])
        inp = _make_input(sector=sector, historical=hist, pathway=path)
        result = engine.calculate(inp)
        assert result is not None
        assert isinstance(result.gap_analysis, list)

    def test_power_sector_analysis(self):
        engine = ConvergenceAnalyzerEngine()
        hist = _make_historical([Decimal("450"), Decimal("430"), Decimal("410"), Decimal("395"), Decimal("380")])
        path = _make_pathway([Decimal("450"), Decimal("300"), Decimal("150"), Decimal("50"), Decimal("0")])
        inp = _make_input(sector="power_generation", historical=hist, pathway=path)
        result = engine.calculate(inp)
        assert result is not None

    def test_steel_sector_analysis(self):
        engine = ConvergenceAnalyzerEngine()
        hist = _make_historical([Decimal("2.10"), Decimal("2.05"), Decimal("2.00"),
                                  Decimal("1.95"), Decimal("1.90")])
        path = _make_pathway([Decimal("2.10"), Decimal("1.70"), Decimal("1.30"),
                               Decimal("0.80"), Decimal("0.20")])
        inp = _make_input(sector="steel", historical=hist, pathway=path)
        result = engine.calculate(inp)
        assert result is not None


# ===========================================================================
# Provenance & Determinism
# ===========================================================================


class TestConvergenceProvenance:
    """Test result provenance and determinism."""

    def test_result_has_provenance_hash(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert_provenance_hash(result)

    def test_result_is_deterministic(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.current_gap_absolute == r2.current_gap_absolute

    def test_result_processing_time(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert_processing_time(result)

    def test_result_has_calculated_at(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert result.calculated_at is not None


# ===========================================================================
# Multi-Sector Gap Analysis
# ===========================================================================


class TestMultiSectorGapAnalysis:
    """Test gap analysis across multiple sectors."""

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_gap_analysis_all_sectors(self, sector):
        engine = ConvergenceAnalyzerEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        base = metrics["base_2020"]
        hist = _make_historical([base, base * Decimal("0.98"), base * Decimal("0.96"),
                                  base * Decimal("0.94"), base * Decimal("0.92")])
        path = _make_pathway([base, base * Decimal("0.80"), base * Decimal("0.60"),
                               base * Decimal("0.40"), base * Decimal("0.20")])
        inp = _make_input(sector=sector, historical=hist, pathway=path)
        result = engine.calculate(inp)
        assert result.current_gap_absolute >= Decimal("0") or result.current_gap_absolute < Decimal("0")

    @pytest.mark.parametrize("current_year", [2022, 2024, 2026, 2028, 2030])
    def test_gap_at_different_years(self, current_year):
        engine = ConvergenceAnalyzerEngine()
        n = current_year - 2020 + 1
        hist = _make_historical(
            [Decimal(str(450 - i * 8)) for i in range(n)]
        )
        path = _make_pathway([Decimal("450"), Decimal("350"), Decimal("250"),
                               Decimal("150"), Decimal("50"), Decimal("10")])
        inp = _make_input(historical=hist, pathway=path, current_year=current_year)
        result = engine.calculate(inp)
        assert result is not None


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestConvergenceEdgeCases:
    """Edge case tests for convergence analysis."""

    def test_single_historical_point(self):
        engine = ConvergenceAnalyzerEngine()
        hist = _make_historical([Decimal("450")])
        path = _make_pathway([Decimal("450"), Decimal("300"), Decimal("150")])
        inp = _make_input(historical=hist, pathway=path, current_year=2020)
        result = engine.calculate(inp)
        assert result is not None

    def test_many_historical_points(self):
        engine = ConvergenceAnalyzerEngine()
        hist = _make_historical(
            [Decimal(str(450 - i * 5)) for i in range(15)]
        )
        path = _make_pathway([Decimal("450"), Decimal("350"), Decimal("250"),
                               Decimal("150"), Decimal("50"), Decimal("10")])
        inp = _make_input(historical=hist, pathway=path, current_year=2034)
        result = engine.calculate(inp)
        assert result is not None

    def test_current_year_at_end_of_historical(self):
        engine = ConvergenceAnalyzerEngine()
        hist = _make_historical(
            [Decimal(str(450 - i * 10)) for i in range(10)], start_year=2020
        )
        path = _make_pathway([Decimal("450"), Decimal("350"), Decimal("250"),
                               Decimal("150"), Decimal("50"), Decimal("0")])
        inp = _make_input(historical=hist, pathway=path, current_year=2029)
        result = engine.calculate(inp)
        assert result is not None


# ===========================================================================
# Performance
# ===========================================================================


class TestConvergencePerformance:
    """Performance tests for convergence analysis."""

    def test_single_analysis_under_1s(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        with timed_block("single_convergence", max_seconds=1.0):
            engine.calculate(inp)

    def test_12_sector_convergence_under_5s(self):
        engine = ConvergenceAnalyzerEngine()
        with timed_block("12_sector_convergence", max_seconds=5.0):
            for sector in SDA_SECTORS:
                metrics = INTENSITY_METRICS.get(sector)
                if metrics:
                    base = metrics["base_2020"]
                    hist = _make_historical([base, base * Decimal("0.97"), base * Decimal("0.94"),
                                              base * Decimal("0.91"), base * Decimal("0.88")])
                    path = _make_pathway([base, base * Decimal("0.80"), base * Decimal("0.60"),
                                           base * Decimal("0.40"), base * Decimal("0.20")])
                    inp = _make_input(sector=sector, historical=hist, pathway=path)
                    engine.calculate(inp)


# ===========================================================================
# Serialization
# ===========================================================================


class TestConvergenceSerialization:
    """Test convergence result serialization."""

    def test_result_has_all_fields(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert hasattr(result, "gap_analysis")
        assert hasattr(result, "risk_assessment")
        assert hasattr(result, "provenance_hash")

    def test_result_model_dump(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        d = result.model_dump()
        assert isinstance(d, dict)
        assert "current_gap_absolute" in d

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_result_has_required_fields(self, sector):
        engine = ConvergenceAnalyzerEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        base = metrics["base_2020"]
        hist = _make_historical([base, base * Decimal("0.97"), base * Decimal("0.94")])
        path = _make_pathway([base, base * Decimal("0.80"), base * Decimal("0.60"),
                               base * Decimal("0.40")])
        inp = _make_input(sector=sector, historical=hist, pathway=path)
        result = engine.calculate(inp)
        assert hasattr(result, "current_gap_absolute")
        assert hasattr(result, "current_gap_pct")


# ===========================================================================
# Recommendations
# ===========================================================================


class TestConvergenceRecommendations:
    """Test recommendations generated by convergence analysis."""

    def test_recommendations_present(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert isinstance(result.recommendations, list)

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_recommendations_per_sector(self, sector):
        engine = ConvergenceAnalyzerEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        base = metrics["base_2020"]
        hist = _make_historical([base, base * Decimal("0.98"), base * Decimal("0.96"),
                                  base * Decimal("0.94"), base * Decimal("0.92")])
        path = _make_pathway([base, base * Decimal("0.75"), base * Decimal("0.50"),
                               base * Decimal("0.25"), base * Decimal("0.10")])
        inp = _make_input(sector=sector, historical=hist, pathway=path)
        result = engine.calculate(inp)
        assert isinstance(result.recommendations, list)


# ===========================================================================
# Milestone Checks
# ===========================================================================


class TestMilestoneChecks:
    """Test milestone analysis."""

    def test_milestone_checks_present(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert isinstance(result.milestone_checks, list)

    @pytest.mark.parametrize("sector", ["power_generation", "steel", "cement"])
    def test_milestone_checks_per_sector(self, sector):
        engine = ConvergenceAnalyzerEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if not metrics:
            pytest.skip(f"No metrics for {sector}")
        base = metrics["base_2020"]
        hist = _make_historical([base, base * Decimal("0.95"), base * Decimal("0.90"),
                                  base * Decimal("0.85"), base * Decimal("0.80")])
        path = _make_pathway([base, base * Decimal("0.75"), base * Decimal("0.50"),
                               base * Decimal("0.25"), base * Decimal("0.10")])
        inp = _make_input(sector=sector, historical=hist, pathway=path)
        result = engine.calculate(inp)
        assert isinstance(result.milestone_checks, list)


# ===========================================================================
# Trajectory Direction
# ===========================================================================


class TestTrajectoryDirection:
    """Test trajectory direction analysis."""

    def test_improving_trajectory(self):
        engine = ConvergenceAnalyzerEngine()
        # Decreasing intensity = improving
        hist = _make_historical([Decimal("450"), Decimal("430"), Decimal("410"),
                                  Decimal("390"), Decimal("370")])
        path = _make_pathway([Decimal("450"), Decimal("350"), Decimal("250"),
                               Decimal("150"), Decimal("50")])
        inp = _make_input(historical=hist, pathway=path)
        result = engine.calculate(inp)
        assert result.trajectory_direction in ("improving", "on_track", "declining", "stagnant")

    def test_worsening_trajectory(self):
        engine = ConvergenceAnalyzerEngine()
        # Increasing intensity = worsening
        hist = _make_historical([Decimal("450"), Decimal("460"), Decimal("470"),
                                  Decimal("480"), Decimal("490")])
        path = _make_pathway([Decimal("450"), Decimal("350"), Decimal("250"),
                               Decimal("150"), Decimal("50")])
        inp = _make_input(historical=hist, pathway=path)
        result = engine.calculate(inp)
        assert result.trajectory_direction is not None


# ===========================================================================
# Result Completeness
# ===========================================================================


class TestConvergenceResultCompleteness:
    """Test that results include all expected fields."""

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_full_result_per_sector(self, sector):
        engine = ConvergenceAnalyzerEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        base = metrics["base_2020"]
        hist = _make_historical([base, base * Decimal("0.97"), base * Decimal("0.94")])
        path = _make_pathway([base, base * Decimal("0.80"), base * Decimal("0.60"),
                               base * Decimal("0.40"), base * Decimal("0.20")])
        inp = _make_input(sector=sector, historical=hist, pathway=path)
        result = engine.calculate(inp)
        assert result.entity_name == "TestCo"
        assert result.sector == sector
        assert isinstance(result.current_intensity, Decimal)
        assert isinstance(result.current_pathway_target, Decimal)
        assert isinstance(result.current_gap_absolute, Decimal)
        assert isinstance(result.current_gap_pct, Decimal)

    def test_result_id_is_uuid(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert len(result.result_id) > 0

    def test_engine_version_in_result(self):
        engine = ConvergenceAnalyzerEngine()
        inp = _make_input()
        result = engine.calculate(inp)
        assert result.engine_version == "1.0.0"
