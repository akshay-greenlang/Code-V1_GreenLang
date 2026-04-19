# -*- coding: utf-8 -*-
"""
Test suite for PACK-028 Sector Pathway Pack - Scenario Comparison Engine.

Tests multi-scenario comparison, pairwise deltas, risk-return analysis,
investment analysis, and optimal pathway recommendation.

Author:  GreenLang Test Engineering
Pack:    PACK-028 Sector Pathway Pack
Engine:  8 of 8 - scenario_comparison_engine.py
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.scenario_comparison_engine import (
    ScenarioComparisonEngine,
    ComparisonInput,
    ComparisonResult,
    ScenarioPathwayData,
    ScenarioSummary,
    ScenarioPairDelta,
    ScenarioRiskReturn,
    OptimalPathwayRecommendation,
    InvestmentAnalysis,
    ScenarioId,
)

from .conftest import (
    assert_decimal_close,
    assert_provenance_hash,
    assert_processing_time,
    INTENSITY_METRICS,
    SDA_SECTORS,
    timed_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pathway(scenario, target_2050=Decimal("0"), reduction=Decimal("5.0"),
                  capex=None, target_2030=None):
    kw = dict(
        scenario=ScenarioId(scenario),
        target_intensity_2050=target_2050,
        annual_reduction_rate_pct=reduction,
    )
    if capex is not None:
        kw["estimated_capex_eur"] = capex
    if target_2030 is not None:
        kw["target_intensity_2030"] = target_2030
    return ScenarioPathwayData(**kw)


def _make_input(sector="power_generation", entity_name="TestCo",
                pathways=None, include_risk=True, include_investment=True,
                include_optimal=True, current_intensity=None,
                annual_revenue_eur=None, carbon_price_exposure=None):
    if pathways is None:
        pathways = [
            _make_pathway("nze", Decimal("0"), Decimal("7.0")),
            _make_pathway("wb2c", Decimal("30"), Decimal("5.0")),
            _make_pathway("aps", Decimal("80"), Decimal("3.0")),
        ]
    kw = dict(
        entity_name=entity_name,
        sector=sector,
        scenario_pathways=pathways,
        include_risk_assessment=include_risk,
        include_investment_analysis=include_investment,
        include_optimal_recommendation=include_optimal,
    )
    if current_intensity is not None:
        kw["current_intensity"] = current_intensity
    if annual_revenue_eur is not None:
        kw["annual_revenue_eur"] = annual_revenue_eur
    if carbon_price_exposure is not None:
        kw["carbon_price_exposure_tco2e"] = carbon_price_exposure
    return ComparisonInput(**kw)


# ===========================================================================
# Engine Instantiation
# ===========================================================================


class TestScenarioCompInstantiation:
    """Engine instantiation tests."""

    def test_engine_instantiates(self):
        engine = ScenarioComparisonEngine()
        assert engine is not None

    def test_engine_has_calculate(self):
        engine = ScenarioComparisonEngine()
        assert hasattr(engine, "calculate")

    def test_engine_version(self):
        engine = ScenarioComparisonEngine()
        assert engine.engine_version == "1.0.0"

    def test_get_scenario_metadata(self):
        engine = ScenarioComparisonEngine()
        metadata = engine.get_scenario_metadata()
        assert len(metadata) >= 5


# ===========================================================================
# Basic Comparison
# ===========================================================================


class TestBasicComparison:
    """Test basic scenario comparison."""

    def test_two_scenario_comparison(self):
        engine = ScenarioComparisonEngine()
        pathways = [
            _make_pathway("nze", Decimal("0"), Decimal("7.0")),
            _make_pathway("aps", Decimal("80"), Decimal("3.0")),
        ]
        result = engine.calculate(_make_input(pathways=pathways))
        assert result.scenarios_compared == 2

    def test_three_scenario_comparison(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input())
        assert result.scenarios_compared == 3

    def test_five_scenario_comparison(self):
        engine = ScenarioComparisonEngine()
        pathways = [
            _make_pathway("nze", Decimal("0"), Decimal("7.0")),
            _make_pathway("wb2c", Decimal("30"), Decimal("5.0")),
            _make_pathway("2c", Decimal("50"), Decimal("4.0")),
            _make_pathway("aps", Decimal("80"), Decimal("3.0")),
            _make_pathway("steps", Decimal("150"), Decimal("1.5")),
        ]
        result = engine.calculate(_make_input(pathways=pathways))
        assert result.scenarios_compared == 5


# ===========================================================================
# Scenario Summaries
# ===========================================================================


class TestScenarioSummaries:
    """Test scenario summary generation."""

    def test_summaries_match_count(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input())
        assert len(result.scenario_summaries) == 3

    def test_summary_structure(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input())
        summary = result.scenario_summaries[0]
        assert isinstance(summary, ScenarioSummary)
        assert hasattr(summary, "scenario")


# ===========================================================================
# Pairwise Deltas
# ===========================================================================


class TestPairwiseDeltas:
    """Test pairwise scenario deltas."""

    def test_pairwise_deltas_exist(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input())
        assert len(result.pairwise_deltas) > 0

    def test_pairwise_count(self):
        """n scenarios => n*(n-1)/2 pairs."""
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input())
        n = result.scenarios_compared
        expected_pairs = n * (n - 1) // 2
        assert len(result.pairwise_deltas) == expected_pairs

    def test_delta_structure(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input())
        delta = result.pairwise_deltas[0]
        assert isinstance(delta, ScenarioPairDelta)


# ===========================================================================
# Risk-Return Analysis
# ===========================================================================


class TestRiskReturn:
    """Test risk-return analysis."""

    def test_risk_return_exists(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input(include_risk=True))
        assert len(result.risk_return) > 0

    def test_risk_return_per_scenario(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input())
        assert len(result.risk_return) == result.scenarios_compared

    def test_risk_return_structure(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input())
        rr = result.risk_return[0]
        assert isinstance(rr, ScenarioRiskReturn)


# ===========================================================================
# Investment Analysis
# ===========================================================================


class TestInvestmentAnalysis:
    """Test investment analysis."""

    def test_investment_analysis_exists(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input(include_investment=True))
        assert result.investment_analysis is not None

    def test_investment_analysis_disabled(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input(include_investment=False))
        assert result.investment_analysis is None

    def test_investment_with_revenue(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input(
            annual_revenue_eur=Decimal("5000000000")))
        assert result.investment_analysis is not None


# ===========================================================================
# Optimal Recommendation
# ===========================================================================


class TestOptimalRecommendation:
    """Test optimal pathway recommendation."""

    def test_optimal_recommendation_exists(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input(include_optimal=True))
        assert result.optimal_recommendation is not None

    def test_optimal_recommendation_disabled(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input(include_optimal=False))
        assert result.optimal_recommendation is None

    def test_optimal_has_scenario(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input())
        if result.optimal_recommendation:
            assert hasattr(result.optimal_recommendation, "recommended_scenario")


# ===========================================================================
# Sector Parametrized Tests
# ===========================================================================


class TestSectorComparisons:
    """Test comparisons across sectors."""

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_sector_comparison(self, sector):
        engine = ScenarioComparisonEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        result = engine.calculate(_make_input(
            sector=sector,
            current_intensity=metrics["base_2020"],
        ))
        assert result.scenarios_compared >= 2


# ===========================================================================
# Result Structure & Provenance
# ===========================================================================


class TestScenarioCompResultStructure:
    """Test result structure and provenance."""

    def test_result_provenance(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input())
        assert_provenance_hash(result)

    def test_result_processing_time(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input())
        assert_processing_time(result)

    def test_result_entity_name(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input(entity_name="ScenCo"))
        assert result.entity_name == "ScenCo"

    def test_result_has_recommendations(self):
        engine = ScenarioComparisonEngine()
        result = engine.calculate(_make_input())
        assert isinstance(result.recommendations, list)

    def test_result_deterministic(self):
        engine = ScenarioComparisonEngine()
        inp = _make_input()
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.scenarios_compared == r2.scenarios_compared


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestScenarioCompEdgeCases:
    """Edge case tests."""

    def test_minimum_two_scenarios(self):
        engine = ScenarioComparisonEngine()
        pathways = [
            _make_pathway("nze", Decimal("0"), Decimal("7.0")),
            _make_pathway("steps", Decimal("150"), Decimal("1.5")),
        ]
        result = engine.calculate(_make_input(pathways=pathways))
        assert result.scenarios_compared == 2

    def test_with_capex_estimates(self):
        engine = ScenarioComparisonEngine()
        pathways = [
            _make_pathway("nze", Decimal("0"), Decimal("7.0"), capex=Decimal("5000000000")),
            _make_pathway("aps", Decimal("80"), Decimal("3.0"), capex=Decimal("2000000000")),
        ]
        result = engine.calculate(_make_input(pathways=pathways))
        assert result.scenarios_compared == 2


# ===========================================================================
# Performance Tests
# ===========================================================================


class TestScenarioCompPerformance:
    """Performance tests."""

    def test_single_comparison_under_100ms(self):
        engine = ScenarioComparisonEngine()
        with timed_block("single_scenario_comp", max_seconds=0.1):
            engine.calculate(_make_input())

    def test_50_comparisons_under_3s(self):
        engine = ScenarioComparisonEngine()
        with timed_block("50_scenario_comps", max_seconds=3.0):
            for _ in range(50):
                engine.calculate(_make_input())
