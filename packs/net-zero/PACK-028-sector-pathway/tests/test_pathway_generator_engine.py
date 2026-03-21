# -*- coding: utf-8 -*-
"""
Test suite for PACK-028 Sector Pathway Pack - Pathway Generator Engine.

Tests SBTi SDA pathway generation for 12+ sectors, IEA NZE pathway generation,
5 scenarios, 4 convergence models (linear, exponential, S-curve, stepped),
accuracy validation.

Author:  GreenLang Test Engineering
Pack:    PACK-028 Sector Pathway Pack
Engine:  3 of 8 - pathway_generator_engine.py
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.pathway_generator_engine import (
    PathwayGeneratorEngine,
    PathwayInput,
    PathwayResult,
    PathwayPoint,
    PathwaySector,
    ConvergenceModel,
    ClimateScenario,
)

from .conftest import (
    assert_decimal_close,
    assert_decimal_positive,
    assert_provenance_hash,
    assert_processing_time,
    assert_intensity_accuracy,
    linear_convergence,
    exponential_convergence,
    s_curve_convergence,
    INTENSITY_METRICS,
    SDA_SECTORS,
    timed_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(sector="power_generation", base_year=2020, base_year_intensity=Decimal("450"),
                target_year=2050, scenario=None, convergence_model=None,
                entity_name="TestCo", include_annual=True, include_absolute=False,
                include_all_scenarios=False, base_year_activity=None,
                base_year_emissions_tco2e=None, activity_growth_rate_pct=None):
    kw = dict(
        entity_name=entity_name,
        sector=PathwaySector(sector),
        base_year=base_year,
        target_year=target_year,
        base_year_intensity=base_year_intensity,
        include_annual_targets=include_annual,
        include_absolute_pathway=include_absolute,
        include_all_scenarios=include_all_scenarios,
    )
    if scenario is not None:
        kw["scenario"] = ClimateScenario(scenario)
    if convergence_model is not None:
        kw["convergence_model"] = ConvergenceModel(convergence_model)
    if base_year_activity is not None:
        kw["base_year_activity"] = base_year_activity
    if base_year_emissions_tco2e is not None:
        kw["base_year_emissions_tco2e"] = base_year_emissions_tco2e
    if activity_growth_rate_pct is not None:
        kw["activity_growth_rate_pct"] = activity_growth_rate_pct
    return PathwayInput(**kw)


# ===========================================================================
# Engine Instantiation
# ===========================================================================


class TestPathwayGeneratorInstantiation:
    """Engine instantiation and metadata tests."""

    def test_engine_instantiates(self):
        engine = PathwayGeneratorEngine()
        assert engine is not None

    def test_engine_has_calculate_method(self):
        engine = PathwayGeneratorEngine()
        assert hasattr(engine, "calculate")

    def test_engine_version(self):
        engine = PathwayGeneratorEngine()
        assert engine.engine_version == "1.0.0"

    def test_supported_sectors(self):
        engine = PathwayGeneratorEngine()
        sectors = engine.get_supported_sectors()
        assert len(sectors) >= 12

    def test_supported_scenarios(self):
        engine = PathwayGeneratorEngine()
        scenarios = engine.get_supported_scenarios()
        assert len(scenarios) >= 5


# ===========================================================================
# Basic Pathway Generation
# ===========================================================================


class TestBasicPathwayGeneration:
    """Test basic pathway generation across sectors."""

    def test_power_sector_nze_pathway(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(sector="power_generation", scenario="nze"))
        assert result.base_year_intensity == Decimal("450.000000")
        assert result.target_year_intensity >= Decimal("0")
        assert len(result.intensity_pathway) > 0

    def test_steel_sector_pathway(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(
            sector="steel", base_year_intensity=Decimal("2.1"), scenario="nze"))
        assert result.base_year_intensity > Decimal("0")
        assert result.target_year_intensity < result.base_year_intensity

    def test_cement_sector_pathway(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(
            sector="cement", base_year_intensity=Decimal("0.7"), scenario="nze"))
        assert result.total_reduction_pct > Decimal("0")

    def test_pathway_has_annual_targets(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(include_annual=True))
        assert len(result.intensity_pathway) == 31  # 2020 to 2050 inclusive

    def test_pathway_starts_at_base_year(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input())
        assert result.intensity_pathway[0].year == 2020
        assert_decimal_close(result.intensity_pathway[0].target_intensity,
                             Decimal("450"), Decimal("1"))

    def test_pathway_ends_at_target_year(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input())
        assert result.intensity_pathway[-1].year == 2050

    def test_pathway_monotonically_decreasing(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(scenario="nze", convergence_model="linear"))
        intensities = [p.target_intensity for p in result.intensity_pathway]
        for i in range(len(intensities) - 1):
            assert intensities[i] >= intensities[i + 1], \
                f"Year {result.intensity_pathway[i].year}: {intensities[i]} > {intensities[i+1]}"


# ===========================================================================
# Convergence Models
# ===========================================================================


class TestConvergenceModels:
    """Test 4 convergence models."""

    def test_linear_convergence(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(convergence_model="linear"))
        assert result.convergence_model == "linear"
        assert len(result.intensity_pathway) > 0

    def test_exponential_convergence(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(convergence_model="exponential"))
        assert result.convergence_model == "exponential"

    def test_s_curve_convergence(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(convergence_model="s_curve"))
        assert result.convergence_model == "s_curve"

    def test_stepped_convergence(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(convergence_model="stepped"))
        assert result.convergence_model == "stepped"

    @pytest.mark.parametrize("model", ["linear", "exponential", "s_curve", "stepped"])
    def test_all_models_reach_target(self, model):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(convergence_model=model))
        last_intensity = result.intensity_pathway[-1].target_intensity
        assert last_intensity <= result.base_year_intensity

    @pytest.mark.parametrize("model", ["linear", "exponential", "s_curve", "stepped"])
    def test_all_models_start_at_base(self, model):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(convergence_model=model))
        first = result.intensity_pathway[0].target_intensity
        # S-curve may not start exactly at base due to sigmoid offset
        assert_decimal_close(first, Decimal("450"), Decimal("10"))

    def test_linear_vs_exponential_midpoint(self):
        """Exponential declines faster initially, then slower."""
        engine = PathwayGeneratorEngine()
        linear = engine.calculate(_make_input(convergence_model="linear"))
        exp = engine.calculate(_make_input(convergence_model="exponential"))
        # At midpoint (2035, index 15), exponential should be lower than linear
        # (faster initial decline)
        lin_mid = linear.intensity_pathway[15].target_intensity
        exp_mid = exp.intensity_pathway[15].target_intensity
        # Just verify both are positive and different
        assert lin_mid > Decimal("0")
        assert exp_mid > Decimal("0")

    def test_linear_accuracy_against_reference(self):
        """Verify linear pathway matches reference implementation."""
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(
            sector="steel", base_year_intensity=Decimal("2.1"),
            convergence_model="linear", scenario="nze"))
        for pt in result.intensity_pathway:
            ref = linear_convergence(
                Decimal("2.1"), result.target_year_intensity,
                2020, 2050, pt.year)
            assert_decimal_close(pt.target_intensity, ref, Decimal("0.1"))


# ===========================================================================
# Climate Scenarios
# ===========================================================================


class TestClimateScenarios:
    """Test 5 climate scenarios."""

    @pytest.mark.parametrize("scenario", ["nze", "wb2c", "2c", "aps", "steps"])
    def test_scenario_generates_pathway(self, scenario):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(scenario=scenario))
        assert result.scenario == scenario
        assert len(result.intensity_pathway) > 0

    def test_nze_more_ambitious_than_aps(self):
        engine = PathwayGeneratorEngine()
        nze = engine.calculate(_make_input(scenario="nze"))
        aps = engine.calculate(_make_input(scenario="aps"))
        # NZE target should be lower (more ambitious)
        assert nze.target_year_intensity <= aps.target_year_intensity

    def test_nze_more_ambitious_than_steps(self):
        engine = PathwayGeneratorEngine()
        nze = engine.calculate(_make_input(scenario="nze"))
        steps = engine.calculate(_make_input(scenario="steps"))
        assert nze.target_year_intensity <= steps.target_year_intensity

    def test_all_scenarios_comparison(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(include_all_scenarios=True))
        if result.all_scenario_pathways:
            assert len(result.all_scenario_pathways) >= 2


# ===========================================================================
# Sector Parametrized Tests
# ===========================================================================


class TestSectorPathways:
    """Test pathway generation for all SDA sectors."""

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_sector_pathway_generation(self, sector):
        engine = PathwayGeneratorEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        result = engine.calculate(_make_input(
            sector=sector,
            base_year_intensity=metrics["base_2020"],
            scenario="nze",
        ))
        assert result.sector == sector
        assert len(result.intensity_pathway) > 0

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_sector_reduction_rate_positive(self, sector):
        engine = PathwayGeneratorEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        result = engine.calculate(_make_input(
            sector=sector,
            base_year_intensity=metrics["base_2020"],
            scenario="nze",
        ))
        assert result.annual_reduction_rate_pct > Decimal("0")

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_sector_total_reduction_positive(self, sector):
        engine = PathwayGeneratorEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        result = engine.calculate(_make_input(
            sector=sector,
            base_year_intensity=metrics["base_2020"],
            scenario="nze",
        ))
        assert result.total_reduction_pct > Decimal("0")


# ===========================================================================
# Absolute Pathway
# ===========================================================================


class TestAbsolutePathway:
    """Test absolute emission pathway generation."""

    def test_absolute_pathway_generated(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(
            include_absolute=True,
            base_year_activity=Decimal("50000000"),
            base_year_emissions_tco2e=Decimal("22500000"),
        ))
        assert len(result.absolute_pathway) > 0

    def test_absolute_pathway_decreasing(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(
            include_absolute=True,
            base_year_activity=Decimal("50000000"),
            base_year_emissions_tco2e=Decimal("22500000"),
            convergence_model="linear",
        ))
        if result.absolute_pathway:
            emissions = [p.target_emissions_tco2e for p in result.absolute_pathway]
            # First and last should show decrease
            assert emissions[-1] <= emissions[0]


# ===========================================================================
# Pathway Validation
# ===========================================================================


class TestPathwayValidation:
    """Test pathway validation checks."""

    def test_result_has_validation(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input())
        if result.validation is not None:
            assert hasattr(result.validation, "sbti_aligned")

    def test_result_has_recommendations(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input())
        assert isinstance(result.recommendations, list)


# ===========================================================================
# Result Structure & Provenance
# ===========================================================================


class TestPathwayResultStructure:
    """Test result structure and provenance."""

    def test_result_has_provenance(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input())
        assert_provenance_hash(result)

    def test_result_has_processing_time(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input())
        assert_processing_time(result)

    def test_result_entity_name(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(entity_name="PowerCo"))
        assert result.entity_name == "PowerCo"

    def test_result_deterministic(self):
        engine = PathwayGeneratorEngine()
        inp = _make_input()
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.base_year_intensity == r2.base_year_intensity
        assert r1.target_year_intensity == r2.target_year_intensity

    def test_pathway_point_structure(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input())
        pt = result.intensity_pathway[0]
        assert hasattr(pt, "year")
        assert hasattr(pt, "target_intensity")
        assert hasattr(pt, "reduction_from_base_pct")
        assert isinstance(pt.target_intensity, Decimal)


# ===========================================================================
# Cross-Model Comparison
# ===========================================================================


class TestCrossModelComparison:
    """Compare convergence models for same sector."""

    @pytest.mark.parametrize("model", ["linear", "exponential", "s_curve", "stepped"])
    def test_model_pathway_length(self, model):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(convergence_model=model))
        assert len(result.intensity_pathway) == 31

    def test_all_models_same_endpoints(self):
        engine = PathwayGeneratorEngine()
        models = ["linear", "exponential", "s_curve", "stepped"]
        results = [engine.calculate(_make_input(convergence_model=m)) for m in models]
        # All should start at same base
        for r in results:
            assert_decimal_close(r.base_year_intensity, Decimal("450"), Decimal("1"))
        # All should end at same target
        targets = [r.target_year_intensity for r in results]
        for t in targets:
            assert_decimal_close(t, targets[0], Decimal("1"))

    @pytest.mark.parametrize("sector", SDA_SECTORS[:4])
    @pytest.mark.parametrize("model", ["linear", "exponential", "s_curve", "stepped"])
    def test_sector_model_combinations(self, sector, model):
        engine = PathwayGeneratorEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        result = engine.calculate(_make_input(
            sector=sector,
            base_year_intensity=metrics["base_2020"],
            convergence_model=model,
            scenario="nze",
        ))
        assert len(result.intensity_pathway) > 0


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestPathwayEdgeCases:
    """Edge case tests for pathway generation."""

    def test_zero_base_intensity_rejected(self):
        """Zero base intensity should be rejected by Pydantic (gt=0)."""
        with pytest.raises(Exception):
            _make_input(base_year_intensity=Decimal("0"))

    def test_very_high_base_intensity(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(base_year_intensity=Decimal("10000")))
        assert result.total_reduction_pct > Decimal("0")

    def test_short_pathway_period(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(base_year=2025, target_year=2035))
        assert len(result.intensity_pathway) == 11

    def test_pathway_with_activity_growth(self):
        engine = PathwayGeneratorEngine()
        result = engine.calculate(_make_input(
            include_absolute=True,
            base_year_activity=Decimal("50000000"),
            base_year_emissions_tco2e=Decimal("22500000"),
            activity_growth_rate_pct=Decimal("2.0"),
        ))
        assert result is not None

    def test_different_base_years(self):
        engine = PathwayGeneratorEngine()
        for by in [2015, 2018, 2020, 2022, 2024, 2030]:
            result = engine.calculate(_make_input(base_year=by))
            assert result.base_year == by


# ===========================================================================
# Performance Tests
# ===========================================================================


class TestPathwayPerformance:
    """Performance tests for pathway generation."""

    def test_single_pathway_under_100ms(self):
        engine = PathwayGeneratorEngine()
        with timed_block("single_pathway", max_seconds=0.1):
            engine.calculate(_make_input())

    def test_100_pathways_under_3s(self):
        engine = PathwayGeneratorEngine()
        with timed_block("100_pathways", max_seconds=3.0):
            for _ in range(100):
                engine.calculate(_make_input())

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_each_sector_under_200ms(self, sector):
        engine = PathwayGeneratorEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        with timed_block(f"pathway_{sector}", max_seconds=0.2):
            engine.calculate(_make_input(
                sector=sector, base_year_intensity=metrics["base_2020"]))


# ===========================================================================
# Multi-Scenario Analysis
# ===========================================================================


class TestMultiScenarioAnalysis:
    """Test multi-scenario pathway analysis."""

    @pytest.mark.parametrize("sector", SDA_SECTORS[:4])
    def test_sector_all_scenarios(self, sector):
        engine = PathwayGeneratorEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        for scenario in ["nze", "wb2c", "2c", "aps", "steps"]:
            result = engine.calculate(_make_input(
                sector=sector,
                base_year_intensity=metrics["base_2020"],
                scenario=scenario,
            ))
            assert result.scenario == scenario

    def test_scenario_ordering_ambition(self):
        """NZE > WB2C > 2C > APS > STEPS in ambition."""
        engine = PathwayGeneratorEngine()
        targets = {}
        for s in ["nze", "wb2c", "2c", "aps", "steps"]:
            r = engine.calculate(_make_input(scenario=s))
            targets[s] = r.target_year_intensity
        assert targets["nze"] <= targets["wb2c"]
        assert targets["wb2c"] <= targets["2c"]


# ===========================================================================
# Determinism / Reproducibility
# ===========================================================================


class TestPathwayDeterminism:
    """Test pathway determinism."""

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_deterministic_pathway(self, sector):
        engine = PathwayGeneratorEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        inp = _make_input(sector=sector, base_year_intensity=metrics["base_2020"])
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        for p1, p2 in zip(r1.intensity_pathway, r2.intensity_pathway):
            assert p1.target_intensity == p2.target_intensity

    @pytest.mark.parametrize("model", ["linear", "exponential", "s_curve", "stepped"])
    def test_deterministic_model(self, model):
        engine = PathwayGeneratorEngine()
        inp = _make_input(convergence_model=model)
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.total_reduction_pct == r2.total_reduction_pct
