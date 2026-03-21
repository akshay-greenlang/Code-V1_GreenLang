# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - Scenario Modeling Engine.

Tests Monte Carlo pathway analysis with simulation runs across
1.5C, 2C, and BAU scenarios with sensitivity analysis and probability
distributions.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~40 tests
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.scenario_modeling_engine import (
    ScenarioModelingEngine,
    ScenarioModelingInput,
    ScenarioModelingResult,
    ScenarioType,
    ParameterDistribution,
    SensitivityDriver,
    ScenarioTrajectory,
    ClimateRiskScore,
    MACCResult,
    DistributionType,
)

from .conftest import (
    assert_decimal_close, assert_decimal_positive,
    assert_provenance_hash, timed_block,
)


def _default_input(**overrides):
    """Build a ScenarioModelingInput with sensible defaults."""
    defaults = dict(
        base_year_emissions_tco2e=Decimal("867000"),
        scope1_tco2e=Decimal("125000"),
        scope2_tco2e=Decimal("85000"),
        scope3_tco2e=Decimal("657000"),
        mc_runs=100,
        random_seed=42,
    )
    defaults.update(overrides)
    return ScenarioModelingInput(**defaults)


# ===========================================================================
# Tests -- Engine Instantiation
# ===========================================================================


class TestScenarioModelingEngineInstantiation:
    def test_engine_instantiates(self):
        engine = ScenarioModelingEngine()
        assert engine is not None

    def test_engine_has_calculate_method(self):
        engine = ScenarioModelingEngine()
        assert hasattr(engine, "calculate")

    def test_engine_supports_scenarios(self):
        """ScenarioType enum should include the three standard scenarios."""
        members = [m.name for m in ScenarioType]
        assert "AGGRESSIVE_1_5C" in members
        assert "MODERATE_2C" in members
        assert "CONSERVATIVE_BAU" in members


# ===========================================================================
# Tests -- Scenario Configuration
# ===========================================================================


class TestScenarioConfiguration:
    def test_scenario_type_enum_values(self):
        assert ScenarioType.AGGRESSIVE_1_5C.value is not None
        assert ScenarioType.MODERATE_2C.value is not None
        assert ScenarioType.CONSERVATIVE_BAU.value is not None
        assert ScenarioType.CUSTOM.value is not None

    def test_distribution_types(self):
        assert DistributionType.NORMAL.value is not None
        assert DistributionType.LOG_NORMAL.value is not None
        assert DistributionType.TRIANGULAR.value is not None

    def test_parameter_distribution_creation(self):
        dist = ParameterDistribution(
            name="carbon_price",
            distribution=DistributionType.LOG_NORMAL,
            mean=Decimal("100"),
            std=Decimal("30"),
        )
        assert dist.name == "carbon_price"
        assert dist.distribution == DistributionType.LOG_NORMAL


# ===========================================================================
# Tests -- Monte Carlo Simulation
# ===========================================================================


class TestMonteCarloSimulation:
    def test_basic_calculation(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        assert isinstance(result, ScenarioModelingResult)
        assert result.mc_runs_completed > 0

    def test_configurable_run_count(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input(mc_runs=100))
        assert result.mc_runs_completed >= 100

    def test_random_seed_deterministic(self):
        """Same seed must produce same trajectory results."""
        engine = ScenarioModelingEngine()
        r1 = engine.calculate(_default_input(random_seed=42))
        r2 = engine.calculate(_default_input(random_seed=42))
        # With same seed, trajectories should match
        if r1.scenario_trajectories and r2.scenario_trajectories:
            t1 = r1.scenario_trajectories[0]
            t2 = r2.scenario_trajectories[0]
            assert t1.final_year_emissions_p50 == t2.final_year_emissions_p50

    def test_scenario_trajectories_generated(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        assert len(result.scenario_trajectories) > 0

    def test_each_trajectory_has_scenario(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        for traj in result.scenario_trajectories:
            assert traj.scenario != ""

    def test_trajectory_has_points(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        for traj in result.scenario_trajectories:
            assert len(traj.trajectory) > 0

    def test_trajectory_point_percentiles(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        for traj in result.scenario_trajectories:
            for pt in traj.trajectory:
                # p10 <= p25 <= p50 <= p75 <= p90
                assert pt.p10_tco2e <= pt.p50_tco2e
                assert pt.p50_tco2e <= pt.p90_tco2e


# ===========================================================================
# Tests -- Scenario Results
# ===========================================================================


class TestScenarioResults:
    def test_best_scenario_identified(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        assert result.best_scenario != ""

    def test_target_achievement_probability(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        for traj in result.scenario_trajectories:
            assert Decimal("0") <= traj.target_achievement_probability <= Decimal("100")

    def test_carbon_budget_tracked(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        for traj in result.scenario_trajectories:
            assert traj.carbon_budget_consumed_pct >= Decimal("0")

    def test_investment_estimates(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        for traj in result.scenario_trajectories:
            assert traj.total_investment_p50_usd >= Decimal("0")

    def test_climate_risks_generated(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        assert isinstance(result.climate_risks, list)


# ===========================================================================
# Tests -- Sensitivity Analysis
# ===========================================================================


class TestSensitivityAnalysis:
    def test_sensitivity_drivers_generated(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        assert len(result.sensitivity_drivers) > 0

    def test_sensitivity_driver_fields(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        for driver in result.sensitivity_drivers:
            assert driver.parameter != ""
            assert driver.rank >= 0

    def test_sobol_indices_present(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        for driver in result.sensitivity_drivers:
            assert driver.sobol_first_order >= Decimal("0")
            assert driver.sobol_total >= Decimal("0")

    def test_drivers_ranked(self):
        """Sensitivity drivers should be ranked."""
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        if len(result.sensitivity_drivers) > 1:
            ranks = [d.rank for d in result.sensitivity_drivers]
            assert ranks == sorted(ranks)


# ===========================================================================
# Tests -- MACC (Marginal Abatement Cost Curve)
# ===========================================================================


class TestMACCResults:
    def test_macc_generated(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        assert result.macc is not None

    def test_macc_total_abatement(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        assert result.macc.total_abatement_tco2e >= Decimal("0")


# ===========================================================================
# Tests -- Performance & Provenance
# ===========================================================================


class TestScenarioPerformance:
    def test_calculation_completes(self):
        engine = ScenarioModelingEngine()
        with timed_block("100 Monte Carlo runs", max_seconds=60.0):
            result = engine.calculate(_default_input(mc_runs=100))
        assert result.mc_runs_completed >= 100

    def test_provenance_hash(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        assert_provenance_hash(result)

    def test_result_serializable(self):
        """Result must be serializable via model_dump."""
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        assert hasattr(result, "model_dump")
        data = result.model_dump(mode="json")
        assert isinstance(data, dict)

    def test_processing_time(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        assert result.processing_time_ms >= 0

    def test_regulatory_citations(self):
        engine = ScenarioModelingEngine()
        result = engine.calculate(_default_input())
        assert len(result.regulatory_citations) > 0
