# -*- coding: utf-8 -*-
"""
Tests for ScenarioModelingEngine - PACK-022 Engine 1

Multi-scenario Monte Carlo pathway analysis with uncertainty
quantification, sensitivity analysis, and decision matrix scoring.

Coverage targets: 85%+ of ScenarioModelingEngine methods.
"""

import pytest
from decimal import Decimal

from engines.scenario_modeling_engine import (
    ScenarioModelingEngine,
    ScenarioModelingInput,
    ScenarioModelingResult,
    ScenarioOutput,
    YearStatistics,
    SensitivityEntry,
    ScenarioComparison,
    DecisionMatrixEntry,
    ScenarioType,
    UncertaintyLevel,
    ParameterType,
    SimulationStatus,
    CustomScenarioConfig,
    ScenarioParameterOverride,
    DEFAULT_SCENARIO_PARAMS,
    UNCERTAINTY_MULTIPLIERS,
    DEFAULT_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a ScenarioModelingEngine instance."""
    return ScenarioModelingEngine()


@pytest.fixture
def basic_input():
    """Create a basic ScenarioModelingInput with all 3 default scenarios."""
    return ScenarioModelingInput(
        entity_name="TestCorp",
        base_year=2025,
        base_year_emissions_tco2e=Decimal("50000"),
        target_year=2050,
        scenarios=[
            ScenarioType.AGGRESSIVE,
            ScenarioType.MODERATE,
            ScenarioType.CONSERVATIVE,
        ],
        num_simulations=100,
        random_seed=42,
        uncertainty_level=UncertaintyLevel.MEDIUM,
        projection_interval_years=5,
    )


@pytest.fixture
def small_input():
    """Minimal input with fewer simulations for fast testing."""
    return ScenarioModelingInput(
        entity_name="SmallCo",
        base_year=2025,
        base_year_emissions_tco2e=Decimal("1000"),
        target_year=2050,
        scenarios=[ScenarioType.MODERATE],
        num_simulations=100,
        random_seed=99,
    )


@pytest.fixture
def custom_scenario_input():
    """Input that includes a custom scenario."""
    return ScenarioModelingInput(
        entity_name="CustomCorp",
        base_year=2025,
        base_year_emissions_tco2e=Decimal("20000"),
        target_year=2050,
        scenarios=[ScenarioType.MODERATE, ScenarioType.CUSTOM],
        custom_scenario=CustomScenarioConfig(
            name="Custom Net Zero",
            annual_reduction_rate=Decimal("0.06"),
            carbon_price_base_usd=Decimal("60"),
            carbon_price_2030_usd=Decimal("120"),
            carbon_price_2050_usd=Decimal("220"),
            technology_learning_rate=Decimal("0.03"),
            mac_base_usd_per_tco2e=Decimal("50"),
            activity_growth_rate=Decimal("0.02"),
            ambition_score=Decimal("80"),
            risk_score=Decimal("55"),
        ),
        num_simulations=100,
        random_seed=42,
    )


# ---------------------------------------------------------------------------
# TestScenarioModelingBasic
# ---------------------------------------------------------------------------


class TestScenarioModelingBasic:
    """Basic functionality tests for ScenarioModelingEngine."""

    def test_engine_instantiation(self):
        """Engine can be instantiated."""
        engine = ScenarioModelingEngine()
        assert engine.engine_version == "1.0.0"

    def test_calculate_returns_result(self, engine, basic_input):
        """calculate() returns a ScenarioModelingResult."""
        result = engine.calculate(basic_input)
        assert isinstance(result, ScenarioModelingResult)

    def test_result_has_entity_name(self, engine, basic_input):
        """Result carries the entity name from input."""
        result = engine.calculate(basic_input)
        assert result.entity_name == "TestCorp"

    def test_result_has_base_and_target_year(self, engine, basic_input):
        """Result carries base and target year."""
        result = engine.calculate(basic_input)
        assert result.base_year == 2025
        assert result.target_year == 2050

    def test_result_has_provenance_hash(self, engine, basic_input):
        """Result contains a valid 64-char hex provenance hash."""
        result = engine.calculate(basic_input)
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_result_has_processing_time(self, engine, basic_input):
        """Result has positive processing time."""
        result = engine.calculate(basic_input)
        assert result.processing_time_ms > 0.0

    def test_result_simulation_status_completed(self, engine, basic_input):
        """Status is 'completed' for successful run."""
        result = engine.calculate(basic_input)
        assert result.simulation_status == SimulationStatus.COMPLETED.value

    def test_result_has_three_scenarios(self, engine, basic_input):
        """Result contains exactly the requested number of scenarios."""
        result = engine.calculate(basic_input)
        assert len(result.scenarios) == 3

    def test_result_num_simulations(self, engine, basic_input):
        """Result records the requested number of simulations."""
        result = engine.calculate(basic_input)
        assert result.num_simulations == 100

    def test_result_has_base_year_emissions(self, engine, basic_input):
        """Result records the base year emissions."""
        result = engine.calculate(basic_input)
        assert float(result.base_year_emissions_tco2e) == pytest.approx(50000.0, rel=1e-3)


# ---------------------------------------------------------------------------
# TestScenarioOutputs
# ---------------------------------------------------------------------------


class TestScenarioOutputs:
    """Tests for individual scenario outputs."""

    def test_scenario_types_are_correct(self, engine, basic_input):
        """Each scenario output has the correct type string."""
        result = engine.calculate(basic_input)
        types = {s.scenario_type for s in result.scenarios}
        assert types == {"aggressive", "moderate", "conservative"}

    def test_scenario_has_year_statistics(self, engine, basic_input):
        """Each scenario has multiple year statistics."""
        result = engine.calculate(basic_input)
        for s in result.scenarios:
            assert len(s.year_statistics) > 0

    def test_year_statistics_include_base_year(self, engine, basic_input):
        """Year statistics include the base year."""
        result = engine.calculate(basic_input)
        for s in result.scenarios:
            years = [ys.year for ys in s.year_statistics]
            assert 2025 in years

    def test_year_statistics_include_target_year(self, engine, basic_input):
        """Year statistics include the target year."""
        result = engine.calculate(basic_input)
        for s in result.scenarios:
            years = [ys.year for ys in s.year_statistics]
            assert 2050 in years

    def test_year_statistics_include_reference_years(self, engine, basic_input):
        """Year statistics include 2030 and 2040 reference years."""
        result = engine.calculate(basic_input)
        for s in result.scenarios:
            years = [ys.year for ys in s.year_statistics]
            assert 2030 in years
            assert 2040 in years

    def test_base_year_emissions_match_input(self, engine, basic_input):
        """At the base year, mean emissions should approximate base value."""
        result = engine.calculate(basic_input)
        for s in result.scenarios:
            base_stat = [ys for ys in s.year_statistics if ys.year == 2025][0]
            assert float(base_stat.mean_tco2e) == pytest.approx(50000.0, rel=0.15)

    def test_emissions_decrease_over_time(self, engine, basic_input):
        """Mean emissions in the final year are lower than the base year."""
        result = engine.calculate(basic_input)
        for s in result.scenarios:
            base_stat = [ys for ys in s.year_statistics if ys.year == 2025][0]
            final_stat = [ys for ys in s.year_statistics if ys.year == 2050][0]
            assert float(final_stat.mean_tco2e) < float(base_stat.mean_tco2e)

    def test_aggressive_has_lower_2050_than_conservative(self, engine, basic_input):
        """Aggressive scenario has lower residual emissions than conservative."""
        result = engine.calculate(basic_input)
        agg = [s for s in result.scenarios if s.scenario_type == "aggressive"][0]
        cons = [s for s in result.scenarios if s.scenario_type == "conservative"][0]
        assert float(agg.residual_emissions_2050_mean_tco2e) < float(
            cons.residual_emissions_2050_mean_tco2e
        )

    def test_scenario_has_cumulative_cost(self, engine, basic_input):
        """Scenarios have non-negative cumulative cost."""
        result = engine.calculate(basic_input)
        for s in result.scenarios:
            assert float(s.total_cumulative_cost_mean_usd) >= 0

    def test_scenario_has_cumulative_abatement(self, engine, basic_input):
        """Scenarios have non-negative cumulative abatement."""
        result = engine.calculate(basic_input)
        for s in result.scenarios:
            assert float(s.total_cumulative_abatement_mean_tco2e) >= 0

    def test_scenario_annual_reduction_rate_matches(self, engine, basic_input):
        """Scenario output records the correct annual reduction rate."""
        result = engine.calculate(basic_input)
        agg = [s for s in result.scenarios if s.scenario_type == "aggressive"][0]
        assert float(agg.annual_reduction_rate) == pytest.approx(0.072, rel=1e-3)


# ---------------------------------------------------------------------------
# TestYearStatisticsPercentiles
# ---------------------------------------------------------------------------


class TestYearStatisticsPercentiles:
    """Tests for percentile ordering in year statistics."""

    def test_percentiles_ordered_correctly(self, engine, basic_input):
        """p10 <= p25 <= median <= mean ~= median, p75 >= p25, p90 >= p75."""
        result = engine.calculate(basic_input)
        for s in result.scenarios:
            for ys in s.year_statistics:
                assert float(ys.p10_tco2e) <= float(ys.p25_tco2e) + 1e-6
                assert float(ys.p25_tco2e) <= float(ys.p75_tco2e) + 1e-6
                assert float(ys.p75_tco2e) <= float(ys.p90_tco2e) + 1e-6

    def test_std_dev_non_negative(self, engine, basic_input):
        """Standard deviation is non-negative."""
        result = engine.calculate(basic_input)
        for s in result.scenarios:
            for ys in s.year_statistics:
                assert float(ys.std_dev_tco2e) >= 0


# ---------------------------------------------------------------------------
# TestSensitivityAnalysis
# ---------------------------------------------------------------------------


class TestSensitivityAnalysis:
    """Tests for sensitivity ranking in the result."""

    def test_sensitivity_ranking_populated(self, engine, basic_input):
        """Sensitivity ranking has entries."""
        result = engine.calculate(basic_input)
        assert len(result.sensitivity_ranking) > 0

    def test_sensitivity_ranking_ordered_by_index(self, engine, basic_input):
        """Sensitivity entries are ordered by descending sensitivity index."""
        result = engine.calculate(basic_input)
        indices = [float(e.sensitivity_index) for e in result.sensitivity_ranking]
        for i in range(len(indices) - 1):
            assert indices[i] >= indices[i + 1] - 1e-9

    def test_sensitivity_ranks_sequential(self, engine, basic_input):
        """Ranks start at 1 and are sequential."""
        result = engine.calculate(basic_input)
        ranks = [e.rank for e in result.sensitivity_ranking]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_sensitivity_has_known_parameters(self, engine, basic_input):
        """Sensitivity includes expected parameter names."""
        result = engine.calculate(basic_input)
        params = {e.parameter for e in result.sensitivity_ranking}
        assert "annual_reduction_rate" in params


# ---------------------------------------------------------------------------
# TestScenarioComparisons
# ---------------------------------------------------------------------------


class TestScenarioComparisons:
    """Tests for pairwise scenario comparisons."""

    def test_comparisons_populated(self, engine, basic_input):
        """Result has pairwise comparisons for 3 scenarios."""
        result = engine.calculate(basic_input)
        # C(3,2) = 3 pairs
        assert len(result.comparisons) == 3

    def test_comparison_has_scenario_labels(self, engine, basic_input):
        """Each comparison references two scenario types."""
        result = engine.calculate(basic_input)
        for comp in result.comparisons:
            assert comp.scenario_a != ""
            assert comp.scenario_b != ""
            assert comp.scenario_a != comp.scenario_b

    def test_comparison_cost_per_additional_non_negative(self, engine, basic_input):
        """Cost per additional tCO2e abated is non-negative."""
        result = engine.calculate(basic_input)
        for comp in result.comparisons:
            assert float(comp.cost_per_additional_tco2e_usd) >= 0


# ---------------------------------------------------------------------------
# TestDecisionMatrix
# ---------------------------------------------------------------------------


class TestDecisionMatrix:
    """Tests for the decision matrix scoring."""

    def test_decision_matrix_populated(self, engine, basic_input):
        """Decision matrix has entries for each scenario."""
        result = engine.calculate(basic_input)
        assert len(result.decision_matrix) == 3

    def test_decision_matrix_ranks_sequential(self, engine, basic_input):
        """Decision matrix ranks are 1, 2, 3."""
        result = engine.calculate(basic_input)
        ranks = sorted(e.rank for e in result.decision_matrix)
        assert ranks == [1, 2, 3]

    def test_recommended_scenario_is_rank_1(self, engine, basic_input):
        """Recommended scenario is the one with rank 1."""
        result = engine.calculate(basic_input)
        rank1 = [e for e in result.decision_matrix if e.rank == 1][0]
        assert result.recommended_scenario == rank1.scenario_type

    def test_decision_scores_non_negative(self, engine, basic_input):
        """All decision scores are non-negative."""
        result = engine.calculate(basic_input)
        for e in result.decision_matrix:
            assert float(e.cost_score) >= 0
            assert float(e.risk_score) >= 0
            assert float(e.ambition_score) >= 0
            assert float(e.weighted_total) >= 0


# ---------------------------------------------------------------------------
# TestCustomScenario
# ---------------------------------------------------------------------------


class TestCustomScenario:
    """Tests for custom scenario handling."""

    def test_custom_scenario_included(self, engine, custom_scenario_input):
        """Custom scenario appears in result."""
        result = engine.calculate(custom_scenario_input)
        types = {s.scenario_type for s in result.scenarios}
        assert "custom" in types

    def test_custom_scenario_name(self, engine, custom_scenario_input):
        """Custom scenario carries the configured name."""
        result = engine.calculate(custom_scenario_input)
        custom = [s for s in result.scenarios if s.scenario_type == "custom"][0]
        assert custom.scenario_name == "Custom Net Zero"

    def test_custom_scenario_reduction_rate(self, engine, custom_scenario_input):
        """Custom scenario uses the configured reduction rate."""
        result = engine.calculate(custom_scenario_input)
        custom = [s for s in result.scenarios if s.scenario_type == "custom"][0]
        assert float(custom.annual_reduction_rate) == pytest.approx(0.06, rel=1e-3)


# ---------------------------------------------------------------------------
# TestUtilityMethods
# ---------------------------------------------------------------------------


class TestUtilityMethods:
    """Tests for engine utility methods."""

    def test_get_default_params_aggressive(self, engine):
        """get_default_params returns params for aggressive."""
        params = engine.get_default_params(ScenarioType.AGGRESSIVE)
        assert "annual_reduction_rate" in params
        assert params["annual_reduction_rate"] == "0.072"

    def test_get_default_params_unknown_raises(self, engine):
        """get_default_params raises ValueError for CUSTOM."""
        with pytest.raises(ValueError):
            engine.get_default_params(ScenarioType.CUSTOM)

    def test_get_uncertainty_bounds(self, engine):
        """get_uncertainty_bounds returns bounds for all parameter types."""
        bounds = engine.get_uncertainty_bounds(UncertaintyLevel.MEDIUM)
        assert len(bounds) == 6
        for key, val in bounds.items():
            assert float(val) > 0

    def test_get_summary(self, engine, basic_input):
        """get_summary returns a dict with expected keys."""
        result = engine.calculate(basic_input)
        summary = engine.get_summary(result)
        assert summary["entity_name"] == "TestCorp"
        assert summary["num_scenarios"] == 3
        assert summary["recommended_scenario"] != ""
        assert "provenance_hash" in summary


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for ScenarioModelingEngine."""

    def test_single_scenario(self, engine):
        """Engine works with a single scenario."""
        inp = ScenarioModelingInput(
            entity_name="Solo",
            base_year=2025,
            base_year_emissions_tco2e=Decimal("5000"),
            target_year=2050,
            scenarios=[ScenarioType.CONSERVATIVE],
            num_simulations=100,
            random_seed=1,
        )
        result = engine.calculate(inp)
        assert len(result.scenarios) == 1
        assert len(result.comparisons) == 0

    def test_high_uncertainty(self, engine):
        """Engine operates under HIGH uncertainty."""
        inp = ScenarioModelingInput(
            entity_name="HighUnc",
            base_year=2025,
            base_year_emissions_tco2e=Decimal("10000"),
            target_year=2050,
            scenarios=[ScenarioType.MODERATE],
            num_simulations=100,
            random_seed=7,
            uncertainty_level=UncertaintyLevel.HIGH,
        )
        result = engine.calculate(inp)
        assert result.simulation_status == SimulationStatus.COMPLETED.value

    def test_low_uncertainty(self, engine):
        """Engine operates under LOW uncertainty with tighter bounds."""
        inp = ScenarioModelingInput(
            entity_name="LowUnc",
            base_year=2025,
            base_year_emissions_tco2e=Decimal("10000"),
            target_year=2050,
            scenarios=[ScenarioType.MODERATE],
            num_simulations=100,
            random_seed=7,
            uncertainty_level=UncertaintyLevel.LOW,
        )
        result = engine.calculate(inp)
        assert result.simulation_status == SimulationStatus.COMPLETED.value

    def test_small_emissions(self, engine):
        """Engine handles very small emissions values."""
        inp = ScenarioModelingInput(
            entity_name="Tiny",
            base_year=2025,
            base_year_emissions_tco2e=Decimal("0.01"),
            target_year=2050,
            scenarios=[ScenarioType.MODERATE],
            num_simulations=100,
            random_seed=42,
        )
        result = engine.calculate(inp)
        assert result.simulation_status == SimulationStatus.COMPLETED.value

    def test_large_emissions(self, engine):
        """Engine handles very large emissions values."""
        inp = ScenarioModelingInput(
            entity_name="MegaCorp",
            base_year=2025,
            base_year_emissions_tco2e=Decimal("500000000"),
            target_year=2050,
            scenarios=[ScenarioType.AGGRESSIVE],
            num_simulations=100,
            random_seed=42,
        )
        result = engine.calculate(inp)
        assert result.simulation_status == SimulationStatus.COMPLETED.value

    def test_short_projection_interval(self, engine):
        """Projection interval of 1 year produces yearly stats."""
        inp = ScenarioModelingInput(
            entity_name="Yearly",
            base_year=2025,
            base_year_emissions_tco2e=Decimal("10000"),
            target_year=2035,
            scenarios=[ScenarioType.MODERATE],
            num_simulations=100,
            random_seed=42,
            projection_interval_years=1,
        )
        result = engine.calculate(inp)
        mod = result.scenarios[0]
        years = [ys.year for ys in mod.year_statistics]
        # Should include every year from 2025 to 2035
        for y in range(2025, 2036):
            assert y in years

    def test_parameter_overrides(self, engine):
        """Parameter overrides are applied to scenario."""
        inp = ScenarioModelingInput(
            entity_name="Override",
            base_year=2025,
            base_year_emissions_tco2e=Decimal("10000"),
            target_year=2050,
            scenarios=[ScenarioType.MODERATE],
            num_simulations=100,
            random_seed=42,
            parameter_overrides={
                "moderate": [
                    ScenarioParameterOverride(
                        parameter_name="annual_reduction_rate",
                        value=Decimal("0.08"),
                    )
                ]
            },
        )
        result = engine.calculate(inp)
        mod = result.scenarios[0]
        assert float(mod.annual_reduction_rate) == pytest.approx(0.08, rel=1e-3)


# ---------------------------------------------------------------------------
# TestEnums
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for enum definitions."""

    def test_scenario_type_values(self):
        """ScenarioType has expected values."""
        assert ScenarioType.AGGRESSIVE.value == "aggressive"
        assert ScenarioType.MODERATE.value == "moderate"
        assert ScenarioType.CONSERVATIVE.value == "conservative"
        assert ScenarioType.CUSTOM.value == "custom"

    def test_uncertainty_level_values(self):
        """UncertaintyLevel has expected values."""
        assert UncertaintyLevel.LOW.value == "low"
        assert UncertaintyLevel.MEDIUM.value == "medium"
        assert UncertaintyLevel.HIGH.value == "high"

    def test_parameter_type_values(self):
        """ParameterType has expected values."""
        assert ParameterType.EMISSION_FACTOR.value == "emission_factor"
        assert ParameterType.CARBON_PRICE.value == "carbon_price"

    def test_simulation_status_values(self):
        """SimulationStatus has expected values."""
        assert SimulationStatus.COMPLETED.value == "completed"
        assert SimulationStatus.FAILED.value == "failed"


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for Pydantic input validation."""

    def test_target_year_must_be_after_base(self):
        """target_year <= base_year raises ValidationError."""
        with pytest.raises(Exception):
            ScenarioModelingInput(
                entity_name="Bad",
                base_year=2025,
                base_year_emissions_tco2e=Decimal("10000"),
                target_year=2025,
            )

    def test_num_simulations_minimum(self):
        """num_simulations below 100 raises ValidationError."""
        with pytest.raises(Exception):
            ScenarioModelingInput(
                entity_name="Bad",
                base_year=2025,
                base_year_emissions_tco2e=Decimal("10000"),
                target_year=2050,
                num_simulations=10,
            )

    def test_emissions_must_be_positive(self):
        """Zero or negative emissions raises ValidationError."""
        with pytest.raises(Exception):
            ScenarioModelingInput(
                entity_name="Bad",
                base_year=2025,
                base_year_emissions_tco2e=Decimal("0"),
                target_year=2050,
            )

    def test_empty_entity_name_rejected(self):
        """Empty entity name is rejected."""
        with pytest.raises(Exception):
            ScenarioModelingInput(
                entity_name="",
                base_year=2025,
                base_year_emissions_tco2e=Decimal("10000"),
                target_year=2050,
            )
