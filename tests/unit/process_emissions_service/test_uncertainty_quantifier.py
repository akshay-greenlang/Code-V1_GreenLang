# -*- coding: utf-8 -*-
"""
Unit tests for UncertaintyQuantifierEngine (Engine 5 of 6).

AGENT-MRV-004: Process Emissions Agent (GL-MRV-SCOPE1-004)

Tests Monte Carlo simulation, analytical error propagation, DQI scoring,
sensitivity analysis, confidence intervals, process-specific uncertainty
ranges, and edge cases.

Total: 90 tests across 7 test classes.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import pytest

from greenlang.process_emissions.uncertainty_quantifier import (
    UncertaintyQuantifierEngine,
    UncertaintyResult,
    SensitivityResult,
    _ACTIVITY_DATA_UNCERTAINTY,
    _EMISSION_FACTOR_UNCERTAINTY,
    _GWP_UNCERTAINTY,
    _PROCESS_UNCERTAINTY_RANGES,
    _DEFAULT_PROCESS_UNCERTAINTY,
    _DQI_CRITERIA,
    _DQI_MULTIPLIERS,
    _CONFIDENCE_Z_SCORES,
    _DEFAULT_ITERATIONS,
    _MIN_ITERATIONS,
    _MAX_ITERATIONS,
    ProcessCalculationMethod,
    ProcessCalculationTier,
    ActivityDataSource,
    DQICategory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> UncertaintyQuantifierEngine:
    """Create a fresh UncertaintyQuantifierEngine instance."""
    return UncertaintyQuantifierEngine()


@pytest.fixture
def cement_input() -> Dict[str, Any]:
    """Cement production calculation input for uncertainty analysis."""
    return {
        "total_co2e_kg": 500000.0,
        "process_type": "CEMENT",
        "calculation_method": "EMISSION_FACTOR",
        "tier": "TIER_2",
        "activity_data_source": "metered",
        "production_tonnes": 10000.0,
        "emission_factor": 0.507,
        "gwp_value": 1.0,
    }


@pytest.fixture
def iron_steel_input() -> Dict[str, Any]:
    """Iron & steel (BF-BOF) calculation input."""
    return {
        "total_co2e_kg": 1800000.0,
        "process_type": "IRON_STEEL_BF_BOF",
        "calculation_method": "MASS_BALANCE",
        "tier": "TIER_2",
        "activity_data_source": "metered",
        "production_tonnes": 1000.0,
        "emission_factor": 1.8,
        "carbon_input_tonnes": 600.0,
        "carbon_output_tonnes": 100.0,
        "gwp_value": 1.0,
    }


@pytest.fixture
def aluminum_input() -> Dict[str, Any]:
    """Aluminum smelting (prebake) calculation input."""
    return {
        "total_co2e_kg": 1600000.0,
        "process_type": "ALUMINUM_PREBAKE",
        "calculation_method": "EMISSION_FACTOR",
        "tier": "TIER_1",
        "activity_data_source": "estimated",
        "production_tonnes": 1000.0,
        "emission_factor": 1.6,
        "gwp_value": 1.0,
    }


@pytest.fixture
def nitric_acid_input() -> Dict[str, Any]:
    """Nitric acid production calculation input."""
    return {
        "total_co2e_kg": 2000000.0,
        "process_type": "NITRIC_ACID",
        "calculation_method": "EMISSION_FACTOR",
        "tier": "TIER_1",
        "activity_data_source": "estimated",
        "production_tonnes": 100000.0,
        "emission_factor": 0.007,
        "gwp_value": 273.0,
    }


@pytest.fixture
def semiconductor_input() -> Dict[str, Any]:
    """Semiconductor manufacturing calculation input."""
    return {
        "total_co2e_kg": 50000.0,
        "process_type": "SEMICONDUCTOR",
        "calculation_method": "EMISSION_FACTOR",
        "tier": "TIER_1",
        "activity_data_source": "estimated",
        "production_tonnes": 500.0,
        "emission_factor": 0.0001,
        "gwp_value": 7380.0,
    }


@pytest.fixture
def ammonia_input() -> Dict[str, Any]:
    """Ammonia production calculation input."""
    return {
        "total_co2e_kg": 169400.0,
        "process_type": "AMMONIA",
        "calculation_method": "EMISSION_FACTOR",
        "tier": "TIER_2",
        "activity_data_source": "metered",
        "production_tonnes": 100.0,
        "emission_factor": 1.694,
        "gwp_value": 1.0,
    }


# ===========================================================================
# TestMonteCarlo (20 tests)
# ===========================================================================


class TestMonteCarlo:
    """Test Monte Carlo simulation with various process types."""

    def test_cement_mc_returns_result(self, engine, cement_input):
        """Monte Carlo for cement returns an UncertaintyResult."""
        result = engine.run_monte_carlo(cement_input, n_iterations=500, seed=42)
        assert isinstance(result, UncertaintyResult)

    def test_cement_mc_mean_close_to_point_estimate(self, engine, cement_input):
        """MC mean should be close to the central estimate for cement."""
        result = engine.run_monte_carlo(cement_input, n_iterations=2000, seed=42)
        point_estimate = Decimal("500000.0")
        assert result.monte_carlo_mean is not None
        relative_diff = abs(float(result.monte_carlo_mean) - float(point_estimate)) / float(point_estimate)
        assert relative_diff < 0.10, f"MC mean deviates >10% from point estimate: {relative_diff:.4f}"

    def test_cement_mc_std_positive(self, engine, cement_input):
        """MC standard deviation must be positive for non-zero emissions."""
        result = engine.run_monte_carlo(cement_input, n_iterations=500, seed=42)
        assert result.monte_carlo_std is not None
        assert result.monte_carlo_std > Decimal("0")

    def test_iron_steel_mc_returns_result(self, engine, iron_steel_input):
        """Monte Carlo for iron/steel BF-BOF returns a result."""
        result = engine.run_monte_carlo(iron_steel_input, n_iterations=500, seed=42)
        assert isinstance(result, UncertaintyResult)
        assert result.monte_carlo_mean is not None

    def test_iron_steel_mc_mean_reasonable(self, engine, iron_steel_input):
        """Iron/steel MC mean should be reasonably close to point estimate."""
        result = engine.run_monte_carlo(iron_steel_input, n_iterations=2000, seed=42)
        point = 1800000.0
        relative_diff = abs(float(result.monte_carlo_mean) - point) / point
        assert relative_diff < 0.15

    def test_aluminum_mc_returns_result(self, engine, aluminum_input):
        """Monte Carlo for aluminum returns a result."""
        result = engine.run_monte_carlo(aluminum_input, n_iterations=500, seed=42)
        assert isinstance(result, UncertaintyResult)

    def test_nitric_acid_mc_returns_result(self, engine, nitric_acid_input):
        """Monte Carlo for nitric acid returns a result."""
        result = engine.run_monte_carlo(nitric_acid_input, n_iterations=500, seed=42)
        assert isinstance(result, UncertaintyResult)

    def test_semiconductor_mc_returns_result(self, engine, semiconductor_input):
        """Monte Carlo for semiconductor returns a result."""
        result = engine.run_monte_carlo(semiconductor_input, n_iterations=500, seed=42)
        assert isinstance(result, UncertaintyResult)

    def test_ammonia_mc_returns_result(self, engine, ammonia_input):
        """Monte Carlo for ammonia returns a result."""
        result = engine.run_monte_carlo(ammonia_input, n_iterations=500, seed=42)
        assert isinstance(result, UncertaintyResult)

    def test_mass_balance_mc_method(self, engine, iron_steel_input):
        """Mass balance method uses correct MC path."""
        result = engine.run_monte_carlo(iron_steel_input, n_iterations=500, seed=42)
        assert result.calculation_method == "MASS_BALANCE"

    def test_stoichiometric_mc_method(self, engine):
        """Stoichiometric method uses correct MC path."""
        data = {
            "total_co2e_kg": 300000.0,
            "process_type": "LIME",
            "calculation_method": "STOICHIOMETRIC",
            "tier": "TIER_2",
            "activity_data_source": "metered",
            "production_tonnes": 5000.0,
            "stoichiometric_factor": 0.785,
        }
        result = engine.run_monte_carlo(data, n_iterations=500, seed=42)
        assert result.calculation_method == "STOICHIOMETRIC"

    def test_direct_measurement_mc_method(self, engine):
        """Direct measurement method uses correct MC path."""
        data = {
            "total_co2e_kg": 100000.0,
            "process_type": "CEMENT",
            "calculation_method": "DIRECT_MEASUREMENT",
            "tier": "TIER_3",
            "activity_data_source": "metered",
        }
        result = engine.run_monte_carlo(data, n_iterations=500, seed=42)
        assert result.calculation_method == "DIRECT_MEASUREMENT"

    def test_mc_confidence_intervals_present(self, engine, cement_input):
        """MC result should contain confidence intervals for 90/95/99."""
        result = engine.run_monte_carlo(cement_input, n_iterations=1000, seed=42)
        assert "90" in result.confidence_intervals
        assert "95" in result.confidence_intervals
        assert "99" in result.confidence_intervals

    def test_mc_provenance_hash_not_empty(self, engine, cement_input):
        """MC result should have a SHA-256 provenance hash."""
        result = engine.run_monte_carlo(cement_input, n_iterations=500, seed=42)
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_mc_result_id_prefix(self, engine, cement_input):
        """MC result ID should start with 'uq_'."""
        result = engine.run_monte_carlo(cement_input, n_iterations=500, seed=42)
        assert result.result_id.startswith("uq_")

    def test_mc_combined_uncertainty_positive(self, engine, cement_input):
        """Combined uncertainty should be positive for non-zero emissions."""
        result = engine.run_monte_carlo(cement_input, n_iterations=500, seed=42)
        assert result.combined_uncertainty_pct > Decimal("0")

    def test_mc_history_recorded(self, engine, cement_input):
        """MC assessment should be added to history."""
        engine.clear_history()
        engine.run_monte_carlo(cement_input, n_iterations=500, seed=42)
        history = engine.get_assessment_history()
        assert len(history) == 1

    def test_mc_with_abatement(self, engine):
        """MC simulation with abatement efficiency parameter."""
        data = {
            "total_co2e_kg": 300000.0,
            "process_type": "NITRIC_ACID",
            "calculation_method": "EMISSION_FACTOR",
            "tier": "TIER_2",
            "activity_data_source": "metered",
            "production_tonnes": 50000.0,
            "emission_factor": 0.007,
            "abatement_efficiency": 0.85,
            "gwp_value": 273.0,
        }
        result = engine.run_monte_carlo(data, n_iterations=500, seed=42)
        assert result.monte_carlo_mean is not None

    def test_mc_to_dict_serialization(self, engine, cement_input):
        """MC result should serialize to dict."""
        result = engine.run_monte_carlo(cement_input, n_iterations=500, seed=42)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "result_id" in d
        assert "emissions_value" in d
        assert "confidence_intervals" in d

    def test_mc_iteration_count_validated(self, engine, cement_input):
        """MC should reject iterations outside valid range."""
        with pytest.raises(ValueError, match="n_iterations"):
            engine.run_monte_carlo(cement_input, n_iterations=50, seed=42)

        with pytest.raises(ValueError, match="n_iterations"):
            engine.run_monte_carlo(cement_input, n_iterations=200000, seed=42)


# ===========================================================================
# TestDQIScoring (15 tests)
# ===========================================================================


class TestDQIScoring:
    """Test Data Quality Indicator scoring across 5 dimensions."""

    def test_empty_inputs_returns_default(self, engine):
        """Empty DQI inputs should return default score of 3.0."""
        score = engine.calculate_dqi({})
        assert score == 3.0

    def test_all_best_quality(self, engine):
        """All best-quality inputs should yield a score near 1.0."""
        dqi = {
            "reliability": "direct_measurement",
            "completeness": "above_95_pct",
            "temporal_correlation": "same_year",
            "geographical_correlation": "same_facility",
            "technological_correlation": "same_process",
        }
        score = engine.calculate_dqi(dqi)
        assert score == 1.0

    def test_all_worst_quality(self, engine):
        """All worst-quality inputs should yield a score of 5.0."""
        dqi = {
            "reliability": "unknown",
            "completeness": "below_40_pct",
            "temporal_correlation": "older_than_5_years",
            "geographical_correlation": "unknown",
            "technological_correlation": "unknown",
        }
        score = engine.calculate_dqi(dqi)
        assert score == 5.0

    def test_reliability_direct_measurement(self, engine):
        """Reliability 'direct_measurement' should score 1."""
        score = engine.calculate_dqi({"reliability": "direct_measurement"})
        assert score < 3.0  # Better than default

    def test_reliability_unknown(self, engine):
        """Reliability 'unknown' should score 5."""
        score = engine.calculate_dqi({"reliability": "unknown"})
        assert score > 3.0  # Worse than default

    def test_completeness_above_95(self, engine):
        """Completeness 'above_95_pct' should score 1."""
        score = engine.calculate_dqi({"completeness": "above_95_pct"})
        assert score < 3.0

    def test_completeness_below_40(self, engine):
        """Completeness 'below_40_pct' should score 5."""
        score = engine.calculate_dqi({"completeness": "below_40_pct"})
        assert score > 3.0

    def test_temporal_same_year(self, engine):
        """Temporal 'same_year' should score 1."""
        score = engine.calculate_dqi({"temporal_correlation": "same_year"})
        assert score < 3.0

    def test_temporal_older_than_5(self, engine):
        """Temporal 'older_than_5_years' should score 5."""
        score = engine.calculate_dqi({"temporal_correlation": "older_than_5_years"})
        assert score > 3.0

    def test_geo_same_facility(self, engine):
        """Geographical 'same_facility' should score 1."""
        score = engine.calculate_dqi({"geographical_correlation": "same_facility"})
        assert score < 3.0

    def test_tech_same_process(self, engine):
        """Technological 'same_process' should score 1."""
        score = engine.calculate_dqi({"technological_correlation": "same_process"})
        assert score < 3.0

    def test_numeric_dqi_scores(self, engine):
        """Direct numeric DQI scores should be accepted and clamped."""
        score = engine.calculate_dqi({
            "reliability": 2,
            "completeness": 3,
            "temporal_correlation": 1,
            "geographical_correlation": 4,
            "technological_correlation": 2,
        })
        assert 1.0 <= score <= 5.0

    def test_composite_geometric_mean(self, engine):
        """Composite score should be geometric mean of individual scores."""
        dqi = {
            "reliability": 1,
            "completeness": 1,
            "temporal_correlation": 1,
            "geographical_correlation": 1,
            "technological_correlation": 1,
        }
        score = engine.calculate_dqi(dqi)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_partial_inputs_defaults_to_3(self, engine):
        """Missing dimensions should default to 3."""
        score_partial = engine.calculate_dqi({"reliability": 1})
        # Should be geometric mean of [1, 3, 3, 3, 3]
        expected = math.exp(
            (math.log(1) + math.log(3) + math.log(3) + math.log(3) + math.log(3)) / 5
        )
        assert score_partial == pytest.approx(expected, abs=0.01)

    def test_unrecognized_string_defaults_to_3(self, engine):
        """Unrecognized string values should default to score 3."""
        score = engine.calculate_dqi({"reliability": "not_a_valid_criterion"})
        # Default of 3 for unknown reliability, plus 4 defaults of 3
        assert score == pytest.approx(3.0, abs=0.01)


# ===========================================================================
# TestAnalyticalPropagation (10 tests)
# ===========================================================================


class TestAnalyticalPropagation:
    """Test analytical error propagation (IPCC Approach 1)."""

    def test_multiplicative_components(self, engine):
        """Multiplicative components combine via root-sum-of-squares."""
        components = [
            {"name": "activity_data", "value": 10000, "uncertainty_pct": 5, "relationship": "multiplicative"},
            {"name": "emission_factor", "value": 0.5, "uncertainty_pct": 10, "relationship": "multiplicative"},
        ]
        result = engine.calculate_analytical_uncertainty(components)
        expected = math.sqrt(5**2 + 10**2)
        assert float(result.combined_uncertainty_pct) == pytest.approx(expected, abs=0.1)

    def test_single_component(self, engine):
        """Single component uncertainty equals that component's uncertainty."""
        components = [
            {"name": "activity_data", "value": 10000, "uncertainty_pct": 5},
        ]
        result = engine.calculate_analytical_uncertainty(components)
        assert float(result.combined_uncertainty_pct) == pytest.approx(5.0, abs=0.1)

    def test_three_multiplicative_components(self, engine):
        """Three multiplicative components combine correctly."""
        components = [
            {"name": "ad", "value": 1000, "uncertainty_pct": 2, "relationship": "multiplicative"},
            {"name": "ef", "value": 0.5, "uncertainty_pct": 10, "relationship": "multiplicative"},
            {"name": "gwp", "value": 298, "uncertainty_pct": 10, "relationship": "multiplicative"},
        ]
        result = engine.calculate_analytical_uncertainty(components)
        expected = math.sqrt(2**2 + 10**2 + 10**2)
        assert float(result.combined_uncertainty_pct) == pytest.approx(expected, abs=0.1)

    def test_additive_components(self, engine):
        """Additive components combine correctly."""
        components = [
            {"name": "co2", "value": 500, "uncertainty_pct": 5, "relationship": "additive"},
            {"name": "ch4", "value": 500, "uncertainty_pct": 20, "relationship": "additive"},
        ]
        result = engine.calculate_analytical_uncertainty(components)
        assert float(result.combined_uncertainty_pct) > 0

    def test_empty_components_raises(self, engine):
        """Empty components list should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.calculate_analytical_uncertainty([])

    def test_missing_name_raises(self, engine):
        """Component missing 'name' should raise ValueError."""
        with pytest.raises(ValueError, match="must have"):
            engine.calculate_analytical_uncertainty([{"value": 100}])

    def test_missing_value_raises(self, engine):
        """Component missing 'value' should raise ValueError."""
        with pytest.raises(ValueError, match="must have"):
            engine.calculate_analytical_uncertainty([{"name": "test"}])

    def test_analytical_result_has_provenance(self, engine):
        """Analytical result should have a provenance hash."""
        components = [
            {"name": "ad", "value": 1000, "uncertainty_pct": 5},
        ]
        result = engine.calculate_analytical_uncertainty(components)
        assert len(result.provenance_hash) == 64

    def test_analytical_confidence_intervals(self, engine):
        """Analytical result should include confidence intervals."""
        components = [
            {"name": "ad", "value": 10000, "uncertainty_pct": 10},
        ]
        result = engine.calculate_analytical_uncertainty(components)
        assert "90" in result.confidence_intervals
        assert "95" in result.confidence_intervals
        assert "99" in result.confidence_intervals

    def test_analytical_contribution_analysis(self, engine):
        """Analytical result should include contribution analysis."""
        components = [
            {"name": "ad", "value": 1000, "uncertainty_pct": 5, "relationship": "multiplicative"},
            {"name": "ef", "value": 0.5, "uncertainty_pct": 10, "relationship": "multiplicative"},
        ]
        result = engine.calculate_analytical_uncertainty(components)
        assert "ad" in result.contribution_analysis
        assert "ef" in result.contribution_analysis
        total = sum(float(v) for v in result.contribution_analysis.values())
        assert total == pytest.approx(1.0, abs=0.01)


# ===========================================================================
# TestSensitivityAnalysis (10 tests)
# ===========================================================================


class TestSensitivityAnalysis:
    """Test sensitivity analysis results."""

    def test_single_parameter(self, engine, cement_input):
        """Sensitivity analysis for a single parameter."""
        results = engine.sensitivity_analysis(
            cement_input, ["production_tonnes"],
        )
        assert len(results) == 1
        assert results[0].parameter == "production_tonnes"

    def test_multiple_parameters(self, engine, cement_input):
        """Sensitivity analysis for multiple parameters."""
        results = engine.sensitivity_analysis(
            cement_input,
            ["production_tonnes", "emission_factor"],
        )
        assert len(results) == 2

    def test_sensitivity_coefficient_production(self, engine, cement_input):
        """Production tonnage should have sensitivity coefficient near 1.0."""
        results = engine.sensitivity_analysis(
            cement_input, ["production_tonnes"], perturbation_pct=10.0,
        )
        coeff = float(results[0].sensitivity_coefficient)
        assert coeff == pytest.approx(1.0, abs=0.1)

    def test_sensitivity_coefficient_ef(self, engine, cement_input):
        """Emission factor should have sensitivity coefficient near 1.0."""
        results = engine.sensitivity_analysis(
            cement_input, ["emission_factor"], perturbation_pct=10.0,
        )
        coeff = float(results[0].sensitivity_coefficient)
        assert coeff == pytest.approx(1.0, abs=0.1)

    def test_sensitivity_to_dict(self, engine, cement_input):
        """SensitivityResult should serialize to dict."""
        results = engine.sensitivity_analysis(
            cement_input, ["production_tonnes"],
        )
        d = results[0].to_dict()
        assert isinstance(d, dict)
        assert "parameter" in d
        assert "sensitivity_coefficient" in d

    def test_empty_parameters_raises(self, engine, cement_input):
        """Empty parameters list should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.sensitivity_analysis(cement_input, [])

    def test_zero_emissions_raises(self, engine):
        """Zero emissions should raise ValueError for sensitivity."""
        data = {"total_co2e_kg": 0, "production_tonnes": 100}
        with pytest.raises(ValueError, match="must be > 0"):
            engine.sensitivity_analysis(data, ["production_tonnes"])

    def test_custom_perturbation(self, engine, cement_input):
        """Custom perturbation percentage should be applied."""
        results = engine.sensitivity_analysis(
            cement_input, ["production_tonnes"], perturbation_pct=5.0,
        )
        assert float(results[0].perturbation_pct) == pytest.approx(5.0, abs=0.01)

    def test_abatement_inverse_relationship(self, engine):
        """Abatement efficiency should have inverse sensitivity."""
        data = {
            "total_co2e_kg": 100000.0,
            "abatement_efficiency": 0.5,
        }
        results = engine.sensitivity_analysis(
            data, ["abatement_efficiency"], perturbation_pct=10.0,
        )
        # Increasing abatement should decrease emissions
        assert float(results[0].emissions_change_pct) < 0

    def test_perturbed_emissions_changes(self, engine, cement_input):
        """Perturbed emissions should differ from base emissions."""
        results = engine.sensitivity_analysis(
            cement_input, ["production_tonnes"],
        )
        assert results[0].perturbed_emissions != results[0].base_emissions


# ===========================================================================
# TestConfidenceIntervals (10 tests)
# ===========================================================================


class TestConfidenceIntervals:
    """Test 90%, 95%, 99% confidence intervals are correctly ordered."""

    def test_ci_90_95_99_ordering(self, engine):
        """90% CI should be narrower than 95% which is narrower than 99%."""
        samples = [float(x) for x in range(1, 101)]
        cis = engine.get_confidence_intervals(samples, [0.90, 0.95, 0.99])
        width_90 = float(cis["90"][1]) - float(cis["90"][0])
        width_95 = float(cis["95"][1]) - float(cis["95"][0])
        width_99 = float(cis["99"][1]) - float(cis["99"][0])
        assert width_90 <= width_95
        assert width_95 <= width_99

    def test_ci_lower_less_than_upper(self, engine):
        """Lower bound should be less than upper bound."""
        samples = [float(x) for x in range(1, 101)]
        cis = engine.get_confidence_intervals(samples)
        for key, (lower, upper) in cis.items():
            assert lower <= upper, f"CI-{key}: {lower} > {upper}"

    def test_ci_90_contains_median(self, engine):
        """90% CI should contain the median."""
        samples = [float(x) for x in range(1, 101)]
        cis = engine.get_confidence_intervals(samples, [0.90])
        median = 50.5
        assert float(cis["90"][0]) < median < float(cis["90"][1])

    def test_ci_default_levels(self, engine):
        """Default confidence levels should be 90, 95, 99."""
        samples = [float(x) for x in range(1, 101)]
        cis = engine.get_confidence_intervals(samples)
        assert "90" in cis
        assert "95" in cis
        assert "99" in cis

    def test_ci_custom_levels(self, engine):
        """Custom confidence levels should be supported."""
        samples = [float(x) for x in range(1, 101)]
        cis = engine.get_confidence_intervals(samples, [0.80, 0.90])
        assert "80" in cis
        assert "90" in cis

    def test_ci_empty_samples_raises(self, engine):
        """Empty samples should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.get_confidence_intervals([])

    def test_ci_invalid_level_raises(self, engine):
        """Invalid confidence level (>=1 or <=0) should raise ValueError."""
        with pytest.raises(ValueError, match="must be in"):
            engine.get_confidence_intervals([1, 2, 3], [1.0])
        with pytest.raises(ValueError, match="must be in"):
            engine.get_confidence_intervals([1, 2, 3], [0.0])

    def test_ci_returns_decimal_tuples(self, engine):
        """CI bounds should be Decimal values."""
        samples = [float(x) for x in range(1, 101)]
        cis = engine.get_confidence_intervals(samples)
        for key, (lower, upper) in cis.items():
            assert isinstance(lower, Decimal)
            assert isinstance(upper, Decimal)

    def test_mc_ci_ordering(self, engine, cement_input):
        """MC confidence intervals should be properly ordered."""
        result = engine.run_monte_carlo(cement_input, n_iterations=1000, seed=42)
        cis = result.confidence_intervals
        if "90" in cis and "95" in cis:
            w90 = float(cis["90"][1]) - float(cis["90"][0])
            w95 = float(cis["95"][1]) - float(cis["95"][0])
            assert w90 <= w95 + 1.0  # Small tolerance for MC variation

    def test_ci_single_sample(self, engine):
        """Single sample should produce valid (degenerate) intervals."""
        cis = engine.get_confidence_intervals([42.0])
        for key, (lower, upper) in cis.items():
            assert lower <= upper


# ===========================================================================
# TestProcessSpecificRanges (15 tests)
# ===========================================================================


class TestProcessSpecificRanges:
    """Test uncertainty ranges for process-specific parameters."""

    def test_cement_tier1_range(self, engine):
        """Cement Tier 1 uncertainty should be 2-5%."""
        low, high = engine.get_process_uncertainty_range("CEMENT", "TIER_1")
        assert low == pytest.approx(2.0, abs=0.1)
        assert high == pytest.approx(5.0, abs=0.1)

    def test_cement_tier3_range(self, engine):
        """Cement Tier 3 uncertainty should be 1-2%."""
        low, high = engine.get_process_uncertainty_range("CEMENT", "TIER_3")
        assert low == pytest.approx(1.0, abs=0.1)
        assert high == pytest.approx(2.0, abs=0.1)

    def test_lime_tier1_range(self, engine):
        """Lime Tier 1 uncertainty should be 3-7%."""
        low, high = engine.get_process_uncertainty_range("LIME", "TIER_1")
        assert low == pytest.approx(3.0, abs=0.1)
        assert high == pytest.approx(7.0, abs=0.1)

    def test_semiconductor_tier1_range(self, engine):
        """Semiconductor Tier 1 uncertainty should be 15-40%."""
        low, high = engine.get_process_uncertainty_range("SEMICONDUCTOR", "TIER_1")
        assert low == pytest.approx(15.0, abs=0.1)
        assert high == pytest.approx(40.0, abs=0.1)

    def test_semiconductor_tier3_range(self, engine):
        """Semiconductor Tier 3 uncertainty should be 10-20%."""
        low, high = engine.get_process_uncertainty_range("SEMICONDUCTOR", "TIER_3")
        assert low == pytest.approx(10.0, abs=0.1)
        assert high == pytest.approx(20.0, abs=0.1)

    def test_nitric_acid_tier1_range(self, engine):
        """Nitric acid Tier 1 uncertainty should be 10-30%."""
        low, high = engine.get_process_uncertainty_range("NITRIC_ACID", "TIER_1")
        assert low == pytest.approx(10.0, abs=0.1)
        assert high == pytest.approx(30.0, abs=0.1)

    def test_ammonia_tier1_range(self, engine):
        """Ammonia Tier 1 uncertainty should be 3-8%."""
        low, high = engine.get_process_uncertainty_range("AMMONIA", "TIER_1")
        assert low == pytest.approx(3.0, abs=0.1)
        assert high == pytest.approx(8.0, abs=0.1)

    def test_iron_steel_bf_bof_tier1_range(self, engine):
        """Iron/steel BF-BOF Tier 1 should be 5-15%."""
        low, high = engine.get_process_uncertainty_range("IRON_STEEL_BF_BOF", "TIER_1")
        assert low == pytest.approx(5.0, abs=0.1)
        assert high == pytest.approx(15.0, abs=0.1)

    def test_iron_steel_eaf_tier1_range(self, engine):
        """Iron/steel EAF Tier 1 should be 8-15%."""
        low, high = engine.get_process_uncertainty_range("IRON_STEEL_EAF", "TIER_1")
        assert low == pytest.approx(8.0, abs=0.1)
        assert high == pytest.approx(15.0, abs=0.1)

    def test_aluminum_prebake_tier1_range(self, engine):
        """Aluminum prebake Tier 1 should be 5-20%."""
        low, high = engine.get_process_uncertainty_range("ALUMINUM_PREBAKE", "TIER_1")
        assert low == pytest.approx(5.0, abs=0.1)
        assert high == pytest.approx(20.0, abs=0.1)

    def test_aluminum_soderberg_tier1_range(self, engine):
        """Aluminum Soderberg Tier 1 should be 8-20%."""
        low, high = engine.get_process_uncertainty_range("ALUMINUM_SODERBERG", "TIER_1")
        assert low == pytest.approx(8.0, abs=0.1)
        assert high == pytest.approx(20.0, abs=0.1)

    def test_unknown_process_uses_default(self, engine):
        """Unknown process should fall back to default range."""
        low, high = engine.get_process_uncertainty_range("UNKNOWN_PROCESS", "TIER_1")
        expected_low = _DEFAULT_PROCESS_UNCERTAINTY["TIER_1"][0] * 100
        expected_high = _DEFAULT_PROCESS_UNCERTAINTY["TIER_1"][1] * 100
        assert low == pytest.approx(expected_low, abs=0.1)
        assert high == pytest.approx(expected_high, abs=0.1)

    def test_higher_tier_lower_uncertainty(self, engine):
        """Higher tier should have lower uncertainty than lower tier."""
        low_t1, high_t1 = engine.get_process_uncertainty_range("CEMENT", "TIER_1")
        low_t3, high_t3 = engine.get_process_uncertainty_range("CEMENT", "TIER_3")
        assert high_t3 <= high_t1

    def test_glass_tier1_range(self, engine):
        """Glass Tier 1 should be 5-10%."""
        low, high = engine.get_process_uncertainty_range("GLASS", "TIER_1")
        assert low == pytest.approx(5.0, abs=0.1)
        assert high == pytest.approx(10.0, abs=0.1)

    def test_adipic_acid_tier1_range(self, engine):
        """Adipic acid Tier 1 should be 10-25%."""
        low, high = engine.get_process_uncertainty_range("ADIPIC_ACID", "TIER_1")
        assert low == pytest.approx(10.0, abs=0.1)
        assert high == pytest.approx(25.0, abs=0.1)


# ===========================================================================
# TestEdgeCases (10 tests)
# ===========================================================================


class TestEdgeCases:
    """Test edge cases including seed reproducibility and zero uncertainty."""

    def test_seed_reproducibility(self, engine, cement_input):
        """Same seed should produce identical results."""
        r1 = engine.run_monte_carlo(cement_input, n_iterations=500, seed=12345)
        r2 = engine.run_monte_carlo(cement_input, n_iterations=500, seed=12345)
        assert r1.monte_carlo_mean == r2.monte_carlo_mean
        assert r1.monte_carlo_std == r2.monte_carlo_std

    def test_different_seeds_different_results(self, engine, cement_input):
        """Different seeds should produce different MC results."""
        r1 = engine.run_monte_carlo(cement_input, n_iterations=500, seed=1)
        r2 = engine.run_monte_carlo(cement_input, n_iterations=500, seed=2)
        # Mean may differ slightly due to different random draws
        assert r1.monte_carlo_mean != r2.monte_carlo_mean

    def test_zero_emissions(self, engine):
        """Zero emissions should produce zero uncertainty."""
        data = {
            "total_co2e_kg": 0,
            "process_type": "CEMENT",
            "calculation_method": "EMISSION_FACTOR",
            "tier": "TIER_1",
        }
        result = engine.run_monte_carlo(data, n_iterations=500, seed=42)
        assert result.monte_carlo_mean is None or result.monte_carlo_mean == Decimal("0")

    def test_very_small_emissions(self, engine):
        """Very small emissions should still produce valid results."""
        data = {
            "total_co2e_kg": 0.001,
            "process_type": "CEMENT",
            "calculation_method": "EMISSION_FACTOR",
            "tier": "TIER_1",
            "activity_data_source": "metered",
        }
        result = engine.run_monte_carlo(data, n_iterations=500, seed=42)
        assert result is not None
        assert result.combined_uncertainty_pct >= Decimal("0")

    def test_very_large_emissions(self, engine):
        """Very large emissions should produce valid results."""
        data = {
            "total_co2e_kg": 1e12,
            "process_type": "CEMENT",
            "calculation_method": "EMISSION_FACTOR",
            "tier": "TIER_1",
            "activity_data_source": "metered",
        }
        result = engine.run_monte_carlo(data, n_iterations=500, seed=42)
        assert result is not None
        assert result.monte_carlo_mean is not None

    def test_minimum_iterations(self, engine, cement_input):
        """Minimum iteration count (100) should work."""
        result = engine.run_monte_carlo(cement_input, n_iterations=100, seed=42)
        assert result.monte_carlo_iterations == 100

    def test_default_seed(self, engine, cement_input):
        """None seed should default to 42."""
        result = engine.run_monte_carlo(cement_input, n_iterations=500, seed=None)
        assert result.monte_carlo_seed == 42

    def test_clear_history(self, engine, cement_input):
        """Clear history should remove all assessments."""
        engine.run_monte_carlo(cement_input, n_iterations=500, seed=42)
        count = engine.clear_history()
        assert count >= 1
        assert len(engine.get_assessment_history()) == 0

    def test_combine_uncertainties_rss(self, engine):
        """combine_uncertainties should use root-sum-of-squares."""
        components = [Decimal("0.03"), Decimal("0.04")]
        result = engine.combine_uncertainties(components)
        expected = Decimal(str(math.sqrt(0.03**2 + 0.04**2)))
        assert float(result) == pytest.approx(float(expected), abs=0.001)

    def test_combine_uncertainties_empty_raises(self, engine):
        """combine_uncertainties with empty list should raise."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.combine_uncertainties([])
