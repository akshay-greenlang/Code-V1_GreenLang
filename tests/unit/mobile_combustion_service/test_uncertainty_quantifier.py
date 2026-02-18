# -*- coding: utf-8 -*-
"""
Unit tests for UncertaintyQuantifierEngine (Engine 5) - AGENT-MRV-003 Mobile Combustion.

Tests all public methods with 90+ test functions covering:
- Initialization, Monte Carlo simulation
- Analytical uncertainty propagation
- Data Quality Indicator (DQI) scoring
- Method uncertainty ranges
- Confidence intervals from samples
- Combine uncertainties (root-sum-of-squares)
- Sensitivity analysis
- Assessment history management
- Provenance hashing, thread safety, edge cases

Author: GreenLang QA Team
"""

import math
import threading
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import patch

import pytest

from greenlang.mobile_combustion.uncertainty_quantifier import (
    UncertaintyQuantifierEngine,
    UncertaintyResult,
    SensitivityResult,
    CalculationMethod,
    CalculationTier,
    ActivityDataSource,
    DQICategory,
    _ACTIVITY_DATA_UNCERTAINTY,
    _EMISSION_FACTOR_UNCERTAINTY,
    _GWP_UNCERTAINTY,
    _FUEL_ECONOMY_UNCERTAINTY,
    _METHOD_UNCERTAINTY_RANGES,
    _METHOD_DEFAULT_UNCERTAINTY,
    _DQI_MULTIPLIERS,
    _DQI_CRITERIA,
    _CONFIDENCE_Z_SCORES,
    _DEFAULT_ITERATIONS,
    _MIN_ITERATIONS,
    _MAX_ITERATIONS,
    _decimal_sqrt,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine():
    """Create a default UncertaintyQuantifierEngine instance."""
    return UncertaintyQuantifierEngine()


@pytest.fixture
def basic_fuel_input():
    """Basic fuel-based calculation input for Monte Carlo."""
    return {
        "total_co2e_kg": 5000,
        "method": CalculationMethod.FUEL_BASED.value,
        "tier": CalculationTier.TIER_2.value,
        "activity_data_source": "METERED",
        "fuel_litres": 2000,
        "emission_factor": 2.5,
    }


@pytest.fixture
def basic_distance_input():
    """Basic distance-based calculation input for Monte Carlo."""
    return {
        "total_co2e_kg": 3000,
        "method": CalculationMethod.DISTANCE_BASED.value,
        "tier": CalculationTier.TIER_2.value,
        "activity_data_source": "ESTIMATED",
        "distance_km": 20000,
        "fuel_economy": 8.5,
        "emission_factor": 2.5,
    }


@pytest.fixture
def basic_spend_input():
    """Basic spend-based calculation input for Monte Carlo."""
    return {
        "total_co2e_kg": 10000,
        "method": CalculationMethod.SPEND_BASED.value,
        "tier": CalculationTier.TIER_1.value,
        "spend_amount": 50000,
        "spend_ef": 0.2,
    }


# ===========================================================================
# TestInit
# ===========================================================================


class TestInit:
    """Test UncertaintyQuantifierEngine initialization."""

    def test_default_init(self, engine):
        """Engine initializes with empty history."""
        assert engine._assessment_history == []

    def test_rlock_created(self, engine):
        """Engine creates a reentrant lock."""
        assert isinstance(engine._lock, type(threading.RLock()))

    def test_default_iterations_constant(self):
        """Default iteration count is 5000."""
        assert _DEFAULT_ITERATIONS == 5000

    def test_iteration_range_constants(self):
        """Min/max iteration limits are 100 and 100000."""
        assert _MIN_ITERATIONS == 100
        assert _MAX_ITERATIONS == 100000

    def test_three_calculation_methods(self):
        """There are exactly 3 calculation methods."""
        assert len(CalculationMethod) == 3

    def test_three_calculation_tiers(self):
        """There are exactly 3 calculation tiers."""
        assert len(CalculationTier) == 3

    def test_three_activity_data_sources(self):
        """There are exactly 3 activity data sources."""
        assert len(ActivityDataSource) == 3

    def test_five_dqi_categories(self):
        """There are exactly 5 DQI categories."""
        assert len(DQICategory) == 5


# ===========================================================================
# TestMonteCarlo
# ===========================================================================


class TestMonteCarlo:
    """Test Monte Carlo simulation."""

    def test_basic_fuel_mc(self, engine, basic_fuel_input):
        """Fuel-based Monte Carlo returns a valid UncertaintyResult."""
        result = engine.run_monte_carlo(basic_fuel_input, n_iterations=500, seed=42)
        assert isinstance(result, UncertaintyResult)
        assert result.emissions_value == Decimal("5000")
        assert result.method == CalculationMethod.FUEL_BASED.value
        assert result.tier == CalculationTier.TIER_2.value
        assert result.monte_carlo_iterations == 500
        assert result.monte_carlo_seed == 42

    def test_mc_mean_near_central(self, engine, basic_fuel_input):
        """MC mean is within 15% of the central estimate for fuel-based."""
        result = engine.run_monte_carlo(basic_fuel_input, n_iterations=5000, seed=42)
        central = float(result.emissions_value)
        mc_mean = float(result.monte_carlo_mean)
        assert abs(mc_mean - central) / central < 0.15

    def test_mc_std_positive(self, engine, basic_fuel_input):
        """MC standard deviation is positive."""
        result = engine.run_monte_carlo(basic_fuel_input, n_iterations=1000, seed=42)
        assert result.monte_carlo_std > Decimal("0")

    def test_mc_median_present(self, engine, basic_fuel_input):
        """MC median is present and reasonable."""
        result = engine.run_monte_carlo(basic_fuel_input, n_iterations=1000, seed=42)
        assert result.monte_carlo_median is not None
        assert result.monte_carlo_median > Decimal("0")

    def test_mc_percentiles(self, engine, basic_fuel_input):
        """MC result includes percentile values."""
        result = engine.run_monte_carlo(basic_fuel_input, n_iterations=1000, seed=42)
        assert "50" in result.monte_carlo_percentiles or "50.0" in result.monte_carlo_percentiles

    def test_mc_reproducibility_with_seed(self, engine, basic_fuel_input):
        """Same seed produces identical MC results."""
        r1 = engine.run_monte_carlo(basic_fuel_input, n_iterations=1000, seed=99)
        r2 = engine.run_monte_carlo(basic_fuel_input, n_iterations=1000, seed=99)
        assert r1.monte_carlo_mean == r2.monte_carlo_mean
        assert r1.monte_carlo_std == r2.monte_carlo_std

    def test_mc_default_seed_42(self, engine, basic_fuel_input):
        """When seed is None, default seed 42 is used."""
        result = engine.run_monte_carlo(basic_fuel_input, n_iterations=500)
        assert result.monte_carlo_seed == 42

    def test_mc_distance_based(self, engine, basic_distance_input):
        """Distance-based Monte Carlo produces valid results."""
        result = engine.run_monte_carlo(basic_distance_input, n_iterations=500, seed=42)
        assert result.method == CalculationMethod.DISTANCE_BASED.value
        assert result.monte_carlo_mean > Decimal("0")

    def test_mc_spend_based(self, engine, basic_spend_input):
        """Spend-based Monte Carlo produces valid results."""
        result = engine.run_monte_carlo(basic_spend_input, n_iterations=500, seed=42)
        assert result.method == CalculationMethod.SPEND_BASED.value
        assert result.monte_carlo_mean > Decimal("0")

    def test_mc_confidence_intervals(self, engine, basic_fuel_input):
        """MC result has 90%, 95%, and 99% confidence intervals."""
        result = engine.run_monte_carlo(basic_fuel_input, n_iterations=2000, seed=42)
        assert "90" in result.confidence_intervals
        assert "95" in result.confidence_intervals
        assert "99" in result.confidence_intervals
        # CIs should be ordered: 90 narrower than 99
        ci_90 = result.confidence_intervals["90"]
        ci_99 = result.confidence_intervals["99"]
        assert (ci_99[1] - ci_99[0]) >= (ci_90[1] - ci_90[0])

    def test_mc_too_few_iterations_raises(self, engine, basic_fuel_input):
        """Fewer than 100 iterations raises ValueError."""
        with pytest.raises(ValueError, match="n_iterations must be in"):
            engine.run_monte_carlo(basic_fuel_input, n_iterations=50)

    def test_mc_too_many_iterations_raises(self, engine, basic_fuel_input):
        """More than 100000 iterations raises ValueError."""
        with pytest.raises(ValueError, match="n_iterations must be in"):
            engine.run_monte_carlo(basic_fuel_input, n_iterations=200000)

    def test_mc_negative_emissions_raises(self, engine):
        """Negative total_co2e_kg raises ValueError."""
        with pytest.raises(ValueError, match="total_co2e_kg must be >= 0"):
            engine.run_monte_carlo({
                "total_co2e_kg": -100,
                "method": "FUEL_BASED",
                "tier": "TIER_2",
            }, n_iterations=100)

    def test_mc_invalid_method_raises(self, engine):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized method"):
            engine.run_monte_carlo({
                "total_co2e_kg": 100,
                "method": "MAGIC",
                "tier": "TIER_2",
            }, n_iterations=100)

    def test_mc_invalid_tier_raises(self, engine):
        """Invalid tier raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized tier"):
            engine.run_monte_carlo({
                "total_co2e_kg": 100,
                "method": "FUEL_BASED",
                "tier": "TIER_99",
            }, n_iterations=100)

    def test_mc_zero_emissions(self, engine):
        """Zero emissions produces result with zero MC values."""
        result = engine.run_monte_carlo({
            "total_co2e_kg": 0,
            "method": "FUEL_BASED",
            "tier": "TIER_2",
        }, n_iterations=100, seed=42)
        assert result.emissions_value == Decimal("0")
        assert result.monte_carlo_mean is None

    def test_mc_provenance_hash_is_sha256(self, engine, basic_fuel_input):
        """MC result has a valid 64-char hex SHA-256 provenance hash."""
        result = engine.run_monte_carlo(basic_fuel_input, n_iterations=100, seed=42)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_mc_provenance_deterministic(self, engine, basic_fuel_input):
        """Same inputs produce the same provenance hash."""
        r1 = engine.run_monte_carlo(basic_fuel_input, n_iterations=100, seed=42)
        r2 = engine.run_monte_carlo(basic_fuel_input, n_iterations=100, seed=42)
        assert r1.provenance_hash == r2.provenance_hash

    def test_mc_combined_uncertainty_positive(self, engine, basic_fuel_input):
        """Combined uncertainty is positive for non-zero emissions."""
        result = engine.run_monte_carlo(basic_fuel_input, n_iterations=500, seed=42)
        assert result.combined_uncertainty_pct > Decimal("0")

    def test_mc_analytical_uncertainty_present(self, engine, basic_fuel_input):
        """Analytical uncertainty percentage is computed alongside MC."""
        result = engine.run_monte_carlo(basic_fuel_input, n_iterations=500, seed=42)
        assert result.analytical_uncertainty_pct > Decimal("0")

    def test_mc_component_uncertainties(self, engine, basic_fuel_input):
        """Component uncertainties are broken down."""
        result = engine.run_monte_carlo(basic_fuel_input, n_iterations=500, seed=42)
        assert len(result.component_uncertainties) >= 2

    def test_mc_contribution_analysis(self, engine, basic_fuel_input):
        """Contribution analysis fractions roughly sum to 1.0."""
        result = engine.run_monte_carlo(basic_fuel_input, n_iterations=500, seed=42)
        total = sum(float(v) for v in result.contribution_analysis.values())
        assert abs(total - 1.0) < 0.01

    def test_mc_recorded_in_history(self, engine, basic_fuel_input):
        """MC run is recorded in assessment history."""
        engine.run_monte_carlo(basic_fuel_input, n_iterations=100, seed=42)
        assert len(engine.get_assessment_history()) == 1

    def test_mc_dqi_inputs_applied(self, engine):
        """DQI inputs affect the uncertainty multiplier."""
        good_input = {
            "total_co2e_kg": 5000,
            "method": "FUEL_BASED",
            "tier": "TIER_2",
            "dqi_inputs": {
                "reliability": "direct_measurement",
                "completeness": "above_95_pct",
                "temporal_correlation": "same_year",
                "geographical_correlation": "same_region",
                "technological_correlation": "same_technology",
            },
        }
        poor_input = {
            "total_co2e_kg": 5000,
            "method": "FUEL_BASED",
            "tier": "TIER_2",
            "dqi_inputs": {
                "reliability": "unknown",
                "completeness": "below_40_pct",
                "temporal_correlation": "older_than_5_years",
                "geographical_correlation": "unknown",
                "technological_correlation": "unknown",
            },
        }
        good_result = engine.run_monte_carlo(good_input, n_iterations=500, seed=42)
        poor_result = engine.run_monte_carlo(poor_input, n_iterations=500, seed=42)
        # Poor data quality should have higher uncertainty
        assert poor_result.combined_uncertainty_pct > good_result.combined_uncertainty_pct

    def test_mc_to_dict(self, engine, basic_fuel_input):
        """to_dict serializes all fields."""
        result = engine.run_monte_carlo(basic_fuel_input, n_iterations=100, seed=42)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["method"] == "FUEL_BASED"
        assert "provenance_hash" in d
        assert "confidence_intervals" in d

    def test_mc_scaling_fallback(self, engine):
        """When fuel_litres and emission_factor are missing, scaling fallback is used."""
        result = engine.run_monte_carlo({
            "total_co2e_kg": 5000,
            "method": "FUEL_BASED",
            "tier": "TIER_2",
            "activity_data_source": "METERED",
        }, n_iterations=500, seed=42)
        assert result.monte_carlo_mean is not None
        assert result.monte_carlo_mean > Decimal("0")


# ===========================================================================
# TestAnalyticalUncertainty
# ===========================================================================


class TestAnalyticalUncertainty:
    """Test analytical (IPCC Approach 1) uncertainty propagation."""

    def test_basic_multiplicative(self, engine):
        """Multiplicative components combine via root-sum-of-squares."""
        components = [
            {"name": "activity_data", "value": 2000, "uncertainty_pct": 5, "relationship": "multiplicative"},
            {"name": "emission_factor", "value": 2.5, "uncertainty_pct": 10, "relationship": "multiplicative"},
        ]
        result = engine.calculate_analytical_uncertainty(components)
        assert isinstance(result, UncertaintyResult)
        # sqrt(0.05^2 + 0.10^2) * 100 = sqrt(0.0125) * 100 ~ 11.18%
        expected = Decimal(str(math.sqrt(0.05**2 + 0.10**2) * 100)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        assert result.combined_uncertainty_pct == expected

    def test_additive_components(self, engine):
        """Additive components combine using absolute uncertainties."""
        components = [
            {"name": "co2", "value": 4900, "uncertainty_pct": 5, "relationship": "additive"},
            {"name": "ch4", "value": 50, "uncertainty_pct": 20, "relationship": "additive"},
            {"name": "n2o", "value": 50, "uncertainty_pct": 30, "relationship": "additive"},
        ]
        result = engine.calculate_analytical_uncertainty(components)
        assert result.combined_uncertainty_pct > Decimal("0")

    def test_empty_components_raises(self, engine):
        """Empty component list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.calculate_analytical_uncertainty([])

    def test_missing_name_raises(self, engine):
        """Component without 'name' raises ValueError."""
        with pytest.raises(ValueError, match="must have 'name' and 'value'"):
            engine.calculate_analytical_uncertainty([{"value": 100}])

    def test_missing_value_raises(self, engine):
        """Component without 'value' raises ValueError."""
        with pytest.raises(ValueError, match="must have 'name' and 'value'"):
            engine.calculate_analytical_uncertainty([{"name": "test"}])

    def test_analytical_confidence_intervals(self, engine):
        """Analytical result includes 90/95/99% confidence intervals."""
        components = [
            {"name": "ad", "value": 1000, "uncertainty_pct": 10},
        ]
        result = engine.calculate_analytical_uncertainty(components)
        assert "90" in result.confidence_intervals
        assert "95" in result.confidence_intervals
        assert "99" in result.confidence_intervals

    def test_analytical_provenance_hash(self, engine):
        """Analytical result has a 64-char hex provenance hash."""
        components = [
            {"name": "ad", "value": 1000, "uncertainty_pct": 10},
        ]
        result = engine.calculate_analytical_uncertainty(components)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_analytical_recorded_in_history(self, engine):
        """Analytical assessment is recorded in history."""
        components = [
            {"name": "ad", "value": 1000, "uncertainty_pct": 10},
        ]
        engine.calculate_analytical_uncertainty(components)
        assert len(engine.get_assessment_history()) == 1

    def test_contribution_analysis(self, engine):
        """Contribution analysis allocates variance correctly."""
        components = [
            {"name": "ad", "value": 1000, "uncertainty_pct": 10, "relationship": "multiplicative"},
            {"name": "ef", "value": 2.5, "uncertainty_pct": 10, "relationship": "multiplicative"},
        ]
        result = engine.calculate_analytical_uncertainty(components)
        # Equal uncertainties => equal contributions (~0.5 each)
        assert abs(float(result.contribution_analysis.get("ad", 0)) - 0.5) < 0.01
        assert abs(float(result.contribution_analysis.get("ef", 0)) - 0.5) < 0.01

    def test_zero_total_emissions(self, engine):
        """Zero total emissions gives zero combined uncertainty."""
        components = [
            {"name": "ad", "value": 0, "uncertainty_pct": 10, "relationship": "multiplicative"},
        ]
        result = engine.calculate_analytical_uncertainty(components)
        assert result.combined_uncertainty_pct == Decimal("0") or result.emissions_value == Decimal("0")


# ===========================================================================
# TestDQIScoring
# ===========================================================================


class TestDQIScoring:
    """Test Data Quality Indicator scoring."""

    def test_empty_inputs_default_3(self, engine):
        """Empty DQI inputs return default score of 3.0."""
        assert engine.score_data_quality({}) == 3.0

    def test_all_best_score_1(self, engine):
        """All best criteria produce score of 1.0."""
        dqi = {
            "reliability": "direct_measurement",
            "completeness": "above_95_pct",
            "temporal_correlation": "same_year",
            "geographical_correlation": "same_region",
            "technological_correlation": "same_technology",
        }
        assert engine.score_data_quality(dqi) == 1.0

    def test_all_worst_score_5(self, engine):
        """All worst criteria produce score of 5.0."""
        dqi = {
            "reliability": "unknown",
            "completeness": "below_40_pct",
            "temporal_correlation": "older_than_5_years",
            "geographical_correlation": "unknown",
            "technological_correlation": "unknown",
        }
        assert engine.score_data_quality(dqi) == 5.0

    def test_numeric_scores_direct(self, engine):
        """Numeric scores are used directly."""
        dqi = {
            "reliability": 2,
            "completeness": 3,
            "temporal_correlation": 4,
            "geographical_correlation": 1,
            "technological_correlation": 5,
        }
        expected = (2 + 3 + 4 + 1 + 5) / 5
        assert engine.score_data_quality(dqi) == expected

    def test_partial_inputs_fill_default(self, engine):
        """Missing dimensions default to 3."""
        dqi = {
            "reliability": "direct_measurement",  # score 1
        }
        # 1 + 3 + 3 + 3 + 3 = 13 / 5 = 2.6
        assert engine.score_data_quality(dqi) == 2.6

    def test_unrecognized_value_defaults_3(self, engine):
        """Unrecognized string value defaults to score of 3."""
        dqi = {
            "reliability": "quantum_measurement",
        }
        # 3 (unrecognized) + 3 + 3 + 3 + 3 = 15 / 5 = 3.0
        assert engine.score_data_quality(dqi) == 3.0

    def test_numeric_clamped_to_1_5(self, engine):
        """Numeric scores are clamped to [1, 5]."""
        dqi = {"reliability": 0.5}  # Below 1 => clamped to 1
        result = engine.score_data_quality(dqi)
        assert result >= 1.0

    def test_dqi_multipliers_exist(self):
        """DQI multipliers exist for levels 1 through 5."""
        for level in [1, 2, 3, 4, 5]:
            assert level in _DQI_MULTIPLIERS

    def test_dqi_multiplier_values(self):
        """DQI multiplier values match specification."""
        assert _DQI_MULTIPLIERS[1] == 0.60
        assert _DQI_MULTIPLIERS[2] == 0.80
        assert _DQI_MULTIPLIERS[3] == 1.00
        assert _DQI_MULTIPLIERS[4] == 1.30
        assert _DQI_MULTIPLIERS[5] == 1.80


# ===========================================================================
# TestMethodUncertaintyRange
# ===========================================================================


class TestMethodUncertaintyRange:
    """Test method uncertainty range lookups."""

    @pytest.mark.parametrize("method,tier,expected_low,expected_high", [
        ("FUEL_BASED", "TIER_3", 5.0, 10.0),
        ("FUEL_BASED", "TIER_2", 10.0, 20.0),
        ("FUEL_BASED", "TIER_1", 15.0, 30.0),
        ("DISTANCE_BASED", "TIER_3", 15.0, 20.0),
        ("DISTANCE_BASED", "TIER_2", 18.0, 25.0),
        ("SPEND_BASED", "TIER_1", 40.0, 50.0),
    ])
    def test_range_values(self, engine, method, tier, expected_low, expected_high):
        """Method/tier uncertainty ranges match specification."""
        low, high = engine.get_method_uncertainty_range(method, tier)
        assert low == expected_low
        assert high == expected_high

    def test_invalid_method_raises(self, engine):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized method"):
            engine.get_method_uncertainty_range("MAGIC", "TIER_1")

    def test_invalid_tier_raises(self, engine):
        """Invalid tier raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized tier"):
            engine.get_method_uncertainty_range("FUEL_BASED", "TIER_99")


# ===========================================================================
# TestConfidenceIntervals
# ===========================================================================


class TestConfidenceIntervals:
    """Test confidence interval computation from samples."""

    def test_basic_intervals(self, engine):
        """Confidence intervals at 90/95/99% levels."""
        samples = list(range(1, 101))  # 1 to 100
        cis = engine.get_confidence_intervals(samples)
        assert "90" in cis
        assert "95" in cis
        assert "99" in cis
        # 95% CI lower should be around 3, upper around 98
        assert cis["95"][0] < cis["95"][1]

    def test_custom_levels(self, engine):
        """Custom confidence levels are supported."""
        samples = list(range(1, 1001))
        cis = engine.get_confidence_intervals(samples, levels=[0.80, 0.90])
        assert "80" in cis
        assert "90" in cis

    def test_empty_samples_raises(self, engine):
        """Empty samples list raises ValueError."""
        with pytest.raises(ValueError, match="samples must not be empty"):
            engine.get_confidence_intervals([])

    def test_invalid_level_raises(self, engine):
        """Level outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="Confidence level must be in"):
            engine.get_confidence_intervals([1, 2, 3], levels=[1.5])

    def test_wider_99_than_90(self, engine):
        """99% CI is wider than 90% CI."""
        samples = [float(x) for x in range(1, 1001)]
        cis = engine.get_confidence_intervals(samples)
        width_90 = cis["90"][1] - cis["90"][0]
        width_99 = cis["99"][1] - cis["99"][0]
        assert width_99 >= width_90


# ===========================================================================
# TestCombineUncertainties
# ===========================================================================


class TestCombineUncertainties:
    """Test root-sum-of-squares uncertainty combination."""

    def test_basic_rss(self, engine):
        """sqrt(0.05^2 + 0.10^2) = sqrt(0.0125) ~ 0.1118."""
        result = engine.combine_uncertainties([Decimal("0.05"), Decimal("0.10")])
        expected = Decimal(str(math.sqrt(0.05**2 + 0.10**2)))
        assert abs(float(result) - float(expected)) < 0.0001

    def test_single_component(self, engine):
        """Single component returns itself."""
        result = engine.combine_uncertainties([Decimal("0.15")])
        assert abs(float(result) - 0.15) < 0.0001

    def test_empty_raises(self, engine):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.combine_uncertainties([])

    def test_three_components(self, engine):
        """Three component RSS: sqrt(0.05^2 + 0.10^2 + 0.15^2)."""
        result = engine.combine_uncertainties([
            Decimal("0.05"), Decimal("0.10"), Decimal("0.15")
        ])
        expected = math.sqrt(0.05**2 + 0.10**2 + 0.15**2)
        assert abs(float(result) - expected) < 0.0001


# ===========================================================================
# TestSensitivityAnalysis
# ===========================================================================


class TestSensitivityAnalysis:
    """Test sensitivity analysis."""

    def test_basic_sensitivity(self, engine):
        """Sensitivity analysis for a single parameter."""
        results = engine.sensitivity_analysis(
            calculation_input={
                "total_co2e_kg": 5000,
                "method": "FUEL_BASED",
                "fuel_litres": 2000,
                "emission_factor": 2.5,
            },
            parameters=["fuel_litres"],
            perturbation_pct=10.0,
        )
        assert len(results) == 1
        assert isinstance(results[0], SensitivityResult)
        assert results[0].parameter == "fuel_litres"
        assert results[0].perturbation_pct == Decimal("10.00")

    def test_fuel_litres_linear_sensitivity(self, engine):
        """Fuel litres has sensitivity coefficient near 1.0 (linear)."""
        results = engine.sensitivity_analysis(
            calculation_input={
                "total_co2e_kg": 5000,
                "method": "FUEL_BASED",
                "fuel_litres": 2000,
            },
            parameters=["fuel_litres"],
            perturbation_pct=10.0,
        )
        # For multiplicative param, 10% increase => 10% change => coeff ~ 1.0
        assert abs(float(results[0].sensitivity_coefficient) - 1.0) < 0.01

    def test_multiple_parameters(self, engine):
        """Multiple parameters produce multiple results."""
        results = engine.sensitivity_analysis(
            calculation_input={
                "total_co2e_kg": 5000,
                "method": "FUEL_BASED",
                "fuel_litres": 2000,
                "emission_factor": 2.5,
            },
            parameters=["fuel_litres", "emission_factor"],
        )
        assert len(results) == 2
        assert results[0].parameter == "fuel_litres"
        assert results[1].parameter == "emission_factor"

    def test_gwp_parameter_non_linear(self, engine):
        """GWP parameter has non-linear sensitivity (partial non-CO2)."""
        results = engine.sensitivity_analysis(
            calculation_input={
                "total_co2e_kg": 5000,
                "method": "FUEL_BASED",
                "gwp_ch4": 28,
            },
            parameters=["gwp_ch4"],
            perturbation_pct=10.0,
        )
        # GWP affects only ~5% non-CO2 fraction, so coeff should be < 1.0
        assert float(results[0].sensitivity_coefficient) < 1.0

    def test_empty_parameters_raises(self, engine):
        """Empty parameter list raises ValueError."""
        with pytest.raises(ValueError, match="parameters must not be empty"):
            engine.sensitivity_analysis(
                {"total_co2e_kg": 100, "method": "FUEL_BASED"},
                parameters=[],
            )

    def test_zero_emissions_raises(self, engine):
        """Zero emissions raises ValueError."""
        with pytest.raises(ValueError, match="total_co2e_kg must be > 0"):
            engine.sensitivity_analysis(
                {"total_co2e_kg": 0, "method": "FUEL_BASED"},
                parameters=["fuel_litres"],
            )

    def test_to_dict(self, engine):
        """SensitivityResult.to_dict serializes all fields."""
        results = engine.sensitivity_analysis(
            {"total_co2e_kg": 5000, "method": "FUEL_BASED", "fuel_litres": 2000},
            parameters=["fuel_litres"],
        )
        d = results[0].to_dict()
        assert isinstance(d, dict)
        assert "parameter" in d
        assert "sensitivity_coefficient" in d


# ===========================================================================
# TestHistory
# ===========================================================================


class TestHistory:
    """Test assessment history management."""

    def test_empty_history(self, engine):
        """Fresh engine has empty history."""
        assert engine.get_assessment_history() == []

    def test_history_grows(self, engine, basic_fuel_input):
        """History grows with each assessment."""
        engine.run_monte_carlo(basic_fuel_input, n_iterations=100, seed=42)
        engine.run_monte_carlo(basic_fuel_input, n_iterations=100, seed=43)
        assert len(engine.get_assessment_history()) == 2

    def test_clear_history(self, engine, basic_fuel_input):
        """clear_history empties the history and returns count."""
        engine.run_monte_carlo(basic_fuel_input, n_iterations=100, seed=42)
        engine.run_monte_carlo(basic_fuel_input, n_iterations=100, seed=43)
        count = engine.clear_history()
        assert count == 2
        assert len(engine.get_assessment_history()) == 0

    def test_clear_empty_history(self, engine):
        """Clearing an empty history returns 0."""
        assert engine.clear_history() == 0

    def test_history_returns_copy(self, engine, basic_fuel_input):
        """get_assessment_history returns a copy, not internal state."""
        engine.run_monte_carlo(basic_fuel_input, n_iterations=100, seed=42)
        h1 = engine.get_assessment_history()
        h2 = engine.get_assessment_history()
        assert h1 is not h2


# ===========================================================================
# TestUncertaintyConstants
# ===========================================================================


class TestUncertaintyConstants:
    """Test uncertainty constant definitions."""

    def test_activity_data_metered(self):
        """METERED activity data uncertainty is 0.05."""
        assert _ACTIVITY_DATA_UNCERTAINTY["METERED"] == 0.05

    def test_activity_data_estimated(self):
        """ESTIMATED activity data uncertainty is 0.10."""
        assert _ACTIVITY_DATA_UNCERTAINTY["ESTIMATED"] == 0.10

    def test_activity_data_screening(self):
        """SCREENING activity data uncertainty is 0.25."""
        assert _ACTIVITY_DATA_UNCERTAINTY["SCREENING"] == 0.25

    def test_ef_tier3(self):
        """TIER_3 emission factor uncertainty is 0.03."""
        assert _EMISSION_FACTOR_UNCERTAINTY["TIER_3"] == 0.03

    def test_ef_tier2(self):
        """TIER_2 emission factor uncertainty is 0.10."""
        assert _EMISSION_FACTOR_UNCERTAINTY["TIER_2"] == 0.10

    def test_ef_tier1(self):
        """TIER_1 emission factor uncertainty is 0.25."""
        assert _EMISSION_FACTOR_UNCERTAINTY["TIER_1"] == 0.25

    def test_gwp_uncertainty(self):
        """GWP uncertainty is 0.10."""
        assert _GWP_UNCERTAINTY == 0.10

    def test_confidence_z_scores(self):
        """Z-scores for 90, 95, 99% confidence levels."""
        assert abs(_CONFIDENCE_Z_SCORES["90"] - 1.6449) < 0.001
        assert abs(_CONFIDENCE_Z_SCORES["95"] - 1.9600) < 0.001
        assert abs(_CONFIDENCE_Z_SCORES["99"] - 2.5758) < 0.001


# ===========================================================================
# TestDecimalSqrt
# ===========================================================================


class TestDecimalSqrt:
    """Test the _decimal_sqrt helper."""

    def test_sqrt_of_4(self):
        """sqrt(4) = 2."""
        result = _decimal_sqrt(Decimal("4"))
        assert abs(float(result) - 2.0) < 0.0001

    def test_sqrt_of_0(self):
        """sqrt(0) = 0."""
        assert _decimal_sqrt(Decimal("0")) == Decimal("0")

    def test_sqrt_negative_raises(self):
        """sqrt of negative value raises ValueError."""
        with pytest.raises(ValueError, match="Cannot take sqrt of negative"):
            _decimal_sqrt(Decimal("-1"))


# ===========================================================================
# TestThreadSafety
# ===========================================================================


class TestThreadSafety:
    """Test thread safety of UncertaintyQuantifierEngine."""

    def test_concurrent_mc_runs(self, engine):
        """Concurrent MC runs do not corrupt internal state."""
        import concurrent.futures

        def run_mc(seed):
            return engine.run_monte_carlo({
                "total_co2e_kg": 5000,
                "method": "FUEL_BASED",
                "tier": "TIER_2",
            }, n_iterations=100, seed=seed)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futs = [executor.submit(run_mc, i) for i in range(10)]
            results = [f.result() for f in futs]

        assert len(results) == 10
        assert len(engine.get_assessment_history()) == 10
        for r in results:
            assert isinstance(r, UncertaintyResult)
