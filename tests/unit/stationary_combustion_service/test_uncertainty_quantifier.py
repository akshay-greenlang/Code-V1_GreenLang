# -*- coding: utf-8 -*-
"""
Unit tests for UncertaintyQuantifierEngine (Engine 5)

AGENT-MRV-001 Stationary Combustion Agent

Tests Monte Carlo simulation, analytical propagation, data quality scoring,
confidence intervals, contribution analysis, reproducibility, and summary
statistics. 40+ tests covering all public methods and edge cases.

Target: 85%+ code coverage of uncertainty_quantifier.py
"""

import math
import threading
from unittest.mock import patch, MagicMock

import pytest

from greenlang.stationary_combustion.uncertainty_quantifier import (
    UncertaintyQuantifierEngine,
    UncertaintyResult,
    DEFAULT_UNCERTAINTIES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Create a fresh UncertaintyQuantifierEngine with no external config."""
    return UncertaintyQuantifierEngine(config=None)


@pytest.fixture
def metered_tier1_input():
    """Standard calculation result: metered activity data, TIER_1 EF."""
    return {
        "total_co2e_kg": 5000.0,
        "activity_data_source": "metered",
        "ef_tier": "TIER_1",
        "heating_value_type": "default",
        "oxidation_factor_type": "default",
    }


@pytest.fixture
def invoiced_tier2_input():
    """Invoiced data, TIER_2 EF, measured HV."""
    return {
        "total_co2e_kg": 12000.0,
        "activity_data_source": "invoiced",
        "ef_tier": "TIER_2",
        "heating_value_type": "measured",
        "oxidation_factor_type": "default",
    }


@pytest.fixture
def estimated_tier1_input():
    """Worst-case data quality: estimated data, TIER_1 EF."""
    return {
        "total_co2e_kg": 8000.0,
        "activity_data_source": "estimated",
        "ef_tier": "TIER_1",
        "heating_value_type": "default",
        "oxidation_factor_type": "default",
    }


@pytest.fixture
def high_quality_dqi():
    """Best possible data quality info."""
    return {
        "data_source": "metered",
        "measurement_method": "continuous_monitoring",
        "data_age_months": 3,
        "completeness_pct": 99.0,
    }


@pytest.fixture
def low_quality_dqi():
    """Worst possible data quality info."""
    return {
        "data_source": "estimated",
        "measurement_method": "estimate",
        "data_age_months": 48,
        "completeness_pct": 20.0,
    }


# ---------------------------------------------------------------------------
# TestUncertaintyInit
# ---------------------------------------------------------------------------


class TestUncertaintyInit:
    """Test UncertaintyQuantifierEngine initialisation."""

    def test_default_initialisation(self, engine):
        """Engine initialises with sensible defaults."""
        assert engine._default_iterations == 5_000
        assert engine._default_seed == 42
        assert engine._confidence_levels == [90, 95, 99]
        assert engine._history == []

    def test_custom_config_iterations(self):
        """Config object overrides default iterations."""
        cfg = MagicMock()
        cfg.monte_carlo_iterations = 10_000
        cfg.confidence_levels = "90,95,99"
        eng = UncertaintyQuantifierEngine(config=cfg)
        assert eng._default_iterations == 10_000

    def test_custom_config_confidence_levels(self):
        """Config object overrides default confidence levels."""
        cfg = MagicMock()
        cfg.monte_carlo_iterations = 5_000
        cfg.confidence_levels = "80,90,95"
        eng = UncertaintyQuantifierEngine(config=cfg)
        assert eng._confidence_levels == [80, 90, 95]

    def test_none_config_fallback(self):
        """When config is None and get_config raises, defaults are used."""
        with patch(
            "greenlang.stationary_combustion.uncertainty_quantifier.get_config",
            side_effect=Exception("no config"),
        ):
            eng = UncertaintyQuantifierEngine(config=None)
            assert eng._default_iterations == 5_000

    def test_lock_is_reentrant(self, engine):
        """Engine lock is an RLock (reentrant)."""
        assert isinstance(engine._lock, type(threading.RLock()))


# ---------------------------------------------------------------------------
# TestQuantifyUncertainty (primary method)
# ---------------------------------------------------------------------------


class TestQuantifyUncertainty:
    """Test the top-level quantify_uncertainty method."""

    def test_returns_uncertainty_result(self, engine, metered_tier1_input):
        """quantify_uncertainty returns an UncertaintyResult dataclass."""
        result = engine.quantify_uncertainty(metered_tier1_input)
        assert isinstance(result, UncertaintyResult)

    def test_base_value_is_preserved(self, engine, metered_tier1_input):
        """The base_value in the result matches the input total_co2e_kg."""
        result = engine.quantify_uncertainty(metered_tier1_input)
        assert result.base_value == 5000.0

    def test_mean_near_base_value(self, engine, metered_tier1_input):
        """MC mean should be close to the base value (within 5%)."""
        result = engine.quantify_uncertainty(metered_tier1_input)
        assert abs(result.mean - result.base_value) / result.base_value < 0.05

    def test_std_dev_positive(self, engine, metered_tier1_input):
        """Standard deviation should be positive for non-trivial input."""
        result = engine.quantify_uncertainty(metered_tier1_input)
        assert result.std_dev > 0

    def test_cv_is_std_over_mean(self, engine, metered_tier1_input):
        """Coefficient of variation = std_dev / mean."""
        result = engine.quantify_uncertainty(metered_tier1_input)
        expected_cv = result.std_dev / abs(result.mean)
        assert abs(result.cv - expected_cv) < 0.001

    def test_confidence_intervals_present(self, engine, metered_tier1_input):
        """Result contains CIs for 90%, 95%, 99%."""
        result = engine.quantify_uncertainty(metered_tier1_input)
        assert "90" in result.confidence_intervals
        assert "95" in result.confidence_intervals
        assert "99" in result.confidence_intervals

    def test_ci_lower_less_than_upper(self, engine, metered_tier1_input):
        """For each CI, lower bound < upper bound."""
        result = engine.quantify_uncertainty(metered_tier1_input)
        for level, (lo, hi) in result.confidence_intervals.items():
            assert lo <= hi, f"CI {level}%: lower ({lo}) > upper ({hi})"

    def test_wider_ci_at_higher_level(self, engine, metered_tier1_input):
        """99% CI should be wider than or equal to 95%, which >= 90%."""
        result = engine.quantify_uncertainty(metered_tier1_input)
        ci90 = result.confidence_intervals["90"]
        ci95 = result.confidence_intervals["95"]
        ci99 = result.confidence_intervals["99"]
        width_90 = ci90[1] - ci90[0]
        width_95 = ci95[1] - ci95[0]
        width_99 = ci99[1] - ci99[0]
        assert width_95 >= width_90 - 0.01  # Small tolerance
        assert width_99 >= width_95 - 0.01

    def test_provenance_hash_is_sha256(self, engine, metered_tier1_input):
        """Provenance hash is a 64-character hex string (SHA-256)."""
        result = engine.quantify_uncertainty(metered_tier1_input)
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_monte_carlo_iterations_recorded(self, engine, metered_tier1_input):
        """The result records the number of MC iterations used."""
        result = engine.quantify_uncertainty(metered_tier1_input)
        assert result.monte_carlo_iterations == 5_000

    def test_seed_recorded(self, engine, metered_tier1_input):
        """The result records the PRNG seed used."""
        result = engine.quantify_uncertainty(metered_tier1_input)
        assert result.seed == 42

    def test_timestamp_present(self, engine, metered_tier1_input):
        """The result has an ISO-format timestamp."""
        result = engine.quantify_uncertainty(metered_tier1_input)
        assert result.timestamp is not None
        assert "T" in result.timestamp  # ISO format

    def test_data_quality_default_when_no_dqi(self, engine, metered_tier1_input):
        """Without DQI info, default score is 3 (Fair)."""
        result = engine.quantify_uncertainty(metered_tier1_input, data_quality_info=None)
        assert result.data_quality_score == 3

    def test_data_quality_with_dqi(self, engine, metered_tier1_input, high_quality_dqi):
        """With high-quality DQI, score should be 1 (Very Good)."""
        result = engine.quantify_uncertainty(
            metered_tier1_input, data_quality_info=high_quality_dqi,
        )
        assert result.data_quality_score == 1

    def test_history_grows(self, engine, metered_tier1_input):
        """Each call appends to the internal history."""
        engine.quantify_uncertainty(metered_tier1_input)
        engine.quantify_uncertainty(metered_tier1_input)
        assert len(engine._history) == 2


# ---------------------------------------------------------------------------
# TestMonteCarloSimulation
# ---------------------------------------------------------------------------


class TestMonteCarloSimulation:
    """Test monte_carlo_simulation independently."""

    def test_returns_expected_keys(self, engine):
        """MC result dictionary contains required keys."""
        params = {"activity_data": 0.02, "emission_factor": 0.07}
        result = engine.monte_carlo_simulation(5000.0, params, iterations=1000)
        assert "mean" in result
        assert "std_dev" in result
        assert "cv" in result
        assert "percentiles" in result
        assert "confidence_intervals" in result
        assert "samples_count" in result

    def test_samples_count_matches_iterations(self, engine):
        """Number of samples equals the requested iterations."""
        params = {"activity_data": 0.02}
        result = engine.monte_carlo_simulation(5000.0, params, iterations=2000)
        assert result["samples_count"] == 2000

    def test_mean_near_base_value(self, engine):
        """MC mean should be close to the base value for small uncertainties."""
        params = {"activity_data": 0.02, "emission_factor": 0.03}
        result = engine.monte_carlo_simulation(10000.0, params, iterations=5000)
        assert abs(result["mean"] - 10000.0) / 10000.0 < 0.05

    def test_larger_uncertainty_gives_larger_std(self, engine):
        """Higher input uncertainty produces a wider distribution."""
        params_small = {"activity_data": 0.02}
        params_large = {"activity_data": 0.20}
        result_small = engine.monte_carlo_simulation(5000.0, params_small, iterations=5000)
        result_large = engine.monte_carlo_simulation(5000.0, params_large, iterations=5000)
        assert result_large["std_dev"] > result_small["std_dev"]

    def test_zero_base_value_returns_empty(self, engine):
        """When base_value is 0, degenerate result is returned."""
        params = {"activity_data": 0.02}
        result = engine.monte_carlo_simulation(0.0, params, iterations=1000)
        assert result["mean"] == 0.0
        assert result["std_dev"] == 0.0
        assert result["samples_count"] == 0

    def test_empty_params_returns_empty(self, engine):
        """When no parameters, degenerate result is returned."""
        result = engine.monte_carlo_simulation(5000.0, {}, iterations=1000)
        assert result["mean"] == 5000.0
        assert result["std_dev"] == 0.0

    def test_minimum_iterations_clamp(self, engine):
        """Iterations are clamped to at least 100."""
        params = {"activity_data": 0.05}
        result = engine.monte_carlo_simulation(5000.0, params, iterations=10)
        assert result["samples_count"] == 100

    def test_percentile_keys(self, engine):
        """Percentile dict contains standard percentile keys."""
        params = {"activity_data": 0.02}
        result = engine.monte_carlo_simulation(5000.0, params, iterations=1000)
        expected_keys = {"1", "2.5", "5", "10", "25", "50", "75", "90", "95", "97.5", "99"}
        assert expected_keys == set(result["percentiles"].keys())

    def test_percentiles_monotonically_increase(self, engine):
        """Percentile values should be non-decreasing."""
        params = {"activity_data": 0.05, "emission_factor": 0.07}
        result = engine.monte_carlo_simulation(10000.0, params, iterations=5000)
        pct = result["percentiles"]
        keys_sorted = sorted(pct.keys(), key=float)
        values = [pct[k] for k in keys_sorted]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]


# ---------------------------------------------------------------------------
# TestAnalyticalPropagation
# ---------------------------------------------------------------------------


class TestAnalyticalPropagation:
    """Test IPCC Approach 1 analytical error propagation."""

    def test_single_parameter(self, engine):
        """Single parameter: combined = the parameter itself."""
        result = engine.analytical_propagation({"activity_data": 0.05})
        assert abs(result["combined_relative_uncertainty"] - 0.05) < 1e-6

    def test_two_parameters_root_sum_of_squares(self, engine):
        """Two parameters: combined = sqrt(0.02^2 + 0.07^2) = 0.072801..."""
        result = engine.analytical_propagation({
            "activity_data": 0.02,
            "emission_factor": 0.07,
        })
        expected = math.sqrt(0.02**2 + 0.07**2)
        assert abs(result["combined_relative_uncertainty"] - expected) < 1e-5

    def test_four_parameters(self, engine):
        """Full chain: activity + EF + HV + OF."""
        params = {
            "activity_data": 0.02,
            "emission_factor": 0.07,
            "heating_value": 0.05,
            "oxidation_factor": 0.02,
        }
        result = engine.analytical_propagation(params)
        expected = math.sqrt(0.02**2 + 0.07**2 + 0.05**2 + 0.02**2)
        assert abs(result["combined_relative_uncertainty"] - expected) < 1e-5

    def test_combined_percent(self, engine):
        """combined_percent = combined_relative_uncertainty * 100."""
        result = engine.analytical_propagation({"activity_data": 0.10})
        expected_pct = result["combined_relative_uncertainty"] * 100
        assert abs(result["combined_percent"] - expected_pct) < 0.01

    def test_formula_present(self, engine):
        """Result includes the formula string."""
        result = engine.analytical_propagation({"activity_data": 0.05})
        assert "sqrt" in result["formula"]

    def test_input_uncertainties_preserved(self, engine):
        """Input uncertainties are echoed back in the result."""
        params = {"activity_data": 0.02, "emission_factor": 0.07}
        result = engine.analytical_propagation(params)
        assert result["input_uncertainties"]["activity_data"] == 0.02
        assert result["input_uncertainties"]["emission_factor"] == 0.07

    def test_zero_uncertainties(self, engine):
        """All-zero uncertainties produce zero combined uncertainty."""
        result = engine.analytical_propagation({"a": 0.0, "b": 0.0})
        assert result["combined_relative_uncertainty"] == 0.0

    def test_empty_params(self, engine):
        """Empty params dict produces zero combined uncertainty."""
        result = engine.analytical_propagation({})
        assert result["combined_relative_uncertainty"] == 0.0


# ---------------------------------------------------------------------------
# TestConfidenceIntervals
# ---------------------------------------------------------------------------


class TestConfidenceIntervals:
    """Test calculate_confidence_interval standalone method."""

    def test_95_percent_ci(self, engine):
        """95% CI computed from sample list."""
        samples = list(range(1, 101))  # 1..100
        lo, hi = engine.calculate_confidence_interval(samples, 0.95)
        assert lo < 10
        assert hi > 90

    def test_90_percent_ci_narrower_than_99(self, engine):
        """90% CI is narrower than 99% CI."""
        samples = list(range(1, 1001))
        lo90, hi90 = engine.calculate_confidence_interval(samples, 0.90)
        lo99, hi99 = engine.calculate_confidence_interval(samples, 0.99)
        assert (hi99 - lo99) >= (hi90 - lo90)

    def test_empty_samples(self, engine):
        """Empty sample list returns (0.0, 0.0)."""
        lo, hi = engine.calculate_confidence_interval([], 0.95)
        assert lo == 0.0
        assert hi == 0.0

    def test_single_sample(self, engine):
        """Single sample: lower == upper."""
        lo, hi = engine.calculate_confidence_interval([42.0], 0.95)
        assert lo == hi == 42.0

    def test_two_samples(self, engine):
        """Two samples: interval spans both."""
        lo, hi = engine.calculate_confidence_interval([10.0, 20.0], 0.95)
        assert lo <= 10.0
        assert hi >= 10.0


# ---------------------------------------------------------------------------
# TestDataQualityScoring
# ---------------------------------------------------------------------------


class TestDataQualityScoring:
    """Test score_data_quality (GHG Protocol DQI)."""

    def test_best_quality_score_1(self, engine):
        """Best inputs produce score 1 (Very Good)."""
        score = engine.score_data_quality(
            data_source="metered",
            measurement_method="continuous_monitoring",
            data_age_months=3,
            completeness_pct=99.0,
        )
        assert score == 1

    def test_worst_quality_score_5(self, engine):
        """Worst inputs produce score 5 (Very Poor)."""
        score = engine.score_data_quality(
            data_source="estimated",
            measurement_method="estimate",
            data_age_months=60,
            completeness_pct=10.0,
        )
        assert score == 5

    def test_invoiced_periodic(self, engine):
        """Invoiced source + periodic measurement -> score 2."""
        score = engine.score_data_quality(
            data_source="invoiced",
            measurement_method="periodic_measurement",
            data_age_months=6,
            completeness_pct=95.0,
        )
        assert score == 2

    def test_supplier_report_fuel_analysis(self, engine):
        """Supplier report + fuel analysis -> score 3."""
        score = engine.score_data_quality(
            data_source="supplier_report",
            measurement_method="fuel_analysis",
            data_age_months=24,
            completeness_pct=60.0,
        )
        assert score == 3

    def test_industry_average_default_factor(self, engine):
        """Industry average + default factor -> score 4."""
        score = engine.score_data_quality(
            data_source="industry_average",
            measurement_method="default_factor",
            data_age_months=36,
            completeness_pct=40.0,
        )
        assert score == 4

    @pytest.mark.parametrize("source,expected_min,expected_max", [
        ("metered", 1, 3),
        ("invoiced", 1, 4),
        ("estimated", 3, 5),
    ])
    def test_score_range_by_source(self, engine, source, expected_min, expected_max):
        """Score is within expected range based on data source."""
        score = engine.score_data_quality(
            data_source=source,
            measurement_method="periodic_measurement",
            data_age_months=12,
            completeness_pct=80.0,
        )
        assert expected_min <= score <= expected_max

    def test_score_clamped_to_1_5(self, engine):
        """Score is always in [1, 5] regardless of extreme inputs."""
        score_low = engine.score_data_quality(
            data_source="metered",
            measurement_method="continuous_monitoring",
            data_age_months=0,
            completeness_pct=100.0,
        )
        assert 1 <= score_low <= 5

        score_high = engine.score_data_quality(
            data_source="estimated",
            measurement_method="estimate",
            data_age_months=999,
            completeness_pct=0.0,
        )
        assert 1 <= score_high <= 5

    def test_unknown_source_defaults_to_5(self, engine):
        """Unrecognised data source scores 5 for that dimension."""
        score = engine.score_data_quality(
            data_source="unknown_source",
            measurement_method="continuous_monitoring",
            data_age_months=3,
            completeness_pct=99.0,
        )
        # data_source scores 5, others score 1 -> weighted > 1
        assert score >= 2


# ---------------------------------------------------------------------------
# TestContributionAnalysis
# ---------------------------------------------------------------------------


class TestContributionAnalysis:
    """Test analyze_contributions method."""

    def test_single_parameter_100_percent(self, engine):
        """Single parameter contributes 100%."""
        result = engine.analyze_contributions(5000.0, {"activity_data": 0.05})
        assert result["activity_data"] == 100.0

    def test_two_equal_parameters(self, engine):
        """Two equal parameters each contribute 50%."""
        result = engine.analyze_contributions(5000.0, {
            "activity_data": 0.05,
            "emission_factor": 0.05,
        })
        assert abs(result["activity_data"] - 50.0) < 0.01
        assert abs(result["emission_factor"] - 50.0) < 0.01

    def test_contributions_sum_to_100(self, engine):
        """All contributions sum to 100%."""
        params = {
            "activity_data": 0.02,
            "emission_factor": 0.07,
            "heating_value": 0.05,
            "oxidation_factor": 0.02,
        }
        result = engine.analyze_contributions(5000.0, params)
        total = sum(result.values())
        assert abs(total - 100.0) < 0.1

    def test_dominant_parameter(self, engine):
        """The parameter with the largest uncertainty dominates."""
        params = {
            "activity_data": 0.02,
            "emission_factor": 0.20,
        }
        result = engine.analyze_contributions(5000.0, params)
        assert result["emission_factor"] > result["activity_data"]
        assert result["emission_factor"] > 90.0

    def test_zero_uncertainties_gives_zero(self, engine):
        """Zero uncertainties produce zero contributions."""
        result = engine.analyze_contributions(5000.0, {"a": 0.0, "b": 0.0})
        assert result["a"] == 0.0
        assert result["b"] == 0.0


# ---------------------------------------------------------------------------
# TestTierUncertainty
# ---------------------------------------------------------------------------


class TestTierUncertainty:
    """Test uncertainty levels by tier."""

    def test_tier1_uncertainty_7_percent(self, engine):
        """TIER_1 default EF uncertainty is 7%."""
        assert DEFAULT_UNCERTAINTIES["emission_factor"]["TIER_1"] == 0.07

    def test_tier2_uncertainty_3_percent(self, engine):
        """TIER_2 default EF uncertainty is 3%."""
        assert DEFAULT_UNCERTAINTIES["emission_factor"]["TIER_2"] == 0.03

    def test_tier3_uncertainty_1_5_percent(self, engine):
        """TIER_3 default EF uncertainty is 1.5%."""
        assert DEFAULT_UNCERTAINTIES["emission_factor"]["TIER_3"] == 0.015

    def test_metered_activity_2_percent(self):
        """Metered activity data uncertainty is 2%."""
        assert DEFAULT_UNCERTAINTIES["activity_data"]["metered"] == 0.02

    def test_invoiced_activity_5_percent(self):
        """Invoiced activity data uncertainty is 5%."""
        assert DEFAULT_UNCERTAINTIES["activity_data"]["invoiced"] == 0.05

    def test_estimated_activity_20_percent(self):
        """Estimated activity data uncertainty is 20%."""
        assert DEFAULT_UNCERTAINTIES["activity_data"]["estimated"] == 0.20


# ---------------------------------------------------------------------------
# TestCH4N2OUncertainty
# ---------------------------------------------------------------------------


class TestCH4N2OUncertainty:
    """Test CH4 and N2O factor-of-2 uncertainty handling."""

    def test_ch4_factor_of_2(self):
        """CH4 uncertainty is factor-of-2."""
        assert DEFAULT_UNCERTAINTIES["ch4_factor"] == 2.0

    def test_n2o_factor_of_2(self):
        """N2O uncertainty is factor-of-2."""
        assert DEFAULT_UNCERTAINTIES["n2o_factor"] == 2.0

    def test_lognormal_used_for_ch4(self, engine):
        """CH4/N2O parameters use lognormal draw (tested via distribution name)."""
        import random
        rng = random.Random(42)
        # The method should not raise for ch4_factor
        draw = UncertaintyQuantifierEngine._draw_from_distribution(
            "ch4_factor", 2.0, rng,
        )
        assert draw > 0, "Lognormal draw must be positive"

    def test_lognormal_used_for_n2o(self, engine):
        """N2O parameters use lognormal draw."""
        import random
        rng = random.Random(42)
        draw = UncertaintyQuantifierEngine._draw_from_distribution(
            "n2o_factor", 2.0, rng,
        )
        assert draw > 0

    def test_normal_used_for_activity_data(self, engine):
        """Activity data uses normal distribution (mean=1, clamped positive)."""
        import random
        rng = random.Random(42)
        draw = UncertaintyQuantifierEngine._draw_from_distribution(
            "activity_data", 0.02, rng,
        )
        assert draw >= 0.01  # Clamped to min 0.01
        assert abs(draw - 1.0) < 0.5  # Reasonable range for 2% uncertainty


# ---------------------------------------------------------------------------
# TestReproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Test deterministic reproducibility with fixed seed."""

    def test_same_seed_same_mc_result(self):
        """Same seed produces identical MC results."""
        eng1 = UncertaintyQuantifierEngine(config=None)
        eng2 = UncertaintyQuantifierEngine(config=None)

        params = {"activity_data": 0.05, "emission_factor": 0.07}
        r1 = eng1.monte_carlo_simulation(10000.0, params, iterations=1000)
        r2 = eng2.monte_carlo_simulation(10000.0, params, iterations=1000)

        assert r1["mean"] == r2["mean"]
        assert r1["std_dev"] == r2["std_dev"]
        assert r1["cv"] == r2["cv"]

    def test_same_input_same_provenance_hash(self):
        """Same input to quantify_uncertainty produces same provenance hash."""
        eng1 = UncertaintyQuantifierEngine(config=None)
        eng2 = UncertaintyQuantifierEngine(config=None)

        inp = {
            "total_co2e_kg": 5000.0,
            "activity_data_source": "metered",
            "ef_tier": "TIER_1",
        }
        r1 = eng1.quantify_uncertainty(inp)
        r2 = eng2.quantify_uncertainty(inp)
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_input_different_hash(self, engine):
        """Different inputs produce different provenance hashes."""
        inp1 = {"total_co2e_kg": 5000.0, "activity_data_source": "metered"}
        inp2 = {"total_co2e_kg": 10000.0, "activity_data_source": "metered"}
        r1 = engine.quantify_uncertainty(inp1)
        # Create a fresh engine to avoid history contamination
        eng2 = UncertaintyQuantifierEngine(config=None)
        r2 = eng2.quantify_uncertainty(inp2)
        assert r1.provenance_hash != r2.provenance_hash


# ---------------------------------------------------------------------------
# TestSummaryStatistics
# ---------------------------------------------------------------------------


class TestSummaryStatistics:
    """Test get_uncertainty_summary method."""

    def test_empty_summary(self, engine):
        """No analyses -> zero-valued summary."""
        summary = engine.get_uncertainty_summary()
        assert summary["total_analyses"] == 0
        assert summary["average_cv"] == 0.0
        assert summary["average_dq_score"] == 0.0
        assert summary["average_combined_relative_uncertainty"] == 0.0

    def test_summary_after_one_analysis(self, engine, metered_tier1_input):
        """After one analysis, summary reflects it."""
        result = engine.quantify_uncertainty(metered_tier1_input)
        summary = engine.get_uncertainty_summary()
        assert summary["total_analyses"] == 1
        assert summary["average_cv"] == result.cv
        assert summary["average_dq_score"] == result.data_quality_score

    def test_summary_after_multiple_analyses(self, engine):
        """After multiple analyses, averages are computed correctly."""
        inputs = [
            {"total_co2e_kg": 5000.0, "activity_data_source": "metered", "ef_tier": "TIER_1"},
            {"total_co2e_kg": 10000.0, "activity_data_source": "invoiced", "ef_tier": "TIER_2"},
            {"total_co2e_kg": 8000.0, "activity_data_source": "estimated", "ef_tier": "TIER_3"},
        ]
        results = [engine.quantify_uncertainty(inp) for inp in inputs]
        summary = engine.get_uncertainty_summary()

        assert summary["total_analyses"] == 3
        expected_avg_cv = sum(r.cv for r in results) / 3
        assert abs(summary["average_cv"] - expected_avg_cv) < 1e-4


# ---------------------------------------------------------------------------
# TestParameterResolution
# ---------------------------------------------------------------------------


class TestParameterResolution:
    """Test _resolve_parameter_uncertainties internal method."""

    def test_metered_tier1_defaults(self, engine):
        """Metered + TIER_1 resolves to known uncertainty values."""
        params = engine._resolve_parameter_uncertainties({
            "activity_data_source": "metered",
            "ef_tier": "TIER_1",
            "heating_value_type": "default",
            "oxidation_factor_type": "default",
        })
        assert params["activity_data"] == 0.02
        assert params["emission_factor"] == 0.07
        assert params["heating_value"] == 0.05
        assert params["oxidation_factor"] == 0.02

    def test_unknown_source_falls_back_to_estimated(self, engine):
        """Unknown activity_data_source falls back to estimated (0.20)."""
        params = engine._resolve_parameter_uncertainties({
            "activity_data_source": "unknown_source",
        })
        assert params["activity_data"] == 0.20

    def test_unknown_tier_falls_back_to_tier1(self, engine):
        """Unknown ef_tier falls back to TIER_1 (0.07)."""
        params = engine._resolve_parameter_uncertainties({
            "ef_tier": "UNKNOWN_TIER",
        })
        assert params["emission_factor"] == 0.07

    def test_missing_keys_use_defaults(self, engine):
        """Missing keys in calc_result use worst-case defaults."""
        params = engine._resolve_parameter_uncertainties({})
        assert params["activity_data"] == 0.20  # estimated default
        assert params["emission_factor"] == 0.07  # TIER_1 default
        assert params["heating_value"] == 0.05  # default
        assert params["oxidation_factor"] == 0.02  # default


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_base_value(self, engine):
        """Very small base value still produces valid result."""
        result = engine.quantify_uncertainty({"total_co2e_kg": 0.001})
        assert isinstance(result, UncertaintyResult)
        assert result.base_value == 0.001

    def test_very_large_base_value(self, engine):
        """Very large base value still produces valid result."""
        result = engine.quantify_uncertainty({"total_co2e_kg": 1e12})
        assert isinstance(result, UncertaintyResult)
        assert result.mean > 0

    def test_zero_base_value(self, engine):
        """Zero base value produces degenerate but valid result."""
        result = engine.quantify_uncertainty({"total_co2e_kg": 0.0})
        assert result.base_value == 0.0
        assert result.std_dev == 0.0

    def test_missing_total_co2e_kg(self, engine):
        """Missing total_co2e_kg defaults to 0.0."""
        result = engine.quantify_uncertainty({})
        assert result.base_value == 0.0


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Test concurrent access to the engine."""

    def test_concurrent_quantifications(self):
        """Multiple threads can quantify concurrently without error."""
        engine = UncertaintyQuantifierEngine(config=None)
        errors = []

        def worker(idx):
            try:
                inp = {"total_co2e_kg": 1000.0 * (idx + 1), "activity_data_source": "metered"}
                engine.quantify_uncertainty(inp)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(engine._history) == 10

    def test_concurrent_summary_access(self):
        """Summary can be read while quantifications are running."""
        engine = UncertaintyQuantifierEngine(config=None)

        def writer():
            for _ in range(20):
                engine.quantify_uncertainty({"total_co2e_kg": 5000.0})

        def reader():
            for _ in range(20):
                engine.get_uncertainty_summary()

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # No exceptions means thread safety is working
        assert engine.get_uncertainty_summary()["total_analyses"] == 20


# ---------------------------------------------------------------------------
# TestComputeHash
# ---------------------------------------------------------------------------


class TestComputeHash:
    """Test the internal _compute_hash static method."""

    def test_deterministic(self):
        """Same data produces same hash."""
        data = {"a": 1, "b": 2}
        h1 = UncertaintyQuantifierEngine._compute_hash(data)
        h2 = UncertaintyQuantifierEngine._compute_hash(data)
        assert h1 == h2

    def test_different_data_different_hash(self):
        """Different data produces different hash."""
        h1 = UncertaintyQuantifierEngine._compute_hash({"a": 1})
        h2 = UncertaintyQuantifierEngine._compute_hash({"a": 2})
        assert h1 != h2

    def test_hash_length(self):
        """Hash is 64 hex characters (SHA-256)."""
        h = UncertaintyQuantifierEngine._compute_hash({"test": True})
        assert len(h) == 64
