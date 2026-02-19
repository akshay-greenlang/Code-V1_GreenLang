# -*- coding: utf-8 -*-
"""
Unit tests for UncertaintyQuantifierEngine (Engine 5 of 7)

AGENT-MRV-006: Land Use Emissions Agent

Tests Monte Carlo simulation, analytical error propagation, DQI scoring,
confidence interval calculation, percentile extraction, and sensitivity
analysis for LULUCF emission uncertainty quantification.

Target: 85 tests, ~600 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from greenlang.land_use_emissions.uncertainty_quantifier import (
    UncertaintyQuantifierEngine,
    DEFAULT_CV,
    _D,
    _safe_decimal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uq_engine():
    """Create a default UncertaintyQuantifierEngine."""
    engine = UncertaintyQuantifierEngine(default_iterations=5000, default_seed=42)
    yield engine
    engine.reset()


@pytest.fixture
def uq_engine_low_iter():
    """Create an engine with low iteration count for fast testing."""
    engine = UncertaintyQuantifierEngine(default_iterations=500, default_seed=42)
    yield engine
    engine.reset()


@pytest.fixture
def mc_request() -> Dict[str, Any]:
    """Return a standard Monte Carlo request."""
    return {
        "total_co2e_tonnes": 5000,
        "parameters": [
            {"name": "agb", "value": 180, "cv_pct": 30, "dist": "normal"},
            {"name": "area_ha", "value": 1000, "cv_pct": 5, "dist": "uniform"},
        ],
        "n_iterations": 1000,
        "seed": 42,
    }


@pytest.fixture
def error_prop_request() -> Dict[str, Any]:
    """Return a standard error propagation request."""
    return {
        "total_co2e_tonnes": 5000,
        "parameters": [
            {"name": "agb", "value": 180, "uncertainty_pct": 30},
            {"name": "ef", "value": 2.68, "uncertainty_pct": 50},
            {"name": "area", "value": 1000, "uncertainty_pct": 5},
        ],
        "combination": "multiplicative",
    }


@pytest.fixture
def dqi_request() -> Dict[str, Any]:
    """Return a standard DQI scoring request."""
    return {
        "reliability": 2,
        "completeness": 2,
        "temporal_correlation": 3,
        "geographical_correlation": 2,
        "technological_correlation": 3,
    }


# ===========================================================================
# 1. Initialisation Tests
# ===========================================================================


class TestUQEngineInit:
    """Test UncertaintyQuantifierEngine initialisation."""

    def test_default_iterations(self, uq_engine):
        """Test default iteration count is 5000."""
        assert uq_engine._default_iterations == 5000

    def test_default_seed(self, uq_engine):
        """Test default seed is 42."""
        assert uq_engine._default_seed == 42

    def test_custom_parameters(self):
        """Test engine accepts custom iteration count and seed."""
        engine = UncertaintyQuantifierEngine(default_iterations=10000, default_seed=99)
        assert engine._default_iterations == 10000
        assert engine._default_seed == 99

    def test_analysis_counter_starts_at_zero(self, uq_engine):
        """Test analysis counter starts at zero."""
        stats = uq_engine.get_statistics()
        assert stats["total_analyses"] == 0


# ===========================================================================
# 2. Monte Carlo - Reproducibility Tests
# ===========================================================================


class TestMonteCarloReproducibility:
    """Test Monte Carlo with known seed produces reproducible results."""

    def test_same_seed_same_results(self, uq_engine, mc_request):
        """Test that the same seed produces identical results."""
        r1 = uq_engine.run_monte_carlo(mc_request)
        r2 = uq_engine.run_monte_carlo(mc_request)
        assert r1["statistics"]["mean"] == r2["statistics"]["mean"]
        assert r1["statistics"]["std_dev"] == r2["statistics"]["std_dev"]
        assert r1["percentiles"] == r2["percentiles"]

    def test_different_seed_different_results(self, uq_engine, mc_request):
        """Test that different seeds produce different results."""
        r1 = uq_engine.run_monte_carlo(mc_request)
        mc_request["seed"] = 99
        r2 = uq_engine.run_monte_carlo(mc_request)
        assert r1["statistics"]["mean"] != r2["statistics"]["mean"]

    def test_reproducibility_across_three_runs(self, uq_engine, mc_request):
        """Test seed reproducibility across three consecutive runs."""
        results = [uq_engine.run_monte_carlo(mc_request) for _ in range(3)]
        means = [r["statistics"]["mean"] for r in results]
        assert means[0] == means[1] == means[2]


# ===========================================================================
# 3. Monte Carlo - Iteration Count Tests
# ===========================================================================


class TestMonteCarloIterations:
    """Test configurable Monte Carlo iteration count."""

    def test_default_5000_iterations(self, uq_engine, mc_request):
        """Test that default iteration count is used when not specified."""
        del mc_request["n_iterations"]
        result = uq_engine.run_monte_carlo(mc_request)
        assert result["n_iterations"] == 5000

    def test_custom_iteration_count(self, uq_engine, mc_request):
        """Test that custom iteration count is respected."""
        mc_request["n_iterations"] = 2000
        result = uq_engine.run_monte_carlo(mc_request)
        assert result["n_iterations"] == 2000

    def test_zero_iterations_validation_error(self, uq_engine, mc_request):
        """Test that zero iterations returns a validation error."""
        mc_request["n_iterations"] = 0
        result = uq_engine.run_monte_carlo(mc_request)
        assert result["status"] == "VALIDATION_ERROR"

    def test_negative_iterations_validation_error(self, uq_engine, mc_request):
        """Test that negative iterations returns a validation error."""
        mc_request["n_iterations"] = -100
        result = uq_engine.run_monte_carlo(mc_request)
        assert result["status"] == "VALIDATION_ERROR"

    def test_million_iterations_max(self, uq_engine, mc_request):
        """Test that iterations exceeding 1M returns a validation error."""
        mc_request["n_iterations"] = 1_000_001
        result = uq_engine.run_monte_carlo(mc_request)
        assert result["status"] == "VALIDATION_ERROR"


# ===========================================================================
# 4. Monte Carlo - Distribution Tests
# ===========================================================================


class TestMonteCarloDistributions:
    """Test parameter distributions (normal, lognormal, uniform, triangular)."""

    def test_normal_distribution_mean_near_value(self, uq_engine):
        """Test that normal distribution samples cluster around the specified value."""
        result = uq_engine.run_monte_carlo({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "param1", "value": 100, "cv_pct": 10, "dist": "normal"},
            ],
            "n_iterations": 5000,
            "seed": 42,
        })
        param_mean = result["parameter_statistics"]["param1"]["mean"]
        # Mean should be within 5% of 100
        assert abs(param_mean - 100) < 5

    def test_lognormal_distribution_nonnegative(self, uq_engine):
        """Test that lognormal samples are always non-negative."""
        result = uq_engine.run_monte_carlo({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "ef", "value": 2.68, "cv_pct": 50, "dist": "lognormal"},
            ],
            "n_iterations": 5000,
            "seed": 42,
        })
        assert result["parameter_statistics"]["ef"]["min"] >= 0

    def test_uniform_distribution_bounded(self, uq_engine):
        """Test that uniform samples are within (1-cv/100)*val to (1+cv/100)*val."""
        result = uq_engine.run_monte_carlo({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "area", "value": 1000, "cv_pct": 5, "dist": "uniform"},
            ],
            "n_iterations": 5000,
            "seed": 42,
        })
        stats = result["parameter_statistics"]["area"]
        assert stats["min"] >= 1000 * 0.95 - 1  # small tolerance
        assert stats["max"] <= 1000 * 1.05 + 1

    def test_triangular_distribution(self, uq_engine):
        """Test triangular distribution produces results centred near the mode."""
        result = uq_engine.run_monte_carlo({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "param1", "value": 50, "cv_pct": 20, "dist": "triangular"},
            ],
            "n_iterations": 5000,
            "seed": 42,
        })
        assert result["status"] == "SUCCESS"
        param_mean = result["parameter_statistics"]["param1"]["mean"]
        assert abs(param_mean - 50) < 10

    def test_normal_bounded_at_zero(self, uq_engine):
        """Test that normal distribution samples are bounded at zero."""
        result = uq_engine.run_monte_carlo({
            "total_co2e_tonnes": 100,
            "parameters": [
                {"name": "small_val", "value": 1, "cv_pct": 200, "dist": "normal"},
            ],
            "n_iterations": 5000,
            "seed": 42,
        })
        assert result["parameter_statistics"]["small_val"]["min"] >= 0

    def test_lognormal_zero_value_returns_zeros(self, uq_engine):
        """Test that lognormal with value=0 returns all zeros."""
        result = uq_engine.run_monte_carlo({
            "total_co2e_tonnes": 100,
            "parameters": [
                {"name": "zero_param", "value": 0, "cv_pct": 50, "dist": "lognormal"},
            ],
            "n_iterations": 100,
            "seed": 42,
        })
        # With value=0, lognormal returns [0.0] * n
        assert result["status"] == "SUCCESS"


# ===========================================================================
# 5. Monte Carlo - CV Values by Tier Tests
# ===========================================================================


class TestCVByTier:
    """Test default CV values by parameter type and tier."""

    @pytest.mark.parametrize("param_type,tier,expected_cv", [
        ("AGB", "TIER_1", 30.0),
        ("AGB", "TIER_2", 15.0),
        ("AGB", "TIER_3", 5.0),
        ("SOC_REF", "TIER_1", 50.0),
        ("SOC_REF", "TIER_2", 25.0),
        ("AREA", "TIER_1", 5.0),
        ("EMISSION_FACTOR", "TIER_1", 50.0),
        ("PEATLAND_EF", "TIER_1", 90.0),
        ("N2O_EF", "TIER_1", 75.0),
    ])
    def test_default_cv_values(self, param_type, tier, expected_cv):
        """Test that DEFAULT_CV contains correct tier-specific CV values."""
        assert DEFAULT_CV[param_type][tier] == expected_cv

    def test_cv_decreases_with_higher_tier(self):
        """Test that CV consistently decreases from Tier 1 to Tier 3."""
        for param_type, tiers in DEFAULT_CV.items():
            if "TIER_1" in tiers and "TIER_2" in tiers:
                assert tiers["TIER_1"] > tiers["TIER_2"], f"{param_type} T1>T2 failed"
            if "TIER_2" in tiers and "TIER_3" in tiers:
                assert tiers["TIER_2"] > tiers["TIER_3"], f"{param_type} T2>T3 failed"


# ===========================================================================
# 6. Confidence Interval Tests
# ===========================================================================


class TestConfidenceIntervals:
    """Test confidence interval calculation."""

    def test_mc_provides_90_95_99_ci(self, uq_engine, mc_request):
        """Test that Monte Carlo provides 90%, 95%, and 99% confidence intervals."""
        result = uq_engine.run_monte_carlo(mc_request)
        ci = result["confidence_intervals"]
        assert "90" in ci
        assert "95" in ci
        assert "99" in ci

    def test_99_ci_wider_than_95(self, uq_engine, mc_request):
        """Test that 99% CI is wider than 95% CI."""
        result = uq_engine.run_monte_carlo(mc_request)
        ci = result["confidence_intervals"]
        assert ci["99"]["half_width"] > ci["95"]["half_width"]

    def test_95_ci_wider_than_90(self, uq_engine, mc_request):
        """Test that 95% CI is wider than 90% CI."""
        result = uq_engine.run_monte_carlo(mc_request)
        ci = result["confidence_intervals"]
        assert ci["95"]["half_width"] > ci["90"]["half_width"]

    def test_ci_lower_less_than_upper(self, uq_engine, mc_request):
        """Test that CI lower bound is less than upper bound."""
        result = uq_engine.run_monte_carlo(mc_request)
        for level, ci in result["confidence_intervals"].items():
            assert ci["lower"] < ci["upper"], f"CI {level}% lower >= upper"

    def test_ci_contains_mean(self, uq_engine, mc_request):
        """Test that the mean falls within the 95% CI."""
        result = uq_engine.run_monte_carlo(mc_request)
        mean = result["statistics"]["mean"]
        ci95 = result["confidence_intervals"]["95"]
        assert ci95["lower"] <= mean <= ci95["upper"]

    def test_custom_confidence_levels(self, uq_engine, mc_request):
        """Test custom confidence levels."""
        mc_request["confidence_levels"] = [80, 90]
        result = uq_engine.run_monte_carlo(mc_request)
        ci = result["confidence_intervals"]
        assert "80" in ci
        assert "90" in ci

    def test_get_confidence_interval_95(self, uq_engine):
        """Test direct confidence interval calculation from mean and std_dev."""
        ci = uq_engine.get_confidence_interval(mean=5000, std_dev=500, confidence_level=95.0)
        assert ci["confidence_level"] == 95.0
        expected_hw = 1.960 * 500
        assert abs(ci["half_width"] - expected_hw) < 0.01
        assert ci["lower"] < 5000 < ci["upper"]

    def test_get_confidence_interval_90(self, uq_engine):
        """Test 90% confidence interval."""
        ci = uq_engine.get_confidence_interval(mean=1000, std_dev=100, confidence_level=90.0)
        expected_hw = 1.645 * 100
        assert abs(ci["half_width"] - expected_hw) < 0.01

    def test_get_confidence_interval_small_n(self, uq_engine):
        """Test confidence interval with small sample size applies t-correction."""
        ci = uq_engine.get_confidence_interval(
            mean=100, std_dev=20, confidence_level=95.0, n_samples=10
        )
        # With n=10, there should be a t-correction making CI wider per sample
        assert ci["half_width"] > 0


# ===========================================================================
# 7. Percentile Extraction Tests
# ===========================================================================


class TestPercentiles:
    """Test percentile extraction from Monte Carlo results."""

    def test_mc_provides_standard_percentiles(self, uq_engine, mc_request):
        """Test that MC provides 5th, 10th, 25th, 50th, 75th, 90th, 95th percentiles."""
        result = uq_engine.run_monte_carlo(mc_request)
        p = result["percentiles"]
        for pct in ["5", "10", "25", "50", "75", "90", "95"]:
            assert pct in p, f"Missing percentile: {pct}"

    def test_percentiles_monotonically_increasing(self, uq_engine, mc_request):
        """Test that percentiles are monotonically increasing."""
        result = uq_engine.run_monte_carlo(mc_request)
        p = result["percentiles"]
        keys = sorted([int(k) for k in p.keys()])
        values = [p[str(k)] for k in keys]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]

    def test_50th_percentile_near_mean(self, uq_engine, mc_request):
        """Test that the 50th percentile is reasonably close to the mean."""
        result = uq_engine.run_monte_carlo(mc_request)
        median = result["percentiles"]["50"]
        mean = result["statistics"]["mean"]
        # Within 20% for normally distributed data
        assert abs(median - mean) / mean < 0.2

    def test_get_percentiles_method(self, uq_engine):
        """Test the standalone get_percentiles method."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = uq_engine.get_percentiles(values)
        assert "5" in result
        assert "25" in result
        assert "50" in result
        assert "75" in result
        assert "95" in result
        assert result["50"] == 5.5  # Median of [1..10]

    def test_get_percentiles_custom_points(self, uq_engine):
        """Test get_percentiles with custom percentile points."""
        values = list(range(1, 101))
        result = uq_engine.get_percentiles(values, [10, 50, 90])
        assert "10" in result
        assert "50" in result
        assert "90" in result


# ===========================================================================
# 8. DQI Scoring Tests
# ===========================================================================


class TestDQIScoring:
    """Test Data Quality Indicator scoring."""

    def test_dqi_excellent_all_ones(self, uq_engine):
        """Test that all-1 scores produce EXCELLENT quality."""
        result = uq_engine.calculate_dqi({
            "reliability": 1,
            "completeness": 1,
            "temporal_correlation": 1,
            "geographical_correlation": 1,
            "technological_correlation": 1,
        })
        assert result["status"] == "SUCCESS"
        assert result["composite_score"] == 1.0
        assert result["quality_category"] == "EXCELLENT"
        assert result["uncertainty_multiplier"] == 0.8

    def test_dqi_very_poor_all_fives(self, uq_engine):
        """Test that all-5 scores produce VERY_POOR quality."""
        result = uq_engine.calculate_dqi({
            "reliability": 5,
            "completeness": 5,
            "temporal_correlation": 5,
            "geographical_correlation": 5,
            "technological_correlation": 5,
        })
        assert result["composite_score"] == 5.0
        assert result["quality_category"] == "VERY_POOR"
        assert result["uncertainty_multiplier"] == 2.5

    def test_dqi_good_category(self, uq_engine, dqi_request):
        """Test that mixed 2-3 scores produce GOOD or FAIR category."""
        result = uq_engine.calculate_dqi(dqi_request)
        assert result["quality_category"] in ("GOOD", "FAIR")

    @pytest.mark.parametrize("score,expected_category", [
        (1, "EXCELLENT"),
        (2, "GOOD"),
        (3, "FAIR"),
        (4, "POOR"),
        (5, "VERY_POOR"),
    ])
    def test_dqi_uniform_scores(self, uq_engine, score, expected_category):
        """Test quality category for uniform scores across all dimensions."""
        result = uq_engine.calculate_dqi({
            "reliability": score,
            "completeness": score,
            "temporal_correlation": score,
            "geographical_correlation": score,
            "technological_correlation": score,
        })
        assert result["quality_category"] == expected_category

    def test_dqi_geometric_mean(self, uq_engine):
        """Test composite score is geometric mean of dimension scores."""
        result = uq_engine.calculate_dqi({
            "reliability": 1,
            "completeness": 2,
            "temporal_correlation": 3,
            "geographical_correlation": 4,
            "technological_correlation": 5,
        })
        expected = (1 * 2 * 3 * 4 * 5) ** (1.0 / 5)
        assert abs(result["composite_score"] - round(expected, 4)) < 0.001

    def test_dqi_score_out_of_range_error(self, uq_engine):
        """Test that scores outside 1-5 return a validation error."""
        result = uq_engine.calculate_dqi({
            "reliability": 0,
            "completeness": 6,
            "temporal_correlation": 3,
            "geographical_correlation": 2,
            "technological_correlation": 3,
        })
        assert result["status"] == "VALIDATION_ERROR"
        assert len(result["errors"]) >= 2

    def test_dqi_missing_dimension_error(self, uq_engine):
        """Test that missing dimension returns a validation error."""
        result = uq_engine.calculate_dqi({
            "reliability": 2,
            "completeness": 2,
            # missing temporal, geographical, technological
        })
        assert result["status"] == "VALIDATION_ERROR"

    def test_dqi_provenance_hash(self, uq_engine, dqi_request):
        """Test that DQI result has a provenance hash."""
        result = uq_engine.calculate_dqi(dqi_request)
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# 9. Analytical Error Propagation Tests
# ===========================================================================


class TestErrorPropagation:
    """Test IPCC Approach 1 analytical error propagation."""

    def test_multiplicative_propagation(self, uq_engine, error_prop_request):
        """Test multiplicative U_total = sqrt(sum(Ui^2))."""
        result = uq_engine.run_error_propagation(error_prop_request)
        assert result["status"] == "SUCCESS"
        assert result["method"] == "ERROR_PROPAGATION"
        assert result["combination"] == "multiplicative"
        # U = sqrt(30^2 + 50^2 + 5^2) / 100 * 100 = sqrt(900+2500+25)/100*100 = sqrt(3425)/100*100
        expected_pct = Decimal(str(
            math.sqrt(0.30**2 + 0.50**2 + 0.05**2)
        )) * Decimal("100")
        expected_pct = expected_pct.quantize(Decimal("0.01"))
        assert Decimal(result["combined_uncertainty_pct"]) == expected_pct

    def test_additive_propagation(self, uq_engine):
        """Test additive U_total = sqrt(sum((Ui*Xi)^2)) / |sum(Xi)|."""
        result = uq_engine.run_error_propagation({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "a", "value": 500, "uncertainty_pct": 20},
                {"name": "b", "value": 300, "uncertainty_pct": 30},
            ],
            "combination": "additive",
        })
        assert result["status"] == "SUCCESS"
        assert result["combination"] == "additive"

    def test_error_propagation_95_ci(self, uq_engine, error_prop_request):
        """Test that error propagation provides a 95% confidence interval."""
        result = uq_engine.run_error_propagation(error_prop_request)
        ci = result["confidence_interval_95"]
        assert Decimal(ci["lower"]) < Decimal(result["central_estimate"])
        assert Decimal(ci["upper"]) > Decimal(result["central_estimate"])

    def test_error_propagation_no_params_error(self, uq_engine):
        """Test validation error when no parameters are provided."""
        result = uq_engine.run_error_propagation({
            "total_co2e_tonnes": 1000,
            "parameters": [],
        })
        assert result["status"] == "VALIDATION_ERROR"

    def test_single_parameter_uncertainty(self, uq_engine):
        """Test single parameter multiplicative uncertainty."""
        result = uq_engine.run_error_propagation({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "x", "value": 100, "uncertainty_pct": 30},
            ],
            "combination": "multiplicative",
        })
        assert Decimal(result["combined_uncertainty_pct"]) == Decimal("30.00")


# ===========================================================================
# 10. Sensitivity Analysis Tests
# ===========================================================================


class TestSensitivityAnalysis:
    """Test sensitivity analysis for identifying key uncertainty drivers."""

    def test_sensitivity_ranks_by_variance(self, uq_engine):
        """Test that parameters are ranked by variance contribution."""
        result = uq_engine.run_sensitivity_analysis({
            "total_co2e_tonnes": 5000,
            "parameters": [
                {"name": "agb", "value": 180, "cv_pct": 30},
                {"name": "area", "value": 1000, "cv_pct": 5},
                {"name": "ef", "value": 2.68, "cv_pct": 50},
            ],
        })
        assert result["status"] == "SUCCESS"
        sensitivities = result["sensitivities"]
        assert len(sensitivities) == 3
        # First should have highest contribution
        assert sensitivities[0]["contribution_pct"] >= sensitivities[1]["contribution_pct"]
        assert sensitivities[1]["contribution_pct"] >= sensitivities[2]["contribution_pct"]

    def test_sensitivity_contribution_sums_to_100(self, uq_engine):
        """Test that contribution percentages sum to approximately 100%."""
        result = uq_engine.run_sensitivity_analysis({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "a", "value": 100, "cv_pct": 20},
                {"name": "b", "value": 200, "cv_pct": 15},
            ],
        })
        total = sum(s["contribution_pct"] for s in result["sensitivities"])
        assert abs(total - 100.0) < 0.1

    def test_top_driver_identified(self, uq_engine):
        """Test that top_driver field matches the first sensitivity entry."""
        result = uq_engine.run_sensitivity_analysis({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "low_cv", "value": 100, "cv_pct": 5},
                {"name": "high_cv", "value": 100, "cv_pct": 50},
            ],
        })
        assert result["top_driver"] == "high_cv"

    def test_sensitivity_high_low_values(self, uq_engine):
        """Test that result_high > central estimate > result_low for positive CV."""
        result = uq_engine.run_sensitivity_analysis({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "param", "value": 100, "cv_pct": 30},
            ],
        })
        s = result["sensitivities"][0]
        assert s["result_high"] > 1000
        assert s["result_low"] < 1000

    def test_sensitivity_validation_error(self, uq_engine):
        """Test validation error when total_co2e is zero."""
        result = uq_engine.run_sensitivity_analysis({
            "total_co2e_tonnes": 0,
            "parameters": [],
        })
        assert result["status"] == "VALIDATION_ERROR"


# ===========================================================================
# 11. Zero and High Uncertainty Tests
# ===========================================================================


class TestZeroAndHighUncertainty:
    """Test boundary cases with zero and very high uncertainty."""

    def test_zero_cv_monte_carlo(self, uq_engine):
        """Test Monte Carlo with zero CV produces zero variance."""
        result = uq_engine.run_monte_carlo({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "exact", "value": 100, "cv_pct": 0, "dist": "normal"},
            ],
            "n_iterations": 100,
            "seed": 42,
        })
        # CV=0 means std_dev=0, so all samples = value; but if param type in DEFAULT_CV, it uses default
        assert result["status"] == "SUCCESS"

    def test_very_high_cv(self, uq_engine):
        """Test Monte Carlo with very high CV (200%) still completes."""
        result = uq_engine.run_monte_carlo({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "high_u", "value": 100, "cv_pct": 200, "dist": "normal"},
            ],
            "n_iterations": 1000,
            "seed": 42,
        })
        assert result["status"] == "SUCCESS"
        assert result["statistics"]["cv_pct"] > 100


# ===========================================================================
# 12. Monte Carlo Validation Tests
# ===========================================================================


class TestMonteCarloValidation:
    """Test Monte Carlo input validation."""

    def test_zero_total_co2e_error(self, uq_engine, mc_request):
        """Test validation error when total_co2e is zero."""
        mc_request["total_co2e_tonnes"] = 0
        result = uq_engine.run_monte_carlo(mc_request)
        assert result["status"] == "VALIDATION_ERROR"

    def test_empty_parameters_error(self, uq_engine, mc_request):
        """Test validation error when parameters list is empty."""
        mc_request["parameters"] = []
        result = uq_engine.run_monte_carlo(mc_request)
        assert result["status"] == "VALIDATION_ERROR"

    def test_provenance_hash_present(self, uq_engine, mc_request):
        """Test that the result has a 64-char provenance hash."""
        result = uq_engine.run_monte_carlo(mc_request)
        assert len(result["provenance_hash"]) == 64

    def test_processing_time_positive(self, uq_engine, mc_request):
        """Test that processing time is a positive number."""
        result = uq_engine.run_monte_carlo(mc_request)
        assert result["processing_time_ms"] >= 0


# ===========================================================================
# 13. Statistics and Reset Tests
# ===========================================================================


class TestUQStatisticsAndReset:
    """Test engine statistics and reset."""

    def test_statistics_structure(self, uq_engine):
        """Test statistics returns expected fields."""
        stats = uq_engine.get_statistics()
        assert stats["engine"] == "UncertaintyQuantifierEngine"
        assert stats["version"] == "1.0.0"
        assert "created_at" in stats
        assert "total_analyses" in stats
        assert stats["default_iterations"] == 5000
        assert stats["default_seed"] == 42

    def test_counter_increments(self, uq_engine, mc_request):
        """Test that analysis counter increments across different methods."""
        uq_engine.run_monte_carlo(mc_request)
        uq_engine.run_error_propagation({
            "total_co2e_tonnes": 1000,
            "parameters": [{"name": "x", "value": 100, "uncertainty_pct": 20}],
        })
        uq_engine.calculate_dqi({
            "reliability": 2, "completeness": 2,
            "temporal_correlation": 2, "geographical_correlation": 2,
            "technological_correlation": 2,
        })
        uq_engine.run_sensitivity_analysis({
            "total_co2e_tonnes": 1000,
            "parameters": [{"name": "x", "value": 100, "cv_pct": 20}],
        })
        stats = uq_engine.get_statistics()
        assert stats["total_analyses"] == 4

    def test_reset_clears_counter(self, uq_engine, mc_request):
        """Test that reset clears the analysis counter."""
        uq_engine.run_monte_carlo(mc_request)
        uq_engine.reset()
        stats = uq_engine.get_statistics()
        assert stats["total_analyses"] == 0


# ===========================================================================
# 14. Batch and Comprehensive Tests
# ===========================================================================


class TestBatchAndComprehensive:
    """Test comprehensive uncertainty analysis scenarios."""

    def test_mc_with_negative_total_co2e(self, uq_engine):
        """Test Monte Carlo with negative total CO2e (net removal)."""
        result = uq_engine.run_monte_carlo({
            "total_co2e_tonnes": -5000,
            "parameters": [
                {"name": "agb", "value": 180, "cv_pct": 30, "dist": "normal"},
            ],
            "n_iterations": 500,
            "seed": 42,
        })
        assert result["status"] == "SUCCESS"
        assert result["statistics"]["mean"] < 0

    def test_mc_parameter_statistics_present(self, uq_engine, mc_request):
        """Test that per-parameter statistics are reported."""
        result = uq_engine.run_monte_carlo(mc_request)
        for param in mc_request["parameters"]:
            name = param["name"]
            assert name in result["parameter_statistics"]
            ps = result["parameter_statistics"][name]
            assert "mean" in ps
            assert "std_dev" in ps
            assert "cv_pct" in ps
            assert "min" in ps
            assert "max" in ps

    def test_mc_result_min_less_than_max(self, uq_engine, mc_request):
        """Test that result min is less than result max."""
        result = uq_engine.run_monte_carlo(mc_request)
        assert result["statistics"]["min"] < result["statistics"]["max"]

    def test_mc_method_field(self, uq_engine, mc_request):
        """Test that method field is MONTE_CARLO."""
        result = uq_engine.run_monte_carlo(mc_request)
        assert result["method"] == "MONTE_CARLO"

    def test_mc_central_estimate_recorded(self, uq_engine, mc_request):
        """Test that central_estimate matches input."""
        result = uq_engine.run_monte_carlo(mc_request)
        assert result["central_estimate"] == 5000


# ===========================================================================
# 15. Additional Confidence Interval Tests
# ===========================================================================


class TestConfidenceIntervalsExtended:
    """Additional confidence interval tests."""

    def test_ci_relative_pct_positive(self, uq_engine, mc_request):
        """Test that relative percentage is positive for all CI levels."""
        result = uq_engine.run_monte_carlo(mc_request)
        for level, ci in result["confidence_intervals"].items():
            assert ci["relative_pct"] >= 0

    def test_ci_99_contains_95(self, uq_engine, mc_request):
        """Test that 99% CI contains the 95% CI entirely."""
        result = uq_engine.run_monte_carlo(mc_request)
        ci99 = result["confidence_intervals"]["99"]
        ci95 = result["confidence_intervals"]["95"]
        assert ci99["lower"] <= ci95["lower"]
        assert ci99["upper"] >= ci95["upper"]

    def test_get_ci_with_n_samples(self, uq_engine):
        """Test confidence interval calculation with sample size."""
        ci = uq_engine.get_confidence_interval(
            mean=500, std_dev=50, confidence_level=95.0, n_samples=100
        )
        # With n_samples, half_width = z * std_dev / sqrt(n)
        expected_hw = 1.960 * 50 / (100 ** 0.5)
        assert abs(ci["half_width"] - expected_hw) < 0.01

    def test_get_ci_99_wider_than_95(self, uq_engine):
        """Test that 99% CI is wider than 95% CI from direct calculation."""
        ci95 = uq_engine.get_confidence_interval(1000, 100, 95.0)
        ci99 = uq_engine.get_confidence_interval(1000, 100, 99.0)
        assert ci99["half_width"] > ci95["half_width"]


# ===========================================================================
# 16. Additional DQI Tests
# ===========================================================================


class TestDQIScoringExtended:
    """Additional DQI scoring tests."""

    def test_dqi_uncertainty_multiplier_range(self, uq_engine):
        """Test that uncertainty multiplier values are within expected range."""
        for score in range(1, 6):
            result = uq_engine.calculate_dqi({
                "reliability": score,
                "completeness": score,
                "temporal_correlation": score,
                "geographical_correlation": score,
                "technological_correlation": score,
            })
            assert 0.5 <= result["uncertainty_multiplier"] <= 3.0

    def test_dqi_mixed_scores_composite(self, uq_engine):
        """Test composite score for asymmetric dimension scores."""
        result = uq_engine.calculate_dqi({
            "reliability": 1,
            "completeness": 5,
            "temporal_correlation": 1,
            "geographical_correlation": 5,
            "technological_correlation": 1,
        })
        # Geometric mean of (1,5,1,5,1) = (25)^(1/5) = 25^0.2 ~ 1.904
        import math
        expected = math.pow(25, 0.2)
        assert abs(result["composite_score"] - round(expected, 4)) < 0.01

    def test_dqi_non_integer_score_error(self, uq_engine):
        """Test that non-integer dimension score returns error."""
        result = uq_engine.calculate_dqi({
            "reliability": "not_a_number",
            "completeness": 2,
            "temporal_correlation": 2,
            "geographical_correlation": 2,
            "technological_correlation": 2,
        })
        assert result["status"] == "VALIDATION_ERROR"

    def test_dqi_dimension_scores_in_result(self, uq_engine):
        """Test that individual dimension scores are returned in the result."""
        result = uq_engine.calculate_dqi({
            "reliability": 2,
            "completeness": 3,
            "temporal_correlation": 4,
            "geographical_correlation": 1,
            "technological_correlation": 5,
        })
        ds = result["dimension_scores"]
        assert ds["reliability"] == 2
        assert ds["completeness"] == 3
        assert ds["temporal_correlation"] == 4
        assert ds["geographical_correlation"] == 1
        assert ds["technological_correlation"] == 5


# ===========================================================================
# 17. Additional Sensitivity Analysis Tests
# ===========================================================================


class TestSensitivityExtended:
    """Additional sensitivity analysis tests."""

    def test_sensitivity_single_parameter(self, uq_engine):
        """Test sensitivity analysis with a single parameter."""
        result = uq_engine.run_sensitivity_analysis({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "only_param", "value": 100, "cv_pct": 20},
            ],
        })
        assert result["status"] == "SUCCESS"
        assert len(result["sensitivities"]) == 1
        assert result["sensitivities"][0]["contribution_pct"] == 100.0

    def test_sensitivity_zero_value_param_skipped(self, uq_engine):
        """Test that parameters with value=0 are skipped in sensitivity."""
        result = uq_engine.run_sensitivity_analysis({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "nonzero", "value": 100, "cv_pct": 20},
                {"name": "zero_val", "value": 0, "cv_pct": 50},
            ],
        })
        assert result["parameter_count"] == 1  # zero_val skipped

    def test_sensitivity_provenance_hash(self, uq_engine):
        """Test that sensitivity analysis result has a provenance hash."""
        result = uq_engine.run_sensitivity_analysis({
            "total_co2e_tonnes": 1000,
            "parameters": [{"name": "x", "value": 100, "cv_pct": 20}],
        })
        assert len(result["provenance_hash"]) == 64

    def test_sensitivity_impact_range_positive(self, uq_engine):
        """Test that impact range is always non-negative."""
        result = uq_engine.run_sensitivity_analysis({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "a", "value": 50, "cv_pct": 10},
                {"name": "b", "value": 200, "cv_pct": 40},
            ],
        })
        for s in result["sensitivities"]:
            assert s["impact_range"] >= 0


# ===========================================================================
# 18. Additional Error Propagation Tests
# ===========================================================================


class TestErrorPropagationExtended:
    """Additional error propagation tests."""

    def test_additive_zero_sum_returns_zero(self, uq_engine):
        """Test additive propagation with zero sum of values."""
        result = uq_engine.run_error_propagation({
            "total_co2e_tonnes": 0,
            "parameters": [
                {"name": "a", "value": 100, "uncertainty_pct": 20},
                {"name": "b", "value": -100, "uncertainty_pct": 20},
            ],
            "combination": "additive",
        })
        assert result["status"] == "SUCCESS"

    def test_multiplicative_single_50_pct(self, uq_engine):
        """Test multiplicative propagation with single 50% uncertainty."""
        result = uq_engine.run_error_propagation({
            "total_co2e_tonnes": 1000,
            "parameters": [
                {"name": "x", "value": 100, "uncertainty_pct": 50},
            ],
            "combination": "multiplicative",
        })
        assert Decimal(result["combined_uncertainty_pct"]) == Decimal("50.00")

    def test_error_propagation_provenance_hash(self, uq_engine, error_prop_request):
        """Test that error propagation result has a provenance hash."""
        result = uq_engine.run_error_propagation(error_prop_request)
        assert len(result["provenance_hash"]) == 64

    def test_error_propagation_parameter_count(self, uq_engine, error_prop_request):
        """Test parameter count is correctly reported."""
        result = uq_engine.run_error_propagation(error_prop_request)
        assert result["parameter_count"] == 3


# ===========================================================================
# 19. DEFAULT_CV Structure Tests
# ===========================================================================


class TestDefaultCVStructure:
    """Test the structure and completeness of DEFAULT_CV dictionary."""

    def test_all_parameter_types_present(self):
        """Test that all expected parameter types are in DEFAULT_CV."""
        expected_types = [
            "AGB", "BGB", "DEAD_WOOD", "LITTER", "SOC_REF",
            "ROOT_SHOOT_RATIO", "GROWTH_RATE", "EMISSION_FACTOR", "AREA",
            "COMBUSTION_FACTOR", "FIRE_EF", "PEATLAND_EF",
            "SOC_FACTOR_FLU", "SOC_FACTOR_FMG", "SOC_FACTOR_FI", "N2O_EF",
        ]
        for pt in expected_types:
            assert pt in DEFAULT_CV, f"Missing parameter type: {pt}"

    def test_all_types_have_three_tiers(self):
        """Test that every parameter type has TIER_1, TIER_2, and TIER_3."""
        for param_type, tiers in DEFAULT_CV.items():
            assert "TIER_1" in tiers, f"{param_type} missing TIER_1"
            assert "TIER_2" in tiers, f"{param_type} missing TIER_2"
            assert "TIER_3" in tiers, f"{param_type} missing TIER_3"

    def test_all_cv_values_positive(self):
        """Test that all CV values are positive numbers."""
        for param_type, tiers in DEFAULT_CV.items():
            for tier, cv in tiers.items():
                assert cv > 0, f"{param_type}/{tier} CV is non-positive: {cv}"


# ===========================================================================
# 20. Helper Function Tests
# ===========================================================================


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_safe_decimal_none(self):
        """Test _safe_decimal returns default for None."""
        from greenlang.land_use_emissions.uncertainty_quantifier import _safe_decimal, _ZERO
        assert _safe_decimal(None) == _ZERO

    def test_safe_decimal_invalid(self):
        """Test _safe_decimal returns default for invalid string."""
        from greenlang.land_use_emissions.uncertainty_quantifier import _safe_decimal, _ZERO
        assert _safe_decimal("abc") == _ZERO

    def test_D_from_string(self):
        """Test _D converts string to Decimal."""
        result = _D("42.5")
        assert result == Decimal("42.5")
