# -*- coding: utf-8 -*-
"""
Unit tests for UncertaintyQuantifierEngine (Engine 5 of 7) - AGENT-MRV-005

Tests Monte Carlo simulation, DQI scoring, analytical propagation,
sensitivity analysis, confidence intervals, process-specific uncertainty
ranges (equipment +/-30-100%, wastewater +/-40-100%), and seed
reproducibility.

Target: 90 tests, ~870 lines.

Test classes:
    TestMonteCarlo (20)
    TestDQI (15)
    TestAnalytical (10)
    TestSensitivity (10)
    TestConfidenceIntervals (10)
    TestSourceSpecificRanges (15)
    TestEdgeCases (10)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.fugitive_emissions.uncertainty_quantifier import (
    UncertaintyQuantifierEngine,
    FugitiveSourceCategory,
    CalculationMethodType,
    DQIDimension,
    UNCERTAINTY_RANGES,
    DEFAULT_PARAMETER_UNCERTAINTIES,
    DQI_SCORING_MATRIX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Default engine with standard config."""
    return UncertaintyQuantifierEngine()


@pytest.fixture
def engine_seeded():
    """Engine with deterministic seed 42."""
    return UncertaintyQuantifierEngine(config={"monte_carlo_seed": 42})


@pytest.fixture
def engine_custom_iters():
    """Engine with custom iteration count of 2000."""
    return UncertaintyQuantifierEngine(config={
        "monte_carlo_iterations": 2000,
        "monte_carlo_seed": 99,
    })


@pytest.fixture
def equipment_leak_input():
    return {
        "total_co2e_kg": 500000.0,
        "source_type": "EQUIPMENT_LEAK",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "component_count": 5000,
        "emission_factor": 0.0089,
    }


@pytest.fixture
def wastewater_input():
    return {
        "total_co2e_kg": 100000.0,
        "source_type": "WASTEWATER",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
    }


@pytest.fixture
def coal_mine_input():
    return {
        "total_co2e_kg": 250000.0,
        "source_type": "COAL_MINE_METHANE",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
    }


@pytest.fixture
def pneumatic_input():
    return {
        "total_co2e_kg": 50000.0,
        "source_type": "PNEUMATIC_DEVICE",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
    }


@pytest.fixture
def direct_measurement_input():
    return {
        "total_co2e_kg": 75000.0,
        "source_type": "DIRECT_MEASUREMENT",
        "calculation_method": "DIRECT_MEASUREMENT",
    }


@pytest.fixture
def tank_loss_input():
    return {
        "total_co2e_kg": 30000.0,
        "source_type": "TANK_LOSS",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
    }


@pytest.fixture
def ldar_screening_input():
    return {
        "total_co2e_kg": 200000.0,
        "source_type": "LDAR_SCREENING",
        "calculation_method": "SCREENING_RANGES",
    }


# ===========================================================================
# TestMonteCarlo (20 tests)
# ===========================================================================


class TestMonteCarlo:
    """Tests for run_monte_carlo method."""

    def test_basic_run_returns_dict(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=1000,
            seed=42,
        )
        assert isinstance(result, dict)
        assert result["method"] == "monte_carlo"

    def test_iterations_echoed(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=1234,
            seed=42,
        )
        assert result["iterations"] == 1234

    def test_mean_positive(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=2000,
            seed=42,
        )
        assert result["mean_co2e_kg"] > 0

    def test_std_dev_positive(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=2000,
            seed=42,
        )
        assert result["std_dev_kg"] > 0

    def test_confidence_intervals_present(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=1000,
            seed=42,
        )
        ci = result["confidence_intervals"]
        assert len(ci) >= 1

    def test_confidence_interval_bounds_ordered(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=5000,
            seed=42,
        )
        for level_key, bounds in result["confidence_intervals"].items():
            assert bounds["lower"] < bounds["upper"]
            assert bounds["lower"] >= 0

    def test_90_95_99_intervals(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=5000,
            seed=42,
            confidence_levels=[90.0, 95.0, 99.0],
        )
        ci = result["confidence_intervals"]
        assert len(ci) == 3

    def test_mean_near_point_estimate(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=10000,
            seed=42,
        )
        ratio = result["mean_co2e_kg"] / equipment_leak_input["total_co2e_kg"]
        assert 0.5 < ratio < 2.0

    def test_cv_positive(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=1000,
            seed=42,
        )
        assert result["coefficient_of_variation"] > 0

    def test_provenance_hash_sha256(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=100,
            seed=42,
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_processing_time_tracked(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=100,
            seed=42,
        )
        assert result["processing_time_ms"] > 0

    def test_source_type_echoed(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=100,
            seed=42,
        )
        assert result["source_type"] == "EQUIPMENT_LEAK"

    def test_calculation_method_echoed(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=100,
            seed=42,
        )
        assert result["calculation_method"] == "AVERAGE_EMISSION_FACTOR"

    def test_parameter_contributions_present(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=500,
            seed=42,
        )
        assert "parameter_contributions" in result
        assert isinstance(result["parameter_contributions"], dict)

    def test_seed_reproducibility_mean(self, engine, equipment_leak_input):
        r1 = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=1000,
            seed=42,
        )
        r2 = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=1000,
            seed=42,
        )
        assert r1["mean_co2e_kg"] == r2["mean_co2e_kg"]
        assert r1["std_dev_kg"] == r2["std_dev_kg"]

    def test_seed_reproducibility_ci(self, engine, equipment_leak_input):
        r1 = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=1000,
            seed=42,
        )
        r2 = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=1000,
            seed=42,
        )
        assert r1["confidence_intervals"] == r2["confidence_intervals"]

    def test_different_seed_different_mean(self, engine, equipment_leak_input):
        r1 = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=5000,
            seed=42,
        )
        r2 = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=5000,
            seed=123,
        )
        assert r1["mean_co2e_kg"] != r2["mean_co2e_kg"]

    def test_default_seed_from_config(self, engine_seeded, equipment_leak_input):
        r1 = engine_seeded.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=100,
        )
        r2 = engine_seeded.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=100,
        )
        assert r1["mean_co2e_kg"] == r2["mean_co2e_kg"]

    def test_increments_mc_counter(self, engine, equipment_leak_input):
        engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=100,
            seed=42,
        )
        stats = engine.get_statistics()
        assert stats["total_mc_runs"] >= 1

    def test_percentiles_present(self, engine, equipment_leak_input):
        result = engine.run_monte_carlo(
            calculation_input=equipment_leak_input,
            n_iterations=1000,
            seed=42,
        )
        assert "percentiles" in result
        percentiles = result["percentiles"]
        assert "P50" in percentiles


# ===========================================================================
# TestDQI (15 tests)
# ===========================================================================


class TestDQI:
    """Tests for calculate_dqi Data Quality Indicator scoring."""

    def test_all_best_scores_excellent(self, engine):
        result = engine.calculate_dqi({
            "reliability": 1,
            "completeness": 1,
            "temporal_correlation": 1,
            "geographical_correlation": 1,
            "technological_correlation": 1,
        })
        assert result["composite_score"] == 1.0
        assert result["quality_label"] == "excellent"

    def test_all_worst_scores_very_poor(self, engine):
        result = engine.calculate_dqi({
            "reliability": 5,
            "completeness": 5,
            "temporal_correlation": 5,
            "geographical_correlation": 5,
            "technological_correlation": 5,
        })
        assert result["composite_score"] == 5.0
        assert result["quality_label"] == "very_poor"

    def test_mixed_scores_geometric_mean(self, engine):
        scores = {
            "reliability": 2, "completeness": 4,
            "temporal_correlation": 3, "geographical_correlation": 1,
            "technological_correlation": 5,
        }
        result = engine.calculate_dqi(scores)
        product = 2 * 4 * 3 * 1 * 5
        expected = product ** (1.0 / 5)
        assert result["composite_score"] == pytest.approx(expected, rel=0.01)

    def test_uniform_score_3_returns_3(self, engine):
        result = engine.calculate_dqi({
            "reliability": 3, "completeness": 3,
            "temporal_correlation": 3, "geographical_correlation": 3,
            "technological_correlation": 3,
        })
        assert result["composite_score"] == pytest.approx(3.0, rel=0.01)
        assert result["quality_label"] == "fair"

    def test_label_string_input(self, engine):
        result = engine.calculate_dqi({
            "reliability": "direct_measurement",
            "completeness": "above_95",
            "temporal_correlation": "same_year",
            "geographical_correlation": "same_facility",
            "technological_correlation": "same_equipment",
        })
        assert result["composite_score"] <= 2.0

    def test_empty_scores_default_to_3(self, engine):
        result = engine.calculate_dqi({})
        assert result["composite_score"] == pytest.approx(3.0, rel=0.01)

    def test_uncertainty_multiplier_present(self, engine):
        result = engine.calculate_dqi({
            "reliability": 3, "completeness": 3,
            "temporal_correlation": 3, "geographical_correlation": 3,
            "technological_correlation": 3,
        })
        assert "uncertainty_multiplier" in result
        assert result["uncertainty_multiplier"] > 0

    def test_uncertainty_multiplier_best_quality(self, engine):
        result = engine.calculate_dqi({
            "reliability": 1, "completeness": 1,
            "temporal_correlation": 1, "geographical_correlation": 1,
            "technological_correlation": 1,
        })
        # composite=1.0 => multiplier = 0.5
        assert result["uncertainty_multiplier"] == pytest.approx(0.5, abs=0.01)

    def test_uncertainty_multiplier_worst_quality(self, engine):
        result = engine.calculate_dqi({
            "reliability": 5, "completeness": 5,
            "temporal_correlation": 5, "geographical_correlation": 5,
            "technological_correlation": 5,
        })
        # composite=5.0 => multiplier = 2.0
        assert result["uncertainty_multiplier"] == pytest.approx(2.0, abs=0.01)

    def test_provenance_hash_present(self, engine):
        result = engine.calculate_dqi({
            "reliability": 2, "completeness": 2,
            "temporal_correlation": 2, "geographical_correlation": 2,
            "technological_correlation": 2,
        })
        assert len(result["provenance_hash"]) == 64

    def test_increments_dqi_counter(self, engine):
        engine.calculate_dqi({"reliability": 3})
        stats = engine.get_statistics()
        assert stats["total_dqi_scores"] >= 1

    def test_score_clamped_min_1(self, engine):
        result = engine.calculate_dqi({"reliability": 0, "completeness": -5})
        scores = result["dimension_scores"]
        assert scores["reliability"] >= 1
        assert scores["completeness"] >= 1

    def test_score_clamped_max_5(self, engine):
        result = engine.calculate_dqi({"reliability": 10, "completeness": 99})
        scores = result["dimension_scores"]
        assert scores["reliability"] <= 5
        assert scores["completeness"] <= 5

    def test_quality_label_good(self, engine):
        result = engine.calculate_dqi({
            "reliability": 2, "completeness": 2,
            "temporal_correlation": 2, "geographical_correlation": 2,
            "technological_correlation": 2,
        })
        assert result["quality_label"] == "good"

    def test_quality_label_poor(self, engine):
        result = engine.calculate_dqi({
            "reliability": 4, "completeness": 4,
            "temporal_correlation": 4, "geographical_correlation": 4,
            "technological_correlation": 5,
        })
        assert result["quality_label"] in ("poor", "very_poor")


# ===========================================================================
# TestAnalytical (10 tests)
# ===========================================================================


class TestAnalytical:
    """Tests for analytical_propagation (IPCC Approach 1)."""

    def test_method_label(self, engine, equipment_leak_input):
        result = engine.analytical_propagation(equipment_leak_input)
        assert result["method"] == "analytical_propagation"

    def test_combined_uncertainty_positive(self, engine, equipment_leak_input):
        result = engine.analytical_propagation(equipment_leak_input)
        assert result["combined_uncertainty_pct"] > 0

    def test_95_ci_present(self, engine, equipment_leak_input):
        result = engine.analytical_propagation(equipment_leak_input)
        ci = result["confidence_intervals"]
        assert "95" in ci
        assert ci["95"]["lower"] < ci["95"]["upper"]

    def test_lower_bound_non_negative(self, engine, equipment_leak_input):
        result = engine.analytical_propagation(equipment_leak_input)
        assert result["confidence_intervals"]["95"]["lower"] >= 0

    def test_mean_equals_point_estimate(self, engine, equipment_leak_input):
        result = engine.analytical_propagation(equipment_leak_input)
        assert result["mean_co2e_kg"] == equipment_leak_input["total_co2e_kg"]

    def test_std_dev_positive(self, engine, equipment_leak_input):
        result = engine.analytical_propagation(equipment_leak_input)
        assert result["std_dev_kg"] > 0

    def test_cv_positive(self, engine, equipment_leak_input):
        result = engine.analytical_propagation(equipment_leak_input)
        assert result["coefficient_of_variation"] > 0

    def test_parameter_uncertainties_present(self, engine, equipment_leak_input):
        result = engine.analytical_propagation(equipment_leak_input)
        assert "parameter_uncertainties" in result
        assert len(result["parameter_uncertainties"]) > 0

    def test_provenance_hash(self, engine, equipment_leak_input):
        result = engine.analytical_propagation(equipment_leak_input)
        assert len(result["provenance_hash"]) == 64

    def test_increments_analytical_counter(self, engine, equipment_leak_input):
        engine.analytical_propagation(equipment_leak_input)
        stats = engine.get_statistics()
        assert stats["total_analytical_runs"] >= 1


# ===========================================================================
# TestSensitivity (10 tests)
# ===========================================================================


class TestSensitivity:
    """Tests for sensitivity_analysis (one-at-a-time tornado chart data)."""

    def test_tornado_data_present(self, engine, equipment_leak_input):
        result = engine.sensitivity_analysis(
            calculation_input=equipment_leak_input,
            n_iterations=500,
            seed=42,
        )
        assert "tornado_data" in result
        assert len(result["tornado_data"]) > 0

    def test_tornado_sorted_descending(self, engine, equipment_leak_input):
        result = engine.sensitivity_analysis(
            calculation_input=equipment_leak_input,
            n_iterations=500,
            seed=42,
        )
        contributions = [
            item["contribution_pct"]
            for item in result["tornado_data"]
        ]
        assert contributions == sorted(contributions, reverse=True)

    def test_parameter_has_low_high(self, engine, equipment_leak_input):
        result = engine.sensitivity_analysis(
            calculation_input=equipment_leak_input,
            n_iterations=500,
            seed=42,
        )
        for item in result["tornado_data"]:
            assert "low_value_co2e_kg" in item
            assert "high_value_co2e_kg" in item
            assert item["low_value_co2e_kg"] <= item["high_value_co2e_kg"]

    def test_baseline_variance_positive(self, engine, equipment_leak_input):
        result = engine.sensitivity_analysis(
            calculation_input=equipment_leak_input,
            n_iterations=500,
            seed=42,
        )
        assert result["baseline_variance"] > 0

    def test_increments_sensitivity_counter(self, engine, equipment_leak_input):
        engine.sensitivity_analysis(
            calculation_input=equipment_leak_input,
            n_iterations=100,
            seed=42,
        )
        stats = engine.get_statistics()
        assert stats["total_sensitivity_runs"] >= 1

    def test_point_estimate_echoed(self, engine, equipment_leak_input):
        result = engine.sensitivity_analysis(
            calculation_input=equipment_leak_input,
            n_iterations=100,
            seed=42,
        )
        assert result["point_estimate_co2e_kg"] == equipment_leak_input["total_co2e_kg"]

    def test_method_label(self, engine, equipment_leak_input):
        result = engine.sensitivity_analysis(
            calculation_input=equipment_leak_input,
            n_iterations=100,
            seed=42,
        )
        assert result["method"] == "sensitivity_analysis"

    def test_provenance_hash(self, engine, equipment_leak_input):
        result = engine.sensitivity_analysis(
            calculation_input=equipment_leak_input,
            n_iterations=100,
            seed=42,
        )
        assert len(result["provenance_hash"]) == 64

    def test_processing_time(self, engine, equipment_leak_input):
        result = engine.sensitivity_analysis(
            calculation_input=equipment_leak_input,
            n_iterations=100,
            seed=42,
        )
        assert result["processing_time_ms"] > 0

    def test_wastewater_sensitivity(self, engine, wastewater_input):
        result = engine.sensitivity_analysis(
            calculation_input=wastewater_input,
            n_iterations=500,
            seed=42,
        )
        assert len(result["tornado_data"]) > 0


# ===========================================================================
# TestConfidenceIntervals (10 tests)
# ===========================================================================


class TestConfidenceIntervals:
    """Tests for get_confidence_intervals."""

    def test_empty_samples_returns_empty(self, engine):
        result = engine.get_confidence_intervals([])
        assert result == {}

    def test_single_sample(self, engine):
        result = engine.get_confidence_intervals([100.0])
        for level_key, bounds in result.items():
            assert bounds["lower"] == bounds["upper"]

    def test_default_levels_90_95_99(self, engine):
        samples = [float(i) for i in range(1000)]
        result = engine.get_confidence_intervals(samples)
        assert len(result) == 3
        keys = sorted(result.keys())
        assert "90" in keys
        assert "95" in keys
        assert "99" in keys

    def test_custom_levels(self, engine):
        samples = [float(i) for i in range(1000)]
        result = engine.get_confidence_intervals(samples, levels=[80.0])
        assert len(result) == 1
        assert "80" in result

    def test_lower_less_than_upper(self, engine):
        samples = [float(i) for i in range(100)]
        result = engine.get_confidence_intervals(samples, levels=[90.0, 95.0])
        for level_key, bounds in result.items():
            assert bounds["lower"] <= bounds["upper"]

    def test_99_wider_than_95(self, engine):
        samples = [float(i) for i in range(1000)]
        result = engine.get_confidence_intervals(
            samples, levels=[95.0, 99.0],
        )
        ci_95 = result["95"]
        ci_99 = result["99"]
        width_95 = ci_95["upper"] - ci_95["lower"]
        width_99 = ci_99["upper"] - ci_99["lower"]
        assert width_99 >= width_95

    def test_95_wider_than_90(self, engine):
        samples = [float(i) for i in range(1000)]
        result = engine.get_confidence_intervals(
            samples, levels=[90.0, 95.0],
        )
        ci_90 = result["90"]
        ci_95 = result["95"]
        width_90 = ci_90["upper"] - ci_90["lower"]
        width_95 = ci_95["upper"] - ci_95["lower"]
        assert width_95 >= width_90

    def test_all_same_values(self, engine):
        samples = [42.0] * 100
        result = engine.get_confidence_intervals(samples, levels=[95.0])
        assert result["95"]["lower"] == result["95"]["upper"]

    def test_sorted_input_not_required(self, engine):
        import random
        rng = random.Random(42)
        samples = [rng.gauss(100, 10) for _ in range(500)]
        result = engine.get_confidence_intervals(samples, levels=[95.0])
        assert result["95"]["lower"] < result["95"]["upper"]

    def test_negative_values_allowed(self, engine):
        samples = [float(i - 500) for i in range(1000)]
        result = engine.get_confidence_intervals(samples, levels=[95.0])
        assert result["95"]["lower"] < result["95"]["upper"]


# ===========================================================================
# TestSourceSpecificRanges (15 tests)
# ===========================================================================


class TestSourceSpecificRanges:
    """Validate hardcoded uncertainty ranges for all source/method combos."""

    def test_equipment_leak_avg_ef(self):
        rng = UNCERTAINTY_RANGES["EQUIPMENT_LEAK"]["AVERAGE_EMISSION_FACTOR"]
        assert rng == (30.0, 100.0)

    def test_equipment_leak_screening(self):
        rng = UNCERTAINTY_RANGES["EQUIPMENT_LEAK"]["SCREENING_RANGES"]
        assert rng == (20.0, 50.0)

    def test_equipment_leak_epa_corr(self):
        rng = UNCERTAINTY_RANGES["EQUIPMENT_LEAK"]["EPA_CORRELATION"]
        assert rng == (25.0, 60.0)

    def test_equipment_leak_direct(self):
        rng = UNCERTAINTY_RANGES["EQUIPMENT_LEAK"]["DIRECT_MEASUREMENT"]
        assert rng == (10.0, 25.0)

    def test_coal_mine_avg_ef(self):
        rng = UNCERTAINTY_RANGES["COAL_MINE_METHANE"]["AVERAGE_EMISSION_FACTOR"]
        assert rng == (50.0, 75.0)

    def test_coal_mine_direct(self):
        rng = UNCERTAINTY_RANGES["COAL_MINE_METHANE"]["DIRECT_MEASUREMENT"]
        assert rng == (15.0, 35.0)

    def test_wastewater_avg_ef(self):
        rng = UNCERTAINTY_RANGES["WASTEWATER"]["AVERAGE_EMISSION_FACTOR"]
        assert rng == (60.0, 100.0)

    def test_pneumatic_avg_ef(self):
        rng = UNCERTAINTY_RANGES["PNEUMATIC_DEVICE"]["AVERAGE_EMISSION_FACTOR"]
        assert rng == (20.0, 40.0)

    def test_tank_loss_avg_ef(self):
        rng = UNCERTAINTY_RANGES["TANK_LOSS"]["AVERAGE_EMISSION_FACTOR"]
        assert rng == (35.0, 50.0)

    def test_all_categories_have_ranges(self):
        for cat in FugitiveSourceCategory:
            assert cat.value in UNCERTAINTY_RANGES, (
                f"Missing range for {cat.value}"
            )

    def test_all_methods_lower_less_than_upper(self):
        for cat_ranges in UNCERTAINTY_RANGES.values():
            for method, (lower, upper) in cat_ranges.items():
                assert lower > 0
                assert upper >= lower, f"Invalid range for {method}"

    def test_default_emission_factor_50(self):
        assert DEFAULT_PARAMETER_UNCERTAINTIES["emission_factor"] == 50.0

    def test_default_activity_data_10(self):
        assert DEFAULT_PARAMETER_UNCERTAINTIES["activity_data"] == 10.0

    def test_dqi_matrix_five_dimensions(self):
        assert len(DQI_SCORING_MATRIX) == 5

    def test_dqi_matrix_five_entries_per_dim(self):
        for dim, entries in DQI_SCORING_MATRIX.items():
            assert len(entries) == 5, f"{dim} should have 5 entries"


# ===========================================================================
# TestEdgeCases (10 tests)
# ===========================================================================


class TestEdgeCases:
    """Edge cases and error handling for uncertainty quantification."""

    def test_zero_co2e_mc(self, engine):
        result = engine.run_monte_carlo(
            calculation_input={
                "total_co2e_kg": 0,
                "source_type": "EQUIPMENT_LEAK",
                "calculation_method": "AVERAGE_EMISSION_FACTOR",
            },
            n_iterations=100,
            seed=42,
        )
        assert result["mean_co2e_kg"] == pytest.approx(0.0, abs=1e-6)

    def test_zero_co2e_analytical(self, engine):
        result = engine.analytical_propagation({
            "total_co2e_kg": 0,
            "source_type": "EQUIPMENT_LEAK",
            "calculation_method": "AVERAGE_EMISSION_FACTOR",
        })
        assert result["mean_co2e_kg"] == 0.0

    def test_very_large_co2e(self, engine):
        result = engine.run_monte_carlo(
            calculation_input={
                "total_co2e_kg": 1e12,
                "source_type": "EQUIPMENT_LEAK",
                "calculation_method": "AVERAGE_EMISSION_FACTOR",
            },
            n_iterations=100,
            seed=42,
        )
        assert result["mean_co2e_kg"] > 0

    def test_unknown_source_type_fallback(self, engine):
        result = engine.run_monte_carlo(
            calculation_input={
                "total_co2e_kg": 1000.0,
                "source_type": "UNKNOWN_SOURCE",
                "calculation_method": "AVERAGE_EMISSION_FACTOR",
            },
            n_iterations=100,
            seed=42,
        )
        # Should fall back to EQUIPMENT_LEAK defaults
        assert result["mean_co2e_kg"] > 0

    def test_unknown_method_fallback(self, engine):
        result = engine.run_monte_carlo(
            calculation_input={
                "total_co2e_kg": 1000.0,
                "source_type": "EQUIPMENT_LEAK",
                "calculation_method": "UNKNOWN_METHOD",
            },
            n_iterations=100,
            seed=42,
        )
        assert result["mean_co2e_kg"] > 0

    def test_custom_parameter_uncertainties(self, engine):
        result = engine.analytical_propagation({
            "total_co2e_kg": 100000.0,
            "source_type": "EQUIPMENT_LEAK",
            "calculation_method": "AVERAGE_EMISSION_FACTOR",
            "emission_factor_uncertainty_pct": 80.0,
            "activity_data_uncertainty_pct": 20.0,
        })
        uncertainties = result["parameter_uncertainties"]
        assert uncertainties["emission_factor"] == 80.0
        assert uncertainties["activity_data"] == 20.0

    def test_statistics_initial_state(self):
        engine = UncertaintyQuantifierEngine()
        stats = engine.get_statistics()
        assert stats["total_mc_runs"] == 0
        assert stats["total_analytical_runs"] == 0
        assert stats["total_dqi_scores"] == 0
        assert stats["total_sensitivity_runs"] == 0

    def test_quantify_dispatches_monte_carlo(self, engine, equipment_leak_input):
        result = engine.quantify_uncertainty(
            calculation_input=equipment_leak_input,
            method="monte_carlo",
            n_iterations=100,
            seed=42,
        )
        assert result["method"] == "monte_carlo"

    def test_quantify_dispatches_analytical(self, engine, equipment_leak_input):
        result = engine.quantify_uncertainty(
            calculation_input=equipment_leak_input,
            method="analytical",
        )
        assert result["method"] == "analytical_propagation"

    def test_dqi_from_mc_input(self, engine):
        result = engine.run_monte_carlo(
            calculation_input={
                "total_co2e_kg": 100000.0,
                "source_type": "EQUIPMENT_LEAK",
                "calculation_method": "AVERAGE_EMISSION_FACTOR",
                "dqi_scores": {
                    "reliability": 2,
                    "completeness": 2,
                    "temporal_correlation": 2,
                    "geographical_correlation": 2,
                    "technological_correlation": 2,
                },
            },
            n_iterations=100,
            seed=42,
        )
        assert result["data_quality_score"] is not None
        assert 1.0 <= result["data_quality_score"] <= 5.0
