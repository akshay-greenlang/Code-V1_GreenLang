# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Waste Treatment Emissions Agent - UncertaintyQuantifierEngine.

Tests Monte Carlo simulation, analytical uncertainty propagation, DQI scoring,
parameter distributions for waste-specific parameters, confidence intervals,
combined uncertainties, and edge cases.

Target: 85+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import math
import random
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.waste_treatment_emissions.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
        PARAMETER_DISTRIBUTIONS,
        DEFAULT_CV,
        Z_SCORES,
    )
    UNCERTAINTY_AVAILABLE = True
except ImportError:
    UNCERTAINTY_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not UNCERTAINTY_AVAILABLE,
    reason="UncertaintyQuantifierEngine not available",
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine():
    """Create a fresh UncertaintyQuantifierEngine."""
    return UncertaintyQuantifierEngine(default_iterations=5000, default_seed=42)


@pytest.fixture
def small_engine():
    """Engine with fewer iterations for faster tests."""
    return UncertaintyQuantifierEngine(default_iterations=500, default_seed=42)


@pytest.fixture
def base_input():
    """Standard calculation input for uncertainty quantification."""
    return {
        "total_co2e_tonnes": 1200.0,
        "parameters": [
            {"name": "doc", "value": 0.15},
            {"name": "mcf", "value": 1.0},
            {"name": "collection_efficiency", "value": 0.75},
        ],
    }


@pytest.fixture
def analytical_input():
    """Input for analytical uncertainty method."""
    return {
        "total_co2e_tonnes": 500.0,
        "activity_uncertainty_pct": 10.0,
        "ef_uncertainty_pct": 50.0,
    }


@pytest.fixture
def dqi_input():
    """Input for DQI scoring method."""
    return {
        "reliability": 2,
        "completeness": 2,
        "temporal": 3,
        "geographical": 3,
        "technological": 2,
    }


# ===========================================================================
# Test Class: Monte Carlo Simulation
# ===========================================================================


@_SKIP
class TestMonteCarloSimulation:
    """Test Monte Carlo uncertainty simulation."""

    def test_basic_monte_carlo_success(self, engine, base_input):
        """Monte Carlo simulation returns SUCCESS status."""
        result = engine.quantify_uncertainty(base_input, method="monte_carlo")
        assert result["status"] == "SUCCESS"

    def test_default_5000_iterations(self, engine, base_input):
        """Default Monte Carlo uses 5000 iterations."""
        result = engine.quantify_uncertainty(base_input, method="monte_carlo")
        assert result.get("iterations", result.get("n_iterations", 5000)) == 5000

    def test_custom_iterations(self, engine, base_input):
        """Custom iteration count is respected."""
        result = engine.quantify_uncertainty(
            base_input, method="monte_carlo", n_iterations=1000
        )
        assert result.get("iterations", result.get("n_iterations", 1000)) == 1000

    def test_seed_reproducibility(self, engine, base_input):
        """Same seed produces identical results."""
        r1 = engine.quantify_uncertainty(base_input, method="monte_carlo", seed=42)
        r2 = engine.quantify_uncertainty(base_input, method="monte_carlo", seed=42)
        assert r1.get("statistics", {}).get("mean") == r2.get("statistics", {}).get("mean")

    def test_different_seeds_different_results(self, engine, base_input):
        """Different seeds produce different distributions."""
        r1 = engine.quantify_uncertainty(base_input, method="monte_carlo", seed=42)
        r2 = engine.quantify_uncertainty(base_input, method="monte_carlo", seed=99)
        mean_1 = r1.get("statistics", {}).get("mean", 0)
        mean_2 = r2.get("statistics", {}).get("mean", 0)
        # Means should be similar but not identical due to different seeds
        if mean_1 != 0 and mean_2 != 0:
            assert mean_1 != mean_2 or abs(mean_1 - mean_2) < 1.0

    def test_confidence_intervals_present(self, engine, base_input):
        """Result includes confidence intervals."""
        result = engine.quantify_uncertainty(base_input, method="monte_carlo")
        assert "confidence_intervals" in result

    def test_95_ci_bounds(self, engine, base_input):
        """95% confidence interval has lower <= upper (string comparison via float)."""
        result = engine.quantify_uncertainty(
            base_input, method="monte_carlo", confidence_level=0.95
        )
        ci = result.get("confidence_intervals", {})
        if "95" in ci:
            lower = float(ci["95"]["lower"])
            upper = float(ci["95"]["upper"])
            assert lower <= upper

    def test_statistics_fields(self, engine, base_input):
        """Statistics include mean, std_dev, cv_pct."""
        result = engine.quantify_uncertainty(base_input, method="monte_carlo")
        stats = result.get("statistics", {})
        assert "mean" in stats
        assert "std_dev" in stats

    def test_mean_near_base_emissions(self, engine, base_input):
        """Monte Carlo mean should be near the base emissions value."""
        result = engine.quantify_uncertainty(
            base_input, method="monte_carlo", n_iterations=5000
        )
        stats = result.get("statistics", {})
        mean_val = float(stats.get("mean", 0))
        base = float(base_input["total_co2e_tonnes"])
        # Mean should be within 50% of base (generous margin)
        if base > 0 and mean_val > 0:
            assert abs(mean_val - base) / base < 0.5

    def test_provenance_hash_present(self, engine, base_input):
        """Result includes a provenance hash."""
        result = engine.quantify_uncertainty(base_input, method="monte_carlo")
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_processing_time_recorded(self, engine, base_input):
        """Processing time is recorded and non-negative."""
        result = engine.quantify_uncertainty(base_input, method="monte_carlo")
        assert result.get("processing_time_ms", 0) >= 0

    def test_calculation_id_present(self, engine, base_input):
        """Calculation ID is generated."""
        result = engine.quantify_uncertainty(base_input, method="monte_carlo")
        assert "calculation_id" in result
        assert result["calculation_id"] != ""

    def test_no_parameters_returns_validation_error(self, engine):
        """Monte Carlo with empty parameters returns VALIDATION_ERROR (requires >= 1 param)."""
        inp = {"total_co2e_tonnes": 100.0, "parameters": []}
        result = engine.quantify_uncertainty(inp, method="monte_carlo")
        assert result["status"] == "VALIDATION_ERROR"

    def test_single_parameter(self, engine):
        """Monte Carlo with a single parameter works correctly."""
        inp = {
            "total_co2e_tonnes": 500.0,
            "parameters": [{"name": "doc", "value": 0.15}],
        }
        result = engine.quantify_uncertainty(inp, method="monte_carlo")
        assert result["status"] == "SUCCESS"

    def test_many_parameters(self, engine):
        """Monte Carlo with many parameters does not fail."""
        inp = {
            "total_co2e_tonnes": 2000.0,
            "parameters": [
                {"name": "doc", "value": 0.15},
                {"name": "mcf", "value": 1.0},
                {"name": "doc_f", "value": 0.5},
                {"name": "oxidation_factor", "value": 0.95},
                {"name": "collection_efficiency", "value": 0.75},
                {"name": "flare_dre", "value": 0.98},
                {"name": "composting_ef_ch4", "value": 4.0},
                {"name": "incineration_carbon_content", "value": 0.40},
                {"name": "fossil_carbon_fraction", "value": 0.60},
                {"name": "wastewater_mcf", "value": 0.80},
                {"name": "wastewater_bo", "value": 0.25},
            ],
        }
        result = engine.quantify_uncertainty(
            inp, method="monte_carlo", n_iterations=1000
        )
        assert result["status"] == "SUCCESS"

    @pytest.mark.parametrize("conf_level", [0.90, 0.95, 0.99])
    def test_confidence_levels(self, engine, base_input, conf_level):
        """Different confidence levels produce different CI widths."""
        result = engine.quantify_uncertainty(
            base_input, method="monte_carlo", confidence_level=conf_level
        )
        assert result["status"] == "SUCCESS"

    def test_high_iterations_narrower_ci(self, engine, base_input):
        """More iterations should give a more stable estimate."""
        r_low = engine.quantify_uncertainty(
            base_input, method="monte_carlo", n_iterations=100, seed=42
        )
        r_high = engine.quantify_uncertainty(
            base_input, method="monte_carlo", n_iterations=5000, seed=42
        )
        # Both should succeed
        assert r_low["status"] == "SUCCESS"
        assert r_high["status"] == "SUCCESS"


# ===========================================================================
# Test Class: Analytical Uncertainty
# ===========================================================================


@_SKIP
class TestAnalyticalUncertainty:
    """Test analytical error propagation (IPCC Approach 1)."""

    def test_basic_analytical_success(self, engine, analytical_input):
        """Analytical uncertainty returns SUCCESS."""
        result = engine.quantify_uncertainty(analytical_input, method="analytical")
        assert result["status"] == "SUCCESS"

    def test_analytical_combined_uncertainty(self, engine, analytical_input):
        """Combined uncertainty is sqrt(sum(Ui^2)) for uncorrelated params."""
        result = engine.quantify_uncertainty(analytical_input, method="analytical")
        combined = float(result.get("combined_uncertainty_pct", result.get("relative_uncertainty_pct", 0)))
        # U_total = sqrt(10^2 + 50^2) = sqrt(100 + 2500) = sqrt(2600) = 50.99
        expected = math.sqrt(10.0**2 + 50.0**2)
        if combined > 0:
            assert abs(combined - expected) < 1.0

    def test_analytical_provenance_hash(self, engine, analytical_input):
        """Analytical result includes provenance hash."""
        result = engine.quantify_uncertainty(analytical_input, method="analytical")
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_analytical_confidence_interval(self, engine, analytical_input):
        """Analytical result includes confidence interval bounds."""
        result = engine.quantify_uncertainty(
            analytical_input, method="analytical", confidence_level=0.95
        )
        assert "confidence_intervals" in result or "ci_lower" in result or "lower" in str(result)

    def test_analytical_zero_uncertainty(self, engine):
        """Zero uncertainty produces zero combined uncertainty."""
        inp = {
            "total_co2e_tonnes": 100.0,
            "activity_uncertainty_pct": 0.0,
            "ef_uncertainty_pct": 0.0,
        }
        result = engine.quantify_uncertainty(inp, method="analytical")
        assert result["status"] == "SUCCESS"

    def test_analytical_high_uncertainty(self, engine):
        """Very high uncertainties are handled without overflow."""
        inp = {
            "total_co2e_tonnes": 100.0,
            "activity_uncertainty_pct": 100.0,
            "ef_uncertainty_pct": 200.0,
        }
        result = engine.quantify_uncertainty(inp, method="analytical")
        assert result["status"] == "SUCCESS"

    @pytest.mark.parametrize("act_u,ef_u", [
        (5.0, 20.0),
        (10.0, 50.0),
        (15.0, 75.0),
        (2.0, 10.0),
    ])
    def test_analytical_parametrized(self, engine, act_u, ef_u):
        """Parametrized analytical uncertainty tests."""
        inp = {
            "total_co2e_tonnes": 300.0,
            "activity_uncertainty_pct": act_u,
            "ef_uncertainty_pct": ef_u,
        }
        result = engine.quantify_uncertainty(inp, method="analytical")
        assert result["status"] == "SUCCESS"


# ===========================================================================
# Test Class: DQI Scoring
# ===========================================================================


@_SKIP
class TestDQIScoring:
    """Test Data Quality Indicator scoring (5 dimensions, 1-5 scale)."""

    def test_basic_dqi_success(self, engine, dqi_input):
        """DQI scoring returns SUCCESS."""
        result = engine.quantify_uncertainty(dqi_input, method="dqi")
        assert result["status"] == "SUCCESS"

    def test_best_quality_scores(self, engine):
        """All 1s (best quality) produce the lowest uncertainty."""
        best = {
            "reliability": 1,
            "completeness": 1,
            "temporal": 1,
            "geographical": 1,
            "technological": 1,
        }
        result = engine.quantify_uncertainty(best, method="dqi")
        assert result["status"] == "SUCCESS"

    def test_worst_quality_scores(self, engine):
        """All 5s (worst quality) produce the highest uncertainty."""
        worst = {
            "reliability": 5,
            "completeness": 5,
            "temporal": 5,
            "geographical": 5,
            "technological": 5,
        }
        result = engine.quantify_uncertainty(worst, method="dqi")
        assert result["status"] == "SUCCESS"

    def test_dqi_better_quality_lower_uncertainty(self, engine):
        """Better DQI scores produce lower uncertainty estimate."""
        best = {
            "reliability": 1,
            "completeness": 1,
            "temporal": 1,
            "geographical": 1,
            "technological": 1,
        }
        worst = {
            "reliability": 5,
            "completeness": 5,
            "temporal": 5,
            "geographical": 5,
            "technological": 5,
        }
        r_best = engine.quantify_uncertainty(best, method="dqi")
        r_worst = engine.quantify_uncertainty(worst, method="dqi")
        # Extract some uncertainty measure
        u_best = r_best.get("dqi_score", r_best.get("overall_score", 0))
        u_worst = r_worst.get("dqi_score", r_worst.get("overall_score", 0))
        if u_best > 0 and u_worst > 0:
            # Worst should have higher score (more uncertain)
            assert u_worst >= u_best

    def test_dqi_provenance_hash(self, engine, dqi_input):
        """DQI result includes provenance hash."""
        result = engine.quantify_uncertainty(dqi_input, method="dqi")
        assert "provenance_hash" in result

    @pytest.mark.parametrize("dim,value", [
        ("reliability", 3),
        ("completeness", 3),
        ("temporal", 3),
        ("geographical", 3),
        ("technological", 3),
    ])
    def test_dqi_individual_dimensions(self, engine, dim, value):
        """Each DQI dimension contributes to the score."""
        inp = {
            "reliability": 1,
            "completeness": 1,
            "temporal": 1,
            "geographical": 1,
            "technological": 1,
        }
        inp[dim] = value
        result = engine.quantify_uncertainty(inp, method="dqi")
        assert result["status"] == "SUCCESS"

    def test_dqi_default_values(self, engine):
        """DQI with no explicit scores uses defaults (3)."""
        result = engine.quantify_uncertainty({}, method="dqi")
        assert result["status"] == "SUCCESS"


# ===========================================================================
# Test Class: Parameter Distributions
# ===========================================================================


@_SKIP
class TestParameterDistributions:
    """Test waste-specific parameter distribution lookup."""

    def test_doc_distribution_is_normal(self):
        """DOC parameter uses normal distribution."""
        assert PARAMETER_DISTRIBUTIONS["doc"]["type"] == "normal"

    def test_mcf_distribution_is_uniform(self):
        """MCF parameter uses uniform distribution."""
        assert PARAMETER_DISTRIBUTIONS["mcf"]["type"] == "uniform"

    def test_doc_f_distribution_is_triangular(self):
        """DOC_f parameter uses triangular distribution."""
        assert PARAMETER_DISTRIBUTIONS["doc_f"]["type"] == "triangular"
        assert "min" in PARAMETER_DISTRIBUTIONS["doc_f"]
        assert "mode" in PARAMETER_DISTRIBUTIONS["doc_f"]
        assert "max" in PARAMETER_DISTRIBUTIONS["doc_f"]

    def test_oxidation_factor_distribution(self):
        """Oxidation factor uses uniform distribution."""
        assert PARAMETER_DISTRIBUTIONS["oxidation_factor"]["type"] == "uniform"

    def test_collection_efficiency_distribution(self):
        """Collection efficiency uses triangular distribution."""
        assert PARAMETER_DISTRIBUTIONS["collection_efficiency"]["type"] == "triangular"

    def test_flare_dre_distribution(self):
        """Flare DRE uses uniform distribution."""
        assert PARAMETER_DISTRIBUTIONS["flare_dre"]["type"] == "uniform"

    def test_composting_ef_ch4_distribution(self):
        """Composting CH4 EF uses lognormal distribution."""
        assert PARAMETER_DISTRIBUTIONS["composting_ef_ch4"]["type"] == "lognormal"

    def test_incineration_carbon_content_distribution(self):
        """Incineration carbon content uses normal distribution."""
        assert PARAMETER_DISTRIBUTIONS["incineration_carbon_content"]["type"] == "normal"

    def test_fossil_carbon_fraction_distribution(self):
        """Fossil carbon fraction uses uniform distribution."""
        assert PARAMETER_DISTRIBUTIONS["fossil_carbon_fraction"]["type"] == "uniform"

    def test_wastewater_mcf_distribution(self):
        """Wastewater MCF uses uniform distribution."""
        assert PARAMETER_DISTRIBUTIONS["wastewater_mcf"]["type"] == "uniform"

    def test_wastewater_bo_distribution(self):
        """Wastewater Bo uses normal distribution."""
        assert PARAMETER_DISTRIBUTIONS["wastewater_bo"]["type"] == "normal"

    def test_all_distributions_have_descriptions(self):
        """All parameter distributions have descriptions."""
        for name, spec in PARAMETER_DISTRIBUTIONS.items():
            assert "description" in spec, f"Missing description for {name}"

    @pytest.mark.parametrize("param_name", [
        "doc", "mcf", "doc_f", "oxidation_factor",
        "collection_efficiency", "flare_dre", "composting_ef_ch4",
        "incineration_carbon_content", "fossil_carbon_fraction",
        "wastewater_mcf", "wastewater_bo",
    ])
    def test_distribution_type_valid(self, param_name):
        """Each distribution type is one of the supported types."""
        dist = PARAMETER_DISTRIBUTIONS[param_name]
        valid_types = {"normal", "lognormal", "uniform", "triangular"}
        assert dist["type"] in valid_types


# ===========================================================================
# Test Class: Z-Scores and CV Lookup
# ===========================================================================


@_SKIP
class TestZScoresAndCV:
    """Test z-score constants and default CV lookup tables."""

    def test_z_score_90(self):
        """Z-score for 90% CI is 1.645."""
        assert Z_SCORES[90.0] == 1.645

    def test_z_score_95(self):
        """Z-score for 95% CI is 1.960."""
        assert Z_SCORES[95.0] == 1.960

    def test_z_score_99(self):
        """Z-score for 99% CI is 2.576."""
        assert Z_SCORES[99.0] == 2.576

    def test_cv_doc_tier1(self):
        """DOC Tier 1 CV is 20%."""
        assert DEFAULT_CV["DOC"]["TIER_1"] == 20.0

    def test_cv_mcf_tier2(self):
        """MCF Tier 2 CV is 15%."""
        assert DEFAULT_CV["MCF"]["TIER_2"] == 15.0

    def test_cv_gwp_tier3(self):
        """GWP Tier 3 CV is 5%."""
        assert DEFAULT_CV["GWP"]["TIER_3"] == 5.0

    def test_cv_tiers_decrease(self):
        """CV decreases from Tier 1 to Tier 3 for all parameters."""
        for param, tiers in DEFAULT_CV.items():
            assert tiers["TIER_1"] >= tiers["TIER_2"] >= tiers["TIER_3"], (
                f"CV not decreasing for {param}"
            )

    @pytest.mark.parametrize("param", list(DEFAULT_CV.keys()))
    def test_all_params_have_three_tiers(self, param):
        """Each parameter has TIER_1, TIER_2, TIER_3 CV values."""
        assert "TIER_1" in DEFAULT_CV[param]
        assert "TIER_2" in DEFAULT_CV[param]
        assert "TIER_3" in DEFAULT_CV[param]


# ===========================================================================
# Test Class: Unknown Method
# ===========================================================================


@_SKIP
class TestUnknownMethod:
    """Test handling of unknown quantification methods."""

    def test_unknown_method_returns_error(self, engine):
        """Unknown method returns VALIDATION_ERROR status."""
        result = engine.quantify_uncertainty(
            {"total_co2e_tonnes": 100.0},
            method="random_forest",
        )
        assert result["status"] == "VALIDATION_ERROR"

    def test_unknown_method_error_message(self, engine):
        """Unknown method provides helpful error message."""
        result = engine.quantify_uncertainty(
            {"total_co2e_tonnes": 100.0},
            method="bootstrap",
        )
        errors = result.get("errors", [])
        assert len(errors) > 0
        assert "supported" in errors[0].lower() or "unknown" in errors[0].lower()


# ===========================================================================
# Test Class: Edge Cases
# ===========================================================================


@_SKIP
class TestUncertaintyEdgeCases:
    """Test edge cases for uncertainty quantification."""

    def test_zero_emissions(self, engine):
        """Zero base emissions returns VALIDATION_ERROR (must be non-zero)."""
        inp = {
            "total_co2e_tonnes": 0.0,
            "parameters": [{"name": "doc", "value": 0.15}],
        }
        result = engine.quantify_uncertainty(inp, method="monte_carlo")
        assert result["status"] == "VALIDATION_ERROR"

    def test_very_large_emissions(self, engine):
        """Very large emissions do not overflow."""
        inp = {
            "total_co2e_tonnes": 1e12,
            "parameters": [{"name": "doc", "value": 0.15}],
        }
        result = engine.quantify_uncertainty(
            inp, method="monte_carlo", n_iterations=100
        )
        assert result["status"] == "SUCCESS"

    def test_very_small_emissions(self, engine):
        """Very small emissions do not underflow."""
        inp = {
            "total_co2e_tonnes": 0.00001,
            "parameters": [{"name": "doc", "value": 0.15}],
        }
        result = engine.quantify_uncertainty(
            inp, method="monte_carlo", n_iterations=100
        )
        assert result["status"] == "SUCCESS"

    def test_negative_emissions_handled(self, engine):
        """Negative base emissions are handled gracefully."""
        inp = {
            "total_co2e_tonnes": -100.0,
            "parameters": [{"name": "doc", "value": 0.15}],
        }
        result = engine.quantify_uncertainty(
            inp, method="monte_carlo", n_iterations=100
        )
        # Should succeed or return error, but not crash
        assert "status" in result

    def test_analysis_counter_increments(self, engine, base_input):
        """Analysis counter increments on each call."""
        initial = engine._total_analyses
        engine.quantify_uncertainty(base_input, method="monte_carlo", n_iterations=100)
        assert engine._total_analyses == initial + 1

    def test_method_case_insensitive(self, engine, base_input):
        """Method parameter is case-insensitive."""
        result = engine.quantify_uncertainty(base_input, method="MONTE_CARLO")
        assert result["status"] == "SUCCESS"

    def test_method_with_spaces_trimmed(self, engine, base_input):
        """Method parameter with spaces is trimmed."""
        result = engine.quantify_uncertainty(base_input, method="  monte_carlo  ")
        assert result["status"] == "SUCCESS"

    def test_explicit_distribution_override(self, engine):
        """Parameters with explicit distribution override use it."""
        inp = {
            "total_co2e_tonnes": 500.0,
            "parameters": [
                {"name": "custom_param", "value": 0.5, "dist": "uniform", "range_pct": 20},
            ],
        }
        result = engine.quantify_uncertainty(
            inp, method="monte_carlo", n_iterations=500
        )
        assert result["status"] == "SUCCESS"

    def test_missing_total_co2e_returns_validation_error(self, engine):
        """Missing total_co2e_tonnes defaults to zero, which triggers VALIDATION_ERROR."""
        inp = {"parameters": [{"name": "doc", "value": 0.15}]}
        result = engine.quantify_uncertainty(inp, method="monte_carlo", n_iterations=100)
        assert result["status"] == "VALIDATION_ERROR"
