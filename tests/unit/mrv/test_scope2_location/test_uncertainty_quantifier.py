# -*- coding: utf-8 -*-
"""
Unit tests for UncertaintyQuantifierEngine (Engine 5 of 7)

AGENT-MRV-009: Scope 2 Location-Based Emissions Agent

Tests Monte Carlo simulation, per-gas Monte Carlo, analytical error
propagation, DQI scoring, propagation helpers, IPCC default uncertainty
ranges, sensitivity analysis, and statistical helper methods.

Target: 30 tests, ~350 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.scope2_location.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
        GRID_EF_UNCERTAINTY,
        ACTIVITY_DATA_UNCERTAINTY,
        TD_LOSS_UNCERTAINTY,
        GWP_UNCERTAINTY,
        PER_GAS_EF_UNCERTAINTY,
        EF_SOURCE_SCORES,
        ACTIVITY_SOURCE_SCORES,
        DATA_QUALITY_TIER_UNCERTAINTY,
        CONSUMPTION_SOURCE_UNCERTAINTY,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a default UncertaintyQuantifierEngine."""
    eng = UncertaintyQuantifierEngine()
    yield eng
    eng.reset()


# ===========================================================================
# 1. TestMonteCarlo
# ===========================================================================


@_SKIP
class TestMonteCarlo:
    """Tests for run_monte_carlo with seed reproducibility."""

    def test_run_monte_carlo_success(self, engine):
        """MC simulation returns SUCCESS with valid inputs."""
        result = engine.run_monte_carlo(
            base_emissions_kg=Decimal("150000"),
            ef_uncertainty_pct=Decimal("0.10"),
            activity_uncertainty_pct=Decimal("0.05"),
            td_uncertainty_pct=Decimal("0.03"),
            iterations=1000,
            seed=42,
        )
        assert result["status"] == "SUCCESS"
        assert result["method"] == "MONTE_CARLO"
        assert result["iterations"] == 1000
        assert result["seed"] == 42
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_reproducibility_with_seed(self, engine):
        """Same seed produces identical results (bit-perfect)."""
        kwargs = dict(
            base_emissions_kg=Decimal("100000"),
            ef_uncertainty_pct=Decimal("0.10"),
            iterations=500,
            seed=99,
        )
        r1 = engine.run_monte_carlo(**kwargs)
        r2 = engine.run_monte_carlo(**kwargs)
        assert r1["mean_co2e_kg"] == r2["mean_co2e_kg"]
        assert r1["std_dev"] == r2["std_dev"]
        assert r1["ci_lower"] == r2["ci_lower"]
        assert r1["ci_upper"] == r2["ci_upper"]

    def test_mean_approx_base(self, engine):
        """MC mean should be approximately equal to the base emissions."""
        base = Decimal("200000")
        result = engine.run_monte_carlo(
            base_emissions_kg=base,
            ef_uncertainty_pct=Decimal("0.05"),
            activity_uncertainty_pct=Decimal("0.02"),
            td_uncertainty_pct=Decimal("0.01"),
            iterations=5000,
            seed=42,
        )
        mean = float(result["mean_co2e_kg"])
        assert abs(mean - float(base)) / float(base) < 0.05

    def test_ci_contains_base(self, engine):
        """95% confidence interval should contain the base value."""
        base = Decimal("100000")
        result = engine.run_monte_carlo(
            base_emissions_kg=base,
            ef_uncertainty_pct=Decimal("0.10"),
            iterations=5000,
            seed=42,
        )
        ci_lower = float(result["ci_lower"])
        ci_upper = float(result["ci_upper"])
        assert ci_lower <= float(base) <= ci_upper

    def test_validation_error_negative_base(self, engine):
        """Negative base emissions returns VALIDATION_ERROR."""
        result = engine.run_monte_carlo(
            base_emissions_kg=Decimal("-100"),
            ef_uncertainty_pct=Decimal("0.10"),
            iterations=500,
            seed=42,
        )
        assert result["status"] == "VALIDATION_ERROR"
        assert len(result["errors"]) > 0

    def test_percentiles_present(self, engine):
        """Result contains percentile data."""
        result = engine.run_monte_carlo(
            base_emissions_kg=Decimal("50000"),
            ef_uncertainty_pct=Decimal("0.10"),
            iterations=500,
            seed=42,
        )
        assert "percentiles" in result
        assert "50" in result["percentiles"] or "50.0" in result["percentiles"]


# ===========================================================================
# 2. TestMonteCarloPerGas
# ===========================================================================


@_SKIP
class TestMonteCarloPerGas:
    """Tests for run_monte_carlo_per_gas."""

    def test_per_gas_success(self, engine):
        """Per-gas MC returns SUCCESS with per-gas breakdown."""
        result = engine.run_monte_carlo_per_gas(
            co2_kg=Decimal("140000"),
            ch4_kg=Decimal("5000"),
            n2o_kg=Decimal("5000"),
            iterations=500,
            seed=42,
        )
        assert result["status"] == "SUCCESS"
        assert result["method"] == "MONTE_CARLO_PER_GAS"
        assert "CO2" in result["per_gas"]
        assert "CH4" in result["per_gas"]
        assert "N2O" in result["per_gas"]
        assert "total" in result

    def test_per_gas_co2_dominates(self, engine):
        """CO2 mean should be close to the input CO2 value."""
        result = engine.run_monte_carlo_per_gas(
            co2_kg=Decimal("140000"),
            ch4_kg=Decimal("100"),
            n2o_kg=Decimal("50"),
            iterations=1000,
            seed=42,
        )
        co2_mean = float(result["per_gas"]["CO2"]["mean_kg"])
        assert abs(co2_mean - 140000) / 140000 < 0.10

    def test_per_gas_provenance(self, engine):
        """Per-gas MC result includes a provenance hash."""
        result = engine.run_monte_carlo_per_gas(
            co2_kg=Decimal("100000"),
            ch4_kg=Decimal("1000"),
            n2o_kg=Decimal("500"),
            iterations=500,
            seed=42,
        )
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# 3. TestAnalytical
# ===========================================================================


@_SKIP
class TestAnalytical:
    """Tests for analytical_propagation and combined_uncertainty."""

    def test_analytical_propagation_success(self, engine):
        """Analytical propagation returns SUCCESS."""
        result = engine.analytical_propagation(
            consumption=Decimal("5000"),
            consumption_uncertainty=Decimal("100"),
            ef=Decimal("0.5"),
            ef_uncertainty=Decimal("0.05"),
            td_loss=Decimal("0.05"),
            td_uncertainty=Decimal("0.01"),
        )
        assert result["status"] == "SUCCESS"
        assert result["method"] == "ANALYTICAL_PROPAGATION"
        assert result["emissions_kg"] > Decimal("0")

    def test_analytical_ci_bounds(self, engine):
        """CI lower < emissions < CI upper."""
        result = engine.analytical_propagation(
            consumption=Decimal("10000"),
            consumption_uncertainty=Decimal("500"),
            ef=Decimal("0.4"),
            ef_uncertainty=Decimal("0.04"),
        )
        assert result["ci_lower_95"] < result["emissions_kg"]
        assert result["ci_upper_95"] > result["emissions_kg"]

    def test_combined_uncertainty_rss(self, engine):
        """combined_uncertainty gives root-sum-of-squares."""
        result = engine.combined_uncertainty([
            Decimal("0.10"), Decimal("0.05"), Decimal("0.03"),
        ])
        expected = Decimal(str(round(math.sqrt(0.01 + 0.0025 + 0.0009), 8)))
        assert abs(float(result) - float(expected)) < 1e-6

    def test_combined_uncertainty_empty(self, engine):
        """Empty list returns zero."""
        result = engine.combined_uncertainty([])
        assert result == Decimal("0")

    def test_parameter_contributions_sum(self, engine):
        """Parameter contributions should approximately sum to 1."""
        result = engine.analytical_propagation(
            consumption=Decimal("5000"),
            consumption_uncertainty=Decimal("250"),
            ef=Decimal("0.5"),
            ef_uncertainty=Decimal("0.05"),
            td_loss=Decimal("0.05"),
            td_uncertainty=Decimal("0.01"),
        )
        contributions = result["parameter_contributions"]
        total = sum(float(v) for v in contributions.values())
        assert abs(total - 1.0) < 0.01


# ===========================================================================
# 4. TestPropagation
# ===========================================================================


@_SKIP
class TestPropagation:
    """Tests for propagate_multiplication and propagate_addition."""

    def test_propagate_multiplication(self, engine):
        """Multiplication propagation returns correct product."""
        z, u_z = engine.propagate_multiplication(
            a=Decimal("1000"), u_a=Decimal("50"),
            b=Decimal("0.5"), u_b=Decimal("0.05"),
        )
        assert z == Decimal("500.000000")
        assert u_z > Decimal("0")

    def test_propagate_addition(self, engine):
        """Addition propagation returns correct sum and uncertainty."""
        z, sigma_z = engine.propagate_addition(
            values=[Decimal("100"), Decimal("200"), Decimal("300")],
            uncertainties=[Decimal("10"), Decimal("20"), Decimal("30")],
        )
        assert z == Decimal("600.000000")
        expected_sigma = math.sqrt(100 + 400 + 900)
        assert abs(float(sigma_z) - expected_sigma) < 0.01

    def test_propagate_addition_length_mismatch(self, engine):
        """Length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            engine.propagate_addition(
                values=[Decimal("100"), Decimal("200")],
                uncertainties=[Decimal("10")],
            )


# ===========================================================================
# 5. TestIPCCDefaults
# ===========================================================================


@_SKIP
class TestIPCCDefaults:
    """Tests for get_ipcc_default_uncertainties and get_data_quality_uncertainty."""

    def test_ipcc_defaults_structure(self, engine):
        """Default uncertainties dict has expected top-level keys."""
        defaults = engine.get_ipcc_default_uncertainties()
        assert "grid_ef" in defaults
        assert "activity_data" in defaults
        assert "td_loss" in defaults
        assert "gwp" in defaults
        assert "per_gas_ef" in defaults

    def test_ipcc_grid_ef_tiers(self, engine):
        """Grid EF has tier_1, tier_2, tier_3."""
        defaults = engine.get_ipcc_default_uncertainties()
        for tier in ("tier_1", "tier_2", "tier_3"):
            assert tier in defaults["grid_ef"]

    def test_data_quality_uncertainty_tier_1(self, engine):
        """tier_1 maps to 0.05 (5%)."""
        result = engine.get_data_quality_uncertainty("tier_1")
        assert result == Decimal("0.05")

    def test_data_quality_uncertainty_unknown(self, engine):
        """Unknown tier defaults to 0.30."""
        result = engine.get_data_quality_uncertainty("unknown_tier")
        assert result == Decimal("0.30")


# ===========================================================================
# 6. TestDQI
# ===========================================================================


@_SKIP
class TestDQI:
    """Tests for calculate_dqi_score."""

    def test_dqi_high_quality(self, engine):
        """High-quality data produces score > 0.8."""
        from datetime import datetime, timezone
        current_year = datetime.now(timezone.utc).year
        score = engine.calculate_dqi_score(
            ef_source="egrid",
            ef_year=current_year,
            activity_source="meter",
            temporal_representativeness=0,
        )
        assert float(score) > 0.8

    def test_dqi_low_quality(self, engine):
        """Low-quality data produces score < 0.5."""
        score = engine.calculate_dqi_score(
            ef_source="unknown",
            ef_year=2010,
            activity_source="unknown",
            temporal_representativeness=5,
        )
        assert float(score) < 0.5

    def test_dqi_score_range(self, engine):
        """DQI score is between 0 and 1."""
        from datetime import datetime, timezone
        current_year = datetime.now(timezone.utc).year
        score = engine.calculate_dqi_score(
            ef_source="iea",
            ef_year=current_year - 2,
            activity_source="invoice",
            temporal_representativeness=1,
        )
        assert Decimal("0") <= score <= Decimal("1")


# ===========================================================================
# 7. TestSensitivity
# ===========================================================================


@_SKIP
class TestSensitivity:
    """Tests for sensitivity_analysis."""

    def test_sensitivity_analysis_success(self, engine):
        """Sensitivity analysis returns SUCCESS."""
        result = engine.sensitivity_analysis(
            base_result=Decimal("100000"),
            parameters={
                "consumption": Decimal("5000"),
                "ef": Decimal("0.5"),
                "td_loss": Decimal("1.05"),
            },
            variation_pct=Decimal("0.10"),
        )
        assert result["status"] == "SUCCESS"
        assert result["method"] == "SENSITIVITY_ANALYSIS"
        assert "consumption" in result["parameters"]
        assert "ef" in result["parameters"]

    def test_sensitivity_coefficient(self, engine):
        """For multiplicative model, sensitivity coefficient should be ~1."""
        result = engine.sensitivity_analysis(
            base_result=Decimal("100000"),
            parameters={"consumption": Decimal("5000")},
            variation_pct=Decimal("0.10"),
        )
        coeff = float(result["parameters"]["consumption"]["sensitivity_coefficient"])
        assert abs(coeff - 1.0) < 0.01

    def test_sensitivity_provenance_hash(self, engine):
        """Sensitivity analysis includes provenance hash."""
        result = engine.sensitivity_analysis(
            base_result=Decimal("50000"),
            parameters={"ef": Decimal("0.5")},
        )
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# 8. TestStatisticalHelpers
# ===========================================================================


@_SKIP
class TestStatisticalHelpers:
    """Tests for calculate_percentiles, calculate_confidence_interval, calculate_statistics."""

    def test_calculate_percentiles(self, engine):
        """Percentile calculation returns expected median."""
        values = [Decimal(str(i)) for i in range(1, 101)]
        result = engine.calculate_percentiles(values, [50])
        assert abs(float(result["50"]) - 50.5) < 1.0

    def test_calculate_percentiles_empty(self, engine):
        """Empty list returns zeros."""
        result = engine.calculate_percentiles([], [25, 50, 75])
        for v in result.values():
            assert v == Decimal("0")

    def test_calculate_confidence_interval(self, engine):
        """CI lower < CI upper for non-trivial data."""
        values = [Decimal(str(i)) for i in range(1, 101)]
        lower, upper = engine.calculate_confidence_interval(values)
        assert lower < upper
        assert float(lower) >= 1.0
        assert float(upper) <= 100.0

    def test_calculate_confidence_interval_empty(self, engine):
        """Empty list returns (0, 0)."""
        lower, upper = engine.calculate_confidence_interval([])
        assert lower == Decimal("0")
        assert upper == Decimal("0")

    def test_calculate_statistics(self, engine):
        """Statistics dict has expected keys and reasonable values."""
        values = [Decimal(str(i)) for i in range(1, 51)]
        stats = engine.calculate_statistics(values)
        assert stats["count"] == 50
        assert float(stats["mean"]) == pytest.approx(25.5, rel=1e-4)
        assert float(stats["min"]) == 1.0
        assert float(stats["max"]) == 50.0
        assert float(stats["std_dev"]) > 0

    def test_calculate_statistics_empty(self, engine):
        """Empty list returns zeros."""
        stats = engine.calculate_statistics([])
        assert stats["count"] == 0
        assert stats["mean"] == Decimal("0")
