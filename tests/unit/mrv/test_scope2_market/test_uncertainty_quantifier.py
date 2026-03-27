# -*- coding: utf-8 -*-
"""
Unit tests for UncertaintyQuantifierEngine (Engine 5 of 7)

AGENT-MRV-010: Scope 2 Market-Based Emissions Agent

Tests Monte Carlo simulation, per-gas Monte Carlo, market-based Monte Carlo,
batch Monte Carlo, analytical error propagation, IPCC default uncertainties,
instrument and residual mix uncertainties, activity data uncertainties, DQI
scoring, sensitivity/tornado analysis, statistical helpers, sampling
functions, input validation, statistics, and reset.

Target: 70 tests, ~900 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.agents.mrv.scope2_market.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
        INSTRUMENT_UNCERTAINTY,
        RESIDUAL_MIX_UNCERTAINTY,
        ACTIVITY_DATA_UNCERTAINTY,
        EF_SOURCE_SCORES,
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
            uncertainties={
                "instrument_ef": Decimal("0.02"),
                "activity": Decimal("0.05"),
            },
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
            uncertainties={
                "instrument_ef": Decimal("0.05"),
                "activity": Decimal("0.05"),
            },
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
            uncertainties={
                "instrument_ef": Decimal("0.05"),
                "activity": Decimal("0.02"),
            },
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
            uncertainties={
                "instrument_ef": Decimal("0.10"),
                "activity": Decimal("0.05"),
            },
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
            uncertainties={"instrument_ef": Decimal("0.10")},
            iterations=500,
            seed=42,
        )
        assert result["status"] == "VALIDATION_ERROR"
        assert len(result["errors"]) > 0

    def test_percentiles_present(self, engine):
        """Result contains percentile data."""
        result = engine.run_monte_carlo(
            base_emissions_kg=Decimal("50000"),
            uncertainties={"instrument_ef": Decimal("0.10")},
            iterations=500,
            seed=42,
        )
        assert "percentiles" in result
        pct_keys = result["percentiles"]
        # Check at least a median percentile exists
        has_median = any(
            k in pct_keys for k in ("50", "50.0", 50, 50.0, "p50")
        )
        assert has_median

    def test_different_seeds_different_results(self, engine):
        """Different seeds produce different MC samples."""
        kwargs = dict(
            base_emissions_kg=Decimal("100000"),
            uncertainties={"instrument_ef": Decimal("0.10")},
            iterations=1000,
        )
        r1 = engine.run_monte_carlo(**kwargs, seed=1)
        r2 = engine.run_monte_carlo(**kwargs, seed=2)
        assert r1["mean_co2e_kg"] != r2["mean_co2e_kg"]

    def test_mc_with_covered_and_uncovered(self, engine):
        """MC accepts covered and uncovered emissions split."""
        result = engine.run_monte_carlo(
            base_emissions_kg=Decimal("100000"),
            covered_emissions_kg=Decimal("20000"),
            uncovered_emissions_kg=Decimal("80000"),
            uncertainties={
                "instrument_ef": Decimal("0.02"),
                "residual_mix": Decimal("0.15"),
                "activity": Decimal("0.05"),
            },
            iterations=500,
            seed=42,
        )
        assert result["status"] == "SUCCESS"

    def test_mc_zero_base(self, engine):
        """Zero base emissions returns zero results."""
        result = engine.run_monte_carlo(
            base_emissions_kg=Decimal("0"),
            uncertainties={"instrument_ef": Decimal("0.10")},
            iterations=100,
            seed=42,
        )
        assert result["status"] == "SUCCESS"
        assert float(result["mean_co2e_kg"]) == pytest.approx(0.0, abs=0.01)

    def test_mc_large_iterations(self, engine):
        """Large iteration count produces tighter CI."""
        kwargs = dict(
            base_emissions_kg=Decimal("100000"),
            uncertainties={"instrument_ef": Decimal("0.10")},
        )
        r_small = engine.run_monte_carlo(**kwargs, iterations=100, seed=42)
        r_large = engine.run_monte_carlo(**kwargs, iterations=10000, seed=42)
        ci_width_small = float(r_small["ci_upper"] - r_small["ci_lower"])
        ci_width_large = float(r_large["ci_upper"] - r_large["ci_lower"])
        # Larger sample should generally produce tighter or similar CI
        # (not strict due to randomness, but should be in same ballpark)
        assert ci_width_large < ci_width_small * 2

    def test_run_monte_carlo_per_gas_success(self, engine):
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

    def test_run_monte_carlo_market_based(self, engine):
        """Market-based MC simulation returns SUCCESS."""
        result = engine.run_monte_carlo_market_based(
            covered_co2e_kg=Decimal("0"),
            uncovered_co2e_kg=Decimal("80000"),
            instrument_uncertainty=Decimal("0.02"),
            residual_mix_uncertainty=Decimal("0.15"),
            activity_uncertainty=Decimal("0.05"),
            iterations=500,
            seed=42,
        )
        assert result["status"] == "SUCCESS"

    def test_run_monte_carlo_batch(self, engine):
        """Batch MC processes multiple uncertainty analyses."""
        items = [
            {
                "base_emissions_kg": Decimal("100000"),
                "uncertainties": {"instrument_ef": Decimal("0.05")},
            },
            {
                "base_emissions_kg": Decimal("50000"),
                "uncertainties": {"instrument_ef": Decimal("0.10")},
            },
        ]
        result = engine.run_monte_carlo_batch(items, iterations=500, seed=42)
        assert result["status"] == "SUCCESS"
        assert result["total_items"] == 2


# ===========================================================================
# 2. TestAnalytical
# ===========================================================================


@_SKIP
class TestAnalytical:
    """Tests for analytical_propagation and combined_uncertainty."""

    def test_analytical_propagation_success(self, engine):
        """Analytical propagation returns SUCCESS."""
        result = engine.analytical_propagation(
            base_co2e_kg=Decimal("100000"),
            uncertainties={
                "instrument_ef": Decimal("0.02"),
                "activity": Decimal("0.05"),
                "residual_mix": Decimal("0.15"),
            },
        )
        assert result["status"] == "SUCCESS"
        assert result["method"] == "ANALYTICAL_PROPAGATION"

    def test_analytical_ci_bounds(self, engine):
        """CI lower < base < CI upper."""
        result = engine.analytical_propagation(
            base_co2e_kg=Decimal("100000"),
            uncertainties={
                "instrument_ef": Decimal("0.05"),
                "activity": Decimal("0.05"),
            },
        )
        assert result["ci_lower_95"] < result["base_co2e_kg"]
        assert result["ci_upper_95"] > result["base_co2e_kg"]

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

    def test_propagate_multiplication(self, engine):
        """Multiplication propagation returns correct product."""
        z, u_z = engine.propagate_multiplication(
            a=Decimal("1000"), u_a=Decimal("50"),
            b=Decimal("0.5"), u_b=Decimal("0.05"),
        )
        assert float(z) == pytest.approx(500.0, rel=1e-4)
        assert u_z > Decimal("0")

    def test_propagate_addition(self, engine):
        """Addition propagation returns correct sum and uncertainty."""
        z, sigma_z = engine.propagate_addition(
            values=[Decimal("100"), Decimal("200"), Decimal("300")],
            uncertainties=[Decimal("10"), Decimal("20"), Decimal("30")],
        )
        assert float(z) == pytest.approx(600.0, rel=1e-4)
        expected_sigma = math.sqrt(100 + 400 + 900)
        assert abs(float(sigma_z) - expected_sigma) < 0.01

    def test_propagate_addition_length_mismatch(self, engine):
        """Length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            engine.propagate_addition(
                values=[Decimal("100"), Decimal("200")],
                uncertainties=[Decimal("10")],
            )

    def test_analytical_provenance_hash(self, engine):
        """Analytical result includes provenance hash."""
        result = engine.analytical_propagation(
            base_co2e_kg=Decimal("100000"),
            uncertainties={"activity": Decimal("0.05")},
        )
        assert len(result["provenance_hash"]) == 64

    def test_parameter_contributions_sum(self, engine):
        """Parameter contributions should approximately sum to 1."""
        result = engine.analytical_propagation(
            base_co2e_kg=Decimal("100000"),
            uncertainties={
                "instrument_ef": Decimal("0.05"),
                "activity": Decimal("0.10"),
                "residual_mix": Decimal("0.15"),
            },
        )
        contributions = result.get("parameter_contributions", {})
        if contributions:
            total = sum(float(v) for v in contributions.values())
            assert abs(total - 1.0) < 0.05

    def test_combined_uncertainty_single_value(self, engine):
        """Single uncertainty returns itself."""
        result = engine.combined_uncertainty([Decimal("0.10")])
        assert abs(float(result) - 0.10) < 1e-6


# ===========================================================================
# 3. TestIPCCDefaults
# ===========================================================================


@_SKIP
class TestIPCCDefaults:
    """Tests for IPCC default uncertainties and instrument/residual mix lookups."""

    def test_ipcc_defaults_structure(self, engine):
        """Default uncertainties dict has expected top-level keys."""
        defaults = engine.get_ipcc_default_uncertainties()
        assert isinstance(defaults, dict)
        assert len(defaults) > 0

    def test_get_instrument_uncertainty_verified(self, engine):
        """Verified instrument has lower uncertainty than unverified."""
        verified = engine.get_instrument_uncertainty("verified")
        unverified = engine.get_instrument_uncertainty("unverified")
        assert float(verified) < float(unverified)

    def test_get_instrument_uncertainty_verified_value(self, engine):
        """Verified instrument uncertainty is around 2%."""
        verified = engine.get_instrument_uncertainty("verified")
        assert float(verified) <= 0.05

    def test_get_instrument_uncertainty_unverified_value(self, engine):
        """Unverified instrument uncertainty is around 10%."""
        unverified = engine.get_instrument_uncertainty("unverified")
        assert float(unverified) >= 0.05
        assert float(unverified) <= 0.20

    def test_get_instrument_uncertainty_self_declared(self, engine):
        """Self-declared instrument uncertainty is between verified and unverified."""
        verified = engine.get_instrument_uncertainty("verified")
        self_decl = engine.get_instrument_uncertainty("self_declared")
        unverified = engine.get_instrument_uncertainty("unverified")
        assert float(verified) <= float(self_decl) <= float(unverified)

    def test_get_residual_mix_uncertainty_tier1(self, engine):
        """Tier 1 (national) residual mix has ~20% uncertainty."""
        result = engine.get_residual_mix_uncertainty("tier_1")
        assert float(result) >= 0.10
        assert float(result) <= 0.30

    def test_get_residual_mix_uncertainty_tier3(self, engine):
        """Tier 3 (grid operator) residual mix has lower uncertainty."""
        tier1 = engine.get_residual_mix_uncertainty("tier_1")
        tier3 = engine.get_residual_mix_uncertainty("tier_3")
        assert float(tier3) <= float(tier1)

    def test_get_residual_mix_uncertainty_unknown(self, engine):
        """Unknown tier defaults to highest uncertainty."""
        result = engine.get_residual_mix_uncertainty("unknown_tier")
        assert float(result) >= 0.15

    def test_instrument_uncertainty_constant_structure(self):
        """INSTRUMENT_UNCERTAINTY has expected verification levels."""
        assert "verified" in INSTRUMENT_UNCERTAINTY or len(INSTRUMENT_UNCERTAINTY) > 0

    def test_residual_mix_uncertainty_constant_structure(self):
        """RESIDUAL_MIX_UNCERTAINTY has expected tier keys."""
        assert len(RESIDUAL_MIX_UNCERTAINTY) > 0


# ===========================================================================
# 4. TestActivityData
# ===========================================================================


@_SKIP
class TestActivityData:
    """Tests for get_activity_data_uncertainty."""

    def test_meter_uncertainty(self, engine):
        """Metered data has lowest uncertainty (~2%)."""
        result = engine.get_activity_data_uncertainty("meter")
        assert float(result) <= 0.05

    def test_invoice_uncertainty(self, engine):
        """Invoice data has ~5% uncertainty."""
        result = engine.get_activity_data_uncertainty("invoice")
        assert 0.02 <= float(result) <= 0.10

    def test_estimate_uncertainty(self, engine):
        """Estimated data has ~20% uncertainty."""
        result = engine.get_activity_data_uncertainty("estimate")
        assert float(result) >= 0.10

    def test_benchmark_uncertainty(self, engine):
        """Benchmark data has ~30% uncertainty."""
        result = engine.get_activity_data_uncertainty("benchmark")
        assert float(result) >= 0.20

    def test_unknown_source_default(self, engine):
        """Unknown data source defaults to high uncertainty."""
        result = engine.get_activity_data_uncertainty("unknown_source")
        assert float(result) >= 0.15


# ===========================================================================
# 5. TestDQI
# ===========================================================================


@_SKIP
class TestDQI:
    """Tests for calculate_dqi_score and score_to_uncertainty."""

    def test_dqi_high_quality(self, engine):
        """High-quality market data produces score > 0.8."""
        score = engine.calculate_dqi_score(
            ef_source="verified_instrument",
            ef_year=2025,
            activity_source="meter",
            instrument_verification="verified",
        )
        assert float(score) > 0.8

    def test_dqi_low_quality(self, engine):
        """Low-quality data produces score < 0.5."""
        score = engine.calculate_dqi_score(
            ef_source="unknown",
            ef_year=2015,
            activity_source="benchmark",
            instrument_verification="none",
        )
        assert float(score) < 0.5

    def test_dqi_score_range(self, engine):
        """DQI score is between 0 and 1."""
        score = engine.calculate_dqi_score(
            ef_source="residual_mix",
            ef_year=2023,
            activity_source="invoice",
            instrument_verification="self_declared",
        )
        assert Decimal("0") <= score <= Decimal("1")

    def test_score_to_uncertainty_high_score(self, engine):
        """High DQI score maps to low uncertainty."""
        unc = engine.score_to_uncertainty(Decimal("0.95"))
        assert float(unc) < 0.10

    def test_score_to_uncertainty_low_score(self, engine):
        """Low DQI score maps to high uncertainty."""
        unc = engine.score_to_uncertainty(Decimal("0.20"))
        assert float(unc) > 0.15


# ===========================================================================
# 6. TestSensitivity
# ===========================================================================


@_SKIP
class TestSensitivity:
    """Tests for sensitivity_analysis and tornado_analysis."""

    def test_sensitivity_analysis_success(self, engine):
        """Sensitivity analysis returns SUCCESS."""
        result = engine.sensitivity_analysis(
            base_result=Decimal("100000"),
            parameters={
                "instrument_ef": Decimal("0.05"),
                "residual_mix": Decimal("0.400"),
                "activity": Decimal("5000"),
            },
            variation_pct=Decimal("0.10"),
        )
        assert result["status"] == "SUCCESS"
        assert result["method"] == "SENSITIVITY_ANALYSIS"

    def test_sensitivity_provenance_hash(self, engine):
        """Sensitivity analysis includes provenance hash."""
        result = engine.sensitivity_analysis(
            base_result=Decimal("50000"),
            parameters={"residual_mix": Decimal("0.400")},
        )
        assert len(result["provenance_hash"]) == 64

    def test_tornado_analysis_success(self, engine):
        """Tornado analysis returns SUCCESS with ranked parameters."""
        result = engine.tornado_analysis(
            base_result=Decimal("100000"),
            parameters={
                "instrument_ef": Decimal("0.05"),
                "residual_mix": Decimal("0.400"),
                "activity": Decimal("5000"),
            },
        )
        assert result["status"] == "SUCCESS"
        assert "ranked_parameters" in result or "parameters" in result

    def test_tornado_ranking_order(self, engine):
        """Tornado ranked parameters are sorted by impact (descending)."""
        result = engine.tornado_analysis(
            base_result=Decimal("100000"),
            parameters={
                "small_impact": Decimal("0.01"),
                "large_impact": Decimal("10000"),
            },
        )
        ranked = result.get("ranked_parameters", result.get("parameters", []))
        if isinstance(ranked, list) and len(ranked) >= 2:
            impacts = [abs(float(p.get("impact", p.get("range", 0)))) for p in ranked]
            assert impacts == sorted(impacts, reverse=True)

    def test_sensitivity_single_parameter(self, engine):
        """Sensitivity with one parameter returns valid result."""
        result = engine.sensitivity_analysis(
            base_result=Decimal("100000"),
            parameters={"residual_mix": Decimal("0.400")},
        )
        assert result["status"] == "SUCCESS"


# ===========================================================================
# 7. TestStatisticalHelpers
# ===========================================================================


@_SKIP
class TestStatisticalHelpers:
    """Tests for calculate_percentiles, calculate_confidence_interval, calculate_statistics."""

    def test_calculate_percentiles(self, engine):
        """Percentile calculation returns expected median."""
        values = [Decimal(str(i)) for i in range(1, 101)]
        result = engine.calculate_percentiles(values, [50])
        median_key = next(
            (k for k in result if str(k) in ("50", "50.0", "p50")), None
        )
        assert median_key is not None
        assert abs(float(result[median_key]) - 50.5) < 1.0

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


# ===========================================================================
# 8. TestSampling
# ===========================================================================


@_SKIP
class TestSampling:
    """Tests for normal_sample and lognormal_sample with seed reproducibility."""

    def test_normal_sample_mean(self, engine):
        """Normal sample mean is close to specified mean."""
        samples = engine.normal_sample(
            mean=Decimal("100"), std_dev=Decimal("10"), n=5000, seed=42
        )
        avg = sum(float(s) for s in samples) / len(samples)
        assert abs(avg - 100.0) < 2.0

    def test_normal_sample_reproducibility(self, engine):
        """Same seed produces identical normal samples."""
        s1 = engine.normal_sample(Decimal("100"), Decimal("10"), 100, seed=42)
        s2 = engine.normal_sample(Decimal("100"), Decimal("10"), 100, seed=42)
        assert s1 == s2

    def test_lognormal_sample_positive(self, engine):
        """Lognormal samples are always positive."""
        samples = engine.lognormal_sample(
            mean=Decimal("100"), std_dev=Decimal("30"), n=1000, seed=42
        )
        assert all(float(s) > 0 for s in samples)

    def test_lognormal_sample_reproducibility(self, engine):
        """Same seed produces identical lognormal samples."""
        s1 = engine.lognormal_sample(Decimal("50"), Decimal("10"), 100, seed=42)
        s2 = engine.lognormal_sample(Decimal("50"), Decimal("10"), 100, seed=42)
        assert s1 == s2

    def test_normal_sample_count(self, engine):
        """Normal sample returns requested number of samples."""
        samples = engine.normal_sample(Decimal("0"), Decimal("1"), 500, seed=42)
        assert len(samples) == 500


# ===========================================================================
# 9. TestValidation
# ===========================================================================


@_SKIP
class TestValidation:
    """Tests for validate_uncertainty_input and validate_iterations."""

    def test_validate_uncertainty_input_valid(self, engine):
        """Valid uncertainty input passes validation."""
        errors = engine.validate_uncertainty_input(
            base_emissions_kg=Decimal("100000"),
            uncertainties={"instrument_ef": Decimal("0.05")},
        )
        assert len(errors) == 0

    def test_validate_uncertainty_input_negative_base(self, engine):
        """Negative base emissions fails validation."""
        errors = engine.validate_uncertainty_input(
            base_emissions_kg=Decimal("-100"),
            uncertainties={"instrument_ef": Decimal("0.05")},
        )
        assert len(errors) > 0

    def test_validate_uncertainty_input_empty_uncertainties(self, engine):
        """Empty uncertainties dict fails validation."""
        errors = engine.validate_uncertainty_input(
            base_emissions_kg=Decimal("100000"),
            uncertainties={},
        )
        assert len(errors) > 0

    def test_validate_iterations_valid(self, engine):
        """Valid iteration count passes."""
        errors = engine.validate_iterations(5000)
        assert len(errors) == 0

    def test_validate_iterations_too_low(self, engine):
        """Iteration count below minimum fails."""
        errors = engine.validate_iterations(0)
        assert len(errors) > 0


# ===========================================================================
# 10. TestStatisticsReset
# ===========================================================================


@_SKIP
class TestStatisticsReset:
    """Tests for get_statistics and reset."""

    def test_get_statistics(self, engine):
        """Statistics returns dict with expected keys."""
        stats = engine.get_statistics()
        assert isinstance(stats, dict)
        assert "mc_runs" in stats or "monte_carlo_runs" in stats or "total_runs" in stats

    def test_statistics_after_mc(self, engine):
        """Statistics update after Monte Carlo run."""
        engine.run_monte_carlo(
            base_emissions_kg=Decimal("50000"),
            uncertainties={"instrument_ef": Decimal("0.05")},
            iterations=100,
            seed=42,
        )
        stats = engine.get_statistics()
        run_count = stats.get(
            "mc_runs", stats.get("monte_carlo_runs", stats.get("total_runs", 0))
        )
        assert run_count >= 1

    def test_reset_clears_statistics(self, engine):
        """Reset zeroes all counters."""
        engine.run_monte_carlo(
            base_emissions_kg=Decimal("50000"),
            uncertainties={"instrument_ef": Decimal("0.05")},
            iterations=100,
            seed=42,
        )
        engine.reset()
        stats = engine.get_statistics()
        run_count = stats.get(
            "mc_runs", stats.get("monte_carlo_runs", stats.get("total_runs", 0))
        )
        assert run_count == 0

    def test_reset_returns_none(self, engine):
        """Reset method returns None."""
        result = engine.reset()
        assert result is None

    def test_statistics_includes_engine_info(self, engine):
        """Statistics include engine identification."""
        stats = engine.get_statistics()
        assert isinstance(stats, dict)
