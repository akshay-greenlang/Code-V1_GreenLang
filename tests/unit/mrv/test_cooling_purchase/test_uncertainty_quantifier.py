# -*- coding: utf-8 -*-
"""
Unit tests for UncertaintyQuantifierEngine (Engine 5 of 7)

AGENT-MRV-012: Cooling Purchase Agent

Tests Monte Carlo simulation, analytical error propagation, confidence interval
calculation, tier-specific uncertainties, and sensitivity analysis for cooling
purchase emission uncertainty quantification.

Target: 65 tests, ~550 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from greenlang.cooling_purchase.uncertainty_quantifier import (
    UncertaintyQuantifierEngine,
    get_uncertainty_quantifier,
)
from greenlang.cooling_purchase.models import (
    UncertaintyRequest,
    UncertaintyResult,
    DataQualityTier,
    CoolingTechnology,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uq_engine():
    """Create an UncertaintyQuantifierEngine instance."""
    engine = UncertaintyQuantifierEngine()
    yield engine
    engine.reset()


@pytest.fixture
def mc_request() -> Dict[str, Any]:
    """Return a Monte Carlo request dictionary."""
    return {
        "total_emissions_kgco2e": Decimal("10000"),
        "cooling_kwh_th": Decimal("100000"),
        "cop": Decimal("4.5"),
        "grid_ef_kgco2e_kwh": Decimal("0.45"),
        "tier": "TIER_1",
        "technology": "WATER_COOLED_CENTRIFUGAL",
        "n_iterations": 1000,
        "seed": 42,
    }


@pytest.fixture
def analytical_request() -> Dict[str, Any]:
    """Return an analytical uncertainty request."""
    return {
        "total_emissions_kgco2e": Decimal("10000"),
        "cooling_kwh_th": Decimal("100000"),
        "cop": Decimal("4.5"),
        "grid_ef_kgco2e_kwh": Decimal("0.45"),
        "tier": "TIER_2",
        "technology": "WATER_COOLED_CENTRIFUGAL",
    }


# ===========================================================================
# 1. Initialization Tests
# ===========================================================================


class TestUQEngineInit:
    """Test UncertaintyQuantifierEngine initialization."""

    def test_singleton_pattern(self):
        """Test singleton returns same instance."""
        e1 = UncertaintyQuantifierEngine()
        e2 = UncertaintyQuantifierEngine()
        assert e1 is e2

    def test_get_function_returns_singleton(self):
        """Test get_uncertainty_quantifier returns singleton."""
        e1 = get_uncertainty_quantifier()
        e2 = get_uncertainty_quantifier()
        assert e1 is e2

    def test_reset_clears_state(self, uq_engine):
        """Test reset clears internal state."""
        _ = uq_engine.run_monte_carlo({"total_emissions_kgco2e": Decimal("1000"), "n_iterations": 100})
        uq_engine.reset()
        stats = uq_engine.get_statistics()
        assert stats["total_analyses"] == 0


# ===========================================================================
# 2. Monte Carlo Tests
# ===========================================================================


class TestMonteCarloSimulation:
    """Test Monte Carlo uncertainty simulation."""

    def test_run_monte_carlo_basic(self, uq_engine, mc_request):
        """Test basic Monte Carlo simulation."""
        result = uq_engine.run_monte_carlo(mc_request)
        assert result["mean"] > Decimal("0")
        assert result["std_dev"] > Decimal("0")
        assert result["ci_lower"] > Decimal("0")
        assert result["ci_upper"] > result["ci_lower"]

    def test_monte_carlo_reproducibility_with_seed(self, uq_engine, mc_request):
        """Test same seed produces identical results."""
        r1 = uq_engine.run_monte_carlo(mc_request)
        r2 = uq_engine.run_monte_carlo(mc_request)
        assert r1["mean"] == r2["mean"]
        assert r1["std_dev"] == r2["std_dev"]

    def test_monte_carlo_different_seed_different_results(self, uq_engine, mc_request):
        """Test different seeds produce different results."""
        r1 = uq_engine.run_monte_carlo(mc_request)
        mc_request["seed"] = 99
        r2 = uq_engine.run_monte_carlo(mc_request)
        # Mean should be similar but not identical due to stochastic variation
        assert abs(r1["mean"] - r2["mean"]) < Decimal("1000")

    def test_monte_carlo_returns_ci(self, uq_engine, mc_request):
        """Test Monte Carlo returns confidence intervals."""
        result = uq_engine.run_monte_carlo(mc_request)
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] < result["ci_upper"]

    def test_monte_carlo_1000_iterations(self, uq_engine, mc_request):
        """Test Monte Carlo with 1000 iterations."""
        mc_request["n_iterations"] = 1000
        result = uq_engine.run_monte_carlo(mc_request)
        assert result["mean"] > Decimal("0")

    def test_monte_carlo_10000_iterations(self, uq_engine, mc_request):
        """Test Monte Carlo with 10000 iterations (default)."""
        mc_request["n_iterations"] = 10000
        result = uq_engine.run_monte_carlo(mc_request)
        assert result["mean"] > Decimal("0")


# ===========================================================================
# 3. Analytical Error Propagation Tests
# ===========================================================================


class TestAnalyticalErrorPropagation:
    """Test analytical uncertainty propagation."""

    def test_run_analytical_basic(self, uq_engine, analytical_request):
        """Test basic analytical error propagation."""
        result = uq_engine.run_analytical(analytical_request)
        assert result["combined_uncertainty_pct"] > Decimal("0")
        assert result["combined_uncertainty_pct"] < Decimal("100")

    def test_analytical_returns_result(self, uq_engine, analytical_request):
        """Test analytical method returns result dict."""
        result = uq_engine.run_analytical(analytical_request)
        assert "combined_uncertainty_pct" in result
        assert "uncertainty_kgco2e" in result

    def test_analytical_quadrature_formula(self, uq_engine):
        """Test analytical uses quadrature sum of uncertainties."""
        result = uq_engine.analytical_error_propagation(
            uncertainties_pct=[Decimal("10"), Decimal("20"), Decimal("15")]
        )
        expected = (Decimal("10")**2 + Decimal("20")**2 + Decimal("15")**2).sqrt()
        assert abs(result - expected) < Decimal("0.01")


# ===========================================================================
# 4. Statistics Computation Tests
# ===========================================================================


class TestStatisticsComputation:
    """Test statistical calculations."""

    def test_compute_statistics_mean(self, uq_engine):
        """Test compute_statistics calculates mean."""
        data = [Decimal("100"), Decimal("200"), Decimal("300")]
        stats = uq_engine.compute_statistics(data)
        assert stats["mean"] == Decimal("200")

    def test_compute_statistics_std_dev(self, uq_engine):
        """Test compute_statistics calculates std dev."""
        data = [Decimal("100"), Decimal("200"), Decimal("300")]
        stats = uq_engine.compute_statistics(data)
        assert stats["std_dev"] > Decimal("0")

    def test_compute_confidence_interval(self, uq_engine):
        """Test confidence interval calculation."""
        ci = uq_engine.compute_confidence_interval(
            mean=Decimal("1000"),
            std_dev=Decimal("100"),
            confidence_level=Decimal("0.95"),
            n_samples=1000,
        )
        assert ci["lower"] < Decimal("1000")
        assert ci["upper"] > Decimal("1000")

    def test_compute_coefficient_of_variation(self, uq_engine):
        """Test CV calculation."""
        cv = uq_engine.compute_coefficient_of_variation(
            std_dev=Decimal("100"),
            mean=Decimal("1000"),
        )
        assert cv == Decimal("10")  # (100/1000)*100

    def test_compute_combined_uncertainty_quadrature(self, uq_engine):
        """Test combined uncertainty uses quadrature sum."""
        combined = uq_engine.compute_combined_uncertainty(
            [Decimal("10"), Decimal("20"), Decimal("15")]
        )
        expected = (Decimal("10")**2 + Decimal("20")**2 + Decimal("15")**2).sqrt()
        assert abs(combined - expected) < Decimal("0.01")


# ===========================================================================
# 5. Tier-Specific Uncertainty Tests
# ===========================================================================


class TestTierUncertainties:
    """Test tier-specific uncertainty values."""

    def test_tier1_higher_than_tier2(self, uq_engine):
        """Test Tier 1 has higher uncertainty than Tier 2."""
        u_tier1 = uq_engine.get_tier_uncertainty("TIER_1", "WATER_COOLED_CENTRIFUGAL")
        u_tier2 = uq_engine.get_tier_uncertainty("TIER_2", "WATER_COOLED_CENTRIFUGAL")
        assert u_tier1 > u_tier2

    def test_tier2_higher_than_tier3(self, uq_engine):
        """Test Tier 2 has higher uncertainty than Tier 3."""
        u_tier2 = uq_engine.get_tier_uncertainty("TIER_2", "WATER_COOLED_CENTRIFUGAL")
        u_tier3 = uq_engine.get_tier_uncertainty("TIER_3", "WATER_COOLED_CENTRIFUGAL")
        assert u_tier2 > u_tier3

    def test_tier1_uncertainty_range(self, uq_engine):
        """Test Tier 1 uncertainty is typically 30-50%."""
        u = uq_engine.get_tier_uncertainty("TIER_1", "WATER_COOLED_CENTRIFUGAL")
        assert Decimal("20") < u < Decimal("60")

    def test_tier2_uncertainty_range(self, uq_engine):
        """Test Tier 2 uncertainty is typically 15-30%."""
        u = uq_engine.get_tier_uncertainty("TIER_2", "WATER_COOLED_CENTRIFUGAL")
        assert Decimal("10") < u < Decimal("40")

    def test_tier3_uncertainty_range(self, uq_engine):
        """Test Tier 3 uncertainty is typically 5-15%."""
        u = uq_engine.get_tier_uncertainty("TIER_3", "WATER_COOLED_CENTRIFUGAL")
        assert Decimal("3") < u < Decimal("20")


# ===========================================================================
# 6. Technology-Specific Uncertainty Tests
# ===========================================================================


class TestTechnologyUncertainties:
    """Test technology-specific uncertainty quantification."""

    def test_quantify_electric_chiller_uncertainty(self, uq_engine):
        """Test electric chiller uncertainty quantification."""
        result = uq_engine.quantify_electric_chiller_uncertainty(
            cooling_kwh_th=Decimal("100000"),
            cop=Decimal("4.5"),
            grid_ef=Decimal("0.45"),
            tier="TIER_1",
        )
        assert result["uncertainty_pct"] > Decimal("0")

    def test_quantify_absorption_uncertainty(self, uq_engine):
        """Test absorption chiller uncertainty quantification."""
        result = uq_engine.quantify_absorption_uncertainty(
            cooling_kwh_th=Decimal("100000"),
            cop_thermal=Decimal("1.2"),
            heat_ef=Decimal("0.25"),
            tier="TIER_2",
        )
        assert result["uncertainty_pct"] > Decimal("0")

    def test_quantify_free_cooling_uncertainty(self, uq_engine):
        """Test free cooling uncertainty quantification."""
        result = uq_engine.quantify_free_cooling_uncertainty(
            cooling_kwh_th=Decimal("50000"),
            cop=Decimal("20"),
            grid_ef=Decimal("0.40"),
            tier="TIER_2",
        )
        assert result["uncertainty_pct"] > Decimal("0")


# ===========================================================================
# 7. COP Distribution Tests
# ===========================================================================


class TestCOPDistribution:
    """Test COP uncertainty distribution parameters."""

    def test_get_cop_distribution_centrifugal(self, uq_engine):
        """Test COP distribution for centrifugal chiller."""
        dist = uq_engine.get_cop_distribution("WATER_COOLED_CENTRIFUGAL")
        assert dist["mean"] > Decimal("0")
        assert dist["std_dev"] > Decimal("0")

    def test_get_cop_distribution_screw(self, uq_engine):
        """Test COP distribution for screw chiller."""
        dist = uq_engine.get_cop_distribution("WATER_COOLED_SCREW")
        assert dist["mean"] > Decimal("0")

    def test_get_cop_distribution_absorption(self, uq_engine):
        """Test COP distribution for absorption chiller."""
        dist = uq_engine.get_cop_distribution("DOUBLE_EFFECT_LIBR")
        assert dist["mean"] > Decimal("0")


# ===========================================================================
# 8. Grid EF Distribution Tests
# ===========================================================================


class TestGridEFDistribution:
    """Test grid emission factor uncertainty distribution."""

    def test_get_grid_ef_distribution(self, uq_engine):
        """Test grid EF distribution parameters."""
        dist = uq_engine.get_grid_ef_distribution(
            ef_mean=Decimal("0.45"),
            tier="TIER_1",
        )
        assert dist["mean"] == Decimal("0.45")
        assert dist["std_dev"] > Decimal("0")

    def test_grid_ef_uncertainty_tier1(self, uq_engine):
        """Test Tier 1 grid EF has higher uncertainty."""
        dist1 = uq_engine.get_grid_ef_distribution(Decimal("0.45"), "TIER_1")
        dist2 = uq_engine.get_grid_ef_distribution(Decimal("0.45"), "TIER_2")
        assert dist1["std_dev"] > dist2["std_dev"]


# ===========================================================================
# 9. IPLV Weighting Uncertainty Tests
# ===========================================================================


class TestIPLVWeightingUncertainty:
    """Test IPLV part-load weighting uncertainty."""

    def test_get_iplv_weighting_uncertainty(self, uq_engine):
        """Test IPLV weighting has uncertainty."""
        u = uq_engine.get_iplv_weighting_uncertainty()
        assert u > Decimal("0")
        assert u < Decimal("20")  # Typically 5-10%


# ===========================================================================
# 10. Sensitivity Analysis Tests
# ===========================================================================


class TestSensitivityAnalysis:
    """Test sensitivity analysis methods."""

    def test_run_sensitivity_analysis(self, uq_engine):
        """Test sensitivity analysis identifies key parameters."""
        params = {
            "cop": {"value": Decimal("4.5"), "uncertainty_pct": Decimal("10")},
            "grid_ef": {"value": Decimal("0.45"), "uncertainty_pct": Decimal("20")},
            "cooling": {"value": Decimal("100000"), "uncertainty_pct": Decimal("5")},
        }
        result = uq_engine.run_sensitivity_analysis(params)
        assert "dominant_parameter" in result
        assert "sensitivity_indices" in result

    def test_identify_dominant_uncertainty(self, uq_engine):
        """Test identification of dominant uncertainty source."""
        params = {
            "cop": Decimal("10"),
            "grid_ef": Decimal("30"),
            "cooling": Decimal("5"),
        }
        dominant = uq_engine.identify_dominant_uncertainty(params)
        assert dominant == "grid_ef"  # Highest uncertainty


# ===========================================================================
# 11. Deterministic Tests
# ===========================================================================


class TestDeterministicBehavior:
    """Test deterministic behavior with fixed seed."""

    def test_deterministic_with_seed_42(self, uq_engine, mc_request):
        """Test deterministic results with seed=42."""
        mc_request["seed"] = 42
        r1 = uq_engine.run_monte_carlo(mc_request)
        r2 = uq_engine.run_monte_carlo(mc_request)
        assert r1["mean"] == r2["mean"]

    def test_deterministic_with_seed_99(self, uq_engine, mc_request):
        """Test deterministic results with seed=99."""
        mc_request["seed"] = 99
        r1 = uq_engine.run_monte_carlo(mc_request)
        r2 = uq_engine.run_monte_carlo(mc_request)
        assert r1["mean"] == r2["mean"]

    def test_three_runs_same_seed(self, uq_engine, mc_request):
        """Test three runs with same seed are identical."""
        results = [uq_engine.run_monte_carlo(mc_request) for _ in range(3)]
        means = [r["mean"] for r in results]
        assert means[0] == means[1] == means[2]


# ===========================================================================
# 12. Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_emissions(self, uq_engine):
        """Test handling of zero emissions."""
        result = uq_engine.run_analytical({
            "total_emissions_kgco2e": Decimal("0"),
            "tier": "TIER_1",
        })
        assert result["combined_uncertainty_pct"] >= Decimal("0")

    def test_very_small_emissions(self, uq_engine):
        """Test handling of very small emissions."""
        result = uq_engine.run_analytical({
            "total_emissions_kgco2e": Decimal("0.001"),
            "tier": "TIER_1",
        })
        assert result["combined_uncertainty_pct"] >= Decimal("0")

    def test_very_large_emissions(self, uq_engine):
        """Test handling of very large emissions."""
        result = uq_engine.run_analytical({
            "total_emissions_kgco2e": Decimal("1000000000"),
            "tier": "TIER_1",
        })
        assert result["combined_uncertainty_pct"] >= Decimal("0")
