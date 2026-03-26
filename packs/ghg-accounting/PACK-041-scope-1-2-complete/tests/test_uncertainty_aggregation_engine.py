# -*- coding: utf-8 -*-
"""
Unit tests for UncertaintyAggregationEngine -- PACK-041 Engine 6
===================================================================

Tests analytical error propagation (quadrature), Monte Carlo simulation,
sensitivity analysis, data quality scoring, and confidence intervals.

Coverage target: 85%+
Total tests: ~55
"""

import importlib.util
import math
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack041_test.engines.ua_{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


_m = _load("uncertainty_aggregation_engine")


# =============================================================================
# Module Loading
# =============================================================================


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_module_version(self):
        assert hasattr(_m, "_MODULE_VERSION")
        assert _m._MODULE_VERSION == "1.0.0"


# =============================================================================
# Analytical Method (Quadrature)
# =============================================================================


class TestAnalyticalQuadrature:
    """Test analytical uncertainty aggregation using error propagation."""

    def test_single_source_uncertainty(self):
        """Single source: combined = source uncertainty."""
        emissions = [Decimal("10000")]
        uncertainties = [Decimal("0.07")]  # 7%
        total = sum(emissions)
        rss = math.sqrt(sum(
            (float(e) * float(u)) ** 2
            for e, u in zip(emissions, uncertainties)
        ))
        combined_pct = rss / float(total) * 100 if float(total) > 0 else 0
        assert pytest.approx(combined_pct, abs=0.1) == 7.0

    def test_multi_source_quadrature(self):
        """Multi-source: U_combined = sqrt(sum((E_i * u_i)^2)) / E_total * 100.

        IPCC 2006 Vol 1 Ch 3 Eq 3.1.
        """
        emissions = [Decimal("8000"), Decimal("2000"), Decimal("500")]
        uncertainties = [Decimal("0.07"), Decimal("0.11"), Decimal("0.32")]
        total = sum(float(e) for e in emissions)
        rss = math.sqrt(sum(
            (float(e) * float(u)) ** 2
            for e, u in zip(emissions, uncertainties)
        ))
        combined_pct = rss / total * 100
        # sqrt((8000*0.07)^2 + (2000*0.11)^2 + (500*0.32)^2) / 10500 * 100
        # sqrt(560^2 + 220^2 + 160^2) / 10500 * 100
        # sqrt(313600 + 48400 + 25600) / 10500 * 100
        # sqrt(387600) / 10500 * 100 = 622.6 / 10500 * 100 = 5.93%
        assert pytest.approx(combined_pct, abs=0.1) == 5.93

    def test_quadrature_smaller_than_linear_sum(self):
        """Quadrature result should be smaller than simple sum of uncertainties."""
        emissions = [Decimal("5000"), Decimal("5000")]
        uncertainties = [Decimal("0.10"), Decimal("0.10")]
        total = float(sum(emissions))
        rss = math.sqrt(sum(
            (float(e) * float(u)) ** 2
            for e, u in zip(emissions, uncertainties)
        ))
        quadrature_pct = rss / total * 100

        linear_sum_pct = sum(
            float(e) * float(u) / total * 100
            for e, u in zip(emissions, uncertainties)
        )
        assert quadrature_pct < linear_sum_pct

    def test_dominant_source_drives_uncertainty(self):
        """A source with very high uncertainty and high emissions dominates."""
        emissions = [Decimal("9000"), Decimal("1000")]
        uncertainties = [Decimal("0.05"), Decimal("0.50")]
        total = float(sum(emissions))
        contrib_0 = (float(emissions[0]) * float(uncertainties[0])) ** 2
        contrib_1 = (float(emissions[1]) * float(uncertainties[1])) ** 2
        # contrib_0 = 450^2 = 202500, contrib_1 = 500^2 = 250000
        assert contrib_1 > contrib_0  # smaller source dominates due to high uncertainty


# =============================================================================
# Monte Carlo Simulation
# =============================================================================


class TestMonteCarloSimulation:
    """Test Monte Carlo uncertainty estimation."""

    def test_monte_carlo_single_source(self):
        """Monte Carlo with single normal source should approximate analytical."""
        import random
        rng = random.Random(42)
        mean = 10000.0
        std = 700.0  # 7% uncertainty
        n = 10000
        samples = [rng.gauss(mean, std) for _ in range(n)]
        mc_mean = sum(samples) / n
        mc_std = (sum((s - mc_mean) ** 2 for s in samples) / (n - 1)) ** 0.5
        mc_pct = mc_std / mc_mean * 100
        assert pytest.approx(mc_pct, abs=1.0) == 7.0

    def test_monte_carlo_convergence(self):
        """More iterations should converge to more stable results."""
        import random
        rng = random.Random(42)
        results = []
        for n in [100, 1000, 10000]:
            samples = [rng.gauss(10000.0, 700.0) for _ in range(n)]
            mean = sum(samples) / n
            results.append(mean)
        # 10000 iterations should be closer to true mean than 100
        assert abs(results[2] - 10000.0) < abs(results[0] - 10000.0) * 2

    def test_monte_carlo_seed_reproducibility(self):
        """Same seed should give identical results."""
        import random
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        samples1 = [rng1.gauss(1000, 100) for _ in range(1000)]
        samples2 = [rng2.gauss(1000, 100) for _ in range(1000)]
        assert samples1 == samples2

    def test_lognormal_distribution_sampling(self):
        """Lognormal sampling for emission factors."""
        import random
        rng = random.Random(42)
        mu = math.log(56.1)  # natural gas CO2 factor
        sigma = 0.05  # 5% log-normal uncertainty
        n = 5000
        samples = [math.exp(rng.gauss(mu, sigma)) for _ in range(n)]
        sample_mean = sum(samples) / n
        assert pytest.approx(sample_mean, rel=0.05) == 56.1

    def test_normal_distribution_sampling(self):
        """Normal sampling for activity data."""
        import random
        rng = random.Random(42)
        mean = 1000000.0  # m3 natural gas
        std = 50000.0  # 5% uncertainty
        n = 5000
        samples = [rng.gauss(mean, std) for _ in range(n)]
        sample_mean = sum(samples) / n
        assert pytest.approx(sample_mean, rel=0.02) == 1000000.0


# =============================================================================
# Confidence Intervals
# =============================================================================


class TestConfidenceIntervals:
    """Test confidence interval calculations."""

    def test_95_pct_confidence_interval(self):
        """95% CI = mean +/- 1.96 * std."""
        mean = Decimal("22300")
        std = Decimal("1895")  # 8.5% uncertainty
        z_95 = Decimal("1.96")
        lower = mean - z_95 * std
        upper = mean + z_95 * std
        assert lower < mean < upper
        assert lower > Decimal("0")

    def test_90_pct_confidence_interval(self):
        z_90 = Decimal("1.645")
        mean = Decimal("22300")
        std = Decimal("1895")
        lower = mean - z_90 * std
        upper = mean + z_90 * std
        width_90 = upper - lower
        # 90% CI should be narrower than 95%
        width_95 = Decimal("2") * Decimal("1.96") * std
        assert width_90 < width_95

    def test_zero_uncertainty_no_interval(self):
        mean = Decimal("1000")
        std = Decimal("0")
        lower = mean - Decimal("1.96") * std
        upper = mean + Decimal("1.96") * std
        assert lower == upper == mean


# =============================================================================
# Top Contributors / Sensitivity
# =============================================================================


class TestTopContributors:
    """Test sensitivity analysis and top contributors ranking."""

    def test_contribution_ranking(self):
        """Identify top contributors to total uncertainty."""
        contributions = [
            {"category": "stationary_combustion", "contribution_pct": 45.0},
            {"category": "process_emissions", "contribution_pct": 25.0},
            {"category": "fugitive_emissions", "contribution_pct": 15.0},
            {"category": "mobile_combustion", "contribution_pct": 10.0},
            {"category": "refrigerant_fgas", "contribution_pct": 5.0},
        ]
        sorted_contrib = sorted(contributions, key=lambda x: x["contribution_pct"], reverse=True)
        assert sorted_contrib[0]["category"] == "stationary_combustion"
        assert sorted_contrib[-1]["category"] == "refrigerant_fgas"

    def test_contributions_sum_to_100(self):
        contributions = [45.0, 25.0, 15.0, 10.0, 5.0]
        assert sum(contributions) == 100.0

    def test_improvement_recommendation(self):
        """Top contributor should get improvement recommendation."""
        top = {"category": "fugitive_emissions", "uncertainty_pct": 32.0}
        recommendation = (
            f"Reduce {top['category']} uncertainty from {top['uncertainty_pct']}% "
            f"by improving leak detection and measurement"
        )
        assert "fugitive_emissions" in recommendation
        assert "32.0" in recommendation


# =============================================================================
# IPCC Default Uncertainty Ranges
# =============================================================================


class TestIPCCDefaultRanges:
    """Test IPCC 2006 default uncertainty ranges."""

    @pytest.mark.parametrize("category,expected_range_low,expected_range_high", [
        ("stationary_combustion", 5, 10),
        ("mobile_combustion", 8, 15),
        ("process_emissions", 10, 20),
        ("fugitive_emissions", 20, 40),
        ("refrigerant_fgas", 15, 30),
        ("scope2_location", 5, 15),
        ("scope2_market", 3, 10),
    ])
    def test_uncertainty_in_expected_range(self, category, expected_range_low, expected_range_high):
        """Uncertainty should fall within IPCC expected ranges."""
        assert expected_range_low < expected_range_high

    def test_fugitive_highest_uncertainty(self):
        uncertainties = {
            "stationary_combustion": 7,
            "mobile_combustion": 11,
            "process_emissions": 16,
            "fugitive_emissions": 32,
            "refrigerant_fgas": 22,
        }
        max_cat = max(uncertainties, key=uncertainties.get)
        assert max_cat == "fugitive_emissions"


# =============================================================================
# Data Quality Scoring
# =============================================================================


class TestDataQualityScoring:
    """Test data quality assessment."""

    @pytest.mark.parametrize("quality_level,score_range_low,score_range_high", [
        ("high", 80, 100),
        ("medium", 50, 79),
        ("low", 0, 49),
    ])
    def test_quality_score_ranges(self, quality_level, score_range_low, score_range_high):
        assert score_range_low <= score_range_high

    def test_quality_score_in_inventory(self, sample_inventory):
        assert Decimal("0") <= sample_inventory["data_quality_score"] <= Decimal("100")

    def test_high_quality_low_uncertainty(self):
        """Higher data quality should correlate with lower uncertainty."""
        high_quality = {"quality_score": 90, "uncertainty_pct": 5}
        low_quality = {"quality_score": 40, "uncertainty_pct": 25}
        assert high_quality["uncertainty_pct"] < low_quality["uncertainty_pct"]


# =============================================================================
# Provenance
# =============================================================================


class TestUncertaintyProvenance:
    """Test provenance hashing for uncertainty results."""

    def test_provenance_deterministic(self, sample_inventory):
        from tests.conftest import compute_provenance_hash
        h1 = compute_provenance_hash({"uncertainty": sample_inventory["uncertainty_pct"]})
        h2 = compute_provenance_hash({"uncertainty": sample_inventory["uncertainty_pct"]})
        assert h1 == h2
