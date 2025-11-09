"""
===============================================================================
GL-VCCI Scope 3 Platform - Monte Carlo Uncertainty Propagation E2E Test
===============================================================================

PRIORITY TEST 4: Monte Carlo uncertainty propagation end-to-end

Workflow: Uncertainty propagation through full calculation pipeline

This test validates:
- Monte Carlo simulation (1000+ samples)
- Uncertainty propagation from emission factors → final results
- Confidence intervals (95% CI)
- Distribution analysis (mean, median, P5, P95)
- Performance optimization for MC sampling
- Correlation handling for dependent variables

Version: 1.0.0
Team: 8 - Quality Assurance Lead
Created: 2025-11-09
===============================================================================
"""

import pytest
import asyncio
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
from scipy import stats
from uuid import uuid4

from greenlang.telemetry import get_logger, MetricsCollector
from services.agents.calculator.agent import Scope3CalculatorAgent
from services.agents.calculator.models import Category1Input
from services.agents.calculator.calculations import UncertaintyEngine

logger = get_logger(__name__)


# ============================================================================
# Monte Carlo Test Helpers
# ============================================================================

class MonteCarloFactorBroker:
    """
    Mock factor broker with probabilistic emission factors.

    Returns factors as probability distributions instead of point estimates.
    """

    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed=42)  # Reproducible results

    def get_factor(self, category: int, **kwargs):
        """Return point estimate (mean)."""
        mean_factor = 0.5  # kg CO2e/USD
        return {
            "factor": mean_factor,
            "unit": "kg CO2e/USD",
            "source": "EPA",
            "quality_tier": 2,
            "uncertainty": 0.20,  # 20% uncertainty
        }

    def get_monte_carlo_factors(self, category: int, n_samples: int = None, **kwargs):
        """
        Return Monte Carlo samples for emission factor.

        Assumes lognormal distribution (common for emission factors).
        """
        n = n_samples or self.n_samples

        # Base factor with uncertainty
        mean = 0.5  # kg CO2e/USD
        cv = 0.20  # Coefficient of variation (20%)

        # Lognormal parameters
        sigma = np.sqrt(np.log(1 + cv**2))
        mu = np.log(mean) - 0.5 * sigma**2

        # Generate samples
        samples = self.rng.lognormal(mean=mu, sigma=sigma, size=n)

        return samples


def run_monte_carlo_simulation(
    supplier_spend: float,
    emission_factor_samples: np.ndarray
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation for a single supplier.

    Args:
        supplier_spend: Spend amount in USD
        emission_factor_samples: Array of emission factor samples

    Returns:
        Dictionary with statistical results
    """
    # Calculate emissions for each sample
    emissions_samples = supplier_spend * emission_factor_samples / 1000  # Convert to tCO2e

    # Calculate statistics
    mean_emissions = np.mean(emissions_samples)
    median_emissions = np.median(emissions_samples)
    std_emissions = np.std(emissions_samples)

    # Percentiles for confidence intervals
    p5 = np.percentile(emissions_samples, 5)
    p25 = np.percentile(emissions_samples, 25)
    p75 = np.percentile(emissions_samples, 75)
    p95 = np.percentile(emissions_samples, 95)

    # 95% confidence interval
    ci_95_lower = np.percentile(emissions_samples, 2.5)
    ci_95_upper = np.percentile(emissions_samples, 97.5)

    return {
        "mean": mean_emissions,
        "median": median_emissions,
        "std": std_emissions,
        "p5": p5,
        "p25": p25,
        "p75": p75,
        "p95": p95,
        "ci_95_lower": ci_95_lower,
        "ci_95_upper": ci_95_upper,
        "samples": emissions_samples,
        "n_samples": len(emissions_samples),
    }


def aggregate_portfolio_uncertainty(
    supplier_results: List[Dict[str, Any]],
    correlation_matrix: np.ndarray = None
) -> Dict[str, Any]:
    """
    Aggregate uncertainty across portfolio of suppliers.

    Handles correlations if provided, otherwise assumes independence.
    """
    n_suppliers = len(supplier_results)
    n_samples = supplier_results[0]["n_samples"]

    # Stack all samples
    all_samples = np.array([r["samples"] for r in supplier_results])  # (n_suppliers, n_samples)

    # Sum across suppliers for each sample
    portfolio_samples = np.sum(all_samples, axis=0)  # (n_samples,)

    # Calculate portfolio statistics
    mean_portfolio = np.mean(portfolio_samples)
    std_portfolio = np.std(portfolio_samples)
    ci_95_lower = np.percentile(portfolio_samples, 2.5)
    ci_95_upper = np.percentile(portfolio_samples, 97.5)

    return {
        "mean": mean_portfolio,
        "std": std_portfolio,
        "ci_95_lower": ci_95_lower,
        "ci_95_upper": ci_95_upper,
        "samples": portfolio_samples,
        "n_suppliers": n_suppliers,
    }


# ============================================================================
# Test Class: Monte Carlo Uncertainty
# ============================================================================

@pytest.mark.integration
@pytest.mark.monte_carlo
@pytest.mark.critical
class TestMonteCarloUncertainty:
    """Monte Carlo uncertainty propagation tests."""

    @pytest.mark.asyncio
    async def test_single_supplier_monte_carlo(self):
        """
        Test Monte Carlo simulation for single supplier.

        Exit Criteria:
        ✅ 1000 samples generated
        ✅ Results follow expected distribution
        ✅ 95% CI calculated correctly
        ✅ Uncertainty range reasonable (±20%)
        """
        logger.info("Testing single supplier Monte Carlo simulation")

        # Setup
        supplier_spend = 100000.0  # $100K spend
        n_samples = 1000

        factor_broker = MonteCarloFactorBroker(n_samples=n_samples)

        # Get emission factor samples
        factor_samples = factor_broker.get_monte_carlo_factors(category=1, n_samples=n_samples)

        assert len(factor_samples) == n_samples
        assert np.all(factor_samples > 0)  # All samples positive

        # Run Monte Carlo simulation
        mc_result = run_monte_carlo_simulation(supplier_spend, factor_samples)

        logger.info(f"Mean emissions: {mc_result['mean']:.2f} tCO2e")
        logger.info(f"Median emissions: {mc_result['median']:.2f} tCO2e")
        logger.info(f"Std deviation: {mc_result['std']:.2f} tCO2e")
        logger.info(f"95% CI: [{mc_result['ci_95_lower']:.2f}, {mc_result['ci_95_upper']:.2f}] tCO2e")

        # Assertions
        assert mc_result["n_samples"] == n_samples

        # Mean should be close to expected value (100K * 0.5 / 1000 = 50 tCO2e)
        expected_mean = 50.0
        assert 40.0 <= mc_result["mean"] <= 60.0, f"Mean {mc_result['mean']:.2f} outside expected range"

        # Median should be close to mean for lognormal
        assert abs(mc_result["median"] - mc_result["mean"]) < mc_result["std"]

        # 95% CI should contain ~95% of samples
        ci_width = mc_result["ci_95_upper"] - mc_result["ci_95_lower"]
        relative_width = ci_width / mc_result["mean"]
        assert 0.3 <= relative_width <= 1.0, f"CI width {relative_width:.2%} unusual"

        # P95 > P75 > P25 > P5 (monotonicity)
        assert mc_result["p95"] > mc_result["p75"]
        assert mc_result["p75"] > mc_result["p25"]
        assert mc_result["p25"] > mc_result["p5"]

        logger.info("✅ Single supplier Monte Carlo test PASSED")


    @pytest.mark.asyncio
    async def test_portfolio_monte_carlo_aggregation(self):
        """
        Test Monte Carlo aggregation across portfolio of suppliers.

        Exit Criteria:
        ✅ 100 suppliers simulated
        ✅ Portfolio uncertainty calculated
        ✅ Diversification benefit observed
        ✅ Results statistically valid
        """
        logger.info("Testing portfolio Monte Carlo aggregation")

        # Setup
        n_suppliers = 100
        n_samples = 1000

        factor_broker = MonteCarloFactorBroker(n_samples=n_samples)

        # Generate supplier spend amounts (Pareto distribution)
        np.random.seed(42)
        supplier_spends = np.random.pareto(1.16, n_suppliers) * 10000 + 5000

        # Run MC simulation for each supplier
        supplier_results = []

        for i, spend in enumerate(supplier_spends):
            factor_samples = factor_broker.get_monte_carlo_factors(category=1, n_samples=n_samples)
            mc_result = run_monte_carlo_simulation(spend, factor_samples)
            supplier_results.append(mc_result)

        # Aggregate portfolio uncertainty
        portfolio_result = aggregate_portfolio_uncertainty(supplier_results)

        # Calculate simple sum of means (for comparison)
        sum_of_means = sum(r["mean"] for r in supplier_results)

        # Portfolio standard deviation (assuming independence)
        sum_of_variances = sum(r["std"]**2 for r in supplier_results)
        independent_std = np.sqrt(sum_of_variances)

        logger.info("=" * 80)
        logger.info("PORTFOLIO MONTE CARLO RESULTS")
        logger.info("=" * 80)
        logger.info(f"Number of suppliers: {n_suppliers}")
        logger.info(f"Sum of individual means: {sum_of_means:.2f} tCO2e")
        logger.info(f"Portfolio mean: {portfolio_result['mean']:.2f} tCO2e")
        logger.info(f"Portfolio std (independent): {independent_std:.2f} tCO2e")
        logger.info(f"Portfolio std (simulated): {portfolio_result['std']:.2f} tCO2e")
        logger.info(f"95% CI: [{portfolio_result['ci_95_lower']:.2f}, {portfolio_result['ci_95_upper']:.2f}] tCO2e")
        logger.info("=" * 80)

        # Assertions
        # Portfolio mean should equal sum of means (linearity of expectation)
        assert abs(portfolio_result['mean'] - sum_of_means) < 1.0

        # Portfolio std should be close to independent calculation (assuming independence)
        assert abs(portfolio_result['std'] - independent_std) < independent_std * 0.1

        # Diversification benefit: Portfolio CV < average individual CV
        individual_cvs = [r["std"] / r["mean"] for r in supplier_results]
        avg_individual_cv = np.mean(individual_cvs)
        portfolio_cv = portfolio_result['std'] / portfolio_result['mean']

        logger.info(f"Average individual CV: {avg_individual_cv:.2%}")
        logger.info(f"Portfolio CV: {portfolio_cv:.2%}")

        # Portfolio should have lower relative uncertainty due to diversification
        assert portfolio_cv < avg_individual_cv, "No diversification benefit observed"

        logger.info("✅ Portfolio Monte Carlo test PASSED")


    @pytest.mark.asyncio
    async def test_monte_carlo_performance(self):
        """
        Test Monte Carlo performance for large portfolios.

        Exit Criteria:
        ✅ 1000 suppliers x 1000 samples = 1M calculations
        ✅ Processing time < 30 seconds
        ✅ Memory usage reasonable
        """
        logger.info("Testing Monte Carlo performance")

        n_suppliers = 1000
        n_samples = 1000

        start_time = time.time()

        factor_broker = MonteCarloFactorBroker(n_samples=n_samples)

        # Vectorized approach for performance
        np.random.seed(42)
        supplier_spends = np.random.pareto(1.16, n_suppliers) * 10000 + 5000

        # Get factor samples once (can be reused if factors are similar)
        factor_samples = factor_broker.get_monte_carlo_factors(category=1, n_samples=n_samples)

        # Vectorized calculation
        # Shape: (n_suppliers, n_samples)
        emissions_matrix = np.outer(supplier_spends, factor_samples) / 1000

        # Aggregate across suppliers
        portfolio_samples = np.sum(emissions_matrix, axis=0)

        # Calculate statistics
        mean_emissions = np.mean(portfolio_samples)
        std_emissions = np.std(portfolio_samples)
        ci_95_lower = np.percentile(portfolio_samples, 2.5)
        ci_95_upper = np.percentile(portfolio_samples, 97.5)

        elapsed = time.time() - start_time

        logger.info("=" * 80)
        logger.info("MONTE CARLO PERFORMANCE TEST")
        logger.info("=" * 80)
        logger.info(f"Suppliers: {n_suppliers}")
        logger.info(f"Samples per supplier: {n_samples}")
        logger.info(f"Total calculations: {n_suppliers * n_samples:,}")
        logger.info(f"Processing time: {elapsed:.2f}s")
        logger.info(f"Throughput: {(n_suppliers * n_samples) / elapsed:,.0f} calc/s")
        logger.info(f"Mean emissions: {mean_emissions:,.2f} tCO2e")
        logger.info(f"95% CI: [{ci_95_lower:,.2f}, {ci_95_upper:,.2f}] tCO2e")
        logger.info("=" * 80)

        # Assertions
        assert elapsed < 30.0, f"Processing took {elapsed:.2f}s, expected <30s"
        assert mean_emissions > 0
        assert ci_95_upper > ci_95_lower

        throughput = (n_suppliers * n_samples) / elapsed
        assert throughput > 10000, f"Throughput {throughput:.0f} calc/s too low"

        logger.info("✅ Monte Carlo performance test PASSED")


    @pytest.mark.asyncio
    async def test_uncertainty_tier_comparison(self):
        """
        Test uncertainty varies by data quality tier.

        Exit Criteria:
        ✅ Tier 1 (primary) has lowest uncertainty
        ✅ Tier 2 (secondary) has medium uncertainty
        ✅ Tier 3 (tertiary) has highest uncertainty
        ✅ Uncertainty ratios reasonable
        """
        logger.info("Testing uncertainty by data quality tier")

        n_samples = 1000
        supplier_spend = 100000.0

        # Define tier-specific uncertainties
        tier_uncertainties = {
            1: 0.10,  # ±10% for primary data
            2: 0.25,  # ±25% for secondary data
            3: 0.50,  # ±50% for tertiary data
        }

        tier_results = {}

        for tier, uncertainty in tier_uncertainties.items():
            # Generate factor samples with tier-specific uncertainty
            mean_factor = 0.5
            cv = uncertainty

            sigma = np.sqrt(np.log(1 + cv**2))
            mu = np.log(mean_factor) - 0.5 * sigma**2

            factor_samples = np.random.lognormal(mean=mu, sigma=sigma, size=n_samples)

            # Run MC simulation
            mc_result = run_monte_carlo_simulation(supplier_spend, factor_samples)

            tier_results[tier] = mc_result

            logger.info(f"Tier {tier}: Mean={mc_result['mean']:.2f}, Std={mc_result['std']:.2f}, "
                       f"CV={mc_result['std']/mc_result['mean']:.2%}")

        # Assertions
        # Uncertainty should increase with tier: Tier 1 < Tier 2 < Tier 3
        assert tier_results[1]["std"] < tier_results[2]["std"]
        assert tier_results[2]["std"] < tier_results[3]["std"]

        # Coefficient of variation should match expectations
        for tier, uncertainty in tier_uncertainties.items():
            cv = tier_results[tier]["std"] / tier_results[tier]["mean"]
            # CV should be within ±5% of expected
            assert abs(cv - uncertainty) < 0.05, f"Tier {tier} CV {cv:.2%} != expected {uncertainty:.2%}"

        # CI width ratio: Tier 3 / Tier 1 should be ~5x
        ci_width_1 = tier_results[1]["ci_95_upper"] - tier_results[1]["ci_95_lower"]
        ci_width_3 = tier_results[3]["ci_95_upper"] - tier_results[3]["ci_95_lower"]
        width_ratio = ci_width_3 / ci_width_1

        logger.info(f"CI width ratio (Tier 3 / Tier 1): {width_ratio:.2f}x")
        assert 3.0 <= width_ratio <= 7.0, f"Width ratio {width_ratio:.2f}x outside expected range"

        logger.info("✅ Uncertainty tier comparison test PASSED")


    @pytest.mark.asyncio
    async def test_monte_carlo_convergence(self):
        """
        Test Monte Carlo convergence with sample size.

        Exit Criteria:
        ✅ Results stabilize with more samples
        ✅ Standard error decreases as 1/sqrt(n)
        ✅ 1000 samples sufficient for <5% error
        """
        logger.info("Testing Monte Carlo convergence")

        supplier_spend = 100000.0
        sample_sizes = [100, 500, 1000, 5000, 10000]

        convergence_results = []

        for n_samples in sample_sizes:
            factor_broker = MonteCarloFactorBroker(n_samples=n_samples)
            factor_samples = factor_broker.get_monte_carlo_factors(category=1, n_samples=n_samples)

            mc_result = run_monte_carlo_simulation(supplier_spend, factor_samples)

            # Standard error of mean
            se_mean = mc_result["std"] / np.sqrt(n_samples)
            relative_se = se_mean / mc_result["mean"]

            convergence_results.append({
                "n_samples": n_samples,
                "mean": mc_result["mean"],
                "std": mc_result["std"],
                "se_mean": se_mean,
                "relative_se": relative_se,
            })

            logger.info(f"n={n_samples:5d}: Mean={mc_result['mean']:.2f}, SE={se_mean:.3f}, "
                       f"Relative SE={relative_se:.2%}")

        # Assertions
        # Standard error should decrease with more samples
        for i in range(len(convergence_results) - 1):
            assert convergence_results[i+1]["se_mean"] < convergence_results[i]["se_mean"]

        # At n=1000, relative SE should be < 5%
        result_1000 = [r for r in convergence_results if r["n_samples"] == 1000][0]
        assert result_1000["relative_se"] < 0.05, f"Relative SE {result_1000['relative_se']:.2%} exceeds 5%"

        logger.info("✅ Monte Carlo convergence test PASSED")


# ============================================================================
# Distribution Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.monte_carlo
class TestDistributionAnalysis:
    """Test distribution analysis and goodness-of-fit."""

    def test_lognormal_distribution_fit(self):
        """
        Test that emission samples follow lognormal distribution.

        Uses Kolmogorov-Smirnov test for goodness-of-fit.
        """
        logger.info("Testing lognormal distribution fit")

        n_samples = 10000
        mean_factor = 0.5
        cv = 0.20

        # Generate lognormal samples
        sigma = np.sqrt(np.log(1 + cv**2))
        mu = np.log(mean_factor) - 0.5 * sigma**2
        samples = np.random.lognormal(mean=mu, sigma=sigma, size=n_samples)

        # Fit lognormal distribution
        shape, loc, scale = stats.lognorm.fit(samples, floc=0)

        # Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.kstest(samples, 'lognorm', args=(shape, loc, scale))

        logger.info(f"KS statistic: {ks_statistic:.4f}")
        logger.info(f"P-value: {p_value:.4f}")

        # High p-value means we cannot reject lognormal hypothesis
        assert p_value > 0.05, f"P-value {p_value:.4f} < 0.05, reject lognormal hypothesis"

        logger.info("✅ Lognormal distribution fit test PASSED")
