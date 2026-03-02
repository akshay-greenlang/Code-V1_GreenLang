# -*- coding: utf-8 -*-
"""
Uncertainty Quantification Module

This module provides comprehensive uncertainty quantification for carbon
emissions calculations, including:
- Factor-specific uncertainties
- Category-based default uncertainties
- Propagation through calculation chains
- Confidence interval calculation
- Advanced sensitivity analysis (Sobol, Morris, Tornado)

References:
- GHG Protocol: Corporate Value Chain (Scope 3) Standard, Chapter 7
- IPCC Guidelines for National Greenhouse Gas Inventories, Volume 1, Chapter 3
- ISO/TS 14067:2018: Greenhouse gases - Carbon footprint of products
- Weidema et al. (2013): "Data quality management for LCA"

Version: 1.1.0
Date: 2026-03-01
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from datetime import datetime

from .models import UncertaintyResult, PedigreeScore, MonteCarloResult
from .constants import (
    DEFAULT_UNCERTAINTIES,
    TIER_UNCERTAINTY_MULTIPLIERS,
    MIN_UNCERTAINTY,
    MAX_UNCERTAINTY,
    BASIC_UNCERTAINTY,
    DistributionType,
)
from .config import get_config
from .pedigree_matrix import PedigreeMatrixEvaluator
from .monte_carlo import MonteCarloSimulator, AnalyticalPropagator
from .sensitivity_analysis import (
    SobolAnalyzer,
    MorrisAnalyzer,
    TornadoDiagramGenerator,
    ConvergenceAnalyzer,
    SobolResult,
    MorrisResult,
    TornadoData,
    ConvergenceResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# UNCERTAINTY QUANTIFIER
# ============================================================================

class UncertaintyQuantifier:
    """
    Quantifier for uncertainty in carbon emissions calculations.

    This class provides methods to:
    - Determine appropriate uncertainties for different data types
    - Propagate uncertainties through calculations
    - Calculate confidence intervals
    - Perform sensitivity analysis
    - Generate uncertainty reports
    """

    def __init__(self):
        """Initialize Uncertainty Quantifier."""
        self.config = get_config()
        self.pedigree_evaluator = PedigreeMatrixEvaluator()
        logger.info("Initialized UncertaintyQuantifier")

    def get_category_uncertainty(self, category: str) -> float:
        """
        Get default uncertainty for a category.

        Args:
            category: Category name (e.g., "metals", "electricity", "transport")

        Returns:
            Relative uncertainty (coefficient of variation)
        """
        category_lower = category.lower().replace(" ", "_")

        # Try exact match
        if category_lower in DEFAULT_UNCERTAINTIES:
            uncertainty = DEFAULT_UNCERTAINTIES[category_lower]
            logger.debug(f"Category uncertainty for '{category}': {uncertainty}")
            return uncertainty

        # Try partial matches
        for key, value in DEFAULT_UNCERTAINTIES.items():
            if key in category_lower or category_lower in key:
                logger.debug(
                    f"Category uncertainty for '{category}' (matched '{key}'): {value}"
                )
                return value

        # Default
        default = DEFAULT_UNCERTAINTIES["default"]
        logger.warning(
            f"Unknown category '{category}', using default uncertainty: {default}"
        )
        return default

    def apply_tier_multiplier(self, base_uncertainty: float, tier: int) -> float:
        """
        Apply tier-based multiplier to uncertainty.

        Args:
            base_uncertainty: Base uncertainty value
            tier: Data tier (1=primary, 2=secondary, 3=estimated)

        Returns:
            Adjusted uncertainty
        """
        if tier not in TIER_UNCERTAINTY_MULTIPLIERS:
            logger.warning(f"Invalid tier {tier}, using tier 3 multiplier")
            tier = 3

        multiplier = TIER_UNCERTAINTY_MULTIPLIERS[tier]
        adjusted_uncertainty = base_uncertainty * multiplier

        logger.debug(
            f"Applied tier {tier} multiplier: "
            f"{base_uncertainty:.4f} × {multiplier} = {adjusted_uncertainty:.4f}"
        )

        return adjusted_uncertainty

    def apply_bounds(self, uncertainty: float) -> float:
        """
        Apply floor and ceiling to uncertainty value.

        Args:
            uncertainty: Uncertainty value

        Returns:
            Bounded uncertainty
        """
        original = uncertainty

        if self.config.uncertainty.apply_floor:
            uncertainty = max(uncertainty, self.config.uncertainty.min_uncertainty)

        if self.config.uncertainty.apply_ceiling:
            uncertainty = min(uncertainty, self.config.uncertainty.max_uncertainty)

        if uncertainty != original:
            logger.debug(
                f"Applied uncertainty bounds: {original:.4f} → {uncertainty:.4f}"
            )

        return uncertainty

    def estimate_uncertainty(
        self,
        category: Optional[str] = None,
        tier: Optional[int] = None,
        pedigree_score: Optional[PedigreeScore] = None,
        custom_uncertainty: Optional[float] = None,
    ) -> float:
        """
        Estimate uncertainty for a data point.

        Priority order:
        1. Custom uncertainty (if provided)
        2. Pedigree-based uncertainty (if pedigree_score provided)
        3. Category + tier uncertainty
        4. Default uncertainty

        Args:
            category: Data category
            tier: Data tier (1-3)
            pedigree_score: Pedigree matrix scores
            custom_uncertainty: Custom uncertainty value

        Returns:
            Estimated relative uncertainty
        """
        # Use custom uncertainty if provided
        if custom_uncertainty is not None:
            uncertainty = custom_uncertainty
            logger.debug(f"Using custom uncertainty: {uncertainty:.4f}")

        # Use pedigree-based uncertainty
        elif pedigree_score is not None:
            uncertainty = self.pedigree_evaluator.calculate_combined_uncertainty(
                pedigree_score, base_uncertainty=0.0
            )
            logger.debug(f"Using pedigree-based uncertainty: {uncertainty:.4f}")

        # Use category-based uncertainty
        elif category is not None:
            uncertainty = self.get_category_uncertainty(category)

            # Apply tier multiplier if provided
            if tier is not None:
                uncertainty = self.apply_tier_multiplier(uncertainty, tier)

            logger.debug(
                f"Using category-based uncertainty: {uncertainty:.4f} "
                f"(category={category}, tier={tier})"
            )

        # Use default
        else:
            uncertainty = self.config.uncertainty.default_uncertainty
            logger.debug(f"Using default uncertainty: {uncertainty:.4f}")

        # Apply bounds
        uncertainty = self.apply_bounds(uncertainty)

        return uncertainty

    def calculate_confidence_interval(
        self,
        mean: float,
        std_dev: float,
        confidence_level: float = 0.95,
        distribution: DistributionType = DistributionType.LOGNORMAL,
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval.

        Args:
            mean: Mean value
            std_dev: Standard deviation
            confidence_level: Confidence level (0.90, 0.95, 0.99)
            distribution: Distribution type

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Z-scores for common confidence levels
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }

        z = z_scores.get(confidence_level, 1.96)

        if distribution == DistributionType.LOGNORMAL and mean > 0:
            # Lognormal confidence interval
            cv = std_dev / mean
            sigma = math.sqrt(math.log(1 + cv**2))
            mu = math.log(mean) - 0.5 * sigma**2

            lower = math.exp(mu - z * sigma)
            upper = math.exp(mu + z * sigma)

        elif distribution == DistributionType.NORMAL:
            # Normal confidence interval
            lower = mean - z * std_dev
            upper = mean + z * std_dev

        else:
            # Default to normal
            lower = mean - z * std_dev
            upper = mean + z * std_dev

        # Ensure non-negative for emissions
        lower = max(0, lower)

        logger.debug(
            f"Calculated {confidence_level*100}% CI: "
            f"[{lower:.2f}, {upper:.2f}] for mean={mean:.2f}"
        )

        return lower, upper

    def quantify_uncertainty(
        self,
        mean: float,
        category: Optional[str] = None,
        tier: Optional[int] = None,
        pedigree_score: Optional[PedigreeScore] = None,
        custom_uncertainty: Optional[float] = None,
        distribution: DistributionType = DistributionType.LOGNORMAL,
    ) -> UncertaintyResult:
        """
        Quantify uncertainty for a value.

        Args:
            mean: Mean value
            category: Data category
            tier: Data tier (1-3)
            pedigree_score: Pedigree matrix scores
            custom_uncertainty: Custom uncertainty
            distribution: Distribution type

        Returns:
            UncertaintyResult

        Example:
            >>> quantifier = UncertaintyQuantifier()
            >>> result = quantifier.quantify_uncertainty(
            ...     mean=1000.0,
            ...     category="electricity",
            ...     tier=2
            ... )
            >>> result.mean
            1000.0
        """
        # Estimate uncertainty
        relative_std_dev = self.estimate_uncertainty(
            category=category,
            tier=tier,
            pedigree_score=pedigree_score,
            custom_uncertainty=custom_uncertainty,
        )

        # Calculate absolute standard deviation
        std_dev = abs(mean * relative_std_dev)
        variance = std_dev**2

        # Calculate confidence intervals
        ci_95_lower, ci_95_upper = self.calculate_confidence_interval(
            mean, std_dev, 0.95, distribution
        )
        ci_90_lower, ci_90_upper = self.calculate_confidence_interval(
            mean, std_dev, 0.90, distribution
        )

        # Calculate median (for lognormal)
        if distribution == DistributionType.LOGNORMAL and mean > 0:
            cv = relative_std_dev
            sigma = math.sqrt(math.log(1 + cv**2))
            mu = math.log(mean) - 0.5 * sigma**2
            median = math.exp(mu)

            dist_params = {"mu": mu, "sigma": sigma}
        else:
            median = mean
            dist_params = {"mean": mean, "std_dev": std_dev}

        # Identify uncertainty sources
        sources = []
        if pedigree_score is not None:
            sources.append("pedigree_matrix")
        if category is not None:
            sources.append(f"category_{category}")
        if tier is not None:
            sources.append(f"tier_{tier}")
        if custom_uncertainty is not None:
            sources.append("custom")
        if not sources:
            sources.append("default")

        # Create result
        result = UncertaintyResult(
            mean=mean,
            median=median,
            std_dev=std_dev,
            relative_std_dev=relative_std_dev,
            variance=variance,
            confidence_95_lower=ci_95_lower,
            confidence_95_upper=ci_95_upper,
            confidence_90_lower=ci_90_lower,
            confidence_90_upper=ci_90_upper,
            distribution_type=distribution,
            distribution_params=dist_params,
            uncertainty_sources=sources,
            pedigree_score=pedigree_score,
            methodology="uncertainty_quantifier",
        )

        logger.info(
            f"Quantified uncertainty: mean={mean:.2f}, std_dev={std_dev:.2f}, "
            f"CV={relative_std_dev:.4f}, CI95=[{ci_95_lower:.2f}, {ci_95_upper:.2f}]"
        )

        return result

    def propagate_simple(
        self,
        activity_mean: float,
        activity_uncertainty: float,
        factor_mean: float,
        factor_uncertainty: float,
        method: str = "analytical",
    ) -> UncertaintyResult:
        """
        Propagate uncertainty through simple multiplication: E = A × F

        Args:
            activity_mean: Activity data mean
            activity_uncertainty: Activity uncertainty (relative)
            factor_mean: Emission factor mean
            factor_uncertainty: Emission factor uncertainty (relative)
            method: Propagation method ("analytical" or "monte_carlo")

        Returns:
            UncertaintyResult for the emission

        Example:
            >>> quantifier = UncertaintyQuantifier()
            >>> result = quantifier.propagate_simple(
            ...     activity_mean=1000.0,
            ...     activity_uncertainty=0.1,
            ...     factor_mean=2.5,
            ...     factor_uncertainty=0.15,
            ...     method="analytical"
            ... )
        """
        if method == "analytical":
            # Analytical propagation (faster)
            activity_std = activity_mean * activity_uncertainty
            factor_std = factor_mean * factor_uncertainty

            emission_mean, emission_std = AnalyticalPropagator.simple_emission(
                activity_mean, activity_std, factor_mean, factor_std
            )

            # Create uncertainty result
            relative_std_dev = emission_std / emission_mean if emission_mean != 0 else 0

            ci_95_lower, ci_95_upper = self.calculate_confidence_interval(
                emission_mean, emission_std, 0.95, DistributionType.LOGNORMAL
            )
            ci_90_lower, ci_90_upper = self.calculate_confidence_interval(
                emission_mean, emission_std, 0.90, DistributionType.LOGNORMAL
            )

            # Calculate median for lognormal
            if emission_mean > 0:
                cv = relative_std_dev
                sigma = math.sqrt(math.log(1 + cv**2))
                mu = math.log(emission_mean) - 0.5 * sigma**2
                median = math.exp(mu)
            else:
                median = emission_mean

            result = UncertaintyResult(
                mean=emission_mean,
                median=median,
                std_dev=emission_std,
                relative_std_dev=relative_std_dev,
                variance=emission_std**2,
                confidence_95_lower=ci_95_lower,
                confidence_95_upper=ci_95_upper,
                confidence_90_lower=ci_90_lower,
                confidence_90_upper=ci_90_upper,
                distribution_type=DistributionType.LOGNORMAL,
                uncertainty_sources=["activity_data", "emission_factor"],
                methodology="analytical_propagation",
            )

            logger.info(
                f"Analytical propagation: E={emission_mean:.2f} ± {emission_std:.2f}"
            )

        else:
            # Monte Carlo propagation (more accurate)
            simulator = MonteCarloSimulator(seed=self.config.monte_carlo.default_seed)
            mc_result = simulator.simple_propagation(
                activity_data=activity_mean,
                activity_uncertainty=activity_uncertainty,
                emission_factor=factor_mean,
                factor_uncertainty=factor_uncertainty,
                iterations=self.config.monte_carlo.default_iterations,
            )

            # Convert to UncertaintyResult
            result = UncertaintyResult(
                mean=mc_result.mean,
                median=mc_result.median,
                std_dev=mc_result.std_dev,
                relative_std_dev=mc_result.coefficient_of_variation,
                variance=mc_result.variance,
                confidence_95_lower=mc_result.p5,
                confidence_95_upper=mc_result.p95,
                confidence_90_lower=mc_result.p10 if mc_result.p10 else mc_result.p5,
                confidence_90_upper=mc_result.p90 if mc_result.p90 else mc_result.p95,
                distribution_type=DistributionType.LOGNORMAL,
                uncertainty_sources=["activity_data", "emission_factor"],
                methodology="monte_carlo_propagation",
            )

            logger.info(
                f"Monte Carlo propagation ({mc_result.iterations} iterations): "
                f"E={result.mean:.2f} ± {result.std_dev:.2f}"
            )

        return result

    def propagate_chain(
        self,
        values: List[float],
        uncertainties: List[float],
        operations: List[str],
    ) -> UncertaintyResult:
        """
        Propagate uncertainty through a chain of operations.

        Args:
            values: List of values
            uncertainties: List of relative uncertainties
            operations: List of operations ("*" or "+") between consecutive values

        Returns:
            UncertaintyResult for the final result

        Example:
            >>> quantifier = UncertaintyQuantifier()
            >>> # Calculate (100 * 2.5) + 50
            >>> result = quantifier.propagate_chain(
            ...     values=[100, 2.5, 50],
            ...     uncertainties=[0.1, 0.15, 0.2],
            ...     operations=["*", "+"]
            ... )
        """
        if len(values) != len(uncertainties):
            raise ValueError("values and uncertainties must have same length")

        if len(operations) != len(values) - 1:
            raise ValueError("operations must have length = len(values) - 1")

        # Start with first value
        result_mean = values[0]
        result_std = values[0] * uncertainties[0]

        # Chain through operations
        for i, op in enumerate(operations):
            next_value = values[i + 1]
            next_uncertainty = uncertainties[i + 1]
            next_std = next_value * next_uncertainty

            if op == "*":
                result_mean, result_std = AnalyticalPropagator.multiply(
                    result_mean, result_std, next_value, next_std
                )
            elif op == "+":
                result_mean, result_std = AnalyticalPropagator.add(
                    result_mean, result_std, next_value, next_std
                )
            else:
                raise ValueError(f"Unsupported operation: {op}")

        # Create uncertainty result
        relative_std_dev = result_std / result_mean if result_mean != 0 else 0

        ci_95_lower, ci_95_upper = self.calculate_confidence_interval(
            result_mean, result_std, 0.95, DistributionType.LOGNORMAL
        )
        ci_90_lower, ci_90_upper = self.calculate_confidence_interval(
            result_mean, result_std, 0.90, DistributionType.LOGNORMAL
        )

        result = UncertaintyResult(
            mean=result_mean,
            median=result_mean,  # Approximation
            std_dev=result_std,
            relative_std_dev=relative_std_dev,
            variance=result_std**2,
            confidence_95_lower=ci_95_lower,
            confidence_95_upper=ci_95_upper,
            confidence_90_lower=ci_90_lower,
            confidence_90_upper=ci_90_upper,
            distribution_type=DistributionType.LOGNORMAL,
            uncertainty_sources=["chain_propagation"],
            methodology="chain_propagation",
        )

        logger.info(
            f"Chain propagation ({len(values)} values, {len(operations)} ops): "
            f"result={result_mean:.2f} ± {result_std:.2f}"
        )

        return result


# ============================================================================
# SENSITIVITY ANALYZER
# ============================================================================

class SensitivityAnalyzer:
    """
    Analyzer for sensitivity of results to input parameters.

    Provides a unified interface to multiple sensitivity analysis methods:
    - Simple contribution analysis (magnitude-based).
    - Sobol variance-based global sensitivity (first-order + total-order).
    - Morris elementary effects screening.
    - Tornado one-way sensitivity diagrams.
    - Top-contributor extraction from any sensitivity index set.

    All numerical computations delegate to the deterministic engines in
    ``sensitivity_analysis.py`` (zero-hallucination, NumPy/SciPy only).
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize SensitivityAnalyzer.

        Args:
            seed: Random seed for reproducibility in Sobol / Morris
                  analyses.  Defaults to None (non-deterministic).
        """
        self._seed = seed
        self._sobol = SobolAnalyzer(seed=seed)
        self._morris = MorrisAnalyzer(seed=seed)
        self._tornado = TornadoDiagramGenerator()
        self._convergence = ConvergenceAnalyzer()
        logger.info(f"Initialized SensitivityAnalyzer with seed={seed}")

    # ------------------------------------------------------------------
    # Legacy interface (kept for backward compatibility)
    # ------------------------------------------------------------------

    def analyze_contribution(
        self, parameters: Dict[str, float], result: float
    ) -> Dict[str, float]:
        """
        Analyze contribution of each parameter to result.

        Simple magnitude-based contribution (no model evaluations).
        This is the original interface retained for backward compatibility.

        Args:
            parameters: Dictionary of parameter values.
            result: Calculated result.

        Returns:
            Dictionary of contribution percentages (values sum to 1.0).
        """
        contributions: Dict[str, float] = {}
        total = sum(abs(v) for v in parameters.values())

        for name, value in parameters.items():
            if total > 0:
                contributions[name] = abs(value) / total
            else:
                contributions[name] = 0.0

        return contributions

    # ------------------------------------------------------------------
    # Sobol analysis
    # ------------------------------------------------------------------

    def run_sobol_analysis(
        self,
        calculation_func: Callable[[Dict[str, float]], float],
        parameters: List[Dict[str, Any]],
        N: int = 1024,
    ) -> SobolResult:
        """
        Run Sobol variance-based global sensitivity analysis.

        Estimates first-order (Si) and total-order (STi) indices using
        Saltelli's quasi-random sampling scheme.  Si measures the direct
        contribution of parameter Xi to output variance; STi includes
        all interactions involving Xi.

        Args:
            calculation_func: Deterministic model f(params) -> scalar.
            parameters: List of parameter descriptors.  Each dict must
                contain 'name', 'mean', 'std_dev', and optionally
                'distribution', 'min', 'max'.
            N: Base sample size (power of 2 recommended).  Total model
                evaluations = N * (2k + 2).

        Returns:
            SobolResult with first_order_indices, total_order_indices,
            interaction_effects, and convergence_info.

        Example:
            >>> analyzer = SensitivityAnalyzer(seed=42)
            >>> def model(p):
            ...     return p['activity'] * p['factor']
            >>> params = [
            ...     {'name': 'activity', 'mean': 1000, 'std_dev': 100},
            ...     {'name': 'factor', 'mean': 2.5, 'std_dev': 0.25},
            ... ]
            >>> result = analyzer.run_sobol_analysis(model, params, N=512)
            >>> result.first_order_indices
        """
        logger.info(
            f"SensitivityAnalyzer: running Sobol analysis, "
            f"N={N}, params={[p['name'] for p in parameters]}"
        )
        return self._sobol.run_analysis(calculation_func, parameters, N=N)

    # ------------------------------------------------------------------
    # Morris screening
    # ------------------------------------------------------------------

    def run_morris_screening(
        self,
        calculation_func: Callable[[Dict[str, float]], float],
        parameters: List[Dict[str, Any]],
        r: int = 10,
        levels: int = 4,
    ) -> MorrisResult:
        """
        Run Morris elementary effects screening.

        Generates r OAT trajectories through the parameter space and
        computes the mean absolute elementary effect (mu*) and its
        standard deviation (sigma) for each parameter.  Parameters are
        classified as important, non-important, or interactive.

        Args:
            calculation_func: Model f(params) -> scalar.
            parameters: Parameter descriptors (name, mean, std_dev, ...).
            r: Number of trajectories (default 10).
            levels: Number of grid levels (default 4).

        Returns:
            MorrisResult with mu_star, sigma, mu_star_conf, and
            classification per parameter.

        Example:
            >>> analyzer = SensitivityAnalyzer(seed=42)
            >>> result = analyzer.run_morris_screening(model, params, r=20)
            >>> result.classification
        """
        logger.info(
            f"SensitivityAnalyzer: running Morris screening, "
            f"r={r}, levels={levels}"
        )
        return self._morris.run_screening(
            calculation_func, parameters, r=r, levels=levels
        )

    # ------------------------------------------------------------------
    # Tornado diagram
    # ------------------------------------------------------------------

    def generate_tornado_data(
        self,
        calculation_func: Callable[[Dict[str, float]], float],
        parameters: List[Dict[str, Any]],
        baseline: Dict[str, float],
        variation_pct: float = 0.10,
    ) -> TornadoData:
        """
        Generate tornado diagram data via one-way parameter sweeps.

        For each parameter, the model is evaluated at
        baseline * (1 +/- variation_pct) while all other parameters
        are held at their baseline values.  Results are sorted by
        descending absolute impact.

        Args:
            calculation_func: Model f(params) -> scalar.
            parameters: Parameter descriptors.
            baseline: Nominal parameter values.
            variation_pct: Fractional variation (default 0.10 = 10%).

        Returns:
            TornadoData sorted by descending impact.

        Example:
            >>> analyzer = SensitivityAnalyzer()
            >>> tornado = analyzer.generate_tornado_data(
            ...     model,
            ...     params,
            ...     baseline={'activity': 1000, 'factor': 2.5},
            ... )
            >>> tornado.parameters[0].name  # Most impactful parameter
        """
        logger.info(
            f"SensitivityAnalyzer: generating tornado data, "
            f"variation={variation_pct:.0%}"
        )
        return self._tornado.generate_one_way(
            calculation_func, parameters, baseline, variation_pct
        )

    # ------------------------------------------------------------------
    # Top contributors
    # ------------------------------------------------------------------

    @staticmethod
    def get_top_contributors(
        sensitivity_indices: Dict[str, float],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Extract the top-N contributors from a set of sensitivity indices.

        Works with any index type (Sobol Si, STi, Pearson r, or mu*).

        Args:
            sensitivity_indices: Dict mapping parameter name to its
                sensitivity index value.
            top_n: Number of top contributors to return (default 5).

        Returns:
            List of dicts with 'rank', 'name', 'index', and
            'contribution_pct' keys, sorted by descending index.

        Example:
            >>> SensitivityAnalyzer.get_top_contributors(
            ...     {'a': 0.5, 'b': 0.3, 'c': 0.1, 'd': 0.05, 'e': 0.03, 'f': 0.02},
            ...     top_n=3,
            ... )
            [{'rank': 1, 'name': 'a', 'index': 0.5, 'contribution_pct': 51.02}, ...]
        """
        if not sensitivity_indices:
            return []

        sorted_items = sorted(
            sensitivity_indices.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        total = sum(abs(v) for _, v in sorted_items)
        results: List[Dict[str, Any]] = []

        for rank, (name, idx) in enumerate(sorted_items[:top_n], start=1):
            pct = (abs(idx) / total * 100.0) if total > 0 else 0.0
            results.append({
                "rank": rank,
                "name": name,
                "index": round(idx, 6),
                "contribution_pct": round(pct, 2),
            })

        return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quantify_uncertainty(
    mean: float,
    category: Optional[str] = None,
    tier: Optional[int] = None,
    pedigree_score: Optional[PedigreeScore] = None,
) -> UncertaintyResult:
    """
    Quick uncertainty quantification.

    Args:
        mean: Mean value
        category: Data category
        tier: Data tier
        pedigree_score: Pedigree scores

    Returns:
        UncertaintyResult
    """
    quantifier = UncertaintyQuantifier()
    return quantifier.quantify_uncertainty(mean, category, tier, pedigree_score)


def propagate_uncertainty(
    activity_mean: float,
    activity_uncertainty: float,
    factor_mean: float,
    factor_uncertainty: float,
) -> UncertaintyResult:
    """
    Quick uncertainty propagation for E = A × F.

    Args:
        activity_mean: Activity data mean
        activity_uncertainty: Activity relative uncertainty
        factor_mean: Emission factor mean
        factor_uncertainty: Emission factor relative uncertainty

    Returns:
        UncertaintyResult
    """
    quantifier = UncertaintyQuantifier()
    return quantifier.propagate_simple(
        activity_mean, activity_uncertainty, factor_mean, factor_uncertainty
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "UncertaintyQuantifier",
    "SensitivityAnalyzer",
    "quantify_uncertainty",
    "propagate_uncertainty",
    # Re-exported from sensitivity_analysis for convenience
    "SobolResult",
    "MorrisResult",
    "TornadoData",
    "ConvergenceResult",
]
