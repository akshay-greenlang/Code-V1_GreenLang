"""
Uncertainty Engine -- Monte Carlo Propagation (GHG Protocol Ch 11)

Implements uncertainty quantification for GHG inventories using Monte Carlo
simulation.  Each emission source is modeled with a lognormal distribution
whose parameters are derived from the data quality tier.

Outputs:
  - 5th / 50th / 95th percentile ranges
  - Standard deviation and coefficient of variation
  - Per-scope uncertainty breakdown
  - Sensitivity ranking (which sources contribute most to uncertainty)
  - Data quality impact assessment

Reference: GHG Protocol Corporate Standard Chapter 11 -- Managing
Inventory Quality.

Example:
    >>> engine = UncertaintyEngine(config)
    >>> result = engine.run_monte_carlo(inventory, iterations=10000)
    >>> print(f"95% CI: {result.p5} -- {result.p95}")
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    DataQualityTier,
    GHGAppConfig,
    Scope,
    UNCERTAINTY_CV_BY_TIER,
)
from .models import (
    GHGInventory,
    ScopeEmissions,
    ScopeUncertainty,
    UncertaintyResult,
    _now,
)

logger = logging.getLogger(__name__)


class UncertaintyEngine:
    """
    Monte Carlo uncertainty propagation across all scopes.

    For each emission source, a lognormal distribution is created with:
      - mu  = ln(mean) - sigma^2/2  (so the mean of the lognormal = mean)
      - sigma = CV / 100  (derived from the data quality tier)

    The simulation samples each source independently, sums across
    categories and scopes, then extracts percentile statistics.
    """

    # Tier -> CV percentage mapping
    TIER_CV_MAP: Dict[DataQualityTier, float] = {
        DataQualityTier.TIER_1: 50.0,
        DataQualityTier.TIER_2: 20.0,
        DataQualityTier.TIER_3: 5.0,
    }

    def __init__(
        self,
        config: Optional[GHGAppConfig] = None,
    ) -> None:
        """
        Initialize UncertaintyEngine.

        Args:
            config: Application configuration.
        """
        self.config = config or GHGAppConfig()
        self._default_iterations = self.config.monte_carlo_iterations
        self._confidence_level = float(self.config.confidence_level)
        self._rng = np.random.default_rng(seed=42)  # Reproducible by default
        logger.info(
            "UncertaintyEngine initialized (iterations=%d, confidence=%.0f%%)",
            self._default_iterations,
            self._confidence_level,
        )

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def run_monte_carlo(
        self,
        inventory: GHGInventory,
        iterations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> UncertaintyResult:
        """
        Run Monte Carlo simulation across all scopes.

        Args:
            inventory: GHG inventory with populated scope emissions.
            iterations: Number of simulation iterations (default from config).
            seed: Random seed for reproducibility.

        Returns:
            UncertaintyResult with full statistical summary.
        """
        start = datetime.utcnow()
        n = iterations or self._default_iterations
        rng = np.random.default_rng(seed=seed) if seed is not None else self._rng

        logger.info(
            "Starting Monte Carlo simulation for inventory %s (%d iterations)",
            inventory.id,
            n,
        )

        # Collect scope-level results
        scope_samples: Dict[str, np.ndarray] = {}
        scope_uncertainties: Dict[str, ScopeUncertainty] = {}

        # Scope 1
        if inventory.scope1 and inventory.scope1.total_tco2e > 0:
            s1_samples = self._simulate_scope(inventory.scope1, n, rng)
            scope_samples["scope_1"] = s1_samples
            scope_uncertainties["scope_1"] = self._summarize_scope(
                Scope.SCOPE_1, s1_samples
            )

        # Scope 2 (location-based)
        if inventory.scope2_location and inventory.scope2_location.total_tco2e > 0:
            s2l_samples = self._simulate_scope(inventory.scope2_location, n, rng)
            scope_samples["scope_2_location"] = s2l_samples
            scope_uncertainties["scope_2_location"] = self._summarize_scope(
                Scope.SCOPE_2_LOCATION, s2l_samples
            )

        # Scope 2 (market-based)
        if inventory.scope2_market and inventory.scope2_market.total_tco2e > 0:
            s2m_samples = self._simulate_scope(inventory.scope2_market, n, rng)
            scope_samples["scope_2_market"] = s2m_samples
            scope_uncertainties["scope_2_market"] = self._summarize_scope(
                Scope.SCOPE_2_MARKET, s2m_samples
            )

        # Scope 3
        if inventory.scope3 and inventory.scope3.total_tco2e > 0:
            s3_samples = self._simulate_scope(inventory.scope3, n, rng)
            scope_samples["scope_3"] = s3_samples
            scope_uncertainties["scope_3"] = self._summarize_scope(
                Scope.SCOPE_3, s3_samples
            )

        # Combine all scopes
        combined = self._combine_samples(scope_samples, n)

        # Sensitivity ranking
        sensitivity = self._compute_sensitivity(inventory, scope_samples, combined)

        result = self._build_result(
            inventory_id=inventory.id,
            iterations=n,
            combined=combined,
            scope_uncertainties=scope_uncertainties,
            sensitivity=sensitivity,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Monte Carlo complete for inventory %s: mean=%.2f, "
            "p5=%.2f, p95=%.2f, CV=%.1f%% in %.1f ms",
            inventory.id,
            result.mean,
            result.p5,
            result.p95,
            result.cv,
            elapsed_ms,
        )
        return result

    def propagate_scope1_uncertainty(
        self,
        scope1_data: ScopeEmissions,
        iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Propagate uncertainty for Scope 1 only.

        Args:
            scope1_data: Scope 1 emissions data.
            iterations: Number of iterations.

        Returns:
            Dict with Scope 1 uncertainty statistics.
        """
        n = iterations or self._default_iterations
        samples = self._simulate_scope(scope1_data, n, self._rng)
        summary = self._summarize_scope(Scope.SCOPE_1, samples)
        return self._scope_uncertainty_to_dict(summary, n)

    def propagate_scope2_uncertainty(
        self,
        scope2_data: ScopeEmissions,
        iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Propagate uncertainty for Scope 2 only.

        Args:
            scope2_data: Scope 2 emissions data.
            iterations: Number of iterations.

        Returns:
            Dict with Scope 2 uncertainty statistics.
        """
        n = iterations or self._default_iterations
        samples = self._simulate_scope(scope2_data, n, self._rng)
        scope = scope2_data.scope
        summary = self._summarize_scope(scope, samples)
        return self._scope_uncertainty_to_dict(summary, n)

    def propagate_scope3_uncertainty(
        self,
        scope3_data: ScopeEmissions,
        iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Propagate uncertainty for Scope 3 only.

        Args:
            scope3_data: Scope 3 emissions data.
            iterations: Number of iterations.

        Returns:
            Dict with Scope 3 uncertainty statistics.
        """
        n = iterations or self._default_iterations
        samples = self._simulate_scope(scope3_data, n, self._rng)
        summary = self._summarize_scope(Scope.SCOPE_3, samples)
        return self._scope_uncertainty_to_dict(summary, n)

    def combine_scope_uncertainties(
        self,
        scope_results: Dict[str, np.ndarray],
        iterations: Optional[int] = None,
    ) -> UncertaintyResult:
        """
        Combine pre-computed scope samples into overall uncertainty.

        Args:
            scope_results: Dict of scope name to sample arrays.
            iterations: Number of iterations.

        Returns:
            Combined UncertaintyResult.
        """
        n = iterations or self._default_iterations
        combined = self._combine_samples(scope_results, n)
        return self._build_result(
            inventory_id="combined",
            iterations=n,
            combined=combined,
            scope_uncertainties={},
            sensitivity=[],
        )

    def get_sensitivity_ranking(
        self,
        inventory: GHGInventory,
        iterations: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank emission sources by their contribution to total uncertainty.

        Uses a one-at-a-time (OAT) approach: fix each source at its
        mean and measure the reduction in total variance.

        Args:
            inventory: GHG inventory.
            iterations: Number of iterations.

        Returns:
            List of sources ranked by uncertainty contribution (descending).
        """
        n = iterations or min(self._default_iterations, 5000)
        rng = self._rng

        sources = self._extract_sources(inventory)
        if not sources:
            return []

        # Full simulation
        full_samples = self._simulate_sources(sources, n, rng)
        full_variance = float(np.var(full_samples))

        ranking: List[Dict[str, Any]] = []

        for i, source in enumerate(sources):
            # Fix this source at its mean; vary all others
            fixed_samples = self._simulate_sources_except(sources, i, n, rng)
            fixed_variance = float(np.var(fixed_samples))

            contribution = max(full_variance - fixed_variance, 0.0)
            contribution_pct = 0.0
            if full_variance > 0:
                contribution_pct = (contribution / full_variance) * 100

            ranking.append({
                "source": source["name"],
                "scope": source["scope"],
                "mean_tco2e": str(Decimal(str(source["mean"])).quantize(Decimal("0.01"))),
                "cv_pct": str(Decimal(str(source["cv"])).quantize(Decimal("0.1"))),
                "variance_contribution_pct": str(
                    Decimal(str(contribution_pct)).quantize(Decimal("0.1"))
                ),
            })

        ranking.sort(
            key=lambda r: float(r["variance_contribution_pct"]),
            reverse=True,
        )
        return ranking

    def assess_data_quality_impact(
        self,
        inventory: GHGInventory,
    ) -> Dict[str, Any]:
        """
        Assess the impact of data quality improvements on uncertainty.

        Simulates what would happen if all sources were upgraded to
        Tier 2 or Tier 3.

        Args:
            inventory: GHG inventory.

        Returns:
            Dict with current and projected uncertainty under improved tiers.
        """
        n = min(self._default_iterations, 5000)

        # Current state
        current_result = self.run_monte_carlo(inventory, iterations=n, seed=42)

        # Simulated improvement: all sources at Tier 2
        improved_t2 = self._simulate_with_tier_override(
            inventory, DataQualityTier.TIER_2, n
        )

        # Simulated improvement: all sources at Tier 3
        improved_t3 = self._simulate_with_tier_override(
            inventory, DataQualityTier.TIER_3, n
        )

        return {
            "current": {
                "cv_pct": str(current_result.cv),
                "p5": str(current_result.p5),
                "p95": str(current_result.p95),
                "range_width": str(current_result.p95 - current_result.p5),
            },
            "if_all_tier_2": {
                "cv_pct": str(improved_t2["cv"]),
                "p5": str(improved_t2["p5"]),
                "p95": str(improved_t2["p95"]),
                "range_width": str(improved_t2["p95"] - improved_t2["p5"]),
                "uncertainty_reduction_pct": str(
                    self._pct_change(current_result.cv, improved_t2["cv"])
                ),
            },
            "if_all_tier_3": {
                "cv_pct": str(improved_t3["cv"]),
                "p5": str(improved_t3["p5"]),
                "p95": str(improved_t3["p95"]),
                "range_width": str(improved_t3["p95"] - improved_t3["p5"]),
                "uncertainty_reduction_pct": str(
                    self._pct_change(current_result.cv, improved_t3["cv"])
                ),
            },
            "recommendation": self._quality_recommendation(current_result.cv),
        }

    # ------------------------------------------------------------------
    # Simulation Core
    # ------------------------------------------------------------------

    def _simulate_scope(
        self,
        scope_data: ScopeEmissions,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Simulate emissions for a single scope using per-category lognormal.

        Each category within the scope is sampled independently, then
        the scope total is the sum of all category samples.
        """
        sources = self._scope_to_sources(scope_data)
        if not sources:
            # If no categories, simulate the total directly
            total = float(scope_data.total_tco2e)
            cv = self.TIER_CV_MAP.get(scope_data.data_quality_tier, 50.0) / 100.0
            return self._sample_lognormal(total, cv, n, rng)

        return self._simulate_sources(sources, n, rng)

    def _simulate_sources(
        self,
        sources: List[Dict[str, Any]],
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sum of independent lognormal samples for each source."""
        total_samples = np.zeros(n)
        for source in sources:
            mean = source["mean"]
            cv = source["cv"] / 100.0
            if mean > 0:
                samples = self._sample_lognormal(mean, cv, n, rng)
                total_samples += samples
        return total_samples

    def _simulate_sources_except(
        self,
        sources: List[Dict[str, Any]],
        exclude_idx: int,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate all sources except one (fixed at mean)."""
        total_samples = np.zeros(n)
        for i, source in enumerate(sources):
            mean = source["mean"]
            if mean <= 0:
                continue
            if i == exclude_idx:
                total_samples += mean  # Fixed at mean
            else:
                cv = source["cv"] / 100.0
                total_samples += self._sample_lognormal(mean, cv, n, rng)
        return total_samples

    @staticmethod
    def _sample_lognormal(
        mean: float,
        cv: float,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample from a lognormal distribution given mean and CV.

        Parameters of the underlying normal:
          sigma2 = ln(1 + cv^2)
          mu = ln(mean) - sigma2/2
        """
        if mean <= 0 or cv <= 0:
            return np.full(n, mean)

        sigma2 = math.log(1 + cv ** 2)
        sigma = math.sqrt(sigma2)
        mu = math.log(mean) - sigma2 / 2
        return rng.lognormal(mu, sigma, n)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scope_to_sources(
        self,
        scope_data: ScopeEmissions,
    ) -> List[Dict[str, Any]]:
        """Convert scope emissions into a list of source dicts."""
        sources: List[Dict[str, Any]] = []
        tier_cv = self.TIER_CV_MAP.get(scope_data.data_quality_tier, 50.0)

        for cat_name, cat_value in scope_data.by_category.items():
            fval = float(cat_value)
            if fval > 0:
                sources.append({
                    "name": cat_name,
                    "scope": scope_data.scope.value,
                    "mean": fval,
                    "cv": tier_cv,
                })
        return sources

    def _extract_sources(
        self,
        inventory: GHGInventory,
    ) -> List[Dict[str, Any]]:
        """Extract all emission sources from an inventory."""
        all_sources: List[Dict[str, Any]] = []
        for scope_data in [
            inventory.scope1,
            inventory.scope2_location,
            inventory.scope3,
        ]:
            if scope_data:
                all_sources.extend(self._scope_to_sources(scope_data))
        return all_sources

    def _combine_samples(
        self,
        scope_samples: Dict[str, np.ndarray],
        n: int,
    ) -> np.ndarray:
        """Sum samples across all scopes."""
        combined = np.zeros(n)
        for samples in scope_samples.values():
            if len(samples) == n:
                combined += samples
        return combined

    def _summarize_scope(
        self,
        scope: Scope,
        samples: np.ndarray,
    ) -> ScopeUncertainty:
        """Build ScopeUncertainty from sample array."""
        if len(samples) == 0:
            return ScopeUncertainty(scope=scope)

        mean = float(np.mean(samples))
        std = float(np.std(samples))
        cv = (std / mean * 100) if mean > 0 else 0.0

        return ScopeUncertainty(
            scope=scope,
            mean=Decimal(str(round(mean, 2))),
            p5=Decimal(str(round(float(np.percentile(samples, 5)), 2))),
            p50=Decimal(str(round(float(np.percentile(samples, 50)), 2))),
            p95=Decimal(str(round(float(np.percentile(samples, 95)), 2))),
            std_dev=Decimal(str(round(std, 2))),
            cv=Decimal(str(round(cv, 1))),
        )

    def _build_result(
        self,
        inventory_id: str,
        iterations: int,
        combined: np.ndarray,
        scope_uncertainties: Dict[str, ScopeUncertainty],
        sensitivity: List[Dict[str, Any]],
    ) -> UncertaintyResult:
        """Build UncertaintyResult from combined samples."""
        if len(combined) == 0 or float(np.sum(combined)) == 0:
            return UncertaintyResult(
                inventory_id=inventory_id,
                iterations=iterations,
            )

        mean = float(np.mean(combined))
        std = float(np.std(combined))
        cv = (std / mean * 100) if mean > 0 else 0.0

        return UncertaintyResult(
            inventory_id=inventory_id,
            iterations=iterations,
            mean=Decimal(str(round(mean, 2))),
            p5=Decimal(str(round(float(np.percentile(combined, 5)), 2))),
            p50=Decimal(str(round(float(np.percentile(combined, 50)), 2))),
            p95=Decimal(str(round(float(np.percentile(combined, 95)), 2))),
            std_dev=Decimal(str(round(std, 2))),
            cv=Decimal(str(round(cv, 1))),
            confidence_level=Decimal(str(self._confidence_level)),
            by_scope=scope_uncertainties,
            sensitivity_ranking=sensitivity,
        )

    def _compute_sensitivity(
        self,
        inventory: GHGInventory,
        scope_samples: Dict[str, np.ndarray],
        combined: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Compute scope-level sensitivity ranking."""
        if len(combined) == 0 or float(np.var(combined)) == 0:
            return []

        total_var = float(np.var(combined))
        ranking: List[Dict[str, Any]] = []

        for scope_name, samples in scope_samples.items():
            scope_var = float(np.var(samples))
            contribution = (scope_var / total_var * 100) if total_var > 0 else 0.0
            ranking.append({
                "source": scope_name,
                "scope": scope_name,
                "mean_tco2e": str(Decimal(str(round(float(np.mean(samples)), 2)))),
                "cv_pct": str(
                    Decimal(str(round(
                        float(np.std(samples)) / max(float(np.mean(samples)), 1e-9) * 100, 1
                    )))
                ),
                "variance_contribution_pct": str(
                    Decimal(str(round(contribution, 1)))
                ),
            })

        ranking.sort(
            key=lambda r: float(r["variance_contribution_pct"]),
            reverse=True,
        )
        return ranking

    def _simulate_with_tier_override(
        self,
        inventory: GHGInventory,
        tier: DataQualityTier,
        n: int,
    ) -> Dict[str, Decimal]:
        """Simulate with all sources overridden to a specific tier."""
        cv = self.TIER_CV_MAP[tier] / 100.0
        rng = np.random.default_rng(seed=42)

        all_samples = np.zeros(n)
        for scope_data in [
            inventory.scope1,
            inventory.scope2_location,
            inventory.scope3,
        ]:
            if scope_data is None:
                continue
            for cat_value in scope_data.by_category.values():
                fval = float(cat_value)
                if fval > 0:
                    all_samples += self._sample_lognormal(fval, cv, n, rng)
            # If no categories, use total
            if not scope_data.by_category and scope_data.total_tco2e > 0:
                all_samples += self._sample_lognormal(
                    float(scope_data.total_tco2e), cv, n, rng
                )

        mean = float(np.mean(all_samples))
        std = float(np.std(all_samples))

        return {
            "cv": Decimal(str(round((std / mean * 100) if mean > 0 else 0.0, 1))),
            "p5": Decimal(str(round(float(np.percentile(all_samples, 5)), 2))),
            "p95": Decimal(str(round(float(np.percentile(all_samples, 95)), 2))),
        }

    @staticmethod
    def _scope_uncertainty_to_dict(
        su: ScopeUncertainty,
        n: int,
    ) -> Dict[str, Any]:
        """Convert ScopeUncertainty to a plain dict."""
        return {
            "scope": su.scope.value,
            "iterations": n,
            "mean": str(su.mean),
            "p5": str(su.p5),
            "p50": str(su.p50),
            "p95": str(su.p95),
            "std_dev": str(su.std_dev),
            "cv_pct": str(su.cv),
        }

    @staticmethod
    def _pct_change(old: Decimal, new: Decimal) -> Decimal:
        """Percentage change from old to new."""
        if old == 0:
            return Decimal("0")
        return ((new - old) / old * 100).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )

    @staticmethod
    def _quality_recommendation(cv: Decimal) -> str:
        """Generate recommendation based on current CV."""
        cv_float = float(cv)
        if cv_float < 10:
            return "Uncertainty is well-controlled. Maintain current data quality."
        if cv_float < 25:
            return (
                "Moderate uncertainty. Consider upgrading the largest "
                "emission sources from Tier 1 to Tier 2 (activity data + published EFs)."
            )
        if cv_float < 50:
            return (
                "High uncertainty. Prioritize Tier 2/3 data for top-5 emission "
                "sources. Focus on Scope 3 categories with highest variance contribution."
            )
        return (
            "Very high uncertainty. Most data appears to be Tier 1 (estimated). "
            "Implement a data quality improvement plan targeting the top emission "
            "sources for supplier-specific or direct measurement data."
        )
