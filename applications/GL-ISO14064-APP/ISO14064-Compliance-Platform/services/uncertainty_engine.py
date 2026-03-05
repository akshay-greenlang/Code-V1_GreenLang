"""
Uncertainty Engine -- ISO 14064-1:2018 Clause 6.3 Implementation

Implements uncertainty quantification for ISO 14064-1 GHG inventories using
Monte Carlo simulation and analytical (IPCC Tier 1) error propagation.

Supports:
  - Monte Carlo simulation with configurable N (default 10,000)
  - Analytical method (IPCC Tier 1 root-sum-of-squares)
  - Lognormal distributions for emission factors
  - Normal distributions for activity data
  - Combined uncertainty across all six ISO categories
  - Per-source uncertainty contribution ranking (sensitivity)
  - Multiple confidence intervals (90%, 95%, 99%)
  - Convergence assessment (coefficient of variation stabilization)

Reference: ISO 14064-1:2018 Clause 6.3 and IPCC 2006 Guidelines Vol 1 Ch 3.

Example:
    >>> engine = UncertaintyEngine(config)
    >>> result = engine.run_monte_carlo("inv-1", category_results)
    >>> print(f"95% CI: [{result.lower_bound}, {result.upper_bound}]")
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
    ISO14064AppConfig,
    ISOCategory,
    ISO_CATEGORY_NAMES,
    UNCERTAINTY_CV_BY_TIER,
)
from .models import (
    CategoryResult,
    UncertaintyResult,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


class UncertaintyEngine:
    """
    Uncertainty quantification engine for ISO 14064-1 inventories.

    Implements Monte Carlo simulation and analytical propagation
    per ISO 14064-1:2018 Clause 6.3.
    """

    TIER_CV_MAP: Dict[DataQualityTier, float] = {
        DataQualityTier.TIER_1: 50.0,
        DataQualityTier.TIER_2: 20.0,
        DataQualityTier.TIER_3: 5.0,
        DataQualityTier.TIER_4: 2.0,
    }

    def __init__(
        self,
        config: Optional[ISO14064AppConfig] = None,
    ) -> None:
        """
        Initialize UncertaintyEngine.

        Args:
            config: Application configuration.
        """
        self.config = config or ISO14064AppConfig()
        self._default_iterations = self.config.monte_carlo_iterations
        self._confidence_levels = self.config.confidence_levels
        self._rng = np.random.default_rng(seed=42)
        logger.info(
            "UncertaintyEngine initialized (iterations=%d, confidence=%s)",
            self._default_iterations,
            self._confidence_levels,
        )

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def run_monte_carlo(
        self,
        inventory_id: str,
        category_results: Dict[str, CategoryResult],
        iterations: Optional[int] = None,
        seed: Optional[int] = None,
        confidence_level: int = 95,
    ) -> UncertaintyResult:
        """
        Run Monte Carlo simulation across all categories.

        Args:
            inventory_id: Inventory ID.
            category_results: CategoryResult objects keyed by category value.
            iterations: Number of iterations.
            seed: Random seed for reproducibility.
            confidence_level: Primary confidence level (default 95%).

        Returns:
            UncertaintyResult with full statistical summary.
        """
        start = datetime.utcnow()
        n = iterations or self._default_iterations
        rng = np.random.default_rng(seed=seed) if seed is not None else self._rng

        logger.info(
            "Starting Monte Carlo for inventory %s (%d iterations)",
            inventory_id, n,
        )

        cat_samples: Dict[str, np.ndarray] = {}
        by_category: Dict[str, Dict[str, Decimal]] = {}

        for cat_key, cat_result in category_results.items():
            if cat_result.total_tco2e <= 0:
                continue

            samples = self._simulate_category(cat_result, n, rng)
            cat_samples[cat_key] = samples

            cat_mean = float(np.mean(samples))
            cat_std = float(np.std(samples))
            cat_cv = (cat_std / cat_mean * 100) if cat_mean > 0 else 0.0
            by_category[cat_key] = {
                "mean": Decimal(str(round(cat_mean, 2))),
                "std_dev": Decimal(str(round(cat_std, 2))),
                "cv_pct": Decimal(str(round(cat_cv, 1))),
                "p5": Decimal(str(round(float(np.percentile(samples, 5)), 2))),
                "p95": Decimal(str(round(float(np.percentile(samples, 95)), 2))),
            }

        combined = self._combine_samples(cat_samples, n)
        mean_val, std_val, cv_val = self._compute_stats(combined)

        # Compute bounds at requested confidence level
        lower_pct = (100 - confidence_level) / 2
        upper_pct = 100 - lower_pct
        lower_bound = float(np.percentile(combined, lower_pct)) if len(combined) > 0 else 0.0
        upper_bound = float(np.percentile(combined, upper_pct)) if len(combined) > 0 else 0.0

        # Sensitivity ranking
        sensitivity = self._compute_sensitivity(cat_samples, combined)

        result = UncertaintyResult(
            inventory_id=inventory_id,
            scope="total",
            lower_bound=Decimal(str(round(lower_bound, 2))),
            central_estimate=Decimal(str(round(mean_val, 2))),
            upper_bound=Decimal(str(round(upper_bound, 2))),
            confidence_level=confidence_level,
            std_dev=Decimal(str(round(std_val, 2))),
            cv_pct=Decimal(str(round(cv_val, 1))),
            methodology="monte_carlo",
            iterations=n,
            by_category=by_category,
            sensitivity_ranking=sensitivity,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Monte Carlo complete for %s: mean=%.2f, CV=%.1f%% in %.1f ms",
            inventory_id, mean_val, cv_val, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Analytical
    # ------------------------------------------------------------------

    def run_analytical(
        self,
        inventory_id: str,
        category_results: Dict[str, CategoryResult],
        confidence_level: int = 95,
    ) -> UncertaintyResult:
        """
        IPCC Tier 1 analytical error propagation.

        Uses root-sum-of-squares to combine independent category uncertainties.

        Args:
            inventory_id: Inventory ID.
            category_results: Category results keyed by category value.
            confidence_level: Confidence level for bounds.

        Returns:
            UncertaintyResult with analytical bounds.
        """
        start = datetime.utcnow()

        total_emissions = Decimal("0")
        sum_var = Decimal("0")
        by_category: Dict[str, Dict[str, Decimal]] = {}

        for cat_key, cat_result in category_results.items():
            if cat_result.total_tco2e <= 0:
                continue

            cv_pct = Decimal(str(
                self.TIER_CV_MAP.get(cat_result.data_quality_tier, 50.0)
            ))
            mean = cat_result.total_tco2e
            std_dev = mean * (cv_pct / Decimal("100"))

            total_emissions += mean
            sum_var += std_dev ** 2

            by_category[cat_key] = {
                "mean": mean,
                "std_dev": std_dev.quantize(Decimal("0.01")),
                "cv_pct": cv_pct,
            }

        combined_std = Decimal("0")
        if sum_var > 0:
            combined_std = sum_var.sqrt()

        cv_percent = Decimal("0")
        if total_emissions > 0:
            cv_percent = (combined_std / total_emissions * 100).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP,
            )

        # Compute bounds
        z = self._z_score_for_level(confidence_level)
        z_dec = Decimal(str(z))
        half_width = z_dec * combined_std
        lower = max(total_emissions - half_width, Decimal("0"))
        upper = total_emissions + half_width

        result = UncertaintyResult(
            inventory_id=inventory_id,
            scope="total",
            lower_bound=lower.quantize(Decimal("0.01")),
            central_estimate=total_emissions.quantize(Decimal("0.01")),
            upper_bound=upper.quantize(Decimal("0.01")),
            confidence_level=confidence_level,
            std_dev=combined_std.quantize(Decimal("0.01")),
            cv_pct=cv_percent,
            methodology="error_propagation",
            iterations=0,
            by_category=by_category,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Analytical uncertainty for %s: mean=%.2f, CV=%.1f%% in %.1f ms",
            inventory_id, total_emissions, cv_percent, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Per-Category
    # ------------------------------------------------------------------

    def uncertainty_by_category(
        self,
        category_result: CategoryResult,
        iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute uncertainty for a single category."""
        n = iterations or self._default_iterations
        samples = self._simulate_category(category_result, n, self._rng)
        mean_val, std_val, cv_val = self._compute_stats(samples)

        return {
            "category": category_result.iso_category.value,
            "mean": round(mean_val, 2),
            "std_dev": round(std_val, 2),
            "cv_percent": round(cv_val, 1),
            "p5": round(float(np.percentile(samples, 5)), 2),
            "p50": round(float(np.percentile(samples, 50)), 2),
            "p95": round(float(np.percentile(samples, 95)), 2),
            "iterations": n,
        }

    def get_sensitivity_ranking(
        self,
        category_results: Dict[str, CategoryResult],
        iterations: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rank categories by variance contribution to total uncertainty."""
        n = iterations or min(self._default_iterations, 5000)
        rng = self._rng

        cat_samples: Dict[str, np.ndarray] = {}
        for cat_key, cat_result in category_results.items():
            if cat_result.total_tco2e > 0:
                cat_samples[cat_key] = self._simulate_category(cat_result, n, rng)

        combined = self._combine_samples(cat_samples, n)
        return self._compute_sensitivity(cat_samples, combined)

    # ------------------------------------------------------------------
    # Simulation Core
    # ------------------------------------------------------------------

    def _simulate_category(
        self,
        cat_result: CategoryResult,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate emissions for a single category."""
        cv = self.TIER_CV_MAP.get(cat_result.data_quality_tier, 50.0) / 100.0

        if cat_result.sources:
            total_samples = np.zeros(n)
            for source in cat_result.sources:
                fval = float(source.total_tco2e)
                if fval > 0:
                    total_samples += self._sample_lognormal(fval, cv, n, rng)
            return total_samples

        total = float(cat_result.total_tco2e)
        return self._sample_lognormal(total, cv, n, rng)

    @staticmethod
    def _sample_lognormal(
        mean: float, cv: float, n: int, rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample from lognormal given mean and CV."""
        if mean <= 0 or cv <= 0:
            return np.full(n, max(mean, 0.0))
        sigma2 = math.log(1 + cv ** 2)
        sigma = math.sqrt(sigma2)
        mu = math.log(mean) - sigma2 / 2
        return rng.lognormal(mu, sigma, n)

    @staticmethod
    def _combine_samples(
        cat_samples: Dict[str, np.ndarray], n: int,
    ) -> np.ndarray:
        """Sum samples across all categories."""
        combined = np.zeros(n)
        for samples in cat_samples.values():
            if len(samples) == n:
                combined += samples
        return combined

    @staticmethod
    def _compute_stats(
        samples: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute mean, std_dev, cv from samples."""
        if len(samples) == 0:
            return 0.0, 0.0, 0.0
        mean_val = float(np.mean(samples))
        std_val = float(np.std(samples))
        cv_val = (std_val / mean_val * 100) if mean_val > 0 else 0.0
        return mean_val, std_val, cv_val

    @staticmethod
    def _compute_sensitivity(
        cat_samples: Dict[str, np.ndarray],
        combined: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Compute sensitivity ranking by variance contribution."""
        total_var = float(np.var(combined))
        if total_var == 0:
            return []

        ranking: List[Dict[str, Any]] = []
        for cat_key, samples in cat_samples.items():
            cat_var = float(np.var(samples))
            contribution_pct = (cat_var / total_var * 100) if total_var > 0 else 0.0
            cat_mean = float(np.mean(samples))

            ranking.append({
                "category": cat_key,
                "mean_tco2e": round(cat_mean, 2),
                "variance_contribution_pct": round(contribution_pct, 1),
            })

        ranking.sort(
            key=lambda r: r["variance_contribution_pct"],
            reverse=True,
        )
        return ranking

    @staticmethod
    def _assess_convergence(combined: np.ndarray) -> Optional[float]:
        """Assess convergence via running mean CV in last 10%."""
        if len(combined) < 100:
            return None
        cumulative_mean = np.cumsum(combined) / np.arange(1, len(combined) + 1)
        tail_start = int(len(cumulative_mean) * 0.9)
        tail = cumulative_mean[tail_start:]
        if len(tail) < 2:
            return None
        t_mean = float(np.mean(tail))
        t_std = float(np.std(tail))
        return round((t_std / t_mean * 100) if t_mean > 0 else 0.0, 3)

    @staticmethod
    def _z_score_for_level(level: int) -> float:
        """Return z-score for confidence level."""
        return {90: 1.645, 95: 1.960, 99: 2.576}.get(level, 1.960)
