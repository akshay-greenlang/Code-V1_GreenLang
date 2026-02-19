# -*- coding: utf-8 -*-
"""
UncertaintyQuantifierEngine - Monte Carlo & DQI Scoring (Engine 5 of 7)

AGENT-MRV-006: Land Use Emissions Agent

Quantifies the uncertainty of land-use emission calculations using
Monte Carlo simulation, analytical error propagation (IPCC Approach 1),
data quality indicator (DQI) scoring, and sensitivity analysis.

Process-Specific Uncertainty Ranges (half-width of 95% CI):
    - AGB stock: +/-30% (Tier 1), +/-15% (Tier 2), +/-5% (Tier 3)
    - SOC reference: +/-50% (Tier 1), +/-25% (Tier 2)
    - Root-to-shoot ratio: +/-30%
    - Biomass growth rate: +/-40%
    - Emission factors: varies by source (10-100%)
    - Area: +/-5% (uniform distribution)
    - Fire combustion factor: +/-40%
    - Peatland EF: +/-50-100%

DQI Scoring (5 dimensions, 1-5 scale):
    - Reliability: forest inventory (1) ... global default (5)
    - Completeness: >95% area coverage (1) ... <40% (5)
    - Temporal correlation: same year (1) ... >10 years old (5)
    - Geographical correlation: same site (1) ... global default (5)
    - Technological correlation: species-specific (1) ... generic (5)

Monte Carlo Simulation:
    - Configurable iterations (default 5000)
    - Normal distributions for stock values (bounded at zero)
    - Lognormal distributions for emission factors (non-negative)
    - Uniform distributions for area
    - Explicit seed support for full reproducibility

Analytical Propagation (IPCC Approach 1):
    Combined relative uncertainty for multiplicative chains:
    U_total = sqrt(sum(Ui^2)) for uncorrelated parameters.

Zero-Hallucination Guarantees:
    - All formulas are deterministic mathematical operations.
    - PRNG seeded explicitly for full reproducibility.
    - Every result carries a SHA-256 provenance hash.

Thread Safety:
    Monte Carlo simulations create per-call Random instances so
    concurrent callers never interfere.

Example:
    >>> from greenlang.land_use_emissions.uncertainty_quantifier import (
    ...     UncertaintyQuantifierEngine,
    ... )
    >>> engine = UncertaintyQuantifierEngine()
    >>> result = engine.run_monte_carlo({
    ...     "total_co2e_tonnes": 5000,
    ...     "parameters": [
    ...         {"name": "agb", "value": 180, "cv_pct": 30, "dist": "normal"},
    ...         {"name": "area_ha", "value": 1000, "cv_pct": 5, "dist": "uniform"},
    ...     ],
    ... })
    >>> print(result["confidence_intervals"]["95"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Land Use Emissions (GL-MRV-SCOPE1-006)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["UncertaintyQuantifierEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.land_use_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.land_use_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Decimal helpers
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert to Decimal."""
    if value is None:
        return default
    try:
        return _D(value)
    except (InvalidOperation, ValueError, TypeError):
        return default


# ===========================================================================
# Default Uncertainty Parameters for LULUCF
# ===========================================================================

#: Default coefficient of variation (CV%) by parameter type and tier.
DEFAULT_CV: Dict[str, Dict[str, float]] = {
    "AGB": {"TIER_1": 30.0, "TIER_2": 15.0, "TIER_3": 5.0},
    "BGB": {"TIER_1": 40.0, "TIER_2": 20.0, "TIER_3": 8.0},
    "DEAD_WOOD": {"TIER_1": 50.0, "TIER_2": 30.0, "TIER_3": 10.0},
    "LITTER": {"TIER_1": 50.0, "TIER_2": 30.0, "TIER_3": 10.0},
    "SOC_REF": {"TIER_1": 50.0, "TIER_2": 25.0, "TIER_3": 10.0},
    "ROOT_SHOOT_RATIO": {"TIER_1": 30.0, "TIER_2": 15.0, "TIER_3": 5.0},
    "GROWTH_RATE": {"TIER_1": 40.0, "TIER_2": 20.0, "TIER_3": 8.0},
    "EMISSION_FACTOR": {"TIER_1": 50.0, "TIER_2": 25.0, "TIER_3": 10.0},
    "AREA": {"TIER_1": 5.0, "TIER_2": 3.0, "TIER_3": 1.0},
    "COMBUSTION_FACTOR": {"TIER_1": 40.0, "TIER_2": 20.0, "TIER_3": 8.0},
    "FIRE_EF": {"TIER_1": 50.0, "TIER_2": 25.0, "TIER_3": 10.0},
    "PEATLAND_EF": {"TIER_1": 90.0, "TIER_2": 50.0, "TIER_3": 20.0},
    "SOC_FACTOR_FLU": {"TIER_1": 30.0, "TIER_2": 15.0, "TIER_3": 5.0},
    "SOC_FACTOR_FMG": {"TIER_1": 30.0, "TIER_2": 15.0, "TIER_3": 5.0},
    "SOC_FACTOR_FI": {"TIER_1": 30.0, "TIER_2": 15.0, "TIER_3": 5.0},
    "N2O_EF": {"TIER_1": 75.0, "TIER_2": 40.0, "TIER_3": 15.0},
}


# ===========================================================================
# UncertaintyQuantifierEngine
# ===========================================================================


class UncertaintyQuantifierEngine:
    """Monte Carlo uncertainty quantification for LULUCF calculations.

    Provides Monte Carlo simulation, analytical error propagation,
    DQI scoring, confidence interval extraction, and sensitivity analysis.

    Thread Safety:
        Monte Carlo uses per-call Random instances. Shared counters
        are protected by a reentrant lock.

    Attributes:
        _default_iterations: Default Monte Carlo iterations.
        _default_seed: Default random seed.
        _lock: Reentrant lock.
        _total_analyses: Counter.

    Example:
        >>> engine = UncertaintyQuantifierEngine()
        >>> result = engine.run_monte_carlo(request)
    """

    def __init__(
        self,
        default_iterations: int = 5000,
        default_seed: int = 42,
    ) -> None:
        """Initialize the UncertaintyQuantifierEngine.

        Args:
            default_iterations: Default Monte Carlo iterations.
            default_seed: Default random seed for reproducibility.
        """
        self._default_iterations = default_iterations
        self._default_seed = default_seed
        self._lock = threading.RLock()
        self._total_analyses: int = 0
        self._created_at = _utcnow()

        logger.info(
            "UncertaintyQuantifierEngine initialized: "
            "iterations=%d, seed=%d",
            default_iterations, default_seed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _increment_analyses(self) -> None:
        """Thread-safe increment of the analysis counter."""
        with self._lock:
            self._total_analyses += 1

    def _generate_samples(
        self,
        rng: random.Random,
        value: float,
        cv_pct: float,
        distribution: str,
        n: int,
    ) -> List[float]:
        """Generate random samples for a parameter.

        Args:
            rng: Random number generator.
            value: Central/mean value.
            cv_pct: Coefficient of variation as percentage.
            distribution: Distribution type (normal, lognormal, uniform).
            n: Number of samples.

        Returns:
            List of sampled values.
        """
        std_dev = abs(value) * (cv_pct / 100.0)

        if distribution == "lognormal":
            # Parameterize lognormal from mean and CV
            if value <= 0:
                return [0.0] * n
            sigma2 = math.log(1 + (cv_pct / 100.0) ** 2)
            sigma = math.sqrt(sigma2)
            mu = math.log(value) - sigma2 / 2
            return [rng.lognormvariate(mu, sigma) for _ in range(n)]

        elif distribution == "uniform":
            low = value * (1 - cv_pct / 100.0)
            high = value * (1 + cv_pct / 100.0)
            return [rng.uniform(low, high) for _ in range(n)]

        elif distribution == "triangular":
            low = value * (1 - cv_pct / 100.0)
            high = value * (1 + cv_pct / 100.0)
            return [rng.triangular(low, high, value) for _ in range(n)]

        else:
            # Normal distribution, bounded at zero for physical quantities
            samples = []
            for _ in range(n):
                sample = rng.gauss(value, std_dev)
                samples.append(max(0.0, sample))
            return samples

    def _calculate_percentiles(
        self,
        values: List[float],
        percentiles: List[float],
    ) -> Dict[str, float]:
        """Calculate percentiles from a list of values.

        Args:
            values: Sorted list of values.
            percentiles: List of percentile points (0-100).

        Returns:
            Dictionary mapping percentile labels to values.
        """
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        result: Dict[str, float] = {}

        for p in percentiles:
            idx = (p / 100.0) * (n - 1)
            lower = int(math.floor(idx))
            upper = min(lower + 1, n - 1)
            frac = idx - lower
            val = sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac
            result[str(int(p)) if p == int(p) else str(p)] = round(val, 8)

        return result

    # ------------------------------------------------------------------
    # Monte Carlo Simulation
    # ------------------------------------------------------------------

    def run_monte_carlo(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for uncertainty quantification.

        The simulation perturbs each parameter according to its distribution
        and CV, recalculates the total result for each iteration, and
        derives confidence intervals from the resulting distribution.

        Required request keys:
            total_co2e_tonnes: Central estimate of total CO2e.
            parameters: List of parameter dictionaries, each with:
                name: Parameter name.
                value: Central value.
                cv_pct: Coefficient of variation (%).
                dist: Distribution type (normal, lognormal, uniform).

        Optional keys:
            n_iterations: Number of iterations (default from config).
            seed: Random seed (default from config).
            confidence_levels: List of confidence levels (default [90,95,99]).
            calculation_type: Type of calculation (for CV defaults).

        Args:
            request: Monte Carlo request dictionary.

        Returns:
            Simulation results with confidence intervals and statistics.
        """
        self._increment_analyses()
        start_time = time.monotonic()
        calc_id = str(uuid4())

        total_co2e = float(request.get("total_co2e_tonnes", 0))
        parameters = request.get("parameters", [])
        n_iterations = int(request.get("n_iterations", self._default_iterations))
        seed = int(request.get("seed", self._default_seed))
        confidence_levels = request.get("confidence_levels", [90, 95, 99])
        tier = str(request.get("tier", "TIER_1")).upper()

        # -- Validate -------------------------------------------------------
        errors: List[str] = []
        if total_co2e == 0:
            errors.append("total_co2e_tonnes must be non-zero")
        if not parameters:
            errors.append("At least one parameter is required")
        if n_iterations <= 0:
            errors.append("n_iterations must be > 0")
        if n_iterations > 1_000_000:
            errors.append("n_iterations must be <= 1000000")

        if errors:
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "errors": errors,
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        # -- Generate samples per parameter --------------------------------
        rng = random.Random(seed)
        param_samples: Dict[str, List[float]] = {}

        for param in parameters:
            name = str(param.get("name", ""))
            value = float(param.get("value", 0))
            cv_pct = float(param.get("cv_pct", 30))
            dist = str(param.get("dist", "normal")).lower()

            # Use default CV if not specified
            if cv_pct == 0 and name.upper() in DEFAULT_CV:
                cv_pct = DEFAULT_CV[name.upper()].get(tier, 30.0)

            samples = self._generate_samples(rng, value, cv_pct, dist, n_iterations)
            param_samples[name] = samples

        # -- Simulate total results ----------------------------------------
        # For multiplicative model: result = product(params)
        # Scale each iteration proportionally
        total_results: List[float] = []

        for i in range(n_iterations):
            # Calculate scaling factor from parameter perturbations
            scale = 1.0
            for param in parameters:
                name = str(param.get("name", ""))
                value = float(param.get("value", 1))
                if value != 0 and name in param_samples:
                    scale *= param_samples[name][i] / value

            total_results.append(total_co2e * scale)

        # -- Calculate statistics ------------------------------------------
        sorted_results = sorted(total_results)
        n = len(sorted_results)
        mean_val = sum(sorted_results) / n
        variance = sum((x - mean_val) ** 2 for x in sorted_results) / (n - 1)
        std_dev = math.sqrt(variance)
        cv_result = (std_dev / abs(mean_val) * 100) if mean_val != 0 else 0

        # Percentiles
        percentile_points = [5, 10, 25, 50, 75, 90, 95]
        percentiles = self._calculate_percentiles(sorted_results, percentile_points)

        # Confidence intervals
        ci: Dict[str, Dict[str, float]] = {}
        for level in confidence_levels:
            alpha = (100 - level) / 2
            lower = self._calculate_percentiles(sorted_results, [alpha])
            upper = self._calculate_percentiles(sorted_results, [100 - alpha])
            lower_key = str(alpha) if alpha != int(alpha) else str(int(alpha))
            upper_key = str(100 - alpha) if (100 - alpha) != int(100 - alpha) else str(int(100 - alpha))
            ci[str(int(level))] = {
                "lower": lower.get(lower_key, 0),
                "upper": upper.get(upper_key, 0),
                "half_width": round(
                    (upper.get(upper_key, 0) - lower.get(lower_key, 0)) / 2, 8
                ),
                "relative_pct": round(
                    (upper.get(upper_key, 0) - lower.get(lower_key, 0))
                    / (2 * abs(mean_val)) * 100, 2
                ) if mean_val != 0 else 0,
            }

        # -- Per-parameter statistics --------------------------------------
        param_stats: Dict[str, Dict[str, Any]] = {}
        for param in parameters:
            name = str(param.get("name", ""))
            if name in param_samples:
                samples = param_samples[name]
                p_mean = sum(samples) / len(samples)
                p_var = sum((x - p_mean) ** 2 for x in samples) / (len(samples) - 1)
                p_std = math.sqrt(p_var)
                param_stats[name] = {
                    "mean": round(p_mean, 8),
                    "std_dev": round(p_std, 8),
                    "cv_pct": round(
                        (p_std / abs(p_mean) * 100) if p_mean != 0 else 0, 2
                    ),
                    "min": round(min(samples), 8),
                    "max": round(max(samples), 8),
                }

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "MONTE_CARLO",
            "n_iterations": n_iterations,
            "seed": seed,
            "central_estimate": total_co2e,
            "statistics": {
                "mean": round(mean_val, 8),
                "std_dev": round(std_dev, 8),
                "cv_pct": round(cv_result, 2),
                "min": round(sorted_results[0], 8),
                "max": round(sorted_results[-1], 8),
                "median": percentiles.get("50", 0),
            },
            "percentiles": percentiles,
            "confidence_intervals": ci,
            "parameter_statistics": param_stats,
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Monte Carlo complete: id=%s, n=%d, mean=%.2f, "
            "cv=%.1f%%, 95%% CI=[%.2f, %.2f], time=%.3fms",
            calc_id, n_iterations, mean_val, cv_result,
            ci.get("95", {}).get("lower", 0),
            ci.get("95", {}).get("upper", 0),
            processing_time,
        )
        return result

    # ------------------------------------------------------------------
    # Analytical Error Propagation
    # ------------------------------------------------------------------

    def run_error_propagation(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run IPCC Approach 1 analytical error propagation.

        For uncorrelated multiplicative parameters:
            U_total = sqrt(sum(Ui^2))
        Where Ui is the relative uncertainty (half-width of 95% CI / mean).

        For additive parameters:
            U_total = sqrt(sum((Ui * Xi)^2)) / |sum(Xi)|

        Required request keys:
            total_co2e_tonnes: Central estimate.
            parameters: List of {name, value, uncertainty_pct} dicts.
            combination: "multiplicative" or "additive".

        Args:
            request: Error propagation request.

        Returns:
            Combined uncertainty with 95% CI.
        """
        self._increment_analyses()
        start_time = time.monotonic()
        calc_id = str(uuid4())

        total_co2e = _safe_decimal(request.get("total_co2e_tonnes"), _ZERO)
        parameters = request.get("parameters", [])
        combination = str(request.get("combination", "multiplicative")).lower()

        if not parameters:
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "errors": ["At least one parameter is required"],
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        if combination == "additive":
            # Additive: U = sqrt(sum((Ui * Xi)^2)) / |sum(Xi)|
            sum_x = _ZERO
            sum_ux_sq = _ZERO

            for param in parameters:
                x = _safe_decimal(param.get("value"), _ZERO)
                u_pct = _safe_decimal(param.get("uncertainty_pct"), _ZERO)
                u_abs = (x * u_pct / _HUNDRED)
                sum_x += x
                sum_ux_sq += u_abs ** 2

            if sum_x != _ZERO:
                combined_abs = _D(str(
                    math.sqrt(float(sum_ux_sq))
                ))
                combined_pct = (combined_abs / abs(sum_x) * _HUNDRED).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            else:
                combined_abs = _ZERO
                combined_pct = _ZERO

        else:
            # Multiplicative: U = sqrt(sum(Ui^2))
            sum_u_sq = _ZERO
            for param in parameters:
                u_pct = _safe_decimal(param.get("uncertainty_pct"), _ZERO)
                sum_u_sq += (u_pct / _HUNDRED) ** 2

            combined_pct = (
                _D(str(math.sqrt(float(sum_u_sq)))) * _HUNDRED
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            combined_abs = (
                total_co2e * combined_pct / _HUNDRED
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # 95% CI
        ci_lower = (total_co2e - combined_abs * _D("1.96")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        ci_upper = (total_co2e + combined_abs * _D("1.96")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "ERROR_PROPAGATION",
            "combination": combination,
            "central_estimate": str(total_co2e),
            "combined_uncertainty_pct": str(combined_pct),
            "combined_uncertainty_abs": str(combined_abs),
            "confidence_interval_95": {
                "lower": str(ci_lower),
                "upper": str(ci_upper),
            },
            "parameter_count": len(parameters),
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Error propagation: id=%s, combined_u=%.1f%%, "
            "95%% CI=[%s, %s], time=%.3fms",
            calc_id, float(combined_pct), ci_lower, ci_upper, processing_time,
        )
        return result

    # ------------------------------------------------------------------
    # DQI Scoring
    # ------------------------------------------------------------------

    def calculate_dqi(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate Data Quality Indicator score.

        Scores 5 dimensions on a 1-5 scale (1 = best, 5 = worst).
        Composite = geometric mean of dimension scores.

        Required request keys:
            reliability: 1-5 score.
            completeness: 1-5 score.
            temporal_correlation: 1-5 score.
            geographical_correlation: 1-5 score.
            technological_correlation: 1-5 score.

        Args:
            request: DQI scoring request.

        Returns:
            Per-dimension and composite DQI scores.
        """
        self._increment_analyses()
        start_time = time.monotonic()
        calc_id = str(uuid4())

        dimensions = [
            "reliability",
            "completeness",
            "temporal_correlation",
            "geographical_correlation",
            "technological_correlation",
        ]

        scores: Dict[str, int] = {}
        errors: List[str] = []

        for dim in dimensions:
            val = request.get(dim, 0)
            try:
                score = int(val)
                if score < 1 or score > 5:
                    errors.append(f"{dim} must be 1-5, got {score}")
                else:
                    scores[dim] = score
            except (ValueError, TypeError):
                errors.append(f"{dim} must be an integer 1-5, got {val}")

        if errors:
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "errors": errors,
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        # Geometric mean
        product = 1.0
        for score in scores.values():
            product *= score
        composite = product ** (1.0 / len(scores))

        # Quality category
        if composite <= 1.5:
            quality = "EXCELLENT"
        elif composite <= 2.5:
            quality = "GOOD"
        elif composite <= 3.5:
            quality = "FAIR"
        elif composite <= 4.5:
            quality = "POOR"
        else:
            quality = "VERY_POOR"

        # Map DQI to approximate uncertainty multiplier
        dqi_to_uncertainty: Dict[str, float] = {
            "EXCELLENT": 0.8,
            "GOOD": 1.0,
            "FAIR": 1.3,
            "POOR": 1.8,
            "VERY_POOR": 2.5,
        }
        uncertainty_multiplier = dqi_to_uncertainty.get(quality, 1.0)

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "dimension_scores": scores,
            "composite_score": round(composite, 4),
            "quality_category": quality,
            "uncertainty_multiplier": uncertainty_multiplier,
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "DQI calculated: id=%s, composite=%.2f, quality=%s, time=%.3fms",
            calc_id, composite, quality, processing_time,
        )
        return result

    # ------------------------------------------------------------------
    # Confidence Intervals
    # ------------------------------------------------------------------

    def get_confidence_interval(
        self,
        mean: float,
        std_dev: float,
        confidence_level: float = 95.0,
        n_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Calculate a confidence interval from mean and standard deviation.

        Uses z-scores for large samples and t-approximation for small.

        Args:
            mean: Sample mean.
            std_dev: Sample standard deviation.
            confidence_level: Confidence level (e.g. 95.0).
            n_samples: Number of samples (for t-distribution).

        Returns:
            Dictionary with lower, upper, and half_width.
        """
        z_scores: Dict[float, float] = {
            90.0: 1.645,
            95.0: 1.960,
            99.0: 2.576,
        }
        z = z_scores.get(confidence_level, 1.960)

        if n_samples and n_samples < 30:
            # Approximate t-correction for small samples
            z = z * (1 + 1 / (4 * max(n_samples - 1, 1)))

        half_width = z * std_dev
        if n_samples:
            half_width = z * std_dev / math.sqrt(n_samples)

        return {
            "lower": round(mean - half_width, 8),
            "upper": round(mean + half_width, 8),
            "half_width": round(half_width, 8),
            "confidence_level": confidence_level,
        }

    # ------------------------------------------------------------------
    # Percentiles
    # ------------------------------------------------------------------

    def get_percentiles(
        self,
        values: List[float],
        percentile_points: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Calculate percentiles from a list of values.

        Args:
            values: List of numeric values.
            percentile_points: Percentile points (default [5,25,50,75,95]).

        Returns:
            Dictionary mapping percentile labels to values.
        """
        if percentile_points is None:
            percentile_points = [5, 25, 50, 75, 95]

        return self._calculate_percentiles(values, percentile_points)

    # ------------------------------------------------------------------
    # Sensitivity Analysis
    # ------------------------------------------------------------------

    def run_sensitivity_analysis(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run sensitivity analysis to identify key uncertainty drivers.

        Perturbs each parameter independently by +/-1 standard deviation
        and measures the impact on the total result. Results are ranked
        by contribution to total variance (tornado chart data).

        Required request keys:
            total_co2e_tonnes: Central estimate.
            parameters: List of {name, value, cv_pct} dicts.

        Args:
            request: Sensitivity analysis request.

        Returns:
            Ranked parameter sensitivities with tornado chart data.
        """
        self._increment_analyses()
        start_time = time.monotonic()
        calc_id = str(uuid4())

        total_co2e = float(request.get("total_co2e_tonnes", 0))
        parameters = request.get("parameters", [])

        if not parameters or total_co2e == 0:
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "errors": ["total_co2e_tonnes and parameters are required"],
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        sensitivities: List[Dict[str, Any]] = []
        total_variance_contribution = 0.0

        for param in parameters:
            name = str(param.get("name", ""))
            value = float(param.get("value", 0))
            cv_pct = float(param.get("cv_pct", 30))

            if value == 0:
                continue

            std_dev = abs(value) * cv_pct / 100.0

            # Perturb +1 std_dev
            perturbed_high = value + std_dev
            scale_high = perturbed_high / value
            result_high = total_co2e * scale_high

            # Perturb -1 std_dev
            perturbed_low = max(0, value - std_dev)
            scale_low = perturbed_low / value if value != 0 else 1.0
            result_low = total_co2e * scale_low

            # Impact range
            impact_range = abs(result_high - result_low)
            variance_contribution = (impact_range / 2) ** 2
            total_variance_contribution += variance_contribution

            sensitivities.append({
                "parameter": name,
                "central_value": value,
                "cv_pct": cv_pct,
                "result_low": round(result_low, 8),
                "result_high": round(result_high, 8),
                "impact_range": round(impact_range, 8),
                "variance_contribution": round(variance_contribution, 8),
            })

        # Rank by variance contribution
        sensitivities.sort(key=lambda x: x["variance_contribution"], reverse=True)

        # Add percentage contribution
        for s in sensitivities:
            if total_variance_contribution > 0:
                s["contribution_pct"] = round(
                    s["variance_contribution"] / total_variance_contribution * 100, 2
                )
            else:
                s["contribution_pct"] = 0

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "SENSITIVITY_ANALYSIS",
            "central_estimate": total_co2e,
            "sensitivities": sensitivities,
            "total_variance": round(total_variance_contribution, 8),
            "top_driver": sensitivities[0]["parameter"] if sensitivities else None,
            "parameter_count": len(sensitivities),
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Sensitivity analysis: id=%s, params=%d, "
            "top_driver=%s, time=%.3fms",
            calc_id, len(sensitivities),
            result["top_driver"], processing_time,
        )
        return result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine usage statistics."""
        with self._lock:
            return {
                "engine": "UncertaintyQuantifierEngine",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_analyses": self._total_analyses,
                "default_iterations": self._default_iterations,
                "default_seed": self._default_seed,
            }

    def reset(self) -> None:
        """Reset engine counters."""
        with self._lock:
            self._total_analyses = 0
        logger.info("UncertaintyQuantifierEngine reset")
