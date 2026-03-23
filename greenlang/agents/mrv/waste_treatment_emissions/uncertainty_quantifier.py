# -*- coding: utf-8 -*-
"""
UncertaintyQuantifierEngine - Monte Carlo & DQI Scoring (Engine 5 of 7)

AGENT-MRV-007: On-site Waste Treatment Emissions Agent

Quantifies the uncertainty of waste treatment emission calculations using
Monte Carlo simulation, analytical error propagation (IPCC Approach 1),
data quality indicator (DQI) scoring, and sensitivity analysis.

Process-Specific Uncertainty Ranges (half-width of 95% CI):
    - DOC (degradable organic carbon): +/-20% (normal)
    - MCF (methane correction factor): +/-15% (uniform)
    - DOC_f (fraction DOC dissimilated): triangular(0.4, 0.5, 0.7)
    - Oxidation factor: uniform(0.9, 1.0)
    - Collection efficiency: triangular(0.50, 0.75, 0.95)
    - Flare DRE: uniform(0.96, 0.995)
    - Composting EF (CH4): +/-50% (lognormal)
    - Incineration carbon content: +/-10% (normal)
    - Fossil carbon fraction: +/-10% (uniform)
    - Wastewater MCF: +/-30% (uniform)
    - Wastewater Bo: +/-25% (normal)

DQI Scoring (5 dimensions, 1-5 scale):
    - Reliability: site measurement (1) ... global default (5)
    - Completeness: >95% waste stream coverage (1) ... <40% (5)
    - Temporal correlation: same year (1) ... >10 years old (5)
    - Geographical correlation: same facility (1) ... global default (5)
    - Technological correlation: same technology (1) ... generic (5)

Monte Carlo Simulation:
    - Configurable iterations (default 5,000)
    - Normal, lognormal, uniform, triangular distributions
    - Explicit seed support for full reproducibility
    - Per-parameter distribution specifications for waste treatment

Analytical Propagation (IPCC Approach 1):
    Combined relative uncertainty for multiplicative chains:
    U_total = sqrt(sum(Ui^2)) for uncorrelated parameters.

Confidence Intervals:
    - 90% CI: z = 1.645
    - 95% CI: z = 1.960 (default)
    - 99% CI: z = 2.576

Zero-Hallucination Guarantees:
    - All formulas are deterministic mathematical operations.
    - PRNG seeded explicitly for full reproducibility.
    - Every result carries a SHA-256 provenance hash.

Thread Safety:
    Monte Carlo simulations create per-call Random instances so
    concurrent callers never interfere.

Example:
    >>> from greenlang.agents.mrv.waste_treatment_emissions.uncertainty_quantifier import (
    ...     UncertaintyQuantifierEngine,
    ... )
    >>> engine = UncertaintyQuantifierEngine()
    >>> result = engine.quantify_uncertainty({
    ...     "total_co2e_tonnes": 1200,
    ...     "parameters": [
    ...         {"name": "doc", "value": 0.15},
    ...         {"name": "mcf", "value": 1.0},
    ...         {"name": "collection_efficiency", "value": 0.75},
    ...     ],
    ... })
    >>> print(result["confidence_intervals"]["95"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-007 On-site Waste Treatment Emissions (GL-MRV-SCOPE1-007)
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
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
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
    from greenlang.agents.mrv.waste_treatment_emissions.config import (
        get_config as _get_config,
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.provenance import (
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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Any JSON-serializable data structure.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
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
    """Convert a value to Decimal safely.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert to Decimal with a fallback default.

    Args:
        value: Value to convert.
        default: Fallback if conversion fails.

    Returns:
        Decimal representation, or default on failure.
    """
    if value is None:
        return default
    try:
        return _D(value)
    except (InvalidOperation, ValueError, TypeError):
        return default


# ===========================================================================
# Waste Treatment Parameter Distributions
# ===========================================================================

#: Distribution specifications for waste treatment uncertainty parameters.
#: Each entry defines the distribution type and its shape parameters.
#: These are based on IPCC 2006 Guidelines Vol 5, GHG Protocol Waste
#: guidance, and peer-reviewed literature on waste treatment emissions.
PARAMETER_DISTRIBUTIONS: Dict[str, Dict[str, Any]] = {
    "doc": {
        "type": "normal",
        "std_pct": 20,
        "description": "Degradable organic carbon (DOC) by waste type",
    },
    "mcf": {
        "type": "uniform",
        "range_pct": 15,
        "description": "Methane correction factor for disposal site",
    },
    "doc_f": {
        "type": "triangular",
        "min": 0.4,
        "mode": 0.5,
        "max": 0.7,
        "description": "Fraction of DOC that is ultimately dissimilated",
    },
    "oxidation_factor": {
        "type": "uniform",
        "min": 0.9,
        "max": 1.0,
        "description": "Landfill oxidation factor (OX)",
    },
    "collection_efficiency": {
        "type": "triangular",
        "min": 0.50,
        "max": 0.95,
        "mode": 0.75,
        "description": "Landfill gas collection system efficiency",
    },
    "flare_dre": {
        "type": "uniform",
        "min": 0.96,
        "max": 0.995,
        "description": "Flare destruction and removal efficiency",
    },
    "composting_ef_ch4": {
        "type": "lognormal",
        "std_pct": 50,
        "description": "Composting CH4 emission factor (kg CH4/t waste)",
    },
    "incineration_carbon_content": {
        "type": "normal",
        "std_pct": 10,
        "description": "Total carbon content of incinerated waste",
    },
    "fossil_carbon_fraction": {
        "type": "uniform",
        "range_pct": 10,
        "description": "Fraction of carbon that is fossil-derived",
    },
    "wastewater_mcf": {
        "type": "uniform",
        "range_pct": 30,
        "description": "MCF for wastewater treatment systems",
    },
    "wastewater_bo": {
        "type": "normal",
        "std_pct": 25,
        "description": "Maximum CH4 producing capacity (Bo) for wastewater",
    },
}


#: Default coefficient of variation (CV%) by parameter type and tier.
#: Tier 1 uses IPCC defaults, Tier 2 uses country-specific data,
#: Tier 3 uses site-specific measurements.
DEFAULT_CV: Dict[str, Dict[str, float]] = {
    "DOC": {"TIER_1": 20.0, "TIER_2": 10.0, "TIER_3": 5.0},
    "MCF": {"TIER_1": 30.0, "TIER_2": 15.0, "TIER_3": 5.0},
    "DOC_F": {"TIER_1": 25.0, "TIER_2": 12.0, "TIER_3": 5.0},
    "OXIDATION_FACTOR": {"TIER_1": 20.0, "TIER_2": 10.0, "TIER_3": 5.0},
    "COLLECTION_EFFICIENCY": {"TIER_1": 25.0, "TIER_2": 12.0, "TIER_3": 5.0},
    "EMISSION_FACTOR": {"TIER_1": 50.0, "TIER_2": 25.0, "TIER_3": 10.0},
    "WASTE_QUANTITY": {"TIER_1": 10.0, "TIER_2": 5.0, "TIER_3": 2.0},
    "WASTE_COMPOSITION": {"TIER_1": 30.0, "TIER_2": 15.0, "TIER_3": 5.0},
    "CARBON_CONTENT": {"TIER_1": 15.0, "TIER_2": 8.0, "TIER_3": 3.0},
    "FOSSIL_FRACTION": {"TIER_1": 20.0, "TIER_2": 10.0, "TIER_3": 5.0},
    "FLARE_DRE": {"TIER_1": 10.0, "TIER_2": 5.0, "TIER_3": 2.0},
    "COMPOSTING_EF": {"TIER_1": 60.0, "TIER_2": 30.0, "TIER_3": 10.0},
    "AD_EF": {"TIER_1": 50.0, "TIER_2": 25.0, "TIER_3": 10.0},
    "INCINERATION_EF": {"TIER_1": 40.0, "TIER_2": 20.0, "TIER_3": 8.0},
    "WASTEWATER_MCF": {"TIER_1": 40.0, "TIER_2": 20.0, "TIER_3": 8.0},
    "WASTEWATER_BO": {"TIER_1": 30.0, "TIER_2": 15.0, "TIER_3": 5.0},
    "N2O_EF": {"TIER_1": 75.0, "TIER_2": 40.0, "TIER_3": 15.0},
    "GWP": {"TIER_1": 5.0, "TIER_2": 5.0, "TIER_3": 5.0},
}

#: Z-scores for common confidence levels.
Z_SCORES: Dict[float, float] = {
    90.0: 1.645,
    95.0: 1.960,
    99.0: 2.576,
}


# ===========================================================================
# UncertaintyQuantifierEngine
# ===========================================================================


class UncertaintyQuantifierEngine:
    """Monte Carlo uncertainty quantification for waste treatment calculations.

    Provides Monte Carlo simulation, analytical error propagation,
    DQI scoring, parameter uncertainty lookup, uncertainty combination,
    confidence interval extraction, and sensitivity analysis for all
    waste treatment emission pathways (landfill, incineration, composting,
    anaerobic digestion, wastewater, open burning, MBT, pyrolysis,
    gasification).

    Thread Safety:
        Monte Carlo uses per-call Random instances. Shared counters
        are protected by a reentrant lock.

    Attributes:
        _default_iterations: Default Monte Carlo iterations.
        _default_seed: Default random seed.
        _lock: Reentrant lock for thread safety.
        _total_analyses: Counter of analyses performed.

    Example:
        >>> engine = UncertaintyQuantifierEngine()
        >>> result = engine.quantify_uncertainty({
        ...     "total_co2e_tonnes": 1200,
        ...     "parameters": [
        ...         {"name": "doc", "value": 0.15},
        ...         {"name": "mcf", "value": 1.0},
        ...     ],
        ... })
        >>> assert result["status"] == "SUCCESS"
    """

    def __init__(
        self,
        default_iterations: int = 5000,
        default_seed: int = 42,
    ) -> None:
        """Initialize the UncertaintyQuantifierEngine.

        Args:
            default_iterations: Default number of Monte Carlo iterations.
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

    def _resolve_distribution(
        self,
        param_name: str,
        param_config: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Resolve distribution type and shape from parameter config.

        If the parameter config includes explicit distribution info, use it.
        Otherwise, look up the name in PARAMETER_DISTRIBUTIONS.

        Args:
            param_name: Parameter name (e.g. "doc", "mcf").
            param_config: Dictionary with optional dist/std_pct/range_pct keys.

        Returns:
            Tuple of (distribution_type, shape_parameters).
        """
        # Explicit override in request takes precedence
        explicit_dist = param_config.get("dist", "").lower()
        if explicit_dist:
            return explicit_dist, param_config

        # Lookup in PARAMETER_DISTRIBUTIONS
        key = param_name.lower()
        if key in PARAMETER_DISTRIBUTIONS:
            dist_spec = PARAMETER_DISTRIBUTIONS[key]
            merged = {**dist_spec, **param_config}
            return dist_spec["type"], merged

        return "normal", param_config

    def _generate_samples(
        self,
        rng: random.Random,
        value: float,
        dist_type: str,
        shape: Dict[str, Any],
        n: int,
    ) -> List[float]:
        """Generate random samples for a parameter from its distribution.

        Supports normal, lognormal, uniform, and triangular distributions.
        Each distribution handles its shape parameters differently:
        - normal: std_pct (as % of mean value)
        - lognormal: std_pct (as % of mean, parameterized via mu/sigma)
        - uniform: range_pct (as % of value) or explicit min/max
        - triangular: explicit min/mode/max or range_pct

        Args:
            rng: Seeded Random number generator instance.
            value: Central/mean value of the parameter.
            dist_type: Distribution type string.
            shape: Shape parameter dictionary.
            n: Number of samples to generate.

        Returns:
            List of sampled values.
        """
        cv_pct = float(shape.get("cv_pct", shape.get("std_pct", 30)))

        if dist_type == "lognormal":
            return self._sample_lognormal(rng, value, cv_pct, n)
        elif dist_type == "uniform":
            return self._sample_uniform(rng, value, shape, n)
        elif dist_type == "triangular":
            return self._sample_triangular(rng, value, shape, n)
        else:
            return self._sample_normal(rng, value, cv_pct, n)

    def _sample_normal(
        self,
        rng: random.Random,
        value: float,
        cv_pct: float,
        n: int,
    ) -> List[float]:
        """Generate normal-distributed samples, bounded at zero.

        Args:
            rng: Random number generator.
            value: Mean value.
            cv_pct: Coefficient of variation as percentage.
            n: Number of samples.

        Returns:
            List of non-negative normally-distributed samples.
        """
        std_dev = abs(value) * (cv_pct / 100.0)
        samples: List[float] = []
        for _ in range(n):
            sample = rng.gauss(value, std_dev)
            samples.append(max(0.0, sample))
        return samples

    def _sample_lognormal(
        self,
        rng: random.Random,
        value: float,
        cv_pct: float,
        n: int,
    ) -> List[float]:
        """Generate lognormal-distributed samples.

        Parameterized from mean and CV so the distribution's expected
        value equals the input value.

        Args:
            rng: Random number generator.
            value: Expected mean value.
            cv_pct: Coefficient of variation as percentage.
            n: Number of samples.

        Returns:
            List of lognormally-distributed samples.
        """
        if value <= 0:
            return [0.0] * n
        sigma2 = math.log(1.0 + (cv_pct / 100.0) ** 2)
        sigma = math.sqrt(sigma2)
        mu = math.log(value) - sigma2 / 2.0
        return [rng.lognormvariate(mu, sigma) for _ in range(n)]

    def _sample_uniform(
        self,
        rng: random.Random,
        value: float,
        shape: Dict[str, Any],
        n: int,
    ) -> List[float]:
        """Generate uniformly-distributed samples.

        Uses explicit min/max if provided, otherwise constructs bounds
        from range_pct around the central value.

        Args:
            rng: Random number generator.
            value: Central value (used when min/max not explicit).
            shape: Shape dict with optional min/max/range_pct.
            n: Number of samples.

        Returns:
            List of uniformly-distributed samples.
        """
        if "min" in shape and "max" in shape:
            low = float(shape["min"])
            high = float(shape["max"])
        else:
            range_pct = float(shape.get("range_pct", shape.get("cv_pct", 15)))
            low = value * (1.0 - range_pct / 100.0)
            high = value * (1.0 + range_pct / 100.0)

        return [rng.uniform(low, high) for _ in range(n)]

    def _sample_triangular(
        self,
        rng: random.Random,
        value: float,
        shape: Dict[str, Any],
        n: int,
    ) -> List[float]:
        """Generate triangular-distributed samples.

        Uses explicit min/mode/max if provided, otherwise constructs
        a symmetric triangular around the central value using cv_pct.

        Args:
            rng: Random number generator.
            value: Central value (used as mode when explicit not given).
            shape: Shape dict with optional min/mode/max or cv_pct.
            n: Number of samples.

        Returns:
            List of triangularly-distributed samples.
        """
        if "min" in shape and "max" in shape:
            low = float(shape["min"])
            high = float(shape["max"])
            mode = float(shape.get("mode", value))
        else:
            cv_pct = float(shape.get("cv_pct", shape.get("range_pct", 20)))
            low = value * (1.0 - cv_pct / 100.0)
            high = value * (1.0 + cv_pct / 100.0)
            mode = value

        return [rng.triangular(low, high, mode) for _ in range(n)]

    def _calculate_percentiles(
        self,
        values: List[float],
        percentiles: List[float],
    ) -> Dict[str, float]:
        """Calculate percentiles from a list of values using linear interpolation.

        Args:
            values: Unsorted list of numeric values.
            percentiles: List of percentile points (0-100).

        Returns:
            Dictionary mapping percentile labels (e.g. "5", "50") to values.
        """
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        result: Dict[str, float] = {}

        for p in percentiles:
            idx = (p / 100.0) * (n - 1)
            lower = int(math.floor(idx))
            upper = min(lower + 1, n - 1)
            frac = idx - lower
            val = sorted_vals[lower] * (1.0 - frac) + sorted_vals[upper] * frac
            label = str(int(p)) if p == int(p) else str(p)
            result[label] = round(val, 8)

        return result

    def _compute_statistics(
        self,
        values: List[float],
    ) -> Dict[str, float]:
        """Compute descriptive statistics for a list of values.

        Args:
            values: List of numeric values (must be non-empty).

        Returns:
            Dictionary with mean, std_dev, cv_pct, min, max, median.
        """
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean_val = sum(sorted_vals) / n
        variance = sum((x - mean_val) ** 2 for x in sorted_vals) / max(n - 1, 1)
        std_dev = math.sqrt(variance)
        cv_pct = (std_dev / abs(mean_val) * 100.0) if mean_val != 0 else 0.0
        median = self._calculate_percentiles(sorted_vals, [50]).get("50", 0.0)

        return {
            "mean": round(mean_val, 8),
            "std_dev": round(std_dev, 8),
            "cv_pct": round(cv_pct, 2),
            "min": round(sorted_vals[0], 8),
            "max": round(sorted_vals[-1], 8),
            "median": median,
        }

    def _build_confidence_intervals(
        self,
        sorted_results: List[float],
        mean_val: float,
        confidence_levels: List[int],
    ) -> Dict[str, Dict[str, float]]:
        """Build confidence intervals from Monte Carlo output distribution.

        Args:
            sorted_results: Sorted list of simulation results.
            mean_val: Mean value of the distribution.
            confidence_levels: List of confidence levels (e.g. [90, 95, 99]).

        Returns:
            Dictionary keyed by confidence level string, each containing
            lower, upper, half_width, and relative_pct.
        """
        ci: Dict[str, Dict[str, float]] = {}

        for level in confidence_levels:
            alpha = (100 - level) / 2.0
            lower_dict = self._calculate_percentiles(sorted_results, [alpha])
            upper_dict = self._calculate_percentiles(sorted_results, [100 - alpha])

            lower_key = str(int(alpha)) if alpha == int(alpha) else str(alpha)
            upper_key = (
                str(int(100 - alpha))
                if (100 - alpha) == int(100 - alpha)
                else str(100 - alpha)
            )

            lower_val = lower_dict.get(lower_key, 0.0)
            upper_val = upper_dict.get(upper_key, 0.0)
            half_width = (upper_val - lower_val) / 2.0
            relative_pct = (
                (half_width / abs(mean_val) * 100.0)
                if mean_val != 0
                else 0.0
            )

            ci[str(int(level))] = {
                "lower": round(lower_val, 8),
                "upper": round(upper_val, 8),
                "half_width": round(half_width, 8),
                "relative_pct": round(relative_pct, 2),
            }

        return ci

    # ------------------------------------------------------------------
    # Primary Entry Point
    # ------------------------------------------------------------------

    def quantify_uncertainty(
        self,
        calculation_input: Dict[str, Any],
        method: str = "monte_carlo",
        n_iterations: int = 5000,
        seed: int = 42,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """Primary entry point for uncertainty quantification.

        Dispatches to the appropriate method (Monte Carlo, analytical,
        or DQI) and returns a unified result structure.

        Args:
            calculation_input: Dictionary containing:
                - total_co2e_tonnes: Central estimate of total CO2e.
                - parameters: List of parameter dicts.
                - For DQI: reliability, completeness, temporal, geographical,
                  technological (each 1-5).
                - For analytical: activity_uncertainty_pct, ef_uncertainty_pct.
            method: Quantification method - "monte_carlo", "analytical",
                or "dqi".
            n_iterations: Number of Monte Carlo iterations (only for MC).
            seed: Random seed for reproducibility (only for MC).
            confidence_level: Confidence level as a fraction (0.90, 0.95, 0.99).

        Returns:
            Unified result dictionary with uncertainty estimates, confidence
            intervals, and provenance hash.

        Raises:
            ValueError: If an unknown method is specified.
        """
        self._increment_analyses()
        start_time = time.monotonic()
        calc_id = str(uuid4())

        method_lower = method.lower().strip()

        logger.info(
            "quantify_uncertainty: id=%s, method=%s, n=%d, seed=%d, cl=%.2f",
            calc_id, method_lower, n_iterations, seed, confidence_level,
        )

        try:
            if method_lower == "monte_carlo":
                base_emissions = _safe_decimal(
                    calculation_input.get("total_co2e_tonnes"), _ZERO
                )
                parameters = calculation_input.get("parameters", [])
                mc_result = self.monte_carlo_simulation(
                    base_emissions=base_emissions,
                    parameters=parameters,
                    n_iterations=n_iterations,
                    seed=seed,
                    confidence_level=confidence_level,
                    _calc_id=calc_id,
                    _start_time=start_time,
                )
                return mc_result

            elif method_lower == "analytical":
                base_emissions = _safe_decimal(
                    calculation_input.get("total_co2e_tonnes"), _ZERO
                )
                activity_u = float(
                    calculation_input.get("activity_uncertainty_pct", 10.0)
                )
                ef_u = float(
                    calculation_input.get("ef_uncertainty_pct", 50.0)
                )
                return self.analytical_uncertainty(
                    base_emissions=base_emissions,
                    activity_uncertainty_pct=activity_u,
                    ef_uncertainty_pct=ef_u,
                    confidence_level=confidence_level,
                    _calc_id=calc_id,
                    _start_time=start_time,
                )

            elif method_lower == "dqi":
                return self.calculate_dqi(
                    reliability=int(calculation_input.get("reliability", 3)),
                    completeness=int(calculation_input.get("completeness", 3)),
                    temporal=int(calculation_input.get("temporal", 3)),
                    geographical=int(calculation_input.get("geographical", 3)),
                    technological=int(calculation_input.get("technological", 3)),
                    _calc_id=calc_id,
                    _start_time=start_time,
                )

            else:
                processing_time = round(
                    (time.monotonic() - start_time) * 1000, 3
                )
                return {
                    "calculation_id": calc_id,
                    "status": "VALIDATION_ERROR",
                    "errors": [
                        f"Unknown method '{method_lower}'. "
                        "Supported: monte_carlo, analytical, dqi"
                    ],
                    "processing_time_ms": processing_time,
                }

        except Exception as exc:
            processing_time = round(
                (time.monotonic() - start_time) * 1000, 3
            )
            logger.error(
                "quantify_uncertainty failed: id=%s, error=%s",
                calc_id, str(exc), exc_info=True,
            )
            return {
                "calculation_id": calc_id,
                "status": "ERROR",
                "errors": [str(exc)],
                "processing_time_ms": processing_time,
            }

    # ------------------------------------------------------------------
    # Monte Carlo Simulation
    # ------------------------------------------------------------------

    def monte_carlo_simulation(
        self,
        base_emissions: Decimal,
        parameters: List[Dict[str, Any]],
        n_iterations: int = 5000,
        seed: int = 42,
        confidence_level: float = 0.95,
        _calc_id: Optional[str] = None,
        _start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for waste treatment emission uncertainty.

        The simulation perturbs each parameter according to its known
        distribution (from PARAMETER_DISTRIBUTIONS or explicit config),
        recalculates the total result for each iteration using a
        multiplicative scaling model, and derives confidence intervals
        from the resulting distribution.

        Args:
            base_emissions: Central estimate of total emissions (tCO2e).
            parameters: List of parameter dicts, each with:
                - name: Parameter name (e.g. "doc", "mcf", "collection_efficiency").
                - value: Central value.
                - Optional: cv_pct, dist, std_pct, range_pct, min, max, mode.
            n_iterations: Number of simulation iterations.
            seed: Random seed for reproducibility.
            confidence_level: Confidence level as fraction (default 0.95).
            _calc_id: Internal calculation ID (auto-generated if None).
            _start_time: Internal start timestamp (auto-captured if None).

        Returns:
            Result dictionary containing:
                - method: "monte_carlo"
                - base_emissions_tco2e: Input central estimate
                - mean_tco2e: Mean of simulated distribution
                - median_tco2e: Median of simulated distribution
                - std_dev_tco2e: Standard deviation
                - ci_lower_tco2e: Lower bound of CI
                - ci_upper_tco2e: Upper bound of CI
                - confidence_level: Confidence level used
                - coefficient_of_variation: CV as Decimal
                - n_iterations: Number of iterations run
                - seed: Seed used
                - percentiles: p5, p10, p25, p50, p75, p90, p95
                - provenance_hash: SHA-256 hash for audit
        """
        calc_id = _calc_id or str(uuid4())
        start_time = _start_time or time.monotonic()
        total_co2e = float(base_emissions)

        # -- Validate --------------------------------------------------
        errors = self._validate_mc_inputs(total_co2e, parameters, n_iterations)
        if errors:
            return self._error_response(calc_id, errors, start_time)

        # -- Generate per-parameter samples ----------------------------
        rng = random.Random(seed)
        param_samples = self._generate_all_param_samples(
            rng, parameters, n_iterations
        )

        # -- Simulate total results ------------------------------------
        total_results = self._run_scaling_simulation(
            total_co2e, parameters, param_samples, n_iterations
        )

        # -- Compute statistics ----------------------------------------
        stats = self._compute_statistics(total_results)

        # -- Percentiles -----------------------------------------------
        percentile_points = [5, 10, 25, 50, 75, 90, 95]
        percentiles = self._calculate_percentiles(total_results, percentile_points)

        # -- Confidence intervals --------------------------------------
        ci_pct = confidence_level * 100.0
        confidence_levels_int = [90, 95, 99]
        if int(ci_pct) not in confidence_levels_int:
            confidence_levels_int.append(int(ci_pct))
        all_cis = self._build_confidence_intervals(
            sorted(total_results), stats["mean"], confidence_levels_int
        )

        # Extract the requested CI
        ci_key = str(int(ci_pct))
        requested_ci = all_cis.get(ci_key, all_cis.get("95", {}))

        # -- Per-parameter statistics ----------------------------------
        param_stats = self._compute_param_stats(parameters, param_samples)

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "monte_carlo",
            "base_emissions_tco2e": str(base_emissions),
            "mean_tco2e": str(_D(stats["mean"]).quantize(_PRECISION)),
            "median_tco2e": str(_D(stats["median"]).quantize(_PRECISION)),
            "std_dev_tco2e": str(_D(stats["std_dev"]).quantize(_PRECISION)),
            "ci_lower_tco2e": str(
                _D(requested_ci.get("lower", 0)).quantize(_PRECISION)
            ),
            "ci_upper_tco2e": str(
                _D(requested_ci.get("upper", 0)).quantize(_PRECISION)
            ),
            "confidence_level": confidence_level,
            "coefficient_of_variation": str(
                _D(stats["cv_pct"]).quantize(Decimal("0.01"))
            ),
            "n_iterations": n_iterations,
            "seed": seed,
            "percentiles": {
                f"p{k}": str(_D(v).quantize(_PRECISION))
                for k, v in percentiles.items()
            },
            "statistics": {
                k: str(_D(v).quantize(_PRECISION)) if isinstance(v, float) else v
                for k, v in stats.items()
            },
            "confidence_intervals": {
                level: {
                    ck: str(_D(cv).quantize(_PRECISION))
                    if isinstance(cv, float) else cv
                    for ck, cv in ci_data.items()
                }
                for level, ci_data in all_cis.items()
            },
            "parameter_statistics": param_stats,
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Monte Carlo complete: id=%s, n=%d, mean=%.4f, "
            "cv=%.1f%%, %d%% CI=[%.4f, %.4f], time=%.3fms",
            calc_id, n_iterations, stats["mean"], stats["cv_pct"],
            int(ci_pct),
            requested_ci.get("lower", 0),
            requested_ci.get("upper", 0),
            processing_time,
        )
        return result

    def _validate_mc_inputs(
        self,
        total_co2e: float,
        parameters: List[Dict[str, Any]],
        n_iterations: int,
    ) -> List[str]:
        """Validate Monte Carlo input parameters.

        Args:
            total_co2e: Central emissions estimate.
            parameters: Parameter list.
            n_iterations: Number of iterations.

        Returns:
            List of error strings (empty if valid).
        """
        errors: List[str] = []
        if total_co2e == 0:
            errors.append("total_co2e_tonnes / base_emissions must be non-zero")
        if not parameters:
            errors.append("At least one parameter is required")
        if n_iterations <= 0:
            errors.append("n_iterations must be > 0")
        if n_iterations > 1_000_000:
            errors.append("n_iterations must be <= 1,000,000")
        return errors

    def _generate_all_param_samples(
        self,
        rng: random.Random,
        parameters: List[Dict[str, Any]],
        n_iterations: int,
    ) -> Dict[str, List[float]]:
        """Generate Monte Carlo samples for all parameters.

        For each parameter, resolves its distribution from the parameter
        config or PARAMETER_DISTRIBUTIONS, then generates n samples.

        Args:
            rng: Seeded random number generator.
            parameters: List of parameter configuration dicts.
            n_iterations: Number of samples per parameter.

        Returns:
            Dictionary mapping parameter names to sample lists.
        """
        param_samples: Dict[str, List[float]] = {}

        for param in parameters:
            name = str(param.get("name", ""))
            value = float(param.get("value", 0))

            dist_type, shape = self._resolve_distribution(name, param)
            samples = self._generate_samples(rng, value, dist_type, shape, n_iterations)
            param_samples[name] = samples

        return param_samples

    def _run_scaling_simulation(
        self,
        total_co2e: float,
        parameters: List[Dict[str, Any]],
        param_samples: Dict[str, List[float]],
        n_iterations: int,
    ) -> List[float]:
        """Simulate total emissions using multiplicative scaling.

        For each iteration, the scaling factor is the product of
        (sampled_value / central_value) for all parameters.

        Args:
            total_co2e: Central estimate of total emissions.
            parameters: Parameter configuration list.
            param_samples: Pre-generated samples per parameter.
            n_iterations: Number of iterations.

        Returns:
            List of simulated total emission values.
        """
        total_results: List[float] = []

        for i in range(n_iterations):
            scale = 1.0
            for param in parameters:
                name = str(param.get("name", ""))
                value = float(param.get("value", 1.0))
                if value != 0 and name in param_samples:
                    scale *= param_samples[name][i] / value
            total_results.append(total_co2e * scale)

        return total_results

    def _compute_param_stats(
        self,
        parameters: List[Dict[str, Any]],
        param_samples: Dict[str, List[float]],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-parameter descriptive statistics from samples.

        Args:
            parameters: Parameter configuration list.
            param_samples: Pre-generated samples per parameter.

        Returns:
            Dictionary mapping parameter names to statistics.
        """
        param_stats: Dict[str, Dict[str, Any]] = {}

        for param in parameters:
            name = str(param.get("name", ""))
            if name not in param_samples:
                continue

            samples = param_samples[name]
            stats = self._compute_statistics(samples)

            param_stats[name] = {
                "mean": str(_D(stats["mean"]).quantize(_PRECISION)),
                "std_dev": str(_D(stats["std_dev"]).quantize(_PRECISION)),
                "cv_pct": str(_D(stats["cv_pct"]).quantize(Decimal("0.01"))),
                "min": str(_D(stats["min"]).quantize(_PRECISION)),
                "max": str(_D(stats["max"]).quantize(_PRECISION)),
            }

        return param_stats

    def _error_response(
        self,
        calc_id: str,
        errors: List[str],
        start_time: float,
    ) -> Dict[str, Any]:
        """Build a standardized error response.

        Args:
            calc_id: Calculation identifier.
            errors: List of error messages.
            start_time: Monotonic start time.

        Returns:
            Error response dictionary.
        """
        processing_time = round((time.monotonic() - start_time) * 1000, 3)
        return {
            "calculation_id": calc_id,
            "status": "VALIDATION_ERROR",
            "errors": errors,
            "processing_time_ms": processing_time,
        }

    # ------------------------------------------------------------------
    # Analytical Error Propagation (IPCC Approach 1)
    # ------------------------------------------------------------------

    def analytical_uncertainty(
        self,
        base_emissions: Decimal,
        activity_uncertainty_pct: float,
        ef_uncertainty_pct: float,
        confidence_level: float = 0.95,
        additional_uncertainties: Optional[List[Dict[str, Any]]] = None,
        _calc_id: Optional[str] = None,
        _start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run IPCC Approach 1 analytical error propagation.

        Combines activity data uncertainty and emission factor uncertainty
        using root-sum-squares for uncorrelated multiplicative parameters:
            U_combined = sqrt(U_activity^2 + U_ef^2 + U_additional_1^2 + ...)

        Additional uncertainties (e.g., MCF, DOC, waste composition) can be
        appended via additional_uncertainties.

        Args:
            base_emissions: Central estimate of total emissions (tCO2e).
            activity_uncertainty_pct: Activity data uncertainty (% of value).
            ef_uncertainty_pct: Emission factor uncertainty (% of value).
            confidence_level: Confidence level as fraction (default 0.95).
            additional_uncertainties: Optional list of dicts with keys:
                - name: Parameter name.
                - uncertainty_pct: Uncertainty as percentage.
            _calc_id: Internal calculation ID.
            _start_time: Internal start timestamp.

        Returns:
            Result dictionary containing combined uncertainty, CI bounds,
            and provenance hash.
        """
        calc_id = _calc_id or str(uuid4())
        start_time = _start_time or time.monotonic()

        # Collect all relative uncertainties (as fractions)
        u_activity = _safe_decimal(activity_uncertainty_pct) / _HUNDRED
        u_ef = _safe_decimal(ef_uncertainty_pct) / _HUNDRED

        sum_u_sq = u_activity ** 2 + u_ef ** 2

        component_details: List[Dict[str, str]] = [
            {
                "name": "activity_data",
                "uncertainty_pct": str(activity_uncertainty_pct),
                "contribution_sq": str(
                    (u_activity ** 2).quantize(_PRECISION)
                ),
            },
            {
                "name": "emission_factor",
                "uncertainty_pct": str(ef_uncertainty_pct),
                "contribution_sq": str(
                    (u_ef ** 2).quantize(_PRECISION)
                ),
            },
        ]

        if additional_uncertainties:
            for addl in additional_uncertainties:
                name = str(addl.get("name", "unknown"))
                u_pct = _safe_decimal(addl.get("uncertainty_pct", 0))
                u_frac = u_pct / _HUNDRED
                sum_u_sq += u_frac ** 2
                component_details.append({
                    "name": name,
                    "uncertainty_pct": str(u_pct),
                    "contribution_sq": str(
                        (u_frac ** 2).quantize(_PRECISION)
                    ),
                })

        # Combined relative uncertainty
        combined_relative = _D(str(math.sqrt(float(sum_u_sq))))
        combined_pct = (combined_relative * _HUNDRED).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Absolute uncertainty
        combined_abs = (base_emissions * combined_relative).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        # Z-score for requested confidence level
        ci_pct = confidence_level * 100.0
        z = _D(str(Z_SCORES.get(ci_pct, 1.960)))

        ci_lower = (base_emissions - combined_abs * z).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        ci_upper = (base_emissions + combined_abs * z).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "analytical",
            "base_emissions_tco2e": str(base_emissions),
            "combined_uncertainty_pct": str(combined_pct),
            "combined_uncertainty_abs_tco2e": str(combined_abs),
            "mean_tco2e": str(base_emissions),
            "ci_lower_tco2e": str(ci_lower),
            "ci_upper_tco2e": str(ci_upper),
            "confidence_level": confidence_level,
            "z_score": str(z),
            "coefficient_of_variation": str(combined_pct),
            "component_uncertainties": component_details,
            "parameter_count": len(component_details),
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Analytical uncertainty: id=%s, combined_u=%.1f%%, "
            "%.0f%% CI=[%s, %s], time=%.3fms",
            calc_id, float(combined_pct), ci_pct,
            ci_lower, ci_upper, processing_time,
        )
        return result

    # ------------------------------------------------------------------
    # Data Quality Indicator (DQI) Scoring
    # ------------------------------------------------------------------

    def calculate_dqi(
        self,
        reliability: int,
        completeness: int,
        temporal: int,
        geographical: int,
        technological: int,
        _calc_id: Optional[str] = None,
        _start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calculate Data Quality Indicator score for waste treatment data.

        Scores 5 dimensions on a 1-5 scale (1 = best, 5 = worst)
        per GHG Protocol Corporate Standard Chapter 4 guidance.
        The composite score is the geometric mean of dimension scores.

        Dimension descriptors for waste treatment:
            Reliability:
                1 = Direct facility measurement, verified
                2 = Direct measurement, unverified
                3 = Calculated from site-specific data
                4 = Estimated from regional/national data
                5 = Rough estimate, global IPCC default
            Completeness:
                1 = >95% of waste streams covered
                2 = 80-95% coverage
                3 = 60-80% coverage
                4 = 40-60% coverage
                5 = <40% coverage
            Temporal:
                1 = Same reporting year
                2 = Within 2 years
                3 = Within 5 years
                4 = Within 10 years
                5 = >10 years old
            Geographical:
                1 = Same facility
                2 = Same region/country
                3 = Same continent
                4 = Different continent, similar conditions
                5 = Global default
            Technological:
                1 = Same treatment technology/process
                2 = Similar technology, same scale
                3 = Similar technology, different scale
                4 = Different technology, same waste type
                5 = Generic across technologies

        Args:
            reliability: Reliability score (1-5).
            completeness: Completeness score (1-5).
            temporal: Temporal correlation score (1-5).
            geographical: Geographical correlation score (1-5).
            technological: Technological correlation score (1-5).
            _calc_id: Internal calculation ID.
            _start_time: Internal start timestamp.

        Returns:
            DQI result with per-dimension scores, composite, quality
            category, uncertainty multiplier, and provenance hash.
        """
        calc_id = _calc_id or str(uuid4())
        start_time = _start_time or time.monotonic()

        dimensions = {
            "reliability": reliability,
            "completeness": completeness,
            "temporal_correlation": temporal,
            "geographical_correlation": geographical,
            "technological_correlation": technological,
        }

        # -- Validate --------------------------------------------------
        errors: List[str] = []
        for dim_name, score in dimensions.items():
            if not isinstance(score, int) or score < 1 or score > 5:
                errors.append(
                    f"{dim_name} must be an integer 1-5, got {score}"
                )

        if errors:
            return self._error_response(calc_id, errors, start_time)

        # -- Geometric mean composite score ----------------------------
        product = 1.0
        for score in dimensions.values():
            product *= score
        composite = product ** (1.0 / len(dimensions))

        # -- Quality category ------------------------------------------
        quality = self._classify_quality(composite)

        # -- Uncertainty multiplier ------------------------------------
        dqi_to_uncertainty: Dict[str, float] = {
            "EXCELLENT": 0.8,
            "GOOD": 1.0,
            "FAIR": 1.3,
            "POOR": 1.8,
            "VERY_POOR": 2.5,
        }
        uncertainty_multiplier = dqi_to_uncertainty.get(quality, 1.0)

        # -- Suggested default uncertainty (%) -------------------------
        # Based on multiplier applied to a base 30% uncertainty
        suggested_uncertainty_pct = round(30.0 * uncertainty_multiplier, 1)

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "dqi",
            "dimension_scores": dimensions,
            "composite_score": round(composite, 4),
            "quality_category": quality,
            "uncertainty_multiplier": uncertainty_multiplier,
            "suggested_uncertainty_pct": suggested_uncertainty_pct,
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "DQI calculated: id=%s, composite=%.2f, quality=%s, "
            "multiplier=%.1f, time=%.3fms",
            calc_id, composite, quality,
            uncertainty_multiplier, processing_time,
        )
        return result

    def _classify_quality(self, composite: float) -> str:
        """Classify composite DQI score into a quality category.

        Args:
            composite: Geometric mean of dimension scores (1.0 - 5.0).

        Returns:
            Quality category string.
        """
        if composite <= 1.5:
            return "EXCELLENT"
        elif composite <= 2.5:
            return "GOOD"
        elif composite <= 3.5:
            return "FAIR"
        elif composite <= 4.5:
            return "POOR"
        else:
            return "VERY_POOR"

    # ------------------------------------------------------------------
    # Parameter Uncertainty Lookup
    # ------------------------------------------------------------------

    def get_parameter_uncertainty(
        self,
        parameter_name: str,
        tier: str = "TIER_1",
    ) -> Dict[str, Any]:
        """Look up the default uncertainty for a waste treatment parameter.

        Combines information from PARAMETER_DISTRIBUTIONS (distribution
        type and shape) and DEFAULT_CV (tier-specific coefficient of
        variation) to return a complete uncertainty specification.

        Args:
            parameter_name: Parameter name (e.g. "doc", "mcf",
                "composting_ef_ch4").
            tier: IPCC methodology tier ("TIER_1", "TIER_2", "TIER_3").

        Returns:
            Dictionary with distribution type, CV, shape parameters, and
            tier information. Returns a default normal/30% CV if the
            parameter is not found.
        """
        key_lower = parameter_name.lower()
        key_upper = parameter_name.upper()
        tier_upper = tier.upper()

        # Distribution info
        dist_info: Dict[str, Any] = {}
        if key_lower in PARAMETER_DISTRIBUTIONS:
            dist_info = dict(PARAMETER_DISTRIBUTIONS[key_lower])

        # CV info by tier
        cv_pct: Optional[float] = None
        if key_upper in DEFAULT_CV:
            tier_cv = DEFAULT_CV[key_upper]
            cv_pct = tier_cv.get(tier_upper, tier_cv.get("TIER_1"))
        elif dist_info:
            cv_pct = float(
                dist_info.get("std_pct", dist_info.get("range_pct", 30))
            )

        result: Dict[str, Any] = {
            "parameter": parameter_name,
            "tier": tier_upper,
            "distribution_type": dist_info.get("type", "normal"),
            "cv_pct": cv_pct if cv_pct is not None else 30.0,
            "description": dist_info.get("description", "No description available"),
            "found_in_registry": bool(dist_info) or (key_upper in DEFAULT_CV),
        }

        # Include shape parameters if available
        shape_keys = ["min", "max", "mode", "std_pct", "range_pct"]
        for sk in shape_keys:
            if sk in dist_info:
                result[sk] = dist_info[sk]

        return result

    # ------------------------------------------------------------------
    # Combine Uncertainties
    # ------------------------------------------------------------------

    def combine_uncertainties(
        self,
        uncertainties: List[Dict[str, Any]],
        combination: str = "multiplicative",
    ) -> Dict[str, Any]:
        """Combine multiple source uncertainties into a single estimate.

        For multiplicative combination (IPCC default for emission calculations):
            U_combined = sqrt(sum(Ui^2))

        For additive combination (summing independent emission sources):
            U_combined = sqrt(sum((Ui * Xi)^2)) / |sum(Xi)|

        Each entry in the uncertainties list requires:
            - name: Source name.
            - uncertainty_pct: Relative uncertainty (% half-width of 95% CI).
            - For additive: value (absolute emission value in tCO2e).

        Args:
            uncertainties: List of uncertainty dictionaries.
            combination: "multiplicative" or "additive".

        Returns:
            Combined uncertainty result with provenance hash.
        """
        self._increment_analyses()
        start_time = time.monotonic()
        calc_id = str(uuid4())

        if not uncertainties:
            return self._error_response(
                calc_id,
                ["At least one uncertainty source is required"],
                start_time,
            )

        combination_lower = combination.lower().strip()
        component_details: List[Dict[str, str]] = []

        if combination_lower == "additive":
            combined = self._combine_additive(uncertainties, component_details)
        else:
            combined = self._combine_multiplicative(
                uncertainties, component_details
            )

        combined_pct = combined["combined_pct"]
        combined_abs = combined.get("combined_abs")

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "combined_uncertainty",
            "combination": combination_lower,
            "combined_uncertainty_pct": str(combined_pct),
            "component_uncertainties": component_details,
            "source_count": len(uncertainties),
            "processing_time_ms": processing_time,
        }

        if combined_abs is not None:
            result["combined_uncertainty_abs"] = str(combined_abs)

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Combined uncertainty: id=%s, mode=%s, combined=%.2f%%, "
            "sources=%d, time=%.3fms",
            calc_id, combination_lower, float(combined_pct),
            len(uncertainties), processing_time,
        )
        return result

    def _combine_multiplicative(
        self,
        uncertainties: List[Dict[str, Any]],
        component_details: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Combine uncertainties using multiplicative model (root-sum-squares).

        Args:
            uncertainties: List of uncertainty sources.
            component_details: Output list for per-component detail dicts.

        Returns:
            Dict with combined_pct as Decimal.
        """
        sum_u_sq = _ZERO

        for source in uncertainties:
            name = str(source.get("name", "unknown"))
            u_pct = _safe_decimal(source.get("uncertainty_pct", 0))
            u_frac = u_pct / _HUNDRED
            u_sq = u_frac ** 2
            sum_u_sq += u_sq

            component_details.append({
                "name": name,
                "uncertainty_pct": str(u_pct),
                "contribution_sq": str(u_sq.quantize(_PRECISION)),
            })

        combined_frac = _D(str(math.sqrt(float(sum_u_sq))))
        combined_pct = (combined_frac * _HUNDRED).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return {"combined_pct": combined_pct}

    def _combine_additive(
        self,
        uncertainties: List[Dict[str, Any]],
        component_details: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Combine uncertainties using additive model.

        Args:
            uncertainties: List of uncertainty sources with value fields.
            component_details: Output list for per-component detail dicts.

        Returns:
            Dict with combined_pct and combined_abs as Decimals.
        """
        sum_x = _ZERO
        sum_ux_sq = _ZERO

        for source in uncertainties:
            name = str(source.get("name", "unknown"))
            x = _safe_decimal(source.get("value", 0))
            u_pct = _safe_decimal(source.get("uncertainty_pct", 0))
            u_abs = x * u_pct / _HUNDRED
            sum_x += x
            sum_ux_sq += u_abs ** 2

            component_details.append({
                "name": name,
                "value": str(x),
                "uncertainty_pct": str(u_pct),
                "uncertainty_abs": str(u_abs.quantize(_PRECISION)),
            })

        if sum_x != _ZERO:
            combined_abs = _D(str(math.sqrt(float(sum_ux_sq))))
            combined_pct = (combined_abs / abs(sum_x) * _HUNDRED).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            combined_abs = _ZERO
            combined_pct = _ZERO

        return {
            "combined_pct": combined_pct,
            "combined_abs": combined_abs.quantize(_PRECISION),
        }

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
            request: Sensitivity analysis request dictionary.

        Returns:
            Ranked parameter sensitivities with tornado chart data,
            variance contributions, and provenance hash.
        """
        self._increment_analyses()
        start_time = time.monotonic()
        calc_id = str(uuid4())

        total_co2e = float(request.get("total_co2e_tonnes", 0))
        parameters = request.get("parameters", [])

        if not parameters or total_co2e == 0:
            return self._error_response(
                calc_id,
                ["total_co2e_tonnes and parameters are required"],
                start_time,
            )

        sensitivities = self._compute_sensitivities(total_co2e, parameters)
        self._rank_and_annotate(sensitivities)

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        total_variance = sum(
            s["variance_contribution"] for s in sensitivities
        )

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "sensitivity_analysis",
            "central_estimate": total_co2e,
            "sensitivities": sensitivities,
            "total_variance": round(total_variance, 8),
            "top_driver": (
                sensitivities[0]["parameter"] if sensitivities else None
            ),
            "parameter_count": len(sensitivities),
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Sensitivity analysis: id=%s, params=%d, top_driver=%s, "
            "time=%.3fms",
            calc_id, len(sensitivities),
            result["top_driver"], processing_time,
        )
        return result

    def _compute_sensitivities(
        self,
        total_co2e: float,
        parameters: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute per-parameter sensitivity by +/-1 std dev perturbation.

        Args:
            total_co2e: Central estimate of total emissions.
            parameters: Parameter configuration list.

        Returns:
            List of sensitivity dicts (unsorted).
        """
        sensitivities: List[Dict[str, Any]] = []

        for param in parameters:
            name = str(param.get("name", ""))
            value = float(param.get("value", 0))
            cv_pct = float(param.get("cv_pct", 0))

            # If cv_pct not explicitly given, look up default
            if cv_pct == 0:
                param_info = self.get_parameter_uncertainty(name)
                cv_pct = float(param_info.get("cv_pct", 30.0))

            if value == 0:
                continue

            std_dev = abs(value) * cv_pct / 100.0

            # Perturb +1 std_dev
            perturbed_high = value + std_dev
            scale_high = perturbed_high / value
            result_high = total_co2e * scale_high

            # Perturb -1 std_dev (bounded at zero)
            perturbed_low = max(0.0, value - std_dev)
            scale_low = perturbed_low / value
            result_low = total_co2e * scale_low

            impact_range = abs(result_high - result_low)
            variance_contribution = (impact_range / 2.0) ** 2

            sensitivities.append({
                "parameter": name,
                "central_value": value,
                "cv_pct": cv_pct,
                "result_low": round(result_low, 8),
                "result_high": round(result_high, 8),
                "impact_range": round(impact_range, 8),
                "variance_contribution": round(variance_contribution, 8),
            })

        return sensitivities

    def _rank_and_annotate(
        self,
        sensitivities: List[Dict[str, Any]],
    ) -> None:
        """Sort sensitivities by variance contribution and add pct annotation.

        Modifies the list in-place.

        Args:
            sensitivities: List of sensitivity dicts.
        """
        sensitivities.sort(
            key=lambda x: x["variance_contribution"], reverse=True
        )

        total_variance = sum(
            s["variance_contribution"] for s in sensitivities
        )

        for s in sensitivities:
            if total_variance > 0:
                s["contribution_pct"] = round(
                    s["variance_contribution"] / total_variance * 100.0, 2
                )
            else:
                s["contribution_pct"] = 0.0

    # ------------------------------------------------------------------
    # Confidence Interval (standalone utility)
    # ------------------------------------------------------------------

    def get_confidence_interval(
        self,
        mean: float,
        std_dev: float,
        confidence_level: float = 95.0,
        n_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Calculate a confidence interval from mean and standard deviation.

        Uses z-scores for large samples (n >= 30) and an approximate
        t-correction for small samples.

        Args:
            mean: Sample mean.
            std_dev: Sample standard deviation.
            confidence_level: Confidence level as percentage (90.0, 95.0, 99.0).
            n_samples: Number of samples (for t-distribution correction).

        Returns:
            Dictionary with lower, upper, half_width, and confidence_level.
        """
        z = Z_SCORES.get(confidence_level, 1.960)

        if n_samples and n_samples < 30:
            # Approximate t-correction for small sample sizes
            z = z * (1.0 + 1.0 / (4.0 * max(n_samples - 1, 1)))

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
    # Percentiles (standalone utility)
    # ------------------------------------------------------------------

    def get_percentiles(
        self,
        values: List[float],
        percentile_points: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Calculate percentiles from a list of values.

        Args:
            values: List of numeric values.
            percentile_points: Percentile points 0-100 (default [5,25,50,75,95]).

        Returns:
            Dictionary mapping percentile labels to values.
        """
        if percentile_points is None:
            percentile_points = [5, 25, 50, 75, 95]

        return self._calculate_percentiles(values, percentile_points)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine usage statistics.

        Returns:
            Dictionary with engine name, version, creation time,
            total analyses count, and configuration.
        """
        with self._lock:
            return {
                "engine": "UncertaintyQuantifierEngine",
                "domain": "waste_treatment_emissions",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_analyses": self._total_analyses,
                "default_iterations": self._default_iterations,
                "default_seed": self._default_seed,
                "supported_methods": [
                    "monte_carlo",
                    "analytical",
                    "dqi",
                    "sensitivity_analysis",
                    "combine_uncertainties",
                ],
                "registered_parameters": len(PARAMETER_DISTRIBUTIONS),
                "registered_cv_tiers": len(DEFAULT_CV),
            }

    def reset(self) -> None:
        """Reset engine counters to zero.

        Thread-safe reset of the analysis counter. Does not affect
        configuration or default parameters.
        """
        with self._lock:
            self._total_analyses = 0
        logger.info("UncertaintyQuantifierEngine reset")
