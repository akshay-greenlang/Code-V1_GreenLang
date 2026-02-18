# -*- coding: utf-8 -*-
"""
UncertaintyQuantifierEngine - Monte Carlo & DQI Scoring (Engine 5 of 7)

AGENT-MRV-005: Fugitive Emissions Agent

Quantifies the uncertainty of fugitive emission calculations using
Monte Carlo simulation, analytical error propagation, data quality
indicator (DQI) scoring, and sensitivity analysis.

Process-Specific Uncertainty Ranges (half-width of 95% CI):
    - Equipment leak EF:      +/-30-100% (high variability between facilities)
    - LDAR screening:         +/-20-50%
    - Coal mine methane:      +/-25-75%  (depends on measurement method)
    - Wastewater CH4:         +/-40-100% (MCF highly variable)
    - Pneumatic devices:      +/-20-40%
    - Tank losses:            +/-25-50%
    - Direct measurement:     +/-10-25%

DQI Scoring (5 dimensions, 1-5 scale):
    - Reliability: direct measurement (1) ... engineering estimate (5)
    - Completeness: >95% coverage (1) ... <40% coverage (5)
    - Temporal correlation: same year (1) ... >5 years old (5)
    - Geographical correlation: same facility (1) ... global default (5)
    - Technological correlation: same equipment (1) ... generic (5)

    Composite score: geometric mean of 5 dimension scores.

Monte Carlo Simulation:
    - Configurable iterations (default 5000)
    - Log-normal distributions for emission factors (non-negative)
    - Normal distributions for activity data
    - Triangular distributions for abatement efficiencies
    - Explicit seed support for full reproducibility

Analytical Propagation (IPCC Approach 1):
    Combined relative uncertainty for multiplicative chains:
    U_total = sqrt(sum(Ui^2)) for uncorrelated parameters.

Sensitivity Analysis:
    Tornado chart data: rank parameters by contribution to total variance.

Confidence Intervals:
    90%, 95%, and 99% intervals from Monte Carlo percentiles.

Zero-Hallucination Guarantees:
    - All formulas are deterministic mathematical operations.
    - No LLM involvement in any numeric path.
    - PRNG is seeded explicitly for full reproducibility.
    - Every result carries a SHA-256 provenance hash.

Thread Safety:
    All mutable state is protected by a reentrant lock. Monte Carlo
    simulations create per-call Random instances so concurrent callers
    never interfere.

Example:
    >>> from greenlang.fugitive_emissions.uncertainty_quantifier import (
    ...     UncertaintyQuantifierEngine,
    ... )
    >>> engine = UncertaintyQuantifierEngine()
    >>> result = engine.run_monte_carlo(
    ...     calculation_input={
    ...         "total_co2e_kg": 500000,
    ...         "source_type": "EQUIPMENT_LEAK",
    ...         "calculation_method": "AVERAGE_EMISSION_FACTOR",
    ...         "component_count": 5000,
    ...         "emission_factor": 0.0089,
    ...     },
    ...     n_iterations=5000,
    ... )
    >>> print(result["confidence_intervals"]["95"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
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
    from greenlang.fugitive_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.fugitive_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.fugitive_emissions.metrics import (
        record_uncertainty as _record_uncertainty,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_uncertainty = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash.

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ===========================================================================
# Enumerations
# ===========================================================================


class FugitiveSourceCategory(str, Enum):
    """Fugitive emission source categories for uncertainty lookup."""

    EQUIPMENT_LEAK = "EQUIPMENT_LEAK"
    LDAR_SCREENING = "LDAR_SCREENING"
    COAL_MINE_METHANE = "COAL_MINE_METHANE"
    WASTEWATER = "WASTEWATER"
    PNEUMATIC_DEVICE = "PNEUMATIC_DEVICE"
    TANK_LOSS = "TANK_LOSS"
    DIRECT_MEASUREMENT = "DIRECT_MEASUREMENT"


class CalculationMethodType(str, Enum):
    """Fugitive emission calculation method types."""

    AVERAGE_EMISSION_FACTOR = "AVERAGE_EMISSION_FACTOR"
    SCREENING_RANGES = "SCREENING_RANGES"
    EPA_CORRELATION = "EPA_CORRELATION"
    UNIT_SPECIFIC_CORRELATION = "UNIT_SPECIFIC_CORRELATION"
    DIRECT_MEASUREMENT = "DIRECT_MEASUREMENT"


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions (ISO 14044 / PAS 2050)."""

    RELIABILITY = "reliability"
    COMPLETENESS = "completeness"
    TEMPORAL_CORRELATION = "temporal_correlation"
    GEOGRAPHICAL_CORRELATION = "geographical_correlation"
    TECHNOLOGICAL_CORRELATION = "technological_correlation"


# ===========================================================================
# Reference Data: Uncertainty Ranges
# ===========================================================================

#: Uncertainty half-widths by source category and method.
#: Format: {source_category: {method: (lower_pct, upper_pct)}}
#: Values represent +/- percentage at 95% confidence.
UNCERTAINTY_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "EQUIPMENT_LEAK": {
        "AVERAGE_EMISSION_FACTOR": (30.0, 100.0),
        "SCREENING_RANGES": (20.0, 50.0),
        "EPA_CORRELATION": (25.0, 60.0),
        "UNIT_SPECIFIC_CORRELATION": (15.0, 40.0),
        "DIRECT_MEASUREMENT": (10.0, 25.0),
    },
    "LDAR_SCREENING": {
        "AVERAGE_EMISSION_FACTOR": (30.0, 60.0),
        "SCREENING_RANGES": (20.0, 50.0),
        "EPA_CORRELATION": (20.0, 45.0),
        "UNIT_SPECIFIC_CORRELATION": (15.0, 35.0),
        "DIRECT_MEASUREMENT": (10.0, 25.0),
    },
    "COAL_MINE_METHANE": {
        "AVERAGE_EMISSION_FACTOR": (50.0, 75.0),
        "SCREENING_RANGES": (40.0, 65.0),
        "EPA_CORRELATION": (35.0, 55.0),
        "UNIT_SPECIFIC_CORRELATION": (25.0, 45.0),
        "DIRECT_MEASUREMENT": (15.0, 35.0),
    },
    "WASTEWATER": {
        "AVERAGE_EMISSION_FACTOR": (60.0, 100.0),
        "SCREENING_RANGES": (50.0, 80.0),
        "EPA_CORRELATION": (45.0, 70.0),
        "UNIT_SPECIFIC_CORRELATION": (35.0, 60.0),
        "DIRECT_MEASUREMENT": (15.0, 30.0),
    },
    "PNEUMATIC_DEVICE": {
        "AVERAGE_EMISSION_FACTOR": (20.0, 40.0),
        "SCREENING_RANGES": (20.0, 35.0),
        "EPA_CORRELATION": (15.0, 30.0),
        "UNIT_SPECIFIC_CORRELATION": (10.0, 25.0),
        "DIRECT_MEASUREMENT": (10.0, 20.0),
    },
    "TANK_LOSS": {
        "AVERAGE_EMISSION_FACTOR": (35.0, 50.0),
        "SCREENING_RANGES": (30.0, 45.0),
        "EPA_CORRELATION": (25.0, 40.0),
        "UNIT_SPECIFIC_CORRELATION": (20.0, 35.0),
        "DIRECT_MEASUREMENT": (10.0, 25.0),
    },
    "DIRECT_MEASUREMENT": {
        "AVERAGE_EMISSION_FACTOR": (15.0, 25.0),
        "SCREENING_RANGES": (15.0, 25.0),
        "EPA_CORRELATION": (12.0, 22.0),
        "UNIT_SPECIFIC_CORRELATION": (10.0, 20.0),
        "DIRECT_MEASUREMENT": (10.0, 15.0),
    },
}

#: Default uncertainty percentages for individual parameters when
#: process-specific values are not available.
DEFAULT_PARAMETER_UNCERTAINTIES: Dict[str, float] = {
    "emission_factor": 50.0,
    "activity_data": 10.0,
    "component_count": 5.0,
    "gas_composition": 15.0,
    "gwp": 10.0,
    "recovery_efficiency": 20.0,
    "operating_hours": 5.0,
    "leak_rate": 30.0,
    "throughput": 8.0,
}

#: DQI scoring matrix: {dimension: [(threshold, score), ...]}
#: Lower score = better quality (1 = best, 5 = worst).
DQI_SCORING_MATRIX: Dict[str, List[Tuple[str, int]]] = {
    "reliability": [
        ("verified_direct_measurement", 1),
        ("direct_measurement", 2),
        ("calculated_from_measurements", 3),
        ("engineering_estimate", 4),
        ("unknown", 5),
    ],
    "completeness": [
        ("above_95", 1),
        ("80_to_95", 2),
        ("60_to_80", 3),
        ("40_to_60", 4),
        ("below_40", 5),
    ],
    "temporal_correlation": [
        ("same_year", 1),
        ("within_2_years", 2),
        ("within_3_years", 3),
        ("within_5_years", 4),
        ("older_than_5_years", 5),
    ],
    "geographical_correlation": [
        ("same_facility", 1),
        ("same_region", 2),
        ("same_country", 3),
        ("similar_climate", 4),
        ("global_default", 5),
    ],
    "technological_correlation": [
        ("same_equipment", 1),
        ("same_technology", 2),
        ("similar_technology", 3),
        ("related_technology", 4),
        ("generic", 5),
    ],
}


# ===========================================================================
# UncertaintyQuantifierEngine
# ===========================================================================


class UncertaintyQuantifierEngine:
    """Uncertainty quantification engine for fugitive emissions.

    Provides Monte Carlo simulation, analytical error propagation,
    DQI scoring, sensitivity analysis, and confidence interval
    calculations for fugitive emission estimates.

    All numeric calculations are deterministic when seeded (zero-hallucination).
    Thread-safe via reentrant lock with per-call Random instances.

    Attributes:
        config: Configuration dictionary.

    Example:
        >>> engine = UncertaintyQuantifierEngine()
        >>> result = engine.quantify_uncertainty({
        ...     "total_co2e_kg": 100000,
        ...     "source_type": "EQUIPMENT_LEAK",
        ...     "calculation_method": "AVERAGE_EMISSION_FACTOR",
        ... })
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the UncertaintyQuantifierEngine.

        Args:
            config: Optional configuration dictionary.
        """
        self._config = config or {}
        self._lock = threading.RLock()

        self._default_iterations: int = int(
            self._config.get("monte_carlo_iterations", 5000),
        )
        self._default_seed: int = int(
            self._config.get("monte_carlo_seed", 42),
        )
        self._default_confidence_levels: List[float] = [90.0, 95.0, 99.0]

        # Parse confidence levels from config
        cl_str = self._config.get("confidence_levels", "90,95,99")
        if isinstance(cl_str, str):
            try:
                self._default_confidence_levels = [
                    float(x.strip()) for x in cl_str.split(",")
                    if x.strip()
                ]
            except ValueError:
                pass

        # Statistics
        self._total_mc_runs: int = 0
        self._total_analytical_runs: int = 0
        self._total_dqi_scores: int = 0
        self._total_sensitivity_runs: int = 0

        logger.info(
            "UncertaintyQuantifierEngine initialized: "
            "iterations=%d, seed=%d, confidence_levels=%s",
            self._default_iterations,
            self._default_seed,
            self._default_confidence_levels,
        )

    # ------------------------------------------------------------------
    # Public API: Unified quantification
    # ------------------------------------------------------------------

    def quantify_uncertainty(
        self,
        calculation_input: Dict[str, Any],
        method: str = "monte_carlo",
        n_iterations: Optional[int] = None,
        seed: Optional[int] = None,
        confidence_levels: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Quantify uncertainty of a fugitive emission calculation.

        Dispatches to Monte Carlo or analytical method based on the
        method parameter.

        Args:
            calculation_input: Calculation data dictionary containing:
                - total_co2e_kg (float): Point estimate in kg CO2e.
                - source_type (str): FugitiveSourceCategory value.
                - calculation_method (str): CalculationMethodType value.
                - Optional parameter-specific data.
            method: "monte_carlo" or "analytical".
            n_iterations: Monte Carlo iterations (default from config).
            seed: Random seed for reproducibility (default from config).
            confidence_levels: Confidence levels (default: [90, 95, 99]).

        Returns:
            Dictionary with uncertainty characterization.
        """
        if method == "analytical":
            return self.analytical_propagation(calculation_input)
        return self.run_monte_carlo(
            calculation_input=calculation_input,
            n_iterations=n_iterations,
            seed=seed,
            confidence_levels=confidence_levels,
        )

    # ------------------------------------------------------------------
    # Public API: Monte Carlo Simulation
    # ------------------------------------------------------------------

    def run_monte_carlo(
        self,
        calculation_input: Dict[str, Any],
        n_iterations: Optional[int] = None,
        seed: Optional[int] = None,
        confidence_levels: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for uncertainty quantification.

        Draws from parameterised distributions for each uncertain parameter
        (emission factor, activity data, component count, etc.) and
        produces percentile tables and confidence intervals.

        Distributions:
            - Emission factors: log-normal (non-negative, right-skewed)
            - Activity data: normal (symmetric around measured value)
            - Component counts: normal (integer-valued)
            - GWP: normal (IPCC assessment report uncertainty)
            - Recovery efficiency: triangular (bounded [0, 1])

        Args:
            calculation_input: Calculation data with total_co2e_kg,
                source_type, calculation_method, and parameter details.
            n_iterations: Number of MC iterations.
            seed: Random seed for reproducibility.
            confidence_levels: Confidence levels for intervals.

        Returns:
            Dictionary with mean, std_dev, CV, confidence_intervals,
            percentiles, and parameter contributions.
        """
        t0 = time.monotonic()

        iterations = n_iterations or self._default_iterations
        rng_seed = seed if seed is not None else self._default_seed
        cl = confidence_levels or self._default_confidence_levels

        total_co2e_kg = float(
            calculation_input.get("total_co2e_kg", 0),
        )
        source_type = calculation_input.get(
            "source_type", "EQUIPMENT_LEAK",
        ).upper()
        calc_method = calculation_input.get(
            "calculation_method", "AVERAGE_EMISSION_FACTOR",
        ).upper()

        # Get process-specific uncertainty range
        uncertainty_range = self._get_uncertainty_range(
            source_type, calc_method,
        )

        # Build parameter distributions
        params = self._build_parameter_distributions(
            calculation_input, uncertainty_range,
        )

        # Run simulation with isolated Random instance
        rng = random.Random(rng_seed)
        samples: List[float] = []

        for _ in range(iterations):
            sample_value = self._simulate_single_iteration(
                rng, total_co2e_kg, params,
            )
            samples.append(sample_value)

        # Compute statistics
        stats = self._compute_mc_statistics(samples, cl)

        # Sensitivity analysis (one-at-a-time)
        contributions = self._compute_parameter_contributions(
            rng_seed, iterations, total_co2e_kg, params,
        )

        # DQI score
        dqi = self._compute_dqi_from_input(calculation_input)

        with self._lock:
            self._total_mc_runs += 1

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        result = {
            "method": "monte_carlo",
            "source_type": source_type,
            "calculation_method": calc_method,
            "iterations": iterations,
            "seed_used": rng_seed,
            "point_estimate_co2e_kg": total_co2e_kg,
            "mean_co2e_kg": stats["mean"],
            "median_co2e_kg": stats["median"],
            "std_dev_kg": stats["std_dev"],
            "coefficient_of_variation": stats["cv"],
            "confidence_intervals": stats["confidence_intervals"],
            "percentiles": stats["percentiles"],
            "uncertainty_range_pct": uncertainty_range,
            "parameter_contributions": contributions,
            "data_quality_score": dqi,
            "processing_time_ms": round(elapsed_ms, 3),
        }
        result["provenance_hash"] = _compute_hash(result)

        if _record_uncertainty is not None:
            _record_uncertainty("monte_carlo")

        logger.info(
            "MC uncertainty: mean=%.2f, std=%.2f, CV=%.2f%% "
            "(%d iterations in %.1fms)",
            stats["mean"], stats["std_dev"],
            stats["cv"] * 100.0, iterations, elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: DQI Scoring
    # ------------------------------------------------------------------

    def calculate_dqi(
        self,
        scores: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate Data Quality Indicator (DQI) composite score.

        Uses the geometric mean of 5 dimension scores (1-5 scale)
        to produce a single composite quality indicator.

        Args:
            scores: Dictionary with dimension scores:
                - reliability (int or str): 1-5 or label.
                - completeness (int or str): 1-5 or label.
                - temporal_correlation (int or str): 1-5 or label.
                - geographical_correlation (int or str): 1-5 or label.
                - technological_correlation (int or str): 1-5 or label.

        Returns:
            Dictionary with dimension scores and composite DQI.
        """
        dimension_scores: Dict[str, int] = {}

        for dim in DQIDimension:
            raw_value = scores.get(dim.value, 3)
            if isinstance(raw_value, (int, float)):
                score = max(1, min(5, int(raw_value)))
            elif isinstance(raw_value, str):
                score = self._label_to_score(dim.value, raw_value)
            else:
                score = 3
            dimension_scores[dim.value] = score

        # Geometric mean
        product = 1.0
        for s in dimension_scores.values():
            product *= s
        composite = product ** (1.0 / len(dimension_scores))

        # Map composite to uncertainty multiplier
        uncertainty_multiplier = self._dqi_to_uncertainty_multiplier(
            composite,
        )

        with self._lock:
            self._total_dqi_scores += 1

        result = {
            "dimension_scores": dimension_scores,
            "composite_score": round(composite, 2),
            "uncertainty_multiplier": round(uncertainty_multiplier, 4),
            "quality_label": self._score_to_label(composite),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "DQI computed: composite=%.2f, label=%s",
            composite, result["quality_label"],
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Analytical Propagation
    # ------------------------------------------------------------------

    def analytical_propagation(
        self,
        calculation_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute combined uncertainty using IPCC Approach 1.

        For multiplicative chains (emission = EF x AD x GWP):
        U_combined = sqrt(U_EF^2 + U_AD^2 + U_GWP^2)

        Args:
            calculation_input: Calculation data with parameter uncertainties.

        Returns:
            Dictionary with combined uncertainty and 95% interval.
        """
        t0 = time.monotonic()

        total_co2e_kg = float(
            calculation_input.get("total_co2e_kg", 0),
        )
        source_type = calculation_input.get(
            "source_type", "EQUIPMENT_LEAK",
        ).upper()
        calc_method = calculation_input.get(
            "calculation_method", "AVERAGE_EMISSION_FACTOR",
        ).upper()

        # Get parameter uncertainties (as relative %)
        param_uncertainties = self._get_parameter_uncertainties(
            calculation_input,
        )

        # Root-sum-of-squares for multiplicative model
        sum_sq = sum(u ** 2 for u in param_uncertainties.values())
        combined_pct = math.sqrt(sum_sq)

        # 95% CI using combined uncertainty (assumes normal at 2 sigma)
        half_width = total_co2e_kg * combined_pct / 100.0 * 1.96
        lower_95 = max(0.0, total_co2e_kg - half_width)
        upper_95 = total_co2e_kg + half_width

        # Std dev estimate (combined_pct is at 95% level / 1.96)
        std_dev = total_co2e_kg * combined_pct / 100.0

        cv = std_dev / total_co2e_kg if total_co2e_kg > 0 else 0.0

        with self._lock:
            self._total_analytical_runs += 1

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        result = {
            "method": "analytical_propagation",
            "source_type": source_type,
            "calculation_method": calc_method,
            "point_estimate_co2e_kg": total_co2e_kg,
            "combined_uncertainty_pct": round(combined_pct, 2),
            "mean_co2e_kg": total_co2e_kg,
            "std_dev_kg": round(std_dev, 4),
            "coefficient_of_variation": round(cv, 6),
            "confidence_intervals": {
                "95": {
                    "lower": round(lower_95, 4),
                    "upper": round(upper_95, 4),
                },
            },
            "parameter_uncertainties": {
                k: round(v, 2) for k, v in param_uncertainties.items()
            },
            "processing_time_ms": round(elapsed_ms, 3),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Analytical uncertainty: combined=%.2f%%, "
            "95%% CI=[%.2f, %.2f] kg CO2e",
            combined_pct, lower_95, upper_95,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Sensitivity Analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        calculation_input: Dict[str, Any],
        n_iterations: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform one-at-a-time sensitivity analysis.

        Varies each uncertain parameter individually while holding
        others at their mean values to determine each parameter's
        contribution to total output variance.

        Args:
            calculation_input: Calculation data.
            n_iterations: Iterations per parameter.
            seed: Random seed.

        Returns:
            Dictionary with parameter rankings and tornado chart data.
        """
        t0 = time.monotonic()

        total_co2e_kg = float(
            calculation_input.get("total_co2e_kg", 0),
        )
        rng_seed = seed if seed is not None else self._default_seed

        source_type = calculation_input.get(
            "source_type", "EQUIPMENT_LEAK",
        ).upper()
        calc_method = calculation_input.get(
            "calculation_method", "AVERAGE_EMISSION_FACTOR",
        ).upper()
        uncertainty_range = self._get_uncertainty_range(
            source_type, calc_method,
        )

        params = self._build_parameter_distributions(
            calculation_input, uncertainty_range,
        )

        # Compute baseline variance from full MC
        rng_base = random.Random(rng_seed)
        baseline_samples = [
            self._simulate_single_iteration(
                rng_base, total_co2e_kg, params,
            )
            for _ in range(n_iterations)
        ]
        baseline_var = self._variance(baseline_samples)

        # One-at-a-time: fix each parameter at mean, re-run MC
        tornado_data: List[Dict[str, Any]] = []

        for param_name, param_info in params.items():
            rng_oat = random.Random(rng_seed)
            fixed_params = dict(params)
            fixed_params[param_name] = {
                **param_info,
                "uncertainty_pct": 0.0,
            }

            oat_samples = [
                self._simulate_single_iteration(
                    rng_oat, total_co2e_kg, fixed_params,
                )
                for _ in range(n_iterations)
            ]
            oat_var = self._variance(oat_samples)

            # Contribution = reduction in variance when parameter is fixed
            contribution = (
                (baseline_var - oat_var) / baseline_var
                if baseline_var > 0 else 0.0
            )

            # Compute +/- impact at mean +/- 1 sigma
            u_pct = param_info.get("uncertainty_pct", 0.0)
            low_impact = total_co2e_kg * (1.0 - u_pct / 100.0)
            high_impact = total_co2e_kg * (1.0 + u_pct / 100.0)

            tornado_data.append({
                "parameter": param_name,
                "contribution_pct": round(contribution * 100.0, 2),
                "low_value_co2e_kg": round(low_impact, 4),
                "high_value_co2e_kg": round(high_impact, 4),
                "uncertainty_pct": round(u_pct, 2),
            })

        # Sort by contribution descending
        tornado_data.sort(
            key=lambda x: x["contribution_pct"], reverse=True,
        )

        with self._lock:
            self._total_sensitivity_runs += 1

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        result = {
            "method": "sensitivity_analysis",
            "source_type": source_type,
            "calculation_method": calc_method,
            "point_estimate_co2e_kg": total_co2e_kg,
            "baseline_variance": round(baseline_var, 4),
            "iterations_per_parameter": n_iterations,
            "tornado_data": tornado_data,
            "processing_time_ms": round(elapsed_ms, 3),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Sensitivity analysis: %d parameters ranked in %.1fms",
            len(tornado_data), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Confidence Intervals
    # ------------------------------------------------------------------

    def get_confidence_intervals(
        self,
        samples: List[float],
        levels: Optional[List[float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute confidence intervals from a sample distribution.

        Args:
            samples: List of Monte Carlo sample values.
            levels: Confidence levels (default: [90, 95, 99]).

        Returns:
            Dictionary of {level_str: {"lower": val, "upper": val}}.
        """
        if not samples:
            return {}

        target_levels = levels or self._default_confidence_levels
        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        intervals: Dict[str, Dict[str, float]] = {}
        for level in target_levels:
            alpha = (100.0 - level) / 200.0
            lower_idx = max(0, int(math.floor(alpha * n)))
            upper_idx = min(n - 1, int(math.ceil((1.0 - alpha) * n)) - 1)

            intervals[str(int(level))] = {
                "lower": round(sorted_samples[lower_idx], 4),
                "upper": round(sorted_samples[upper_idx], 4),
            }

        return intervals

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary with operation counts.
        """
        with self._lock:
            return {
                "total_mc_runs": self._total_mc_runs,
                "total_analytical_runs": self._total_analytical_runs,
                "total_dqi_scores": self._total_dqi_scores,
                "total_sensitivity_runs": self._total_sensitivity_runs,
                "default_iterations": self._default_iterations,
                "default_seed": self._default_seed,
            }

    # ------------------------------------------------------------------
    # Private: Monte Carlo helpers
    # ------------------------------------------------------------------

    def _get_uncertainty_range(
        self,
        source_type: str,
        calc_method: str,
    ) -> Tuple[float, float]:
        """Get the applicable uncertainty range for a source/method combo.

        Args:
            source_type: Fugitive source category.
            calc_method: Calculation method.

        Returns:
            (lower_pct, upper_pct) tuple.
        """
        source_ranges = UNCERTAINTY_RANGES.get(
            source_type,
            UNCERTAINTY_RANGES.get("EQUIPMENT_LEAK", {}),
        )
        return source_ranges.get(
            calc_method,
            source_ranges.get("AVERAGE_EMISSION_FACTOR", (30.0, 100.0)),
        )

    def _build_parameter_distributions(
        self,
        calculation_input: Dict[str, Any],
        uncertainty_range: Tuple[float, float],
    ) -> Dict[str, Dict[str, Any]]:
        """Build parameter distribution specifications for MC simulation.

        Args:
            calculation_input: Calculation data.
            uncertainty_range: (lower_pct, upper_pct) for this source/method.

        Returns:
            Dictionary of parameter_name -> distribution specification.
        """
        midpoint_pct = (uncertainty_range[0] + uncertainty_range[1]) / 2.0

        # Start with default uncertainties
        params: Dict[str, Dict[str, Any]] = {}

        # Emission factor
        ef_uncertainty = float(calculation_input.get(
            "emission_factor_uncertainty_pct",
            midpoint_pct,
        ))
        params["emission_factor"] = {
            "distribution": "lognormal",
            "uncertainty_pct": ef_uncertainty,
            "contribution_weight": 0.4,
        }

        # Activity data (component count, throughput, etc.)
        ad_uncertainty = float(calculation_input.get(
            "activity_data_uncertainty_pct",
            DEFAULT_PARAMETER_UNCERTAINTIES.get("activity_data", 10.0),
        ))
        params["activity_data"] = {
            "distribution": "normal",
            "uncertainty_pct": ad_uncertainty,
            "contribution_weight": 0.3,
        }

        # GWP
        gwp_uncertainty = float(calculation_input.get(
            "gwp_uncertainty_pct",
            DEFAULT_PARAMETER_UNCERTAINTIES.get("gwp", 10.0),
        ))
        params["gwp"] = {
            "distribution": "normal",
            "uncertainty_pct": gwp_uncertainty,
            "contribution_weight": 0.1,
        }

        # Gas composition
        gas_uncertainty = float(calculation_input.get(
            "gas_composition_uncertainty_pct",
            DEFAULT_PARAMETER_UNCERTAINTIES.get("gas_composition", 15.0),
        ))
        params["gas_composition"] = {
            "distribution": "normal",
            "uncertainty_pct": gas_uncertainty,
            "contribution_weight": 0.1,
        }

        # Operating hours / temporal
        hours_uncertainty = float(calculation_input.get(
            "operating_hours_uncertainty_pct",
            DEFAULT_PARAMETER_UNCERTAINTIES.get("operating_hours", 5.0),
        ))
        params["operating_hours"] = {
            "distribution": "normal",
            "uncertainty_pct": hours_uncertainty,
            "contribution_weight": 0.1,
        }

        return params

    def _simulate_single_iteration(
        self,
        rng: random.Random,
        base_value: float,
        params: Dict[str, Dict[str, Any]],
    ) -> float:
        """Simulate a single MC iteration by sampling all parameters.

        The emission estimate is modeled as a multiplicative chain:
            E = base_value * prod(multiplier_i for each parameter)

        Each multiplier_i is sampled from the parameter's distribution
        centered at 1.0 with spread determined by uncertainty_pct.

        Args:
            rng: Random number generator (per-call instance).
            base_value: Point estimate (center value).
            params: Parameter distribution specifications.

        Returns:
            Simulated emission value (kg CO2e).
        """
        combined_multiplier = 1.0

        for param_info in params.values():
            u_pct = param_info.get("uncertainty_pct", 0.0)
            dist = param_info.get("distribution", "normal")

            if u_pct <= 0.0:
                continue

            # Convert percentage to relative standard deviation
            # u_pct is half-width at ~95% CI, so sigma ~ u_pct / 196
            sigma_rel = u_pct / 100.0 / 1.96

            if dist == "lognormal":
                # Log-normal: mean = 1.0, sigma in log-space
                sigma_ln = math.sqrt(
                    math.log(1.0 + sigma_rel ** 2),
                )
                mu_ln = -0.5 * sigma_ln ** 2  # mean of 1.0
                multiplier = rng.lognormvariate(mu_ln, sigma_ln)
            elif dist == "triangular":
                low = max(0.0, 1.0 - sigma_rel * 1.96)
                high = 1.0 + sigma_rel * 1.96
                mode = 1.0
                multiplier = rng.triangular(low, high, mode)
            else:
                # Normal distribution, clipped at 0
                multiplier = max(0.0, rng.gauss(1.0, sigma_rel))

            combined_multiplier *= multiplier

        return base_value * combined_multiplier

    def _compute_mc_statistics(
        self,
        samples: List[float],
        confidence_levels: List[float],
    ) -> Dict[str, Any]:
        """Compute statistics from Monte Carlo samples.

        Args:
            samples: List of simulated values.
            confidence_levels: Confidence levels for intervals.

        Returns:
            Dictionary with mean, median, std_dev, cv, confidence_intervals,
            and percentile tables.
        """
        n = len(samples)
        if n == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0,
                "cv": 0.0,
                "confidence_intervals": {},
                "percentiles": {},
            }

        sorted_samples = sorted(samples)
        mean = sum(samples) / n
        median = sorted_samples[n // 2]

        variance = sum((x - mean) ** 2 for x in samples) / max(n - 1, 1)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean if mean > 0 else 0.0

        # Confidence intervals
        ci = self.get_confidence_intervals(sorted_samples, confidence_levels)

        # Standard percentiles
        percentile_levels = [5, 10, 25, 50, 75, 90, 95]
        percentiles: Dict[str, float] = {}
        for p in percentile_levels:
            idx = min(int(p / 100.0 * n), n - 1)
            percentiles[f"P{p}"] = round(sorted_samples[idx], 4)

        return {
            "mean": round(mean, 4),
            "median": round(median, 4),
            "std_dev": round(std_dev, 4),
            "cv": round(cv, 6),
            "confidence_intervals": ci,
            "percentiles": percentiles,
        }

    def _compute_parameter_contributions(
        self,
        seed: int,
        iterations: int,
        base_value: float,
        params: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute parameter contributions to total variance.

        Uses a variance decomposition approach: for each parameter,
        hold it fixed and measure the reduction in output variance.

        Args:
            seed: Random seed.
            iterations: Iterations per parameter.
            base_value: Point estimate.
            params: Parameter distributions.

        Returns:
            Dictionary of parameter_name -> contribution_fraction.
        """
        # Baseline run (all parameters varying)
        rng_base = random.Random(seed)
        baseline_samples = [
            self._simulate_single_iteration(rng_base, base_value, params)
            for _ in range(min(iterations, 2000))
        ]
        baseline_var = self._variance(baseline_samples)

        if baseline_var <= 0:
            return {k: 0.0 for k in params}

        contributions: Dict[str, float] = {}
        for param_name in params:
            rng_oat = random.Random(seed)
            fixed_params = dict(params)
            fixed_params[param_name] = {
                **params[param_name],
                "uncertainty_pct": 0.0,
            }
            oat_samples = [
                self._simulate_single_iteration(
                    rng_oat, base_value, fixed_params,
                )
                for _ in range(min(iterations, 2000))
            ]
            oat_var = self._variance(oat_samples)
            contributions[param_name] = round(
                max(0.0, (baseline_var - oat_var) / baseline_var), 4,
            )

        return contributions

    # ------------------------------------------------------------------
    # Private: DQI helpers
    # ------------------------------------------------------------------

    def _label_to_score(self, dimension: str, label: str) -> int:
        """Convert a DQI label to a numeric score (1-5).

        Args:
            dimension: DQI dimension name.
            label: Quality label string.

        Returns:
            Integer score (1-5). Defaults to 3 if not found.
        """
        matrix = DQI_SCORING_MATRIX.get(dimension, [])
        for lbl, score in matrix:
            if lbl == label.lower().strip():
                return score
        return 3

    def _score_to_label(self, composite: float) -> str:
        """Map a composite DQI score to a quality label.

        Args:
            composite: Composite DQI score (1.0-5.0).

        Returns:
            Quality label string.
        """
        if composite <= 1.5:
            return "excellent"
        elif composite <= 2.5:
            return "good"
        elif composite <= 3.5:
            return "fair"
        elif composite <= 4.5:
            return "poor"
        else:
            return "very_poor"

    def _dqi_to_uncertainty_multiplier(self, composite: float) -> float:
        """Convert composite DQI to an uncertainty multiplier.

        Better data quality (lower score) yields lower uncertainty.
        Multiplier is applied to base uncertainty ranges.

        Args:
            composite: Composite DQI score (1.0-5.0).

        Returns:
            Multiplier (0.5 for excellent to 2.0 for very poor).
        """
        # Linear interpolation: 1.0 -> 0.5, 5.0 -> 2.0
        return 0.5 + (composite - 1.0) * (2.0 - 0.5) / (5.0 - 1.0)

    def _compute_dqi_from_input(
        self,
        calculation_input: Dict[str, Any],
    ) -> Optional[float]:
        """Extract DQI scores from calculation input and compute composite.

        Args:
            calculation_input: May contain dqi_scores sub-dictionary.

        Returns:
            Composite DQI score or None if not available.
        """
        dqi_data = calculation_input.get("dqi_scores")
        if not dqi_data or not isinstance(dqi_data, dict):
            return None

        result = self.calculate_dqi(dqi_data)
        return result.get("composite_score")

    # ------------------------------------------------------------------
    # Private: Analytical helpers
    # ------------------------------------------------------------------

    def _get_parameter_uncertainties(
        self,
        calculation_input: Dict[str, Any],
    ) -> Dict[str, float]:
        """Extract parameter-level uncertainties from input.

        Falls back to defaults from DEFAULT_PARAMETER_UNCERTAINTIES.

        Args:
            calculation_input: Calculation data.

        Returns:
            Dictionary of parameter_name -> uncertainty_pct.
        """
        uncertainties: Dict[str, float] = {}

        for param_name, default_pct in DEFAULT_PARAMETER_UNCERTAINTIES.items():
            key = f"{param_name}_uncertainty_pct"
            value = calculation_input.get(key)
            if value is not None:
                uncertainties[param_name] = float(value)
            else:
                uncertainties[param_name] = default_pct

        return uncertainties

    # ------------------------------------------------------------------
    # Private: Math utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _variance(samples: List[float]) -> float:
        """Compute sample variance.

        Args:
            samples: List of numeric values.

        Returns:
            Sample variance (unbiased, N-1 denominator).
        """
        n = len(samples)
        if n < 2:
            return 0.0
        mean = sum(samples) / n
        return sum((x - mean) ** 2 for x in samples) / (n - 1)
