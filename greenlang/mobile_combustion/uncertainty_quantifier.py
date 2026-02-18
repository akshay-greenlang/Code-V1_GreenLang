# -*- coding: utf-8 -*-
"""
UncertaintyQuantifierEngine - Monte Carlo & Analytical Uncertainty (Engine 5 of 7)

AGENT-MRV-003: Mobile Combustion Agent

Quantifies the uncertainty of mobile combustion emission calculations
using two complementary methods:

    1. **Monte Carlo simulation**: Draws from parameterised distributions
       (normal for activity data, emission factors, and GWP) and produces
       full percentile tables with 90/95/99% confidence intervals.
       Configurable iteration count (default 5000) with explicit seed
       support for bit-perfect reproducibility.

    2. **Analytical error propagation** (IPCC Approach 1): Combined
       relative uncertainty for multiplicative chains via root-sum-of-
       squares of (sensitivity x component_uncertainty) terms.

Method-Specific Uncertainty Ranges:
    - Fuel-based (Tier 3):    +/-5-10%   (metered fuel, facility-specific EFs)
    - Fuel-based (Tier 2):    +/-10-20%  (invoiced fuel, country-specific EFs)
    - Fuel-based (Tier 1):    +/-15-30%  (default EFs)
    - Distance-based:         +/-15-25%  (estimated distance, default fuel economy)
    - Spend-based:            +/-30-50%  (financial proxies, high uncertainty)

Activity Data Uncertainty:
    - Metered:    +/-5%  (direct fuel measurement)
    - Estimated:  +/-10% (odometer, GPS, fleet records)
    - Screening:  +/-25% (financial estimates, default assumptions)

Emission Factor Uncertainty:
    - Tier 3: +/-3%  (facility-specific, measured)
    - Tier 2: +/-10% (country-specific)
    - Tier 1: +/-25% (IPCC default)

GWP Uncertainty: +/-10% (IPCC assessment report uncertainty)

Data Quality Indicator (DQI) Scoring (1-5 scale):
    5 dimensions weighted equally:
    - Reliability: direct measurement (1) ... unknown (5)
    - Completeness: >95% (1) ... <40% (5)
    - Temporal: same year (1) ... >5 years (5)
    - Geographical: same region (1) ... unknown (5)
    - Technological: same tech (1) ... unknown (5)

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
    >>> from greenlang.mobile_combustion.uncertainty_quantifier import (
    ...     UncertaintyQuantifierEngine,
    ... )
    >>> engine = UncertaintyQuantifierEngine()
    >>> result = engine.run_monte_carlo(
    ...     calculation_input={
    ...         "total_co2e_kg": 5000,
    ...         "method": "FUEL_BASED",
    ...         "tier": "TIER_2",
    ...         "activity_data_source": "metered",
    ...         "fuel_litres": 2000,
    ...         "emission_factor": 2.5,
    ...     },
    ...     n_iterations=5000,
    ... )
    >>> print(result.confidence_intervals["95"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
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
    from greenlang.mobile_combustion.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.mobile_combustion.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.mobile_combustion.metrics import (
        PROMETHEUS_AVAILABLE as _METRICS_AVAILABLE,
        record_uncertainty as _record_uncertainty,
        observe_calculation_duration as _observe_calculation_duration,
    )
except ImportError:
    _METRICS_AVAILABLE = False
    _record_uncertainty = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _to_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation of the value.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


# ===========================================================================
# Enumerations
# ===========================================================================


class CalculationMethod(str, Enum):
    """Mobile combustion calculation method types.

    Each method has distinct uncertainty characteristics reflecting the
    quality and specificity of underlying data.

    FUEL_BASED: Emissions from fuel consumption * emission factor.
        Most accurate when metered fuel data is available.
    DISTANCE_BASED: Emissions from distance * fuel economy * EF,
        or distance * distance emission factor.
        Moderate uncertainty from fuel economy estimation.
    SPEND_BASED: Emissions from financial spend * spend EF.
        Highest uncertainty due to financial proxy usage.
    """

    FUEL_BASED = "FUEL_BASED"
    DISTANCE_BASED = "DISTANCE_BASED"
    SPEND_BASED = "SPEND_BASED"


class CalculationTier(str, Enum):
    """Calculation tier for emission factor specificity.

    TIER_3: Facility-specific emission factors, measured heating values.
        Lowest uncertainty.
    TIER_2: Country-specific emission factors, default heating values.
        Moderate uncertainty.
    TIER_1: IPCC default emission factors.
        Highest uncertainty.
    """

    TIER_3 = "TIER_3"
    TIER_2 = "TIER_2"
    TIER_1 = "TIER_1"


class ActivityDataSource(str, Enum):
    """Activity data source classification.

    METERED: Direct fuel measurement (flow meters, tank dipstick).
        Uncertainty: +/-5%.
    ESTIMATED: Derived from odometer, GPS, fleet management records.
        Uncertainty: +/-10%.
    SCREENING: Derived from financial data, default assumptions.
        Uncertainty: +/-25%.
    """

    METERED = "METERED"
    ESTIMATED = "ESTIMATED"
    SCREENING = "SCREENING"


class DQICategory(str, Enum):
    """Data Quality Indicator scoring categories.

    RELIABILITY: How the data was collected (measurement vs estimate).
    COMPLETENESS: Fraction of required data that is available.
    TEMPORAL_CORRELATION: Age of the data relative to reporting period.
    GEOGRAPHICAL_CORRELATION: Geographic representativeness.
    TECHNOLOGICAL_CORRELATION: Technology representativeness.
    """

    RELIABILITY = "RELIABILITY"
    COMPLETENESS = "COMPLETENESS"
    TEMPORAL_CORRELATION = "TEMPORAL_CORRELATION"
    GEOGRAPHICAL_CORRELATION = "GEOGRAPHICAL_CORRELATION"
    TECHNOLOGICAL_CORRELATION = "TECHNOLOGICAL_CORRELATION"


# ===========================================================================
# Default Uncertainty Parameters
# ===========================================================================

# Activity data uncertainty (half-width of 95% CI as fraction)
_ACTIVITY_DATA_UNCERTAINTY: Dict[str, float] = {
    ActivityDataSource.METERED.value: 0.05,
    ActivityDataSource.ESTIMATED.value: 0.10,
    ActivityDataSource.SCREENING.value: 0.25,
}

# Emission factor uncertainty by tier (half-width as fraction)
_EMISSION_FACTOR_UNCERTAINTY: Dict[str, float] = {
    CalculationTier.TIER_3.value: 0.03,
    CalculationTier.TIER_2.value: 0.10,
    CalculationTier.TIER_1.value: 0.25,
}

# GWP uncertainty (half-width as fraction)
_GWP_UNCERTAINTY: float = 0.10

# Fuel economy uncertainty for distance-based method (half-width as fraction)
_FUEL_ECONOMY_UNCERTAINTY: float = 0.15

# Distance measurement uncertainty (half-width as fraction)
_DISTANCE_UNCERTAINTY: Dict[str, float] = {
    "gps": 0.02,
    "odometer": 0.05,
    "estimated": 0.15,
    "default": 0.10,
}

# Method-specific total uncertainty ranges (half-width as fraction)
_METHOD_UNCERTAINTY_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    CalculationMethod.FUEL_BASED.value: {
        CalculationTier.TIER_3.value: (0.05, 0.10),
        CalculationTier.TIER_2.value: (0.10, 0.20),
        CalculationTier.TIER_1.value: (0.15, 0.30),
    },
    CalculationMethod.DISTANCE_BASED.value: {
        CalculationTier.TIER_3.value: (0.15, 0.20),
        CalculationTier.TIER_2.value: (0.18, 0.25),
        CalculationTier.TIER_1.value: (0.20, 0.25),
    },
    CalculationMethod.SPEND_BASED.value: {
        CalculationTier.TIER_3.value: (0.30, 0.40),
        CalculationTier.TIER_2.value: (0.35, 0.45),
        CalculationTier.TIER_1.value: (0.40, 0.50),
    },
}

# Method-specific default midpoint uncertainty (half-width as fraction)
_METHOD_DEFAULT_UNCERTAINTY: Dict[str, Dict[str, float]] = {
    CalculationMethod.FUEL_BASED.value: {
        CalculationTier.TIER_3.value: 0.075,
        CalculationTier.TIER_2.value: 0.15,
        CalculationTier.TIER_1.value: 0.225,
    },
    CalculationMethod.DISTANCE_BASED.value: {
        CalculationTier.TIER_3.value: 0.175,
        CalculationTier.TIER_2.value: 0.215,
        CalculationTier.TIER_1.value: 0.225,
    },
    CalculationMethod.SPEND_BASED.value: {
        CalculationTier.TIER_3.value: 0.35,
        CalculationTier.TIER_2.value: 0.40,
        CalculationTier.TIER_1.value: 0.45,
    },
}

# DQI scoring criteria (1 = best, 5 = worst)
_DQI_CRITERIA: Dict[str, Dict[str, int]] = {
    DQICategory.RELIABILITY.value: {
        "direct_measurement": 1,
        "flow_meter": 1,
        "calibrated_instrument": 1,
        "verified_estimate": 2,
        "fleet_management_system": 2,
        "odometer": 2,
        "estimate": 3,
        "engineering_estimate": 3,
        "assumption": 4,
        "expert_judgment": 4,
        "unknown": 5,
    },
    DQICategory.COMPLETENESS.value: {
        "above_95_pct": 1,
        "80_to_95_pct": 2,
        "60_to_80_pct": 3,
        "40_to_60_pct": 4,
        "below_40_pct": 5,
    },
    DQICategory.TEMPORAL_CORRELATION.value: {
        "same_year": 1,
        "within_1_year": 2,
        "within_3_years": 3,
        "within_5_years": 4,
        "older_than_5_years": 5,
    },
    DQICategory.GEOGRAPHICAL_CORRELATION.value: {
        "same_region": 1,
        "same_country": 2,
        "similar_region": 3,
        "different_region": 4,
        "unknown": 5,
    },
    DQICategory.TECHNOLOGICAL_CORRELATION.value: {
        "same_technology": 1,
        "similar_technology": 2,
        "related_technology": 3,
        "different_technology": 4,
        "unknown": 5,
    },
}

# DQI score to uncertainty multiplier mapping
_DQI_MULTIPLIERS: Dict[int, float] = {
    1: 0.60,   # Best quality: reduce uncertainty by 40%
    2: 0.80,   # Good quality: reduce by 20%
    3: 1.00,   # Average: baseline
    4: 1.30,   # Below average: increase by 30%
    5: 1.80,   # Poor quality: increase by 80%
}

# Confidence interval z-scores (two-tailed)
_CONFIDENCE_Z_SCORES: Dict[str, float] = {
    "90": 1.6449,
    "95": 1.9600,
    "99": 2.5758,
}

# Monte Carlo iteration limits
_DEFAULT_ITERATIONS: int = 5000
_MIN_ITERATIONS: int = 100
_MAX_ITERATIONS: int = 100000


# ===========================================================================
# Dataclasses for results
# ===========================================================================


@dataclass
class UncertaintyResult:
    """Complete uncertainty quantification result with provenance.

    Attributes:
        result_id: Unique identifier for this uncertainty assessment.
        emissions_value: Central estimate of emissions.
        emissions_unit: Unit of the emissions value (e.g. "kg_CO2e").
        method: Calculation method used.
        tier: Calculation tier.
        combined_uncertainty_pct: Combined relative uncertainty as a
            percentage (half-width of the 95% CI).
        confidence_intervals: Dictionary mapping confidence level
            strings ("90", "95", "99") to (lower, upper) Decimal tuples.
        monte_carlo_mean: Mean from Monte Carlo simulation (if run).
        monte_carlo_std: Standard deviation from Monte Carlo.
        monte_carlo_median: Median from Monte Carlo.
        monte_carlo_percentiles: Dictionary of percentile values.
        monte_carlo_iterations: Number of iterations executed.
        monte_carlo_seed: Random seed used for reproducibility.
        analytical_uncertainty_pct: Analytical uncertainty as percentage.
        component_uncertainties: Per-component uncertainty breakdown.
        contribution_analysis: Fraction of total variance from each
            component, summing to 1.0.
        data_quality_score: Weighted DQI score (1.0 to 5.0).
        dqi_multiplier: DQI-derived uncertainty multiplier.
        provenance_hash: SHA-256 hash of the assessment.
        timestamp: UTC ISO-formatted timestamp.
        metadata: Additional metadata dictionary.
    """

    result_id: str
    emissions_value: Decimal
    emissions_unit: str
    method: str
    tier: str
    combined_uncertainty_pct: Decimal
    confidence_intervals: Dict[str, Tuple[Decimal, Decimal]]
    monte_carlo_mean: Optional[Decimal]
    monte_carlo_std: Optional[Decimal]
    monte_carlo_median: Optional[Decimal]
    monte_carlo_percentiles: Dict[str, Decimal]
    monte_carlo_iterations: Optional[int]
    monte_carlo_seed: Optional[int]
    analytical_uncertainty_pct: Decimal
    component_uncertainties: Dict[str, Decimal]
    contribution_analysis: Dict[str, Decimal]
    data_quality_score: Decimal
    dqi_multiplier: Decimal
    provenance_hash: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "result_id": self.result_id,
            "emissions_value": str(self.emissions_value),
            "emissions_unit": self.emissions_unit,
            "method": self.method,
            "tier": self.tier,
            "combined_uncertainty_pct": str(self.combined_uncertainty_pct),
            "confidence_intervals": {
                k: [str(v[0]), str(v[1])]
                for k, v in self.confidence_intervals.items()
            },
            "monte_carlo_mean": (
                str(self.monte_carlo_mean)
                if self.monte_carlo_mean is not None else None
            ),
            "monte_carlo_std": (
                str(self.monte_carlo_std)
                if self.monte_carlo_std is not None else None
            ),
            "monte_carlo_median": (
                str(self.monte_carlo_median)
                if self.monte_carlo_median is not None else None
            ),
            "monte_carlo_percentiles": {
                k: str(v) for k, v in self.monte_carlo_percentiles.items()
            },
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "monte_carlo_seed": self.monte_carlo_seed,
            "analytical_uncertainty_pct": str(self.analytical_uncertainty_pct),
            "component_uncertainties": {
                k: str(v) for k, v in self.component_uncertainties.items()
            },
            "contribution_analysis": {
                k: str(v) for k, v in self.contribution_analysis.items()
            },
            "data_quality_score": str(self.data_quality_score),
            "dqi_multiplier": str(self.dqi_multiplier),
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis for a single parameter.

    Attributes:
        parameter: Name of the parameter varied.
        base_value: Original parameter value.
        perturbed_value: Perturbed parameter value.
        perturbation_pct: Perturbation as a percentage of base value.
        base_emissions: Emissions at base value.
        perturbed_emissions: Emissions at perturbed value.
        emissions_change_pct: Percentage change in emissions.
        sensitivity_coefficient: Ratio of % change in output to % change
            in input. Values > 1.0 indicate amplification.
    """

    parameter: str
    base_value: Decimal
    perturbed_value: Decimal
    perturbation_pct: Decimal
    base_emissions: Decimal
    perturbed_emissions: Decimal
    emissions_change_pct: Decimal
    sensitivity_coefficient: Decimal

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "parameter": self.parameter,
            "base_value": str(self.base_value),
            "perturbed_value": str(self.perturbed_value),
            "perturbation_pct": str(self.perturbation_pct),
            "base_emissions": str(self.base_emissions),
            "perturbed_emissions": str(self.perturbed_emissions),
            "emissions_change_pct": str(self.emissions_change_pct),
            "sensitivity_coefficient": str(self.sensitivity_coefficient),
        }


# ===========================================================================
# UncertaintyQuantifierEngine
# ===========================================================================


class UncertaintyQuantifierEngine:
    """Monte Carlo and analytical uncertainty quantification engine for
    mobile combustion emission calculations.

    Provides deterministic, zero-hallucination uncertainty quantification
    using two complementary approaches:

    1. **Monte Carlo Simulation**: Generates N random draws from
       parameterised distributions for each uncertain input, recomputes
       emissions for each draw, and derives percentile-based confidence
       intervals. Uses explicit seeding for full reproducibility.

    2. **Analytical Error Propagation** (IPCC Approach 1): Combines
       component uncertainties using root-sum-of-squares for
       multiplicative relationships, producing a combined relative
       uncertainty estimate.

    Both methods support Data Quality Indicator (DQI) scoring that
    adjusts uncertainty ranges based on the quality of underlying data.

    Supported Methods:
        - FUEL_BASED: fuel_litres * emission_factor * (1 + gwp_adj)
        - DISTANCE_BASED: distance_km * fuel_economy * emission_factor
        - SPEND_BASED: spend * spend_emission_factor

    Thread Safety:
        All mutable state (_assessment_history) is protected by a
        reentrant lock. Monte Carlo simulations create per-call Random
        instances so concurrent callers never interfere.

    Example:
        >>> engine = UncertaintyQuantifierEngine()
        >>> result = engine.run_monte_carlo(
        ...     calculation_input={
        ...         "total_co2e_kg": 5000,
        ...         "method": "FUEL_BASED",
        ...         "tier": "TIER_2",
        ...         "activity_data_source": "metered",
        ...     },
        ...     n_iterations=5000,
        ... )
        >>> print(result.combined_uncertainty_pct)
    """

    def __init__(self) -> None:
        """Initialize the UncertaintyQuantifierEngine."""
        self._assessment_history: List[UncertaintyResult] = []
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "UncertaintyQuantifierEngine initialized: "
            "%d methods, %d tiers, "
            "default iterations=%d",
            len(CalculationMethod),
            len(CalculationTier),
            _DEFAULT_ITERATIONS,
        )

    # ------------------------------------------------------------------
    # Public API: Monte Carlo Simulation
    # ------------------------------------------------------------------

    def run_monte_carlo(
        self,
        calculation_input: Dict[str, Any],
        n_iterations: int = _DEFAULT_ITERATIONS,
        seed: Optional[int] = None,
    ) -> UncertaintyResult:
        """Run Monte Carlo simulation for uncertainty quantification.

        Generates N random draws from parameterised distributions for
        each uncertain input (activity data, emission factor, GWP),
        recomputes emissions for each draw, and derives percentile-based
        confidence intervals.

        Required keys in calculation_input:
            - total_co2e_kg (float/Decimal): Central emission estimate.
            - method (str): FUEL_BASED, DISTANCE_BASED, or SPEND_BASED.
            - tier (str): TIER_1, TIER_2, or TIER_3.

        Optional keys:
            - activity_data_source (str): metered, estimated, screening.
            - fuel_litres (float): Fuel consumed in litres.
            - emission_factor (float): Emission factor value.
            - distance_km (float): Distance in km (distance-based).
            - fuel_economy (float): Fuel economy L/100km (distance-based).
            - spend_amount (float): Financial spend (spend-based).
            - spend_ef (float): Spend emission factor (spend-based).
            - gwp_ch4 (float): GWP for CH4 (default 28).
            - gwp_n2o (float): GWP for N2O (default 265).
            - dqi_inputs (dict): DQI scoring inputs.

        Args:
            calculation_input: Dictionary of calculation parameters.
            n_iterations: Number of Monte Carlo iterations. Must be in
                [100, 100000]. Defaults to 5000.
            seed: Optional random seed for reproducibility. If None,
                a default seed of 42 is used.

        Returns:
            UncertaintyResult with complete uncertainty characterization.

        Raises:
            ValueError: If required keys are missing, method/tier are
                not recognized, or n_iterations is out of range.
        """
        t_start = time.monotonic()

        # Validate iteration count
        if n_iterations < _MIN_ITERATIONS or n_iterations > _MAX_ITERATIONS:
            raise ValueError(
                f"n_iterations must be in [{_MIN_ITERATIONS}, {_MAX_ITERATIONS}], "
                f"got {n_iterations}"
            )

        if seed is None:
            seed = 42

        # Extract and validate inputs
        emissions = _to_decimal(calculation_input.get("total_co2e_kg", 0))
        if emissions < Decimal("0"):
            raise ValueError(
                f"total_co2e_kg must be >= 0, got {emissions}"
            )

        method = calculation_input.get("method", CalculationMethod.FUEL_BASED.value)
        self._validate_method(method)

        tier = calculation_input.get("tier", CalculationTier.TIER_1.value)
        self._validate_tier(tier)

        activity_source = calculation_input.get(
            "activity_data_source", "estimated"
        ).upper()
        if activity_source not in _ACTIVITY_DATA_UNCERTAINTY:
            activity_source = ActivityDataSource.ESTIMATED.value

        # DQI scoring
        dqi_inputs = calculation_input.get("dqi_inputs", {})
        dqi_score = self.score_data_quality(dqi_inputs)
        dqi_level = self._dqi_score_to_level(dqi_score)
        dqi_multiplier = Decimal(str(_DQI_MULTIPLIERS[dqi_level]))

        # Run analytical propagation
        analytical_result = self._analytical_propagation(
            emissions, method, tier, activity_source, dqi_multiplier
        )

        # Run Monte Carlo
        mc_result = self._run_mc_simulation(
            calculation_input, emissions, method, tier,
            activity_source, n_iterations, seed, dqi_multiplier,
        )

        # Combined uncertainty: conservative (max of analytical and MC)
        analytical_unc = analytical_result["combined_pct"]
        mc_unc_pct = Decimal("0")
        if mc_result["std"] is not None and emissions > Decimal("0"):
            mc_unc_pct = (
                Decimal(str(mc_result["std"])) / emissions * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        combined_pct = max(analytical_unc, mc_unc_pct)
        if combined_pct == Decimal("0") and emissions > Decimal("0"):
            default = _METHOD_DEFAULT_UNCERTAINTY.get(method, {}).get(
                tier, 0.20
            )
            combined_pct = Decimal(
                str(default * 100 * float(dqi_multiplier))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Build confidence intervals
        confidence_intervals = self._build_confidence_intervals(
            emissions, combined_pct, mc_result,
        )

        # Component uncertainties
        component_uncertainties = {
            k: Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            for k, v in analytical_result.get("components", {}).items()
        }

        # Contribution analysis
        contributions = self._contribution_analysis(
            method, tier, activity_source, dqi_multiplier
        )
        contribution_dec = {
            k: Decimal(str(v)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            for k, v in contributions.items()
        }

        # MC percentiles
        mc_percentiles = {
            k: Decimal(str(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            for k, v in mc_result.get("percentiles", {}).items()
        }

        # Provenance
        provenance_data = {
            "emissions_value": str(emissions),
            "method": method,
            "tier": tier,
            "activity_source": activity_source,
            "n_iterations": n_iterations,
            "seed": seed,
            "combined_pct": str(combined_pct),
            "dqi_score": str(dqi_score),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        timestamp = _utcnow().isoformat()

        result = UncertaintyResult(
            result_id=f"uq_{uuid4().hex[:12]}",
            emissions_value=emissions,
            emissions_unit="kg_CO2e",
            method=method,
            tier=tier,
            combined_uncertainty_pct=combined_pct,
            confidence_intervals=confidence_intervals,
            monte_carlo_mean=(
                Decimal(str(mc_result["mean"])).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                ) if mc_result["mean"] is not None else None
            ),
            monte_carlo_std=(
                Decimal(str(mc_result["std"])).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                ) if mc_result["std"] is not None else None
            ),
            monte_carlo_median=(
                Decimal(str(mc_result["median"])).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                ) if mc_result["median"] is not None else None
            ),
            monte_carlo_percentiles=mc_percentiles,
            monte_carlo_iterations=n_iterations,
            monte_carlo_seed=seed,
            analytical_uncertainty_pct=analytical_unc,
            component_uncertainties=component_uncertainties,
            contribution_analysis=contribution_dec,
            data_quality_score=Decimal(str(dqi_score)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            dqi_multiplier=dqi_multiplier,
            provenance_hash=provenance_hash,
            timestamp=timestamp,
            metadata={
                "method_range": self.get_method_uncertainty_range(method, tier),
            },
        )

        # Record in history
        with self._lock:
            self._assessment_history.append(result)

        # Provenance tracking
        self._record_provenance("monte_carlo", result.result_id, provenance_data)

        # Metrics
        elapsed = time.monotonic() - t_start
        self._record_metrics(method, elapsed)

        logger.debug(
            "MC uncertainty quantified: method=%s tier=%s "
            "emissions=%.1f combined=%.2f%% analytical=%.2f%% "
            "MC_mean=%.3f DQI=%.1f in %.1fms",
            method, tier, emissions, combined_pct, analytical_unc,
            mc_result["mean"] if mc_result["mean"] is not None else 0,
            dqi_score, elapsed * 1000,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Analytical Uncertainty
    # ------------------------------------------------------------------

    def calculate_analytical_uncertainty(
        self,
        emission_components: List[Dict[str, Any]],
    ) -> UncertaintyResult:
        """Calculate analytical uncertainty from emission components.

        Uses IPCC Approach 1 error propagation to combine component
        uncertainties. For multiplicative relationships (e.g. fuel * EF):

            sigma_rel_total = sqrt(sum(sigma_rel_i^2))

        For additive relationships (e.g. CO2 + CH4_CO2e + N2O_CO2e):

            sigma_total = sqrt(sum(sigma_i^2))

        Each component dict should contain:
            - name (str): Component name.
            - value (float/Decimal): Component value.
            - uncertainty_pct (float): Relative uncertainty as percentage.
            - relationship (str): "multiplicative" or "additive".

        Args:
            emission_components: List of component dictionaries.

        Returns:
            UncertaintyResult with analytical uncertainty characterization.

        Raises:
            ValueError: If emission_components is empty or missing fields.
        """
        t_start = time.monotonic()

        if not emission_components:
            raise ValueError("emission_components must not be empty")

        # Separate multiplicative and additive components
        multiplicative: List[Dict[str, Any]] = []
        additive: List[Dict[str, Any]] = []

        for comp in emission_components:
            if "name" not in comp or "value" not in comp:
                raise ValueError(
                    f"Each component must have 'name' and 'value': {comp}"
                )
            rel = comp.get("relationship", "multiplicative")
            if rel == "additive":
                additive.append(comp)
            else:
                multiplicative.append(comp)

        # Calculate combined multiplicative uncertainty
        mult_unc_squared = Decimal("0")
        component_uncertainties: Dict[str, Decimal] = {}
        for comp in multiplicative:
            unc_pct = _to_decimal(comp.get("uncertainty_pct", 10))
            unc_frac = unc_pct / Decimal("100")
            mult_unc_squared += unc_frac * unc_frac
            component_uncertainties[comp["name"]] = unc_pct

        # Calculate combined additive uncertainty
        add_unc_squared = Decimal("0")
        for comp in additive:
            unc_pct = _to_decimal(comp.get("uncertainty_pct", 10))
            val = _to_decimal(comp["value"])
            abs_unc = val * unc_pct / Decimal("100")
            add_unc_squared += abs_unc * abs_unc
            component_uncertainties[comp["name"]] = unc_pct

        # Calculate total emissions for additive portion
        total_additive = sum(
            _to_decimal(c["value"]) for c in additive
        ) if additive else Decimal("0")

        # Combine
        total_emissions = sum(
            _to_decimal(c["value"]) for c in emission_components
        )

        if total_emissions == Decimal("0"):
            combined_pct = Decimal("0")
        elif total_additive > Decimal("0") and multiplicative:
            # Mixed: combine both
            mult_unc_rel = _decimal_sqrt(mult_unc_squared)
            add_unc_abs = _decimal_sqrt(add_unc_squared)
            add_unc_rel = (
                add_unc_abs / total_additive
                if total_additive > Decimal("0") else Decimal("0")
            )
            combined_rel = _decimal_sqrt(
                mult_unc_rel * mult_unc_rel + add_unc_rel * add_unc_rel
            )
            combined_pct = (combined_rel * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        elif multiplicative:
            combined_pct = (_decimal_sqrt(mult_unc_squared) * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            add_unc_abs = _decimal_sqrt(add_unc_squared)
            combined_pct = (
                (add_unc_abs / total_additive * Decimal("100"))
                if total_additive > Decimal("0") else Decimal("0")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Confidence intervals from analytical
        confidence_intervals = self._analytical_confidence_intervals(
            total_emissions, combined_pct
        )

        # Contribution analysis
        total_var = float(mult_unc_squared) if float(mult_unc_squared) > 0 else 1.0
        contributions: Dict[str, Decimal] = {}
        for comp in multiplicative:
            unc = float(_to_decimal(comp.get("uncertainty_pct", 10))) / 100.0
            frac = (unc * unc) / total_var if total_var > 0 else 0
            contributions[comp["name"]] = Decimal(str(frac)).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )

        # Provenance
        provenance_data = {
            "components": [
                {"name": c["name"], "value": str(c["value"])}
                for c in emission_components
            ],
            "combined_pct": str(combined_pct),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        elapsed = time.monotonic() - t_start

        result = UncertaintyResult(
            result_id=f"uq_{uuid4().hex[:12]}",
            emissions_value=total_emissions,
            emissions_unit="kg_CO2e",
            method="ANALYTICAL",
            tier="N/A",
            combined_uncertainty_pct=combined_pct,
            confidence_intervals=confidence_intervals,
            monte_carlo_mean=None,
            monte_carlo_std=None,
            monte_carlo_median=None,
            monte_carlo_percentiles={},
            monte_carlo_iterations=None,
            monte_carlo_seed=None,
            analytical_uncertainty_pct=combined_pct,
            component_uncertainties=component_uncertainties,
            contribution_analysis=contributions,
            data_quality_score=Decimal("3.00"),
            dqi_multiplier=Decimal("1.00"),
            provenance_hash=provenance_hash,
            timestamp=_utcnow().isoformat(),
        )

        with self._lock:
            self._assessment_history.append(result)

        logger.debug(
            "Analytical uncertainty: %d components, combined=%.2f%% in %.1fms",
            len(emission_components), combined_pct, elapsed * 1000,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Data Quality Scoring
    # ------------------------------------------------------------------

    def score_data_quality(
        self,
        dqi_inputs: Dict[str, Any],
    ) -> float:
        """Score data quality using the 5-dimension DQI framework.

        Computes a weighted average score across the five DQI dimensions:
        reliability, completeness, temporal correlation, geographical
        correlation, and technological correlation.

        Each dimension is scored 1 (best) to 5 (worst). The overall
        score is the arithmetic mean across all provided dimensions.
        Missing dimensions default to 3 (average).

        Args:
            dqi_inputs: Dictionary mapping DQI dimension names to
                their criteria values. Recognized keys:
                - reliability: str (e.g. "direct_measurement")
                - completeness: str (e.g. "above_95_pct")
                - temporal_correlation: str (e.g. "same_year")
                - geographical_correlation: str (e.g. "same_region")
                - technological_correlation: str (e.g. "same_technology")

                Alternatively, direct numeric scores (1-5) may be provided.

        Returns:
            Weighted average DQI score (1.0 to 5.0). Lower is better.
        """
        if not dqi_inputs:
            return 3.0  # Default average quality

        dimension_map = {
            "reliability": DQICategory.RELIABILITY.value,
            "completeness": DQICategory.COMPLETENESS.value,
            "temporal_correlation": DQICategory.TEMPORAL_CORRELATION.value,
            "geographical_correlation": DQICategory.GEOGRAPHICAL_CORRELATION.value,
            "technological_correlation": DQICategory.TECHNOLOGICAL_CORRELATION.value,
        }

        scores: List[float] = []

        for dim_key, dqi_cat in dimension_map.items():
            val = dqi_inputs.get(dim_key)
            if val is None:
                scores.append(3.0)  # Default for missing dimension
                continue

            # If numeric, use directly
            if isinstance(val, (int, float)):
                score = max(1.0, min(5.0, float(val)))
                scores.append(score)
                continue

            # If string, look up in criteria table
            criteria = _DQI_CRITERIA.get(dqi_cat, {})
            if isinstance(val, str) and val in criteria:
                scores.append(float(criteria[val]))
            else:
                scores.append(3.0)  # Default for unrecognized value

        return sum(scores) / len(scores) if scores else 3.0

    # ------------------------------------------------------------------
    # Public API: Method Uncertainty Range
    # ------------------------------------------------------------------

    def get_method_uncertainty_range(
        self,
        method: str,
        tier: str,
    ) -> Tuple[float, float]:
        """Get the expected uncertainty range for a method and tier.

        Returns the lower and upper bounds of the expected combined
        uncertainty (half-width of 95% CI as percentage) for the
        specified calculation method and tier.

        Args:
            method: Calculation method (FUEL_BASED, DISTANCE_BASED,
                SPEND_BASED).
            tier: Calculation tier (TIER_1, TIER_2, TIER_3).

        Returns:
            Tuple of (lower_pct, upper_pct) as floats. For example,
            (5.0, 10.0) means the combined uncertainty is expected
            to be between 5% and 10%.

        Raises:
            ValueError: If method or tier is not recognized.
        """
        self._validate_method(method)
        self._validate_tier(tier)

        ranges = _METHOD_UNCERTAINTY_RANGES.get(method, {})
        low, high = ranges.get(tier, (0.15, 0.30))
        return (low * 100, high * 100)

    # ------------------------------------------------------------------
    # Public API: Confidence Intervals
    # ------------------------------------------------------------------

    def get_confidence_intervals(
        self,
        samples: List[float],
        levels: Optional[List[float]] = None,
    ) -> Dict[str, Tuple[Decimal, Decimal]]:
        """Compute confidence intervals from a set of samples.

        Uses percentile-based approach for non-parametric estimation.

        Args:
            samples: List of numeric samples.
            levels: List of confidence levels as fractions (e.g.
                [0.90, 0.95, 0.99]). Defaults to [0.90, 0.95, 0.99].

        Returns:
            Dictionary mapping confidence level percentage strings
            to (lower, upper) Decimal tuples.

        Raises:
            ValueError: If samples is empty or levels contain invalid
                values.
        """
        if not samples:
            raise ValueError("samples must not be empty")

        if levels is None:
            levels = [0.90, 0.95, 0.99]

        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        result: Dict[str, Tuple[Decimal, Decimal]] = {}

        for level in levels:
            if level <= 0 or level >= 1:
                raise ValueError(
                    f"Confidence level must be in (0, 1), got {level}"
                )

            alpha = 1.0 - level
            lower_idx = max(0, int(math.floor(n * alpha / 2)))
            upper_idx = min(n - 1, int(math.ceil(n * (1 - alpha / 2))) - 1)

            lower_val = Decimal(str(sorted_samples[lower_idx])).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            upper_val = Decimal(str(sorted_samples[upper_idx])).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            key = str(int(level * 100))
            result[key] = (lower_val, upper_val)

        return result

    # ------------------------------------------------------------------
    # Public API: Combine Uncertainties
    # ------------------------------------------------------------------

    def combine_uncertainties(
        self,
        component_uncertainties: List[Decimal],
    ) -> Decimal:
        """Combine component uncertainties using root-sum-of-squares.

        For multiplicative relationships, the combined relative
        uncertainty is:

            sigma_total = sqrt(sum(sigma_i^2))

        Args:
            component_uncertainties: List of relative uncertainties
                as Decimal fractions (e.g. [Decimal("0.05"),
                Decimal("0.10")]).

        Returns:
            Combined relative uncertainty as a Decimal fraction.

        Raises:
            ValueError: If component_uncertainties is empty.
        """
        if not component_uncertainties:
            raise ValueError("component_uncertainties must not be empty")

        sum_sq = Decimal("0")
        for unc in component_uncertainties:
            unc = _to_decimal(unc)
            sum_sq += unc * unc

        return _decimal_sqrt(sum_sq)

    # ------------------------------------------------------------------
    # Public API: Sensitivity Analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        calculation_input: Dict[str, Any],
        parameters: List[str],
        perturbation_pct: float = 10.0,
    ) -> List[SensitivityResult]:
        """Run one-at-a-time sensitivity analysis on parameters.

        Perturbs each specified parameter by +perturbation_pct% while
        holding all others constant, then computes the resulting
        change in emissions.

        Supported parameters:
            - fuel_litres, emission_factor, distance_km, fuel_economy,
              spend_amount, spend_ef, gwp_ch4, gwp_n2o

        Args:
            calculation_input: Baseline calculation parameters. Must
                include "total_co2e_kg" and "method".
            parameters: List of parameter names to analyse.
            perturbation_pct: Perturbation magnitude as a percentage.
                Defaults to 10.0%.

        Returns:
            List of SensitivityResult, one per parameter.

        Raises:
            ValueError: If parameters is empty or baseline emissions
                cannot be determined.
        """
        if not parameters:
            raise ValueError("parameters must not be empty")

        base_emissions = _to_decimal(
            calculation_input.get("total_co2e_kg", 0)
        )
        if base_emissions <= Decimal("0"):
            raise ValueError(
                "total_co2e_kg must be > 0 for sensitivity analysis"
            )

        method = calculation_input.get("method", CalculationMethod.FUEL_BASED.value)
        perturbation_dec = _to_decimal(perturbation_pct) / Decimal("100")

        results: List[SensitivityResult] = []

        for param in parameters:
            base_val = _to_decimal(calculation_input.get(param, 1.0))
            if base_val == Decimal("0"):
                base_val = Decimal("1")

            perturbed_val = base_val * (Decimal("1") + perturbation_dec)

            # Calculate perturbed emissions using proportional scaling
            perturbed_emissions = self._calculate_perturbed_emissions(
                base_emissions, method, param,
                base_val, perturbed_val, calculation_input,
            )

            emissions_change_pct = Decimal("0")
            sensitivity_coeff = Decimal("0")
            if base_emissions > Decimal("0"):
                emissions_change_pct = (
                    (perturbed_emissions - base_emissions) / base_emissions
                    * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                if perturbation_dec > Decimal("0"):
                    sensitivity_coeff = (
                        emissions_change_pct / _to_decimal(perturbation_pct)
                    ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

            results.append(SensitivityResult(
                parameter=param,
                base_value=base_val.quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                ),
                perturbed_value=perturbed_val.quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                ),
                perturbation_pct=_to_decimal(perturbation_pct).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                ),
                base_emissions=base_emissions.quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                ),
                perturbed_emissions=perturbed_emissions.quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                ),
                emissions_change_pct=emissions_change_pct,
                sensitivity_coefficient=sensitivity_coeff,
            ))

        return results

    # ------------------------------------------------------------------
    # Public API: History
    # ------------------------------------------------------------------

    def get_assessment_history(self) -> List[UncertaintyResult]:
        """Return a copy of the assessment history.

        Returns:
            List of all UncertaintyResult objects produced.
        """
        with self._lock:
            return list(self._assessment_history)

    def clear_history(self) -> int:
        """Clear the assessment history.

        Returns:
            Number of records cleared.
        """
        with self._lock:
            count = len(self._assessment_history)
            self._assessment_history.clear()
        logger.info("Assessment history cleared: %d records", count)
        return count

    # ------------------------------------------------------------------
    # Internal: Monte Carlo Simulation
    # ------------------------------------------------------------------

    def _run_mc_simulation(
        self,
        calculation_input: Dict[str, Any],
        emissions: Decimal,
        method: str,
        tier: str,
        activity_source: str,
        n_iterations: int,
        seed: int,
        dqi_multiplier: Decimal,
    ) -> Dict[str, Any]:
        """Run the Monte Carlo simulation inner loop.

        Creates a dedicated Random instance for thread safety and
        generates samples based on the calculation method.

        Args:
            calculation_input: Calculation parameters.
            emissions: Central emission estimate.
            method: Calculation method.
            tier: Calculation tier.
            activity_source: Activity data source type.
            n_iterations: Number of iterations.
            seed: Random seed.
            dqi_multiplier: DQI uncertainty multiplier.

        Returns:
            Dictionary with MC results (mean, std, median, percentiles,
            confidence_intervals).
        """
        rng = random.Random(seed)
        dqi_mult = float(dqi_multiplier)
        emissions_f = float(emissions)

        if emissions_f <= 0:
            return self._empty_mc_result()

        samples: List[float] = []

        if method == CalculationMethod.FUEL_BASED.value:
            samples = self._mc_fuel_based(
                calculation_input, emissions_f, tier,
                activity_source, n_iterations, rng, dqi_mult,
            )
        elif method == CalculationMethod.DISTANCE_BASED.value:
            samples = self._mc_distance_based(
                calculation_input, emissions_f, tier,
                activity_source, n_iterations, rng, dqi_mult,
            )
        elif method == CalculationMethod.SPEND_BASED.value:
            samples = self._mc_spend_based(
                calculation_input, emissions_f, tier,
                n_iterations, rng, dqi_mult,
            )
        else:
            # Fallback: apply combined uncertainty directly
            samples = self._mc_generic(
                emissions_f, method, tier, n_iterations, rng, dqi_mult,
            )

        if not samples:
            return self._empty_mc_result()

        return self._compute_mc_statistics(samples)

    def _mc_fuel_based(
        self,
        inputs: Dict[str, Any],
        emissions: float,
        tier: str,
        activity_source: str,
        n: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Monte Carlo for fuel-based method.

        Emissions = fuel * EF * oxidation_factor
        Each component sampled from Normal distribution.

        Args:
            inputs: Calculation parameters.
            emissions: Central emission estimate.
            tier: Tier for EF uncertainty.
            activity_source: Activity data source for AD uncertainty.
            n: Number of iterations.
            rng: Random instance.
            dqi_mult: DQI multiplier.

        Returns:
            List of simulated emission values.
        """
        fuel = float(inputs.get("fuel_litres", 0))
        ef = float(inputs.get("emission_factor", 0))

        # If individual parameters not available, use scaling approach
        if fuel <= 0 or ef <= 0:
            return self._mc_scaling(
                emissions, tier, activity_source, n, rng, dqi_mult
            )

        ad_unc = _ACTIVITY_DATA_UNCERTAINTY.get(activity_source, 0.10) * dqi_mult
        ef_unc = _EMISSION_FACTOR_UNCERTAINTY.get(tier, 0.10) * dqi_mult

        samples: List[float] = []
        for _ in range(n):
            s_fuel = max(0.0, rng.gauss(fuel, fuel * ad_unc))
            s_ef = max(0.0, rng.gauss(ef, ef * ef_unc))
            samples.append(s_fuel * s_ef)

        return samples

    def _mc_distance_based(
        self,
        inputs: Dict[str, Any],
        emissions: float,
        tier: str,
        activity_source: str,
        n: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Monte Carlo for distance-based method.

        Emissions = distance * fuel_economy / 100 * EF
        Or: distance * distance_ef

        Args:
            inputs: Calculation parameters.
            emissions: Central emission estimate.
            tier: Tier.
            activity_source: Activity data source.
            n: Number of iterations.
            rng: Random instance.
            dqi_mult: DQI multiplier.

        Returns:
            List of simulated emission values.
        """
        distance = float(inputs.get("distance_km", 0))
        fuel_econ = float(inputs.get("fuel_economy", 0))
        ef = float(inputs.get("emission_factor", 0))

        if distance <= 0:
            return self._mc_scaling(
                emissions, tier, activity_source, n, rng, dqi_mult
            )

        dist_unc = _DISTANCE_UNCERTAINTY.get(
            inputs.get("distance_source", "default"), 0.10
        ) * dqi_mult
        fe_unc = _FUEL_ECONOMY_UNCERTAINTY * dqi_mult
        ef_unc = _EMISSION_FACTOR_UNCERTAINTY.get(tier, 0.10) * dqi_mult

        if fuel_econ > 0 and ef > 0:
            samples: List[float] = []
            for _ in range(n):
                s_dist = max(0.0, rng.gauss(distance, distance * dist_unc))
                s_fe = max(0.1, rng.gauss(fuel_econ, fuel_econ * fe_unc))
                s_ef = max(0.0, rng.gauss(ef, ef * ef_unc))
                samples.append(s_dist * s_fe / 100.0 * s_ef)
            return samples
        else:
            # Use distance emission factor approach
            return self._mc_scaling(
                emissions, tier, activity_source, n, rng, dqi_mult
            )

    def _mc_spend_based(
        self,
        inputs: Dict[str, Any],
        emissions: float,
        tier: str,
        n: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Monte Carlo for spend-based method.

        Emissions = spend * spend_EF
        High uncertainty due to financial proxy usage.

        Args:
            inputs: Calculation parameters.
            emissions: Central emission estimate.
            tier: Tier.
            n: Number of iterations.
            rng: Random instance.
            dqi_mult: DQI multiplier.

        Returns:
            List of simulated emission values.
        """
        spend = float(inputs.get("spend_amount", 0))
        spend_ef = float(inputs.get("spend_ef", 0))

        if spend <= 0 or spend_ef <= 0:
            return self._mc_scaling(
                emissions, tier, "SCREENING", n, rng, dqi_mult
            )

        spend_unc = 0.15 * dqi_mult  # Spend data uncertainty
        ef_unc = 0.30 * dqi_mult     # Spend EF uncertainty (high)

        samples: List[float] = []
        for _ in range(n):
            s_spend = max(0.0, rng.gauss(spend, spend * spend_unc))
            s_ef = max(0.0, rng.gauss(spend_ef, spend_ef * ef_unc))
            samples.append(s_spend * s_ef)

        return samples

    def _mc_scaling(
        self,
        emissions: float,
        tier: str,
        activity_source: str,
        n: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Monte Carlo fallback using total emissions scaling.

        When individual parameters are not available, applies combined
        uncertainty directly to the total emission estimate.

        Args:
            emissions: Central emission estimate.
            tier: Tier.
            activity_source: Activity data source.
            n: Number of iterations.
            rng: Random instance.
            dqi_mult: DQI multiplier.

        Returns:
            List of simulated emission values.
        """
        ad_unc = _ACTIVITY_DATA_UNCERTAINTY.get(activity_source, 0.10) * dqi_mult
        ef_unc = _EMISSION_FACTOR_UNCERTAINTY.get(tier, 0.10) * dqi_mult
        gwp_unc = _GWP_UNCERTAINTY * dqi_mult

        combined = math.sqrt(ad_unc**2 + ef_unc**2 + gwp_unc**2)

        samples: List[float] = []
        for _ in range(n):
            s = max(0.0, rng.gauss(emissions, emissions * combined))
            samples.append(s)

        return samples

    def _mc_generic(
        self,
        emissions: float,
        method: str,
        tier: str,
        n: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Generic Monte Carlo using method default uncertainty.

        Args:
            emissions: Central emission estimate.
            method: Calculation method.
            tier: Calculation tier.
            n: Number of iterations.
            rng: Random instance.
            dqi_mult: DQI multiplier.

        Returns:
            List of simulated emission values.
        """
        default = _METHOD_DEFAULT_UNCERTAINTY.get(method, {}).get(tier, 0.20)
        unc = default * dqi_mult

        samples: List[float] = []
        for _ in range(n):
            s = max(0.0, rng.gauss(emissions, emissions * unc))
            samples.append(s)

        return samples

    def _compute_mc_statistics(
        self,
        samples: List[float],
    ) -> Dict[str, Any]:
        """Compute statistics from Monte Carlo samples.

        Args:
            samples: List of simulated emission values.

        Returns:
            Dictionary with mean, std, median, percentiles, and
            confidence intervals.
        """
        n = len(samples)
        sorted_s = sorted(samples)

        mean = sum(sorted_s) / n
        variance = sum((x - mean) ** 2 for x in sorted_s) / (n - 1) if n > 1 else 0
        std = math.sqrt(variance)
        median = sorted_s[n // 2] if n % 2 == 1 else (
            sorted_s[n // 2 - 1] + sorted_s[n // 2]
        ) / 2

        percentiles: Dict[str, float] = {}
        for p in [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]:
            idx = max(0, min(n - 1, int(n * p / 100)))
            percentiles[str(p)] = sorted_s[idx]

        # Confidence intervals from percentiles
        cis: Dict[str, Tuple[float, float]] = {
            "90": (sorted_s[max(0, int(n * 0.05))],
                   sorted_s[min(n - 1, int(n * 0.95))]),
            "95": (sorted_s[max(0, int(n * 0.025))],
                   sorted_s[min(n - 1, int(n * 0.975))]),
            "99": (sorted_s[max(0, int(n * 0.005))],
                   sorted_s[min(n - 1, int(n * 0.995))]),
        }

        return {
            "mean": mean,
            "std": std,
            "median": median,
            "percentiles": percentiles,
            "confidence_intervals": cis,
        }

    def _empty_mc_result(self) -> Dict[str, Any]:
        """Return an empty MC result structure.

        Returns:
            Dictionary with None/empty values for MC results.
        """
        return {
            "mean": None,
            "std": None,
            "median": None,
            "percentiles": {},
            "confidence_intervals": {},
        }

    # ------------------------------------------------------------------
    # Internal: Analytical Error Propagation
    # ------------------------------------------------------------------

    def _analytical_propagation(
        self,
        emissions: Decimal,
        method: str,
        tier: str,
        activity_source: str,
        dqi_multiplier: Decimal,
    ) -> Dict[str, Any]:
        """Compute analytical uncertainty via IPCC Approach 1.

        For multiplicative chains (AD * EF * GWP):
            sigma_rel = sqrt(sigma_AD^2 + sigma_EF^2 + sigma_GWP^2)

        Args:
            emissions: Central emission estimate.
            method: Calculation method.
            tier: Calculation tier.
            activity_source: Activity data source.
            dqi_multiplier: DQI uncertainty multiplier.

        Returns:
            Dictionary with combined_pct and components.
        """
        dqi_mult = float(dqi_multiplier)

        # Component uncertainties (relative, as fractions)
        components: Dict[str, float] = {}

        ad_unc = _ACTIVITY_DATA_UNCERTAINTY.get(
            activity_source, 0.10
        ) * dqi_mult
        ef_unc = _EMISSION_FACTOR_UNCERTAINTY.get(
            tier, 0.10
        ) * dqi_mult
        gwp_unc = _GWP_UNCERTAINTY * dqi_mult

        if method == CalculationMethod.FUEL_BASED.value:
            components["activity_data"] = ad_unc
            components["emission_factor"] = ef_unc
            components["gwp"] = gwp_unc

        elif method == CalculationMethod.DISTANCE_BASED.value:
            dist_unc = 0.10 * dqi_mult
            fe_unc = _FUEL_ECONOMY_UNCERTAINTY * dqi_mult
            components["distance"] = dist_unc
            components["fuel_economy"] = fe_unc
            components["emission_factor"] = ef_unc
            components["gwp"] = gwp_unc

        elif method == CalculationMethod.SPEND_BASED.value:
            spend_unc = 0.15 * dqi_mult
            spend_ef_unc = 0.30 * dqi_mult
            components["spend_data"] = spend_unc
            components["spend_emission_factor"] = spend_ef_unc
            components["gwp"] = gwp_unc

        else:
            components["total"] = 0.20 * dqi_mult

        # Root-sum-of-squares
        sum_sq = sum(v**2 for v in components.values())
        combined_rel = math.sqrt(sum_sq)
        combined_pct = Decimal(str(combined_rel * 100)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Convert components to percentages
        comp_pct = {k: v * 100 for k, v in components.items()}

        return {
            "combined_pct": combined_pct,
            "components": comp_pct,
        }

    # ------------------------------------------------------------------
    # Internal: Confidence Intervals
    # ------------------------------------------------------------------

    def _build_confidence_intervals(
        self,
        emissions: Decimal,
        combined_pct: Decimal,
        mc_result: Dict[str, Any],
    ) -> Dict[str, Tuple[Decimal, Decimal]]:
        """Build confidence intervals from combined uncertainty and MC.

        Uses analytical CIs as baseline, then widens to MC CIs if they
        are broader.

        Args:
            emissions: Central emission estimate.
            combined_pct: Combined uncertainty percentage.
            mc_result: Monte Carlo result dictionary.

        Returns:
            Dictionary of confidence intervals.
        """
        cis = self._analytical_confidence_intervals(emissions, combined_pct)

        # Use MC CIs if available and wider
        mc_cis = mc_result.get("confidence_intervals", {})
        for ci_key in mc_cis:
            if ci_key in cis:
                mc_lower = Decimal(str(mc_cis[ci_key][0])).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                mc_upper = Decimal(str(mc_cis[ci_key][1])).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                ana_lower, ana_upper = cis[ci_key]
                if (mc_upper - mc_lower) > (ana_upper - ana_lower):
                    cis[ci_key] = (mc_lower, mc_upper)

        return cis

    def _analytical_confidence_intervals(
        self,
        emissions: Decimal,
        combined_pct: Decimal,
    ) -> Dict[str, Tuple[Decimal, Decimal]]:
        """Compute confidence intervals from analytical uncertainty.

        Args:
            emissions: Central emission estimate.
            combined_pct: Combined uncertainty as percentage.

        Returns:
            Dictionary of confidence intervals at 90/95/99% levels.
        """
        cis: Dict[str, Tuple[Decimal, Decimal]] = {}

        for ci_label, z_score in _CONFIDENCE_Z_SCORES.items():
            half_width_95 = combined_pct / Decimal("100") * emissions
            scaling = Decimal(str(z_score / 1.9600))
            half_width = (half_width_95 * scaling).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            lower = max(Decimal("0"), emissions - half_width)
            upper = emissions + half_width
            cis[ci_label] = (
                lower.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                upper.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            )

        return cis

    # ------------------------------------------------------------------
    # Internal: Contribution Analysis
    # ------------------------------------------------------------------

    def _contribution_analysis(
        self,
        method: str,
        tier: str,
        activity_source: str,
        dqi_multiplier: Decimal,
    ) -> Dict[str, float]:
        """Compute contribution of each component to total variance.

        The contribution of each component is its squared uncertainty
        divided by the total squared uncertainty.

        Args:
            method: Calculation method.
            tier: Calculation tier.
            activity_source: Activity data source.
            dqi_multiplier: DQI multiplier.

        Returns:
            Dictionary mapping component names to fractional
            contributions (summing to 1.0).
        """
        result = self._analytical_propagation(
            Decimal("100"), method, tier, activity_source, dqi_multiplier
        )
        components = result.get("components", {})

        total_sq = sum(v**2 for v in components.values())
        if total_sq <= 0:
            return {k: 0.0 for k in components}

        return {k: (v**2 / total_sq) for k, v in components.items()}

    # ------------------------------------------------------------------
    # Internal: Sensitivity Calculation
    # ------------------------------------------------------------------

    def _calculate_perturbed_emissions(
        self,
        base_emissions: Decimal,
        method: str,
        param: str,
        base_val: Decimal,
        perturbed_val: Decimal,
        inputs: Dict[str, Any],
    ) -> Decimal:
        """Calculate emissions with a perturbed parameter value.

        For multiplicative relationships, the perturbed emissions scale
        proportionally with the parameter change.

        Args:
            base_emissions: Baseline emission estimate.
            method: Calculation method.
            param: Parameter being perturbed.
            base_val: Original parameter value.
            perturbed_val: Perturbed parameter value.
            inputs: Full calculation input dictionary.

        Returns:
            Perturbed emission estimate.
        """
        if base_val == Decimal("0"):
            return base_emissions

        # For multiplicative parameters, emissions scale linearly
        multiplicative_params = {
            "fuel_litres", "emission_factor", "distance_km",
            "fuel_economy", "spend_amount", "spend_ef",
        }

        if param in multiplicative_params:
            ratio = perturbed_val / base_val
            return (base_emissions * ratio).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        # For GWP parameters, apply proportionally to non-CO2 fraction
        if param in {"gwp_ch4", "gwp_n2o"}:
            # Assume non-CO2 fraction is ~5% for typical mobile combustion
            non_co2_fraction = Decimal("0.05")
            ratio = perturbed_val / base_val
            co2_portion = base_emissions * (Decimal("1") - non_co2_fraction)
            non_co2_portion = base_emissions * non_co2_fraction * ratio
            return (co2_portion + non_co2_portion).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        # Default: proportional scaling
        ratio = perturbed_val / base_val
        return (base_emissions * ratio).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_method(self, method: str) -> None:
        """Validate calculation method.

        Args:
            method: Method string to validate.

        Raises:
            ValueError: If not recognized.
        """
        valid = {m.value for m in CalculationMethod}
        if method not in valid:
            raise ValueError(
                f"Unrecognized method '{method}'. Supported: {sorted(valid)}"
            )

    def _validate_tier(self, tier: str) -> None:
        """Validate calculation tier.

        Args:
            tier: Tier string to validate.

        Raises:
            ValueError: If not recognized.
        """
        valid = {t.value for t in CalculationTier}
        if tier not in valid:
            raise ValueError(
                f"Unrecognized tier '{tier}'. Supported: {sorted(valid)}"
            )

    def _dqi_score_to_level(self, score: float) -> int:
        """Convert a continuous DQI score to a discrete level (1-5).

        Args:
            score: DQI score (1.0 to 5.0).

        Returns:
            Integer level 1-5.
        """
        rounded = max(1, min(5, round(score)))
        return rounded

    # ------------------------------------------------------------------
    # Internal: Provenance and Metrics
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        action: str,
        entity_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Record provenance tracking event if available.

        Args:
            action: Action description.
            entity_id: Entity identifier.
            data: Provenance data dictionary.
        """
        if _PROVENANCE_AVAILABLE and _get_provenance_tracker is not None:
            try:
                tracker = _get_provenance_tracker()
                tracker.record(
                    entity_type="uncertainty",
                    action=action,
                    entity_id=entity_id,
                    data=data,
                    metadata={"engine": "UncertaintyQuantifierEngine"},
                )
            except Exception:
                logger.debug("Provenance recording skipped", exc_info=True)

    def _record_metrics(self, method: str, elapsed: float) -> None:
        """Record Prometheus metrics if available.

        Args:
            method: Method for labelling.
            elapsed: Elapsed time in seconds.
        """
        if _METRICS_AVAILABLE and _record_uncertainty is not None:
            try:
                _record_uncertainty(method, "complete")
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)
        if _METRICS_AVAILABLE and _observe_calculation_duration is not None:
            try:
                _observe_calculation_duration(elapsed)
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)


# ===========================================================================
# Helper: Decimal Square Root
# ===========================================================================


def _decimal_sqrt(value: Decimal) -> Decimal:
    """Compute square root of a Decimal value.

    Uses Python float sqrt and converts back to Decimal for
    sufficient precision in uncertainty calculations.

    Args:
        value: Non-negative Decimal value.

    Returns:
        Square root as Decimal.
    """
    if value < Decimal("0"):
        raise ValueError(f"Cannot take sqrt of negative value: {value}")
    if value == Decimal("0"):
        return Decimal("0")
    return Decimal(str(math.sqrt(float(value))))
