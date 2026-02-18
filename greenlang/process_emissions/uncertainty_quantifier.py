# -*- coding: utf-8 -*-
"""
UncertaintyQuantifierEngine - Monte Carlo & Analytical Uncertainty (Engine 5 of 6)

AGENT-MRV-004: Process Emissions Agent

Quantifies the uncertainty of industrial process emission calculations
using two complementary methods:

    1. **Monte Carlo simulation**: Draws from parameterised distributions
       (normal for activity data and emission factors, log-normal for
       abatement efficiencies) and produces full percentile tables with
       90/95/99% confidence intervals. Configurable iteration count
       (default 5000) with explicit seed support for bit-perfect
       reproducibility.

    2. **Analytical error propagation** (IPCC Approach 1): Combined
       relative uncertainty for multiplicative chains via root-sum-of-
       squares of component uncertainties.

Process-Specific Uncertainty Ranges (half-width of 95% CI):
    - Cement CO2:          +/-2-5%  (Tier 1), +/-1-2% (Tier 3)
    - Lime CO2:            +/-3-7%
    - Iron/steel CO2:      +/-5-15% (depends on route)
    - Aluminum CO2+PFC:    +/-5-20% (PFC high uncertainty)
    - Nitric acid N2O:     +/-10-30% (depends on abatement)
    - Ammonia CO2:         +/-3-8%
    - Semiconductor PFC:   +/-15-40% (highest uncertainty)

Activity Data Uncertainty:
    - Metered/weighed:    +/-1-3%  (direct measurement, calibrated scales)
    - Estimated:          +/-5-10% (production records, invoices)
    - Screening:          +/-15-25% (engineering estimates, defaults)

Emission Factor Uncertainty:
    - Tier 3: +/-2-5%   (facility-specific, measured)
    - Tier 2: +/-5-15%  (country-/region-specific)
    - Tier 1: +/-15-50% (IPCC default)

Abatement Efficiency Uncertainty: +/-5-20% (depends on technology)
GWP Uncertainty: +/-10% (IPCC assessment report uncertainty)

Data Quality Indicator (DQI) Scoring (1-5 scale):
    5 dimensions, geometric mean composite score:
    - Reliability: direct measurement (1) ... unknown (5)
    - Completeness: >95% (1) ... <40% (5)
    - Temporal correlation: same year (1) ... >5 years (5)
    - Geographical correlation: same region (1) ... unknown (5)
    - Technological correlation: same technology (1) ... unknown (5)

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
    >>> from greenlang.process_emissions.uncertainty_quantifier import (
    ...     UncertaintyQuantifierEngine,
    ... )
    >>> engine = UncertaintyQuantifierEngine()
    >>> result = engine.run_monte_carlo(
    ...     calculation_input={
    ...         "total_co2e_kg": 500000,
    ...         "process_type": "CEMENT",
    ...         "calculation_method": "EMISSION_FACTOR",
    ...         "tier": "TIER_2",
    ...         "activity_data_source": "metered",
    ...         "production_tonnes": 10000,
    ...         "emission_factor": 0.507,
    ...     },
    ...     n_iterations=5000,
    ... )
    >>> print(result.confidence_intervals["95"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-004 Process Emissions (GL-MRV-SCOPE1-004)
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
    from greenlang.process_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.process_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.process_emissions.metrics import (
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


def _decimal_sqrt(value: Decimal) -> Decimal:
    """Compute square root of a Decimal value.

    Uses Python float sqrt and converts back to Decimal for
    sufficient precision in uncertainty calculations.

    Args:
        value: Non-negative Decimal value.

    Returns:
        Square root as Decimal.

    Raises:
        ValueError: If value is negative.
    """
    if value < Decimal("0"):
        raise ValueError(f"Cannot take sqrt of negative value: {value}")
    if value == Decimal("0"):
        return Decimal("0")
    return Decimal(str(math.sqrt(float(value))))


# ===========================================================================
# Enumerations
# ===========================================================================


class ProcessCalculationMethod(str, Enum):
    """Process emissions calculation method types.

    EMISSION_FACTOR: Emissions from production * emission factor.
    MASS_BALANCE: Emissions from carbon mass balance across inputs/outputs.
    STOICHIOMETRIC: Emissions from stoichiometric reaction equations.
    DIRECT_MEASUREMENT: Emissions from CEMS or stack testing.
    """

    EMISSION_FACTOR = "EMISSION_FACTOR"
    MASS_BALANCE = "MASS_BALANCE"
    STOICHIOMETRIC = "STOICHIOMETRIC"
    DIRECT_MEASUREMENT = "DIRECT_MEASUREMENT"


class ProcessCalculationTier(str, Enum):
    """Calculation tier for emission factor specificity.

    TIER_3: Facility-specific, measured emission factors.
    TIER_2: Country- or region-specific emission factors.
    TIER_1: IPCC default emission factors.
    """

    TIER_3 = "TIER_3"
    TIER_2 = "TIER_2"
    TIER_1 = "TIER_1"


class ActivityDataSource(str, Enum):
    """Activity data source classification.

    METERED: Direct measurement (calibrated scales, flow meters).
    ESTIMATED: Derived from production records, invoices, purchase orders.
    SCREENING: Engineering estimates, default assumptions, proxy data.
    """

    METERED = "METERED"
    ESTIMATED = "ESTIMATED"
    SCREENING = "SCREENING"


class DQICategory(str, Enum):
    """Data Quality Indicator scoring categories (5 dimensions).

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
    ActivityDataSource.METERED.value: 0.02,
    ActivityDataSource.ESTIMATED.value: 0.075,
    ActivityDataSource.SCREENING.value: 0.20,
}

# Emission factor uncertainty by tier (half-width as fraction)
_EMISSION_FACTOR_UNCERTAINTY: Dict[str, float] = {
    ProcessCalculationTier.TIER_3.value: 0.035,
    ProcessCalculationTier.TIER_2.value: 0.10,
    ProcessCalculationTier.TIER_1.value: 0.30,
}

# GWP uncertainty (half-width as fraction)
_GWP_UNCERTAINTY: float = 0.10

# Abatement efficiency uncertainty (half-width as fraction)
_ABATEMENT_UNCERTAINTY: float = 0.10

# Method-specific uncertainty (half-width as fraction)
_METHOD_UNCERTAINTY: Dict[str, float] = {
    ProcessCalculationMethod.EMISSION_FACTOR.value: 0.05,
    ProcessCalculationMethod.MASS_BALANCE.value: 0.08,
    ProcessCalculationMethod.STOICHIOMETRIC.value: 0.03,
    ProcessCalculationMethod.DIRECT_MEASUREMENT.value: 0.02,
}

# Process-specific uncertainty ranges: (low, high) as half-width fractions
# These represent the *total* combined uncertainty for each process type
# and tier combination, sourced from IPCC 2006 Guidelines.
_PROCESS_UNCERTAINTY_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "CEMENT": {
        ProcessCalculationTier.TIER_3.value: (0.01, 0.02),
        ProcessCalculationTier.TIER_2.value: (0.02, 0.03),
        ProcessCalculationTier.TIER_1.value: (0.02, 0.05),
    },
    "LIME": {
        ProcessCalculationTier.TIER_3.value: (0.02, 0.04),
        ProcessCalculationTier.TIER_2.value: (0.03, 0.05),
        ProcessCalculationTier.TIER_1.value: (0.03, 0.07),
    },
    "GLASS": {
        ProcessCalculationTier.TIER_3.value: (0.02, 0.05),
        ProcessCalculationTier.TIER_2.value: (0.03, 0.08),
        ProcessCalculationTier.TIER_1.value: (0.05, 0.10),
    },
    "CERAMICS": {
        ProcessCalculationTier.TIER_3.value: (0.03, 0.05),
        ProcessCalculationTier.TIER_2.value: (0.05, 0.08),
        ProcessCalculationTier.TIER_1.value: (0.05, 0.12),
    },
    "SODA_ASH": {
        ProcessCalculationTier.TIER_3.value: (0.02, 0.04),
        ProcessCalculationTier.TIER_2.value: (0.03, 0.06),
        ProcessCalculationTier.TIER_1.value: (0.04, 0.08),
    },
    "IRON_STEEL_BF_BOF": {
        ProcessCalculationTier.TIER_3.value: (0.03, 0.05),
        ProcessCalculationTier.TIER_2.value: (0.05, 0.10),
        ProcessCalculationTier.TIER_1.value: (0.05, 0.15),
    },
    "IRON_STEEL_EAF": {
        ProcessCalculationTier.TIER_3.value: (0.05, 0.08),
        ProcessCalculationTier.TIER_2.value: (0.08, 0.12),
        ProcessCalculationTier.TIER_1.value: (0.08, 0.15),
    },
    "IRON_STEEL_DRI": {
        ProcessCalculationTier.TIER_3.value: (0.04, 0.07),
        ProcessCalculationTier.TIER_2.value: (0.07, 0.12),
        ProcessCalculationTier.TIER_1.value: (0.08, 0.15),
    },
    "ALUMINUM_PREBAKE": {
        ProcessCalculationTier.TIER_3.value: (0.03, 0.08),
        ProcessCalculationTier.TIER_2.value: (0.05, 0.12),
        ProcessCalculationTier.TIER_1.value: (0.05, 0.20),
    },
    "ALUMINUM_SODERBERG": {
        ProcessCalculationTier.TIER_3.value: (0.05, 0.10),
        ProcessCalculationTier.TIER_2.value: (0.08, 0.15),
        ProcessCalculationTier.TIER_1.value: (0.08, 0.20),
    },
    "NITRIC_ACID": {
        ProcessCalculationTier.TIER_3.value: (0.05, 0.10),
        ProcessCalculationTier.TIER_2.value: (0.10, 0.20),
        ProcessCalculationTier.TIER_1.value: (0.10, 0.30),
    },
    "ADIPIC_ACID": {
        ProcessCalculationTier.TIER_3.value: (0.05, 0.10),
        ProcessCalculationTier.TIER_2.value: (0.08, 0.15),
        ProcessCalculationTier.TIER_1.value: (0.10, 0.25),
    },
    "AMMONIA": {
        ProcessCalculationTier.TIER_3.value: (0.02, 0.04),
        ProcessCalculationTier.TIER_2.value: (0.03, 0.06),
        ProcessCalculationTier.TIER_1.value: (0.03, 0.08),
    },
    "HYDROGEN": {
        ProcessCalculationTier.TIER_3.value: (0.02, 0.05),
        ProcessCalculationTier.TIER_2.value: (0.04, 0.08),
        ProcessCalculationTier.TIER_1.value: (0.05, 0.10),
    },
    "SEMICONDUCTOR": {
        ProcessCalculationTier.TIER_3.value: (0.10, 0.20),
        ProcessCalculationTier.TIER_2.value: (0.15, 0.30),
        ProcessCalculationTier.TIER_1.value: (0.15, 0.40),
    },
    "MAGNESIUM": {
        ProcessCalculationTier.TIER_3.value: (0.05, 0.10),
        ProcessCalculationTier.TIER_2.value: (0.08, 0.15),
        ProcessCalculationTier.TIER_1.value: (0.10, 0.25),
    },
    "PETROCHEMICAL": {
        ProcessCalculationTier.TIER_3.value: (0.03, 0.06),
        ProcessCalculationTier.TIER_2.value: (0.05, 0.10),
        ProcessCalculationTier.TIER_1.value: (0.08, 0.15),
    },
    "PULP_PAPER": {
        ProcessCalculationTier.TIER_3.value: (0.03, 0.06),
        ProcessCalculationTier.TIER_2.value: (0.05, 0.10),
        ProcessCalculationTier.TIER_1.value: (0.05, 0.12),
    },
}

# Default uncertainty range for unrecognized process types
_DEFAULT_PROCESS_UNCERTAINTY: Dict[str, Tuple[float, float]] = {
    ProcessCalculationTier.TIER_3.value: (0.05, 0.10),
    ProcessCalculationTier.TIER_2.value: (0.10, 0.20),
    ProcessCalculationTier.TIER_1.value: (0.15, 0.30),
}

# Default midpoint uncertainty by process and tier
_PROCESS_DEFAULT_UNCERTAINTY: Dict[str, Dict[str, float]] = {}
for _proc, _tiers in _PROCESS_UNCERTAINTY_RANGES.items():
    _PROCESS_DEFAULT_UNCERTAINTY[_proc] = {}
    for _tier, (_low, _high) in _tiers.items():
        _PROCESS_DEFAULT_UNCERTAINTY[_proc][_tier] = (_low + _high) / 2.0

# DQI scoring criteria (1 = best, 5 = worst)
_DQI_CRITERIA: Dict[str, Dict[str, int]] = {
    DQICategory.RELIABILITY.value: {
        "direct_measurement": 1,
        "cems": 1,
        "calibrated_instrument": 1,
        "stack_test": 1,
        "verified_production_records": 2,
        "mass_balance": 2,
        "stoichiometric": 2,
        "estimate": 3,
        "engineering_estimate": 3,
        "supplier_data": 3,
        "assumption": 4,
        "expert_judgment": 4,
        "ipcc_default": 4,
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
        "same_facility": 1,
        "same_region": 1,
        "same_country": 2,
        "similar_region": 3,
        "different_region": 4,
        "unknown": 5,
    },
    DQICategory.TECHNOLOGICAL_CORRELATION.value: {
        "same_process": 1,
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
        emissions_value: Central estimate of emissions (kg CO2e).
        emissions_unit: Unit of the emissions value.
        process_type: Industrial process type.
        calculation_method: Calculation method used.
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
        data_quality_score: Composite DQI score (1.0 to 5.0).
        dqi_multiplier: DQI-derived uncertainty multiplier.
        provenance_hash: SHA-256 hash of the assessment.
        timestamp: UTC ISO-formatted timestamp.
        metadata: Additional metadata dictionary.
    """

    result_id: str
    emissions_value: Decimal
    emissions_unit: str
    process_type: str
    calculation_method: str
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
            "process_type": self.process_type,
            "calculation_method": self.calculation_method,
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
    industrial process emission calculations.

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
        - EMISSION_FACTOR: production * EF * (1 - abatement) * GWP
        - MASS_BALANCE: C_in - C_out - C_non_co2 = CO2
        - STOICHIOMETRIC: reaction_stoichiometry * production
        - DIRECT_MEASUREMENT: CEMS data * calibration

    Thread Safety:
        All mutable state (_assessment_history) is protected by a
        reentrant lock. Monte Carlo simulations create per-call Random
        instances so concurrent callers never interfere.

    Example:
        >>> engine = UncertaintyQuantifierEngine()
        >>> result = engine.run_monte_carlo(
        ...     calculation_input={
        ...         "total_co2e_kg": 500000,
        ...         "process_type": "CEMENT",
        ...         "calculation_method": "EMISSION_FACTOR",
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
            "%d process types, %d methods, %d tiers, "
            "default iterations=%d",
            len(_PROCESS_UNCERTAINTY_RANGES),
            len(ProcessCalculationMethod),
            len(ProcessCalculationTier),
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
        each uncertain input (activity data, emission factor, abatement
        efficiency, GWP), recomputes emissions for each draw, and derives
        percentile-based confidence intervals.

        Required keys in calculation_input:
            - total_co2e_kg (float/Decimal): Central emission estimate.
            - process_type (str): Industrial process type (e.g. CEMENT).

        Optional keys:
            - calculation_method (str): EMISSION_FACTOR, MASS_BALANCE, etc.
            - tier (str): TIER_1, TIER_2, or TIER_3.
            - activity_data_source (str): metered, estimated, screening.
            - production_tonnes (float): Production quantity.
            - emission_factor (float): Emission factor value.
            - abatement_efficiency (float): Abatement efficiency [0-1].
            - gwp_value (float): GWP value for the primary gas.
            - carbonate_fraction (float): Carbonate content fraction [0-1].
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
            ValueError: If required keys are missing, or n_iterations
                is out of range.
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
            raise ValueError(f"total_co2e_kg must be >= 0, got {emissions}")

        process_type = calculation_input.get("process_type", "UNKNOWN").upper().strip()

        method = calculation_input.get(
            "calculation_method", ProcessCalculationMethod.EMISSION_FACTOR.value
        ).upper().strip()

        tier = calculation_input.get(
            "tier", ProcessCalculationTier.TIER_1.value
        ).upper().strip()

        activity_source = calculation_input.get(
            "activity_data_source", "estimated"
        ).upper().strip()
        if activity_source not in _ACTIVITY_DATA_UNCERTAINTY:
            activity_source = ActivityDataSource.ESTIMATED.value

        # DQI scoring
        dqi_inputs = calculation_input.get("dqi_inputs", {})
        dqi_score = self.calculate_dqi(dqi_inputs)
        dqi_level = self._dqi_score_to_level(dqi_score)
        dqi_multiplier = Decimal(str(_DQI_MULTIPLIERS[dqi_level]))

        # Run analytical propagation
        analytical_result = self._analytical_propagation(
            emissions, process_type, method, tier, activity_source,
            dqi_multiplier, calculation_input,
        )

        # Run Monte Carlo
        mc_result = self._run_mc_simulation(
            calculation_input, emissions, process_type, method, tier,
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
            default = self._get_process_default_uncertainty(process_type, tier)
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
            process_type, method, tier, activity_source,
            dqi_multiplier, calculation_input,
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
            "process_type": process_type,
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
            process_type=process_type,
            calculation_method=method,
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
                "process_range": self.get_process_uncertainty_range(
                    process_type, tier
                ),
            },
        )

        # Record in history
        with self._lock:
            self._assessment_history.append(result)

        # Provenance tracking
        self._record_provenance("monte_carlo", result.result_id, provenance_data)

        # Metrics
        elapsed = time.monotonic() - t_start
        self._record_metrics(process_type, elapsed)

        logger.debug(
            "MC uncertainty quantified: process=%s method=%s tier=%s "
            "emissions=%.1f combined=%.2f%% analytical=%.2f%% "
            "MC_mean=%.3f DQI=%.1f in %.1fms",
            process_type, method, tier, emissions, combined_pct, analytical_unc,
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
        uncertainties. For multiplicative relationships (e.g. AD * EF):
            sigma_rel_total = sqrt(sum(sigma_rel_i^2))

        For additive relationships (e.g. CO2 + N2O_CO2e):
            sigma_total = sqrt(sum((value_i * sigma_rel_i)^2)) / total

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

        # Multiplicative uncertainty (root-sum-of-squares of relative)
        mult_unc_squared = Decimal("0")
        component_uncertainties: Dict[str, Decimal] = {}
        for comp in multiplicative:
            unc_pct = _to_decimal(comp.get("uncertainty_pct", 10))
            unc_frac = unc_pct / Decimal("100")
            mult_unc_squared += unc_frac * unc_frac
            component_uncertainties[comp["name"]] = unc_pct

        # Additive uncertainty
        add_unc_squared = Decimal("0")
        for comp in additive:
            unc_pct = _to_decimal(comp.get("uncertainty_pct", 10))
            val = _to_decimal(comp["value"])
            abs_unc = val * unc_pct / Decimal("100")
            add_unc_squared += abs_unc * abs_unc
            component_uncertainties[comp["name"]] = unc_pct

        total_additive = sum(
            _to_decimal(c["value"]) for c in additive
        ) if additive else Decimal("0")

        total_emissions = sum(
            _to_decimal(c["value"]) for c in emission_components
        )

        if total_emissions == Decimal("0"):
            combined_pct = Decimal("0")
        elif total_additive > Decimal("0") and multiplicative:
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
            process_type="ANALYTICAL",
            calculation_method="ANALYTICAL",
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

    def calculate_dqi(
        self,
        dqi_inputs: Dict[str, Any],
    ) -> float:
        """Score data quality using the 5-dimension DQI framework.

        Computes a geometric mean score across the five DQI dimensions:
        reliability, completeness, temporal correlation, geographical
        correlation, and technological correlation.

        Each dimension is scored 1 (best) to 5 (worst). The overall
        score is the geometric mean across all provided dimensions.
        Missing dimensions default to 3 (average).

        Args:
            dqi_inputs: Dictionary mapping DQI dimension names to
                their criteria values. Recognized keys:
                - reliability: str (e.g. "direct_measurement")
                - completeness: str (e.g. "above_95_pct")
                - temporal_correlation: str (e.g. "same_year")
                - geographical_correlation: str (e.g. "same_facility")
                - technological_correlation: str (e.g. "same_process")

                Alternatively, direct numeric scores (1-5) may be provided.

        Returns:
            Geometric mean DQI score (1.0 to 5.0). Lower is better.
        """
        if not dqi_inputs:
            return 3.0

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
                scores.append(3.0)
                continue

            if isinstance(val, (int, float)):
                score = max(1.0, min(5.0, float(val)))
                scores.append(score)
                continue

            criteria = _DQI_CRITERIA.get(dqi_cat, {})
            if isinstance(val, str) and val in criteria:
                scores.append(float(criteria[val]))
            else:
                scores.append(3.0)

        # Geometric mean
        if not scores:
            return 3.0

        log_sum = sum(math.log(s) for s in scores)
        geo_mean = math.exp(log_sum / len(scores))
        return round(geo_mean, 2)

    # ------------------------------------------------------------------
    # Public API: Process Uncertainty Range
    # ------------------------------------------------------------------

    def get_process_uncertainty_range(
        self,
        process_type: str,
        tier: str,
    ) -> Tuple[float, float]:
        """Get the expected uncertainty range for a process type and tier.

        Returns the lower and upper bounds of expected combined
        uncertainty (half-width of 95% CI as percentage).

        Args:
            process_type: Industrial process type (e.g. CEMENT).
            tier: Calculation tier (TIER_1, TIER_2, TIER_3).

        Returns:
            Tuple of (lower_pct, upper_pct) as floats.
        """
        process_key = process_type.upper().strip()
        tier_key = tier.upper().strip()

        ranges = _PROCESS_UNCERTAINTY_RANGES.get(process_key, {})
        if not ranges:
            ranges = _DEFAULT_PROCESS_UNCERTAINTY  # type: ignore[assignment]

        low, high = ranges.get(tier_key, (0.10, 0.25))
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
            ValueError: If samples is empty or levels invalid.
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
            - production_tonnes, emission_factor, abatement_efficiency,
              gwp_value, carbonate_fraction, carbon_content,
              clinker_ratio, oxidation_factor

        Args:
            calculation_input: Baseline calculation parameters.
            parameters: List of parameter names to analyse.
            perturbation_pct: Perturbation magnitude (default 10%).

        Returns:
            List of SensitivityResult, one per parameter.

        Raises:
            ValueError: If parameters is empty or baseline emissions zero.
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

        perturbation_dec = _to_decimal(perturbation_pct) / Decimal("100")

        results: List[SensitivityResult] = []

        for param in parameters:
            base_val = _to_decimal(calculation_input.get(param, 1.0))
            if base_val == Decimal("0"):
                base_val = Decimal("1")

            perturbed_val = base_val * (Decimal("1") + perturbation_dec)

            perturbed_emissions = self._calculate_perturbed_emissions(
                base_emissions, param, base_val, perturbed_val,
                calculation_input,
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
    # Public API: Combine Uncertainties
    # ------------------------------------------------------------------

    def combine_uncertainties(
        self,
        component_uncertainties: List[Decimal],
    ) -> Decimal:
        """Combine component uncertainties using root-sum-of-squares.

        For multiplicative relationships:
            sigma_total = sqrt(sum(sigma_i^2))

        Args:
            component_uncertainties: List of relative uncertainties as
                Decimal fractions.

        Returns:
            Combined relative uncertainty as a Decimal fraction.

        Raises:
            ValueError: If list is empty.
        """
        if not component_uncertainties:
            raise ValueError("component_uncertainties must not be empty")

        sum_sq = Decimal("0")
        for unc in component_uncertainties:
            unc = _to_decimal(unc)
            sum_sq += unc * unc

        return _decimal_sqrt(sum_sq)

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
        process_type: str,
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
            process_type: Industrial process type.
            method: Calculation method.
            tier: Calculation tier.
            activity_source: Activity data source type.
            n_iterations: Number of iterations.
            seed: Random seed.
            dqi_multiplier: DQI uncertainty multiplier.

        Returns:
            Dictionary with MC results.
        """
        rng = random.Random(seed)
        dqi_mult = float(dqi_multiplier)
        emissions_f = float(emissions)

        if emissions_f <= 0:
            return self._empty_mc_result()

        samples: List[float] = []

        if method == ProcessCalculationMethod.EMISSION_FACTOR.value:
            samples = self._mc_emission_factor(
                calculation_input, emissions_f, process_type, tier,
                activity_source, n_iterations, rng, dqi_mult,
            )
        elif method == ProcessCalculationMethod.MASS_BALANCE.value:
            samples = self._mc_mass_balance(
                calculation_input, emissions_f, process_type, tier,
                activity_source, n_iterations, rng, dqi_mult,
            )
        elif method == ProcessCalculationMethod.STOICHIOMETRIC.value:
            samples = self._mc_stoichiometric(
                calculation_input, emissions_f, process_type, tier,
                activity_source, n_iterations, rng, dqi_mult,
            )
        elif method == ProcessCalculationMethod.DIRECT_MEASUREMENT.value:
            samples = self._mc_direct_measurement(
                calculation_input, emissions_f, process_type,
                n_iterations, rng, dqi_mult,
            )
        else:
            samples = self._mc_generic(
                emissions_f, process_type, tier, n_iterations, rng, dqi_mult,
            )

        if not samples:
            return self._empty_mc_result()

        return self._compute_mc_statistics(samples)

    def _mc_emission_factor(
        self,
        inputs: Dict[str, Any],
        emissions: float,
        process_type: str,
        tier: str,
        activity_source: str,
        n: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Monte Carlo for emission factor method.

        Emissions = production * EF * (1 - abatement_eff) * GWP

        Args:
            inputs: Calculation parameters.
            emissions: Central estimate.
            process_type: Process type.
            tier: Tier.
            activity_source: Activity data source.
            n: Iterations.
            rng: Random instance.
            dqi_mult: DQI multiplier.

        Returns:
            List of simulated emission values.
        """
        production = float(inputs.get("production_tonnes", 0))
        ef = float(inputs.get("emission_factor", 0))
        abatement = float(inputs.get("abatement_efficiency", 0))
        gwp = float(inputs.get("gwp_value", 1.0))

        # If individual params not available, use scaling approach
        if production <= 0 or ef <= 0:
            return self._mc_scaling(
                emissions, process_type, tier, activity_source,
                n, rng, dqi_mult,
            )

        ad_unc = _ACTIVITY_DATA_UNCERTAINTY.get(activity_source, 0.075) * dqi_mult
        ef_unc = _EMISSION_FACTOR_UNCERTAINTY.get(tier, 0.10) * dqi_mult
        abate_unc = _ABATEMENT_UNCERTAINTY * dqi_mult if abatement > 0 else 0
        gwp_unc = _GWP_UNCERTAINTY * dqi_mult if gwp > 1.0 else 0

        samples: List[float] = []
        for _ in range(n):
            s_prod = max(0.0, rng.gauss(production, production * ad_unc))
            s_ef = max(0.0, rng.gauss(ef, ef * ef_unc))
            s_abate = abatement
            if abatement > 0 and abate_unc > 0:
                s_abate = max(0.0, min(1.0, rng.gauss(abatement, abatement * abate_unc)))
            s_gwp = gwp
            if gwp > 1.0 and gwp_unc > 0:
                s_gwp = max(1.0, rng.gauss(gwp, gwp * gwp_unc))
            sample = s_prod * s_ef * (1.0 - s_abate) * s_gwp
            samples.append(max(0.0, sample))

        return samples

    def _mc_mass_balance(
        self,
        inputs: Dict[str, Any],
        emissions: float,
        process_type: str,
        tier: str,
        activity_source: str,
        n: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Monte Carlo for mass balance method.

        Emissions = (C_inputs - C_outputs - C_non_co2) * 44/12

        Uses scaling approach since mass balance inputs vary widely.

        Args:
            inputs: Calculation parameters.
            emissions: Central estimate.
            process_type: Process type.
            tier: Tier.
            activity_source: Source.
            n: Iterations.
            rng: Random instance.
            dqi_mult: DQI multiplier.

        Returns:
            List of simulated values.
        """
        c_in = float(inputs.get("carbon_input_tonnes", 0))
        c_out = float(inputs.get("carbon_output_tonnes", 0))

        if c_in <= 0:
            return self._mc_scaling(
                emissions, process_type, tier, activity_source,
                n, rng, dqi_mult,
            )

        ad_unc = _ACTIVITY_DATA_UNCERTAINTY.get(activity_source, 0.075) * dqi_mult
        # Mass balance has additional measurement uncertainty
        mb_unc = 0.05 * dqi_mult

        samples: List[float] = []
        co2_ratio = 44.0 / 12.0

        for _ in range(n):
            s_c_in = max(0.0, rng.gauss(c_in, c_in * ad_unc))
            s_c_out = max(0.0, rng.gauss(c_out, c_out * (ad_unc + mb_unc)))
            net_c = max(0.0, s_c_in - s_c_out)
            sample = net_c * co2_ratio * 1000.0  # tonnes to kg
            samples.append(sample)

        return samples

    def _mc_stoichiometric(
        self,
        inputs: Dict[str, Any],
        emissions: float,
        process_type: str,
        tier: str,
        activity_source: str,
        n: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Monte Carlo for stoichiometric method.

        Lower uncertainty than emission factor method since reaction
        stoichiometry is well-known. Main uncertainty comes from
        composition variability.

        Args:
            inputs: Calculation parameters.
            emissions: Central estimate.
            process_type: Process type.
            tier: Tier.
            activity_source: Source.
            n: Iterations.
            rng: Random instance.
            dqi_mult: DQI multiplier.

        Returns:
            List of simulated values.
        """
        production = float(inputs.get("production_tonnes", 0))
        stoich_factor = float(inputs.get("stoichiometric_factor", 0))

        if production <= 0 or stoich_factor <= 0:
            return self._mc_scaling(
                emissions, process_type, tier, activity_source,
                n, rng, dqi_mult,
            )

        ad_unc = _ACTIVITY_DATA_UNCERTAINTY.get(activity_source, 0.075) * dqi_mult
        stoich_unc = 0.02 * dqi_mult  # Stoichiometry is well-known
        composition_unc = 0.05 * dqi_mult  # Raw material composition variability

        samples: List[float] = []
        for _ in range(n):
            s_prod = max(0.0, rng.gauss(production, production * ad_unc))
            s_stoich = max(0.0, rng.gauss(stoich_factor, stoich_factor * stoich_unc))
            comp_adj = rng.gauss(1.0, composition_unc)
            sample = s_prod * s_stoich * max(0.5, comp_adj)
            samples.append(max(0.0, sample))

        return samples

    def _mc_direct_measurement(
        self,
        inputs: Dict[str, Any],
        emissions: float,
        process_type: str,
        n: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Monte Carlo for direct measurement (CEMS) method.

        Lowest uncertainty. Main uncertainty from instrument calibration
        and measurement noise.

        Args:
            inputs: Calculation parameters.
            emissions: Central estimate.
            process_type: Process type.
            n: Iterations.
            rng: Random instance.
            dqi_mult: DQI multiplier.

        Returns:
            List of simulated values.
        """
        instrument_unc = 0.02 * dqi_mult
        calibration_unc = 0.01 * dqi_mult
        combined = math.sqrt(instrument_unc**2 + calibration_unc**2)

        samples: List[float] = []
        for _ in range(n):
            s = max(0.0, rng.gauss(emissions, emissions * combined))
            samples.append(s)

        return samples

    def _mc_scaling(
        self,
        emissions: float,
        process_type: str,
        tier: str,
        activity_source: str,
        n: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Monte Carlo fallback using total emissions scaling.

        When individual parameters are not available, applies process-
        and tier-specific combined uncertainty directly.

        Args:
            emissions: Central estimate.
            process_type: Process type.
            tier: Tier.
            activity_source: Source.
            n: Iterations.
            rng: Random instance.
            dqi_mult: DQI multiplier.

        Returns:
            List of simulated values.
        """
        ad_unc = _ACTIVITY_DATA_UNCERTAINTY.get(activity_source, 0.075) * dqi_mult
        ef_unc = _EMISSION_FACTOR_UNCERTAINTY.get(tier, 0.10) * dqi_mult
        gwp_unc = _GWP_UNCERTAINTY * dqi_mult

        combined = math.sqrt(ad_unc**2 + ef_unc**2 + gwp_unc**2)

        # Process-specific additional uncertainty
        process_add = self._get_process_additional_uncertainty(process_type)
        combined = math.sqrt(combined**2 + (process_add * dqi_mult)**2)

        samples: List[float] = []
        for _ in range(n):
            s = max(0.0, rng.gauss(emissions, emissions * combined))
            samples.append(s)

        return samples

    def _mc_generic(
        self,
        emissions: float,
        process_type: str,
        tier: str,
        n: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Generic Monte Carlo using process default uncertainty.

        Args:
            emissions: Central estimate.
            process_type: Process type.
            tier: Tier.
            n: Iterations.
            rng: Random instance.
            dqi_mult: DQI multiplier.

        Returns:
            List of simulated values.
        """
        default = self._get_process_default_uncertainty(process_type, tier)
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
            samples: List of simulated values.

        Returns:
            Dictionary with mean, std, median, percentiles, CIs.
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
        """Return an empty MC result structure."""
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
        process_type: str,
        method: str,
        tier: str,
        activity_source: str,
        dqi_multiplier: Decimal,
        calculation_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute analytical uncertainty via IPCC Approach 1.

        For multiplicative chains (AD * EF * GWP):
            sigma_rel = sqrt(sigma_AD^2 + sigma_EF^2 + sigma_GWP^2)

        Args:
            emissions: Central estimate.
            process_type: Process type.
            method: Calculation method.
            tier: Tier.
            activity_source: Source.
            dqi_multiplier: DQI multiplier.
            calculation_input: Full input dict.

        Returns:
            Dictionary with combined_pct and components.
        """
        dqi_mult = float(dqi_multiplier)
        components: Dict[str, float] = {}

        ad_unc = _ACTIVITY_DATA_UNCERTAINTY.get(activity_source, 0.075) * dqi_mult
        ef_unc = _EMISSION_FACTOR_UNCERTAINTY.get(tier, 0.10) * dqi_mult
        gwp_unc = _GWP_UNCERTAINTY * dqi_mult
        method_unc = _METHOD_UNCERTAINTY.get(method, 0.05) * dqi_mult

        components["activity_data"] = ad_unc
        components["emission_factor"] = ef_unc
        components["gwp"] = gwp_unc
        components["method_uncertainty"] = method_unc

        # Abatement uncertainty if applicable
        abatement = float(calculation_input.get("abatement_efficiency", 0))
        if abatement > 0:
            abate_unc = _ABATEMENT_UNCERTAINTY * dqi_mult
            components["abatement_efficiency"] = abate_unc

        # Process-specific additional uncertainty
        process_add = self._get_process_additional_uncertainty(process_type)
        if process_add > 0:
            components["process_specific"] = process_add * dqi_mult

        # Root-sum-of-squares
        sum_sq = sum(v**2 for v in components.values())
        combined_rel = math.sqrt(sum_sq)
        combined_pct = Decimal(str(combined_rel * 100)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

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

        Args:
            emissions: Central estimate.
            combined_pct: Combined uncertainty percentage.
            mc_result: Monte Carlo result dictionary.

        Returns:
            Dictionary of confidence intervals.
        """
        cis = self._analytical_confidence_intervals(emissions, combined_pct)

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
            emissions: Central estimate.
            combined_pct: Combined uncertainty as percentage.

        Returns:
            Dictionary of CIs at 90/95/99% levels.
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
        process_type: str,
        method: str,
        tier: str,
        activity_source: str,
        dqi_multiplier: Decimal,
        calculation_input: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute contribution of each component to total variance.

        Args:
            process_type: Process type.
            method: Calculation method.
            tier: Tier.
            activity_source: Source.
            dqi_multiplier: DQI multiplier.
            calculation_input: Full input dict.

        Returns:
            Dictionary of fractional contributions summing to 1.0.
        """
        result = self._analytical_propagation(
            Decimal("100"), process_type, method, tier,
            activity_source, dqi_multiplier, calculation_input,
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
        param: str,
        base_val: Decimal,
        perturbed_val: Decimal,
        inputs: Dict[str, Any],
    ) -> Decimal:
        """Calculate emissions with a perturbed parameter value.

        Args:
            base_emissions: Baseline estimate.
            param: Parameter being perturbed.
            base_val: Original value.
            perturbed_val: Perturbed value.
            inputs: Full input dict.

        Returns:
            Perturbed emission estimate.
        """
        if base_val == Decimal("0"):
            return base_emissions

        # Direct multiplicative parameters
        multiplicative_params = {
            "production_tonnes", "emission_factor",
            "stoichiometric_factor", "carbon_content",
            "clinker_ratio", "carbonate_fraction",
        }

        if param in multiplicative_params:
            ratio = perturbed_val / base_val
            return (base_emissions * ratio).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        # Abatement efficiency: inverse relationship
        if param == "abatement_efficiency":
            old_factor = Decimal("1") - base_val
            new_factor = Decimal("1") - perturbed_val
            if old_factor > Decimal("0"):
                ratio = new_factor / old_factor
                return (base_emissions * ratio).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
            return base_emissions

        # GWP: proportional to non-CO2 fraction
        if param == "gwp_value":
            ratio = perturbed_val / base_val
            return (base_emissions * ratio).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        # Oxidation factor
        if param == "oxidation_factor":
            ratio = perturbed_val / base_val
            return (base_emissions * ratio).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        # Default proportional
        ratio = perturbed_val / base_val
        return (base_emissions * ratio).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

    # ------------------------------------------------------------------
    # Internal: Process-Specific Helpers
    # ------------------------------------------------------------------

    def _get_process_default_uncertainty(
        self,
        process_type: str,
        tier: str = "TIER_1",
    ) -> float:
        """Get default midpoint uncertainty for a process and tier.

        Args:
            process_type: Process type.
            tier: Tier.

        Returns:
            Midpoint uncertainty as fraction.
        """
        proc_defaults = _PROCESS_DEFAULT_UNCERTAINTY.get(process_type, {})
        if proc_defaults:
            return proc_defaults.get(tier, 0.20)

        # Fallback
        defaults = _DEFAULT_PROCESS_UNCERTAINTY
        low, high = defaults.get(tier, (0.10, 0.25))
        return (low + high) / 2.0

    def _get_process_additional_uncertainty(
        self,
        process_type: str,
    ) -> float:
        """Get additional process-specific uncertainty component.

        Some processes have inherently higher variability due to
        process-specific factors (e.g. anode effects in aluminum,
        variable N2O formation in nitric acid).

        Args:
            process_type: Process type.

        Returns:
            Additional uncertainty as fraction.
        """
        additional: Dict[str, float] = {
            "SEMICONDUCTOR": 0.10,
            "ALUMINUM_PREBAKE": 0.05,
            "ALUMINUM_SODERBERG": 0.07,
            "NITRIC_ACID": 0.05,
            "ADIPIC_ACID": 0.03,
            "MAGNESIUM": 0.04,
            "IRON_STEEL_BF_BOF": 0.03,
            "IRON_STEEL_EAF": 0.04,
            "IRON_STEEL_DRI": 0.03,
        }
        return additional.get(process_type, 0.0)

    # ------------------------------------------------------------------
    # Internal: Validation and Helpers
    # ------------------------------------------------------------------

    def _dqi_score_to_level(self, score: float) -> int:
        """Convert a continuous DQI score to a discrete level (1-5).

        Args:
            score: DQI score (1.0 to 5.0).

        Returns:
            Integer level 1-5.
        """
        return max(1, min(5, round(score)))

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

    def _record_metrics(self, process_type: str, elapsed: float) -> None:
        """Record Prometheus metrics if available.

        Args:
            process_type: Process type for labelling.
            elapsed: Elapsed time in seconds.
        """
        if _METRICS_AVAILABLE and _record_uncertainty is not None:
            try:
                _record_uncertainty(process_type, "complete")
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)
        if _METRICS_AVAILABLE and _observe_calculation_duration is not None:
            try:
                _observe_calculation_duration(elapsed)
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)
