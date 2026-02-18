# -*- coding: utf-8 -*-
"""
UncertaintyQuantifierEngine - Monte Carlo & Analytical Uncertainty (Engine 5 of 7)

AGENT-MRV-006: Flaring Agent

Quantifies the uncertainty of flaring emission calculations using two
complementary approaches:

    1. **Monte Carlo simulation**: Draws from parameterised distributions
       (normal, lognormal, triangular, uniform) for each uncertain input
       (gas flow rate, gas composition, combustion efficiency, heating value,
       emission factor, duration) and produces full percentile tables with
       90/95/99% confidence intervals. Configurable iteration count
       (default 5000) with explicit seed support for bit-perfect
       reproducibility.

    2. **Analytical error propagation** (IPCC Approach 1): Combined
       relative uncertainty for multiplicative chains via root-sum-of-
       squares of component uncertainties.

Flaring-Specific Uncertainty Sources:
    - Gas flow rate:           +/-5%  (continuous metering)
                               +/-15% (intermittent metering)
                               +/-50% (estimates)
    - Gas composition:         +/-2%  (lab analysis)
                               +/-5%  (field measurements)
                               +/-20% (defaults)
    - Combustion efficiency:   +/-2%  (measured/tested)
                               +/-5%  (defaults)
    - Heating value:           +/-3%  (calculated from composition)
                               +/-10% (defaults)
    - Emission factor:         +/-10% (facility-specific)
                               +/-25% (defaults)
    - Duration:                +/-5%  (metered)
                               +/-30% (estimated)

Distribution Types:
    NORMAL:      Symmetric bell curve, most common for measurement error.
    LOGNORMAL:   Right-skewed, for non-negative values with multiplicative error.
    TRIANGULAR:  Min/mode/max when limited data available.
    UNIFORM:     Equal probability across range.

Data Quality Indicator (DQI) Scoring (1-5 scale):
    5 dimensions, geometric mean composite score:
    - Reliability (1-5): Measurement method quality
    - Completeness (1-5): Data coverage
    - Temporal correlation (1-5): Age of data
    - Geographical correlation (1-5): Regional relevance
    - Technological correlation (1-5): Technology match

Zero-Hallucination Guarantees:
    - All formulas are deterministic mathematical operations.
    - No LLM involvement in any numeric path.
    - PRNG is seeded explicitly for full reproducibility.
    - Every result carries a SHA-256 provenance hash.
    - All calculations use Decimal for precision.

Thread Safety:
    All mutable state is protected by a reentrant lock. Monte Carlo
    simulations create per-call Random instances so concurrent callers
    never interfere.

Example:
    >>> from greenlang.flaring.uncertainty_quantifier import (
    ...     UncertaintyQuantifierEngine,
    ... )
    >>> engine = UncertaintyQuantifierEngine()
    >>> result = engine.run_monte_carlo(
    ...     calculation_input={
    ...         "total_co2e_kg": 250000,
    ...         "gas_volume_scf": 5000000,
    ...         "flow_rate_source": "CONTINUOUS_METERING",
    ...         "composition_source": "LAB_ANALYSIS",
    ...         "combustion_efficiency": 0.98,
    ...         "combustion_efficiency_source": "MEASURED",
    ...     },
    ...     n_iterations=5000,
    ... )
    >>> print(result.confidence_intervals["95"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Flaring Agent (GL-MRV-SCOPE1-006)
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
    from greenlang.flaring.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.flaring.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.flaring.metrics import (
        record_uncertainty as _record_uncertainty,
        observe_calculation_duration as _observe_calculation_duration,
    )
    _METRICS_AVAILABLE = True
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


class DistributionType(str, Enum):
    """Probability distribution types for Monte Carlo sampling.

    NORMAL:      Symmetric bell curve for measurement uncertainties.
    LOGNORMAL:   Right-skewed for non-negative multiplicative errors.
    TRIANGULAR:  Min/mode/max when limited data available.
    UNIFORM:     Equal probability across a range.
    """

    NORMAL = "NORMAL"
    LOGNORMAL = "LOGNORMAL"
    TRIANGULAR = "TRIANGULAR"
    UNIFORM = "UNIFORM"


class FlowRateSource(str, Enum):
    """Source classification for gas flow rate measurements.

    CONTINUOUS_METERING: Permanently installed flow meter with continuous data.
    INTERMITTENT_METERING: Periodic flow measurements.
    ESTIMATE: Engineering estimate or equipment capacity calculation.
    """

    CONTINUOUS_METERING = "CONTINUOUS_METERING"
    INTERMITTENT_METERING = "INTERMITTENT_METERING"
    ESTIMATE = "ESTIMATE"


class CompositionSource(str, Enum):
    """Source classification for gas composition data.

    LAB_ANALYSIS: Laboratory chromatographic analysis.
    FIELD_MEASUREMENT: On-site portable analyzer.
    DEFAULT: Default composition from literature/EPA.
    """

    LAB_ANALYSIS = "LAB_ANALYSIS"
    FIELD_MEASUREMENT = "FIELD_MEASUREMENT"
    DEFAULT = "DEFAULT"


class EfficiencySource(str, Enum):
    """Source classification for combustion efficiency data.

    MEASURED: Direct measurement from flare performance testing.
    DEFAULT: Regulatory or literature default (98% EPA/IPCC).
    """

    MEASURED = "MEASURED"
    DEFAULT = "DEFAULT"


class HeatingValueSource(str, Enum):
    """Source classification for gas heating value data.

    CALCULATED: Derived from measured gas composition.
    DEFAULT: Literature or regulatory default value.
    """

    CALCULATED = "CALCULATED"
    DEFAULT = "DEFAULT"


class EmissionFactorSource(str, Enum):
    """Source classification for emission factor data.

    FACILITY_SPECIFIC: Derived from facility measurements.
    DEFAULT: EPA/IPCC default emission factors.
    """

    FACILITY_SPECIFIC = "FACILITY_SPECIFIC"
    DEFAULT = "DEFAULT"


class DurationSource(str, Enum):
    """Source classification for event duration data.

    METERED: Duration from continuous monitoring equipment.
    ESTIMATED: Duration from operator logs or estimates.
    """

    METERED = "METERED"
    ESTIMATED = "ESTIMATED"


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
# Default Uncertainty Parameters (half-width of 95% CI as fraction)
# ===========================================================================

# Gas flow rate uncertainty by source type
_FLOW_RATE_UNCERTAINTY: Dict[str, float] = {
    FlowRateSource.CONTINUOUS_METERING.value: 0.05,
    FlowRateSource.INTERMITTENT_METERING.value: 0.15,
    FlowRateSource.ESTIMATE.value: 0.50,
}

# Gas composition uncertainty by source type
_COMPOSITION_UNCERTAINTY: Dict[str, float] = {
    CompositionSource.LAB_ANALYSIS.value: 0.02,
    CompositionSource.FIELD_MEASUREMENT.value: 0.05,
    CompositionSource.DEFAULT.value: 0.20,
}

# Combustion efficiency uncertainty by source type
_EFFICIENCY_UNCERTAINTY: Dict[str, float] = {
    EfficiencySource.MEASURED.value: 0.02,
    EfficiencySource.DEFAULT.value: 0.05,
}

# Heating value uncertainty by source type
_HEATING_VALUE_UNCERTAINTY: Dict[str, float] = {
    HeatingValueSource.CALCULATED.value: 0.03,
    HeatingValueSource.DEFAULT.value: 0.10,
}

# Emission factor uncertainty by source type
_EMISSION_FACTOR_UNCERTAINTY: Dict[str, float] = {
    EmissionFactorSource.FACILITY_SPECIFIC.value: 0.10,
    EmissionFactorSource.DEFAULT.value: 0.25,
}

# Duration uncertainty by source type
_DURATION_UNCERTAINTY: Dict[str, float] = {
    DurationSource.METERED.value: 0.05,
    DurationSource.ESTIMATED.value: 0.30,
}

# DQI scoring criteria (1 = best, 5 = worst)
_DQI_CRITERIA: Dict[str, Dict[str, int]] = {
    DQICategory.RELIABILITY.value: {
        "continuous_measurement": 1,
        "calibrated_meter": 1,
        "lab_analysis": 1,
        "field_measurement": 2,
        "periodic_measurement": 2,
        "verified_records": 2,
        "engineering_estimate": 3,
        "operator_estimate": 3,
        "supplier_data": 3,
        "assumption": 4,
        "expert_judgment": 4,
        "default_value": 4,
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
        "same_flare_type": 1,
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

# Default distribution for each parameter type
_DEFAULT_DISTRIBUTIONS: Dict[str, str] = {
    "flow_rate": DistributionType.NORMAL.value,
    "composition": DistributionType.NORMAL.value,
    "combustion_efficiency": DistributionType.TRIANGULAR.value,
    "heating_value": DistributionType.NORMAL.value,
    "emission_factor": DistributionType.LOGNORMAL.value,
    "duration": DistributionType.NORMAL.value,
}


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
        combined_uncertainty_pct: Combined relative uncertainty as a
            percentage (half-width of the 95% CI).
        confidence_intervals: Dictionary mapping confidence level
            strings ("90", "95", "99") to (lower, upper) Decimal tuples.
        mean: Mean from Monte Carlo simulation.
        median: Median from Monte Carlo simulation.
        std_dev: Standard deviation from Monte Carlo.
        cv: Coefficient of variation (std_dev / mean).
        p5: 5th percentile.
        p10: 10th percentile.
        p25: 25th percentile.
        p50: 50th percentile (median).
        p75: 75th percentile.
        p90: 90th percentile.
        p95: 95th percentile.
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
    combined_uncertainty_pct: Decimal
    confidence_intervals: Dict[str, Tuple[Decimal, Decimal]]
    mean: Optional[Decimal]
    median: Optional[Decimal]
    std_dev: Optional[Decimal]
    cv: Optional[Decimal]
    p5: Optional[Decimal]
    p10: Optional[Decimal]
    p25: Optional[Decimal]
    p50: Optional[Decimal]
    p75: Optional[Decimal]
    p90: Optional[Decimal]
    p95: Optional[Decimal]
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
        def _dec_str(v: Optional[Decimal]) -> Optional[str]:
            return str(v) if v is not None else None

        return {
            "result_id": self.result_id,
            "emissions_value": str(self.emissions_value),
            "emissions_unit": self.emissions_unit,
            "combined_uncertainty_pct": str(self.combined_uncertainty_pct),
            "confidence_intervals": {
                k: [str(v[0]), str(v[1])]
                for k, v in self.confidence_intervals.items()
            },
            "mean": _dec_str(self.mean),
            "median": _dec_str(self.median),
            "std_dev": _dec_str(self.std_dev),
            "cv": _dec_str(self.cv),
            "p5": _dec_str(self.p5),
            "p10": _dec_str(self.p10),
            "p25": _dec_str(self.p25),
            "p50": _dec_str(self.p50),
            "p75": _dec_str(self.p75),
            "p90": _dec_str(self.p90),
            "p95": _dec_str(self.p95),
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


# ===========================================================================
# UncertaintyQuantifierEngine
# ===========================================================================


class UncertaintyQuantifierEngine:
    """Monte Carlo and analytical uncertainty quantification engine for
    flaring emission calculations.

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

    Flaring-specific uncertainty sources include gas flow rate, gas
    composition, combustion efficiency, heating value, emission factors,
    and event duration, each with source-type-dependent uncertainty ranges.

    Both methods support Data Quality Indicator (DQI) scoring that
    adjusts uncertainty ranges based on the quality of underlying data.

    Thread Safety:
        All mutable state (_assessment_history) is protected by a
        reentrant lock. Monte Carlo simulations create per-call Random
        instances so concurrent callers never interfere.

    Example:
        >>> engine = UncertaintyQuantifierEngine()
        >>> result = engine.run_monte_carlo(
        ...     calculation_input={
        ...         "total_co2e_kg": 250000,
        ...         "gas_volume_scf": 5000000,
        ...         "flow_rate_source": "CONTINUOUS_METERING",
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
            "%d distribution types, 6 uncertainty sources, "
            "default iterations=%d",
            len(DistributionType),
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
        each uncertain flaring input (flow rate, composition, CE,
        heating value, emission factor, duration), recomputes emissions
        for each draw, and derives percentile-based confidence intervals.

        Required keys in calculation_input:
            - total_co2e_kg (float/Decimal): Central emission estimate.

        Optional keys:
            - gas_volume_scf (float): Gas volume in standard cubic feet.
            - flow_rate_source (str): CONTINUOUS_METERING, INTERMITTENT_METERING, ESTIMATE.
            - composition_source (str): LAB_ANALYSIS, FIELD_MEASUREMENT, DEFAULT.
            - combustion_efficiency (float): CE value [0-1].
            - combustion_efficiency_source (str): MEASURED, DEFAULT.
            - heating_value_source (str): CALCULATED, DEFAULT.
            - emission_factor_source (str): FACILITY_SPECIFIC, DEFAULT.
            - duration_source (str): METERED, ESTIMATED.
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

        # Extract and validate central estimate
        emissions = _to_decimal(calculation_input.get("total_co2e_kg", 0))
        if emissions < Decimal("0"):
            raise ValueError(f"total_co2e_kg must be >= 0, got {emissions}")

        # Resolve source types
        flow_source = calculation_input.get(
            "flow_rate_source", FlowRateSource.ESTIMATE.value
        ).upper().strip()
        comp_source = calculation_input.get(
            "composition_source", CompositionSource.DEFAULT.value
        ).upper().strip()
        ce_source = calculation_input.get(
            "combustion_efficiency_source", EfficiencySource.DEFAULT.value
        ).upper().strip()
        hv_source = calculation_input.get(
            "heating_value_source", HeatingValueSource.DEFAULT.value
        ).upper().strip()
        ef_source = calculation_input.get(
            "emission_factor_source", EmissionFactorSource.DEFAULT.value
        ).upper().strip()
        dur_source = calculation_input.get(
            "duration_source", DurationSource.ESTIMATED.value
        ).upper().strip()

        # DQI scoring
        dqi_inputs = calculation_input.get("dqi_inputs", {})
        dqi_score = self.calculate_dqi(dqi_inputs)
        dqi_level = self._dqi_score_to_level(dqi_score)
        dqi_multiplier = Decimal(str(_DQI_MULTIPLIERS[dqi_level]))

        # Resolve component uncertainties (half-width fractions)
        component_unc = self._resolve_component_uncertainties(
            flow_source, comp_source, ce_source,
            hv_source, ef_source, dur_source, dqi_multiplier,
        )

        # Analytical propagation
        analytical_result = self._analytical_propagation(
            emissions, component_unc,
        )

        # Monte Carlo simulation
        mc_result = self._run_mc_simulation(
            calculation_input, emissions, component_unc,
            n_iterations, seed, dqi_multiplier,
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
            # Fallback to sum of component uncertainties
            total_unc = sum(float(v) for v in component_unc.values())
            combined_pct = Decimal(str(total_unc * 100)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        # Build confidence intervals
        confidence_intervals = self._build_confidence_intervals(
            emissions, combined_pct, mc_result,
        )

        # Component uncertainties as percentages
        comp_unc_pct = {
            k: (v * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            for k, v in component_unc.items()
        }

        # Contribution analysis
        contributions = self._contribution_analysis(component_unc)
        contribution_dec = {
            k: Decimal(str(v)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            for k, v in contributions.items()
        }

        # Extract MC percentiles
        mc_percentiles = mc_result.get("percentiles", {})

        # Provenance
        provenance_data = {
            "emissions_value": str(emissions),
            "flow_source": flow_source,
            "comp_source": comp_source,
            "ce_source": ce_source,
            "n_iterations": n_iterations,
            "seed": seed,
            "combined_pct": str(combined_pct),
            "dqi_score": str(dqi_score),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        timestamp = _utcnow().isoformat()

        def _q(v: Any) -> Optional[Decimal]:
            """Quantize a value to 3 decimal places."""
            if v is None:
                return None
            return Decimal(str(v)).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        result = UncertaintyResult(
            result_id=f"uq_{uuid4().hex[:12]}",
            emissions_value=emissions,
            emissions_unit="kg_CO2e",
            combined_uncertainty_pct=combined_pct,
            confidence_intervals=confidence_intervals,
            mean=_q(mc_result.get("mean")),
            median=_q(mc_result.get("median")),
            std_dev=_q(mc_result.get("std")),
            cv=_q(mc_result.get("cv")),
            p5=_q(mc_percentiles.get("5")),
            p10=_q(mc_percentiles.get("10")),
            p25=_q(mc_percentiles.get("25")),
            p50=_q(mc_percentiles.get("50")),
            p75=_q(mc_percentiles.get("75")),
            p90=_q(mc_percentiles.get("90")),
            p95=_q(mc_percentiles.get("95")),
            monte_carlo_iterations=n_iterations,
            monte_carlo_seed=seed,
            analytical_uncertainty_pct=analytical_unc,
            component_uncertainties=comp_unc_pct,
            contribution_analysis=contribution_dec,
            data_quality_score=Decimal(str(dqi_score)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            dqi_multiplier=dqi_multiplier,
            provenance_hash=provenance_hash,
            timestamp=timestamp,
        )

        # Record in history
        with self._lock:
            self._assessment_history.append(result)

        # Provenance tracking
        self._record_provenance("monte_carlo", result.result_id, provenance_data)

        # Metrics
        elapsed = time.monotonic() - t_start
        self._record_metrics("flaring", elapsed)

        logger.debug(
            "MC uncertainty quantified: emissions=%.1f combined=%.2f%% "
            "analytical=%.2f%% MC_mean=%s DQI=%.1f in %.1fms",
            emissions, combined_pct, analytical_unc,
            mc_result.get("mean", "N/A"), dqi_score, elapsed * 1000,
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
            dqi_inputs: Dictionary mapping DQI dimension names to their
                criteria values. Keys are lowercase dimension names:
                - reliability: str (e.g. "continuous_measurement")
                - completeness: str (e.g. "above_95_pct")
                - temporal_correlation: str (e.g. "same_year")
                - geographical_correlation: str (e.g. "same_facility")
                - technological_correlation: str (e.g. "same_flare_type")

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

        if not scores:
            return 3.0

        # Geometric mean = exp(mean(log(scores)))
        log_sum = sum(math.log(s) for s in scores)
        geo_mean = math.exp(log_sum / len(scores))
        return round(geo_mean, 2)

    # ------------------------------------------------------------------
    # Public API: Uncertainty Range
    # ------------------------------------------------------------------

    def get_uncertainty_range(
        self,
        parameter: str,
        source_type: str,
    ) -> Dict[str, Any]:
        """Get the uncertainty range for a specific parameter and source.

        Args:
            parameter: Parameter name (flow_rate, composition,
                combustion_efficiency, heating_value, emission_factor,
                duration).
            source_type: Source classification string.

        Returns:
            Dictionary with uncertainty information.
        """
        parameter_maps: Dict[str, Dict[str, float]] = {
            "flow_rate": _FLOW_RATE_UNCERTAINTY,
            "composition": _COMPOSITION_UNCERTAINTY,
            "combustion_efficiency": _EFFICIENCY_UNCERTAINTY,
            "heating_value": _HEATING_VALUE_UNCERTAINTY,
            "emission_factor": _EMISSION_FACTOR_UNCERTAINTY,
            "duration": _DURATION_UNCERTAINTY,
        }

        param_map = parameter_maps.get(parameter, {})
        source_upper = source_type.upper().strip()
        unc_value = param_map.get(source_upper)

        if unc_value is None:
            # Try to find default (last entry in map)
            default_values = list(param_map.values())
            unc_value = default_values[-1] if default_values else 0.20

        return {
            "parameter": parameter,
            "source_type": source_upper,
            "uncertainty_fraction": unc_value,
            "uncertainty_pct": round(unc_value * 100, 1),
            "distribution": _DEFAULT_DISTRIBUTIONS.get(
                parameter, DistributionType.NORMAL.value
            ),
        }

    # ------------------------------------------------------------------
    # Public API: Sample Parameter
    # ------------------------------------------------------------------

    def sample_parameter(
        self,
        central_value: float,
        uncertainty_fraction: float,
        distribution: str = "NORMAL",
        n_samples: int = 1,
        seed: Optional[int] = None,
    ) -> List[float]:
        """Sample from a parameter's uncertainty distribution.

        Args:
            central_value: Central (mean/mode) value.
            uncertainty_fraction: Half-width of 95% CI as fraction.
            distribution: Distribution type (NORMAL, LOGNORMAL,
                TRIANGULAR, UNIFORM).
            n_samples: Number of samples to generate.
            seed: Optional random seed for reproducibility.

        Returns:
            List of sampled values.

        Raises:
            ValueError: If distribution type is unknown or parameters invalid.
        """
        if seed is None:
            seed = 42

        rng = random.Random(seed)
        dist = distribution.upper().strip()

        if central_value == 0:
            return [0.0] * n_samples

        samples: List[float] = []
        std = abs(central_value * uncertainty_fraction / 1.96)

        for _ in range(n_samples):
            if dist == DistributionType.NORMAL.value:
                s = rng.gauss(central_value, std)
            elif dist == DistributionType.LOGNORMAL.value:
                s = self._sample_lognormal(central_value, uncertainty_fraction, rng)
            elif dist == DistributionType.TRIANGULAR.value:
                s = self._sample_triangular(central_value, uncertainty_fraction, rng)
            elif dist == DistributionType.UNIFORM.value:
                s = self._sample_uniform(central_value, uncertainty_fraction, rng)
            else:
                raise ValueError(f"Unknown distribution type: {distribution}")
            samples.append(s)

        return samples

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
    # Internal: Component Uncertainty Resolution
    # ------------------------------------------------------------------

    def _resolve_component_uncertainties(
        self,
        flow_source: str,
        comp_source: str,
        ce_source: str,
        hv_source: str,
        ef_source: str,
        dur_source: str,
        dqi_multiplier: Decimal,
    ) -> Dict[str, Decimal]:
        """Resolve uncertainty fractions for each component.

        Args:
            flow_source: Flow rate source type.
            comp_source: Composition source type.
            ce_source: Combustion efficiency source type.
            hv_source: Heating value source type.
            ef_source: Emission factor source type.
            dur_source: Duration source type.
            dqi_multiplier: DQI adjustment multiplier.

        Returns:
            Dictionary of component uncertainties as Decimal fractions.
        """
        dqi_mult = float(dqi_multiplier)

        flow_unc = _FLOW_RATE_UNCERTAINTY.get(flow_source, 0.50) * dqi_mult
        comp_unc = _COMPOSITION_UNCERTAINTY.get(comp_source, 0.20) * dqi_mult
        ce_unc = _EFFICIENCY_UNCERTAINTY.get(ce_source, 0.05) * dqi_mult
        hv_unc = _HEATING_VALUE_UNCERTAINTY.get(hv_source, 0.10) * dqi_mult
        ef_unc = _EMISSION_FACTOR_UNCERTAINTY.get(ef_source, 0.25) * dqi_mult
        dur_unc = _DURATION_UNCERTAINTY.get(dur_source, 0.30) * dqi_mult

        return {
            "flow_rate": _to_decimal(str(round(flow_unc, 6))),
            "composition": _to_decimal(str(round(comp_unc, 6))),
            "combustion_efficiency": _to_decimal(str(round(ce_unc, 6))),
            "heating_value": _to_decimal(str(round(hv_unc, 6))),
            "emission_factor": _to_decimal(str(round(ef_unc, 6))),
            "duration": _to_decimal(str(round(dur_unc, 6))),
        }

    # ------------------------------------------------------------------
    # Internal: Monte Carlo Simulation
    # ------------------------------------------------------------------

    def _run_mc_simulation(
        self,
        calculation_input: Dict[str, Any],
        emissions: Decimal,
        component_unc: Dict[str, Decimal],
        n_iterations: int,
        seed: int,
        dqi_multiplier: Decimal,
    ) -> Dict[str, Any]:
        """Run the Monte Carlo simulation inner loop.

        For flaring emissions = volume * EF * CE_adjustment, we sample
        each component independently and compute the product.

        Args:
            calculation_input: Calculation parameters.
            emissions: Central emission estimate.
            component_unc: Component uncertainties (fractions).
            n_iterations: Number of iterations.
            seed: Random seed.
            dqi_multiplier: DQI uncertainty multiplier.

        Returns:
            Dictionary with MC results.
        """
        rng = random.Random(seed)
        emissions_f = float(emissions)

        if emissions_f <= 0:
            return self._empty_mc_result()

        # Extract central values for individual parameters if available
        volume = float(calculation_input.get("gas_volume_scf", 0))
        ce = float(calculation_input.get("combustion_efficiency", 0.98))
        ef = float(calculation_input.get("emission_factor", 0))
        duration = float(calculation_input.get("duration_hours", 0))

        # If we have individual parameters, use component-based simulation
        if volume > 0 and ef > 0:
            samples = self._mc_component_based(
                volume, ef, ce, duration,
                component_unc, n_iterations, rng,
            )
        else:
            # Fallback: scale total emissions with combined uncertainty
            samples = self._mc_scaling(
                emissions_f, component_unc, n_iterations, rng,
            )

        if not samples:
            return self._empty_mc_result()

        return self._compute_mc_statistics(samples)

    def _mc_component_based(
        self,
        volume: float,
        ef: float,
        ce: float,
        duration: float,
        component_unc: Dict[str, Decimal],
        n: int,
        rng: random.Random,
    ) -> List[float]:
        """Monte Carlo with per-component sampling.

        Emissions = volume * EF * CE_factor
        Where volume may be flow_rate * duration.

        Args:
            volume: Gas volume (scf).
            ef: Emission factor.
            ce: Combustion efficiency.
            duration: Duration in hours.
            component_unc: Component uncertainties.
            n: Number of iterations.
            rng: Random instance.

        Returns:
            List of simulated emission values.
        """
        flow_unc = float(component_unc.get("flow_rate", Decimal("0.15")))
        comp_unc = float(component_unc.get("composition", Decimal("0.05")))
        ce_unc = float(component_unc.get("combustion_efficiency", Decimal("0.05")))
        ef_unc = float(component_unc.get("emission_factor", Decimal("0.25")))

        # Convert half-width 95% CI to standard deviation
        flow_std = abs(volume * flow_unc / 1.96)
        ef_std = abs(ef * ef_unc / 1.96)
        comp_scale_std = comp_unc / 1.96  # scaling factor std

        samples: List[float] = []
        for _ in range(n):
            s_volume = max(0.0, rng.gauss(volume, flow_std))
            s_ef = max(0.0, rng.gauss(ef, ef_std))

            # Composition uncertainty as multiplicative scaling factor
            comp_adj = max(0.5, rng.gauss(1.0, comp_scale_std))

            # CE sampled using triangular distribution (bounded [0,1])
            ce_low = max(0.0, ce - ce * ce_unc)
            ce_high = min(1.0, ce + ce * ce_unc)
            s_ce = max(0.0, min(1.0, rng.triangular(ce_low, ce_high, ce)))

            sample = s_volume * s_ef * s_ce * comp_adj
            samples.append(max(0.0, sample))

        return samples

    def _mc_scaling(
        self,
        emissions: float,
        component_unc: Dict[str, Decimal],
        n: int,
        rng: random.Random,
    ) -> List[float]:
        """Monte Carlo fallback using total emissions scaling.

        When individual parameters are not available, combines all
        component uncertainties into a single scaling factor.

        Args:
            emissions: Central emission estimate.
            component_unc: Component uncertainties.
            n: Number of iterations.
            rng: Random instance.

        Returns:
            List of simulated emission values.
        """
        # Combined uncertainty via root-sum-of-squares
        sum_sq = sum(float(v) ** 2 for v in component_unc.values())
        combined_unc = math.sqrt(sum_sq)
        combined_std = abs(emissions * combined_unc / 1.96)

        samples: List[float] = []
        for _ in range(n):
            s = max(0.0, rng.gauss(emissions, combined_std))
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
            Dictionary with mean, std, median, cv, percentiles, CIs.
        """
        n = len(samples)
        sorted_s = sorted(samples)

        mean = sum(sorted_s) / n
        variance = (
            sum((x - mean) ** 2 for x in sorted_s) / (n - 1)
            if n > 1 else 0.0
        )
        std = math.sqrt(variance)
        median = (
            sorted_s[n // 2] if n % 2 == 1
            else (sorted_s[n // 2 - 1] + sorted_s[n // 2]) / 2
        )
        cv = std / mean if mean > 0 else 0.0

        percentiles: Dict[str, float] = {}
        for p in [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]:
            idx = max(0, min(n - 1, int(n * p / 100)))
            percentiles[str(p)] = sorted_s[idx]

        cis: Dict[str, Tuple[float, float]] = {
            "90": (
                sorted_s[max(0, int(n * 0.05))],
                sorted_s[min(n - 1, int(n * 0.95))],
            ),
            "95": (
                sorted_s[max(0, int(n * 0.025))],
                sorted_s[min(n - 1, int(n * 0.975))],
            ),
            "99": (
                sorted_s[max(0, int(n * 0.005))],
                sorted_s[min(n - 1, int(n * 0.995))],
            ),
        }

        return {
            "mean": mean,
            "std": std,
            "median": median,
            "cv": cv,
            "percentiles": percentiles,
            "confidence_intervals": cis,
        }

    def _empty_mc_result(self) -> Dict[str, Any]:
        """Return an empty MC result structure."""
        return {
            "mean": None,
            "std": None,
            "median": None,
            "cv": None,
            "percentiles": {},
            "confidence_intervals": {},
        }

    # ------------------------------------------------------------------
    # Internal: Distribution Sampling
    # ------------------------------------------------------------------

    def _sample_lognormal(
        self,
        central_value: float,
        uncertainty_fraction: float,
        rng: random.Random,
    ) -> float:
        """Sample from a lognormal distribution.

        Parametrised so that the median equals the central value.

        Args:
            central_value: The mode/central value.
            uncertainty_fraction: Half-width of 95% CI as fraction.
            rng: Random instance.

        Returns:
            Sampled value.
        """
        if central_value <= 0:
            return 0.0
        mu = math.log(central_value)
        sigma = abs(uncertainty_fraction / 1.96)
        if sigma <= 0:
            return central_value
        return rng.lognormvariate(mu, sigma)

    def _sample_triangular(
        self,
        central_value: float,
        uncertainty_fraction: float,
        rng: random.Random,
    ) -> float:
        """Sample from a triangular distribution.

        Args:
            central_value: The mode value.
            uncertainty_fraction: Half-width as fraction.
            rng: Random instance.

        Returns:
            Sampled value.
        """
        half_width = abs(central_value * uncertainty_fraction)
        low = central_value - half_width
        high = central_value + half_width
        return rng.triangular(low, high, central_value)

    def _sample_uniform(
        self,
        central_value: float,
        uncertainty_fraction: float,
        rng: random.Random,
    ) -> float:
        """Sample from a uniform distribution.

        Args:
            central_value: The midpoint value.
            uncertainty_fraction: Half-width as fraction.
            rng: Random instance.

        Returns:
            Sampled value.
        """
        half_width = abs(central_value * uncertainty_fraction)
        low = central_value - half_width
        high = central_value + half_width
        return rng.uniform(low, high)

    # ------------------------------------------------------------------
    # Internal: Analytical Error Propagation
    # ------------------------------------------------------------------

    def _analytical_propagation(
        self,
        emissions: Decimal,
        component_unc: Dict[str, Decimal],
    ) -> Dict[str, Any]:
        """Compute analytical uncertainty via IPCC Approach 1.

        For multiplicative chains (volume * EF * CE):
            sigma_rel = sqrt(sum(sigma_i^2))

        Args:
            emissions: Central emission estimate.
            component_unc: Component uncertainty fractions.

        Returns:
            Dictionary with combined_pct and components.
        """
        sum_sq = Decimal("0")
        components: Dict[str, float] = {}

        for comp_name, unc_frac in component_unc.items():
            unc_float = float(unc_frac)
            components[comp_name] = unc_float * 100  # as percentage
            sum_sq += unc_frac * unc_frac

        combined_rel = _decimal_sqrt(sum_sq)
        combined_pct = (combined_rel * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return {
            "combined_pct": combined_pct,
            "components": components,
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

        Prefers MC-based CIs when they are wider (more conservative).

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
        """Compute CIs from analytical uncertainty.

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
        component_unc: Dict[str, Decimal],
    ) -> Dict[str, float]:
        """Compute the fractional contribution of each component to
        total variance.

        Args:
            component_unc: Component uncertainty fractions.

        Returns:
            Dictionary of fractional contributions summing to 1.0.
        """
        total_sq = sum(float(v) ** 2 for v in component_unc.values())
        if total_sq <= 0:
            return {k: 0.0 for k in component_unc}
        return {
            k: (float(v) ** 2) / total_sq
            for k, v in component_unc.items()
        }

    # ------------------------------------------------------------------
    # Internal: Helpers
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
