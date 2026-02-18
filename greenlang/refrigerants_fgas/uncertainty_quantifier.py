# -*- coding: utf-8 -*-
"""
UncertaintyQuantifierEngine - Monte Carlo & Analytical Uncertainty (Engine 5 of 7)

AGENT-MRV-SCOPE1-002: Refrigerants & F-Gas Agent

Quantifies the uncertainty of refrigerant and F-gas emission calculations
using two complementary methods:

    1. **Monte Carlo simulation**: Draws from parameterised distributions
       (normal for charge, GWP; lognormal for leak rate) and produces
       full percentile tables with 90/95/99% confidence intervals.
       Configurable iteration count (default 5000) with explicit seed
       support for bit-perfect reproducibility.

    2. **Analytical error propagation** (IPCC Approach 1): Combined
       relative uncertainty for multiplicative chains via root-sum-of-
       squares of (sensitivity x component_uncertainty) terms.

Method-Specific Uncertainty Ranges (IPCC Guidelines):
    - Equipment-based:     +/-20-30% (charge uncertainty, leak rate uncertainty)
    - Mass balance:        +/-5-15%  (scale accuracy, inventory counts)
    - Screening:           +/-40-60% (default values, limited specificity)
    - Direct measurement:  +/-5-10%  (instrument accuracy)
    - Top-down:            +/-15-25% (purchasing records, recovery accuracy)

Monte Carlo Parameter Distributions:
    For equipment-based:
        - Charge: Normal(charge_kg, charge_kg * 0.05)
        - Leak rate: Normal(rate, rate * 0.20)
        - GWP: Normal(gwp, gwp * 0.05)
    For mass balance:
        - All inventory items: Normal(value, value * 0.02-0.05)
    For screening:
        - Total charge: Normal(charge, charge * 0.10)
        - Leak rate: Normal(rate, rate * 0.30)

Data Quality Indicator (DQI) Scoring (1-5 scale):
    5: Direct measurement, facility-specific factors
    4: Supplier data, equipment specifications
    3: Published defaults, industry averages
    2: Estimated, proxy data
    1: Expert judgment, no data

Confidence Intervals: 90%, 95%, 99%

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
    >>> from greenlang.refrigerants_fgas.uncertainty_quantifier import (
    ...     UncertaintyQuantifierEngine,
    ... )
    >>> engine = UncertaintyQuantifierEngine()
    >>> result = engine.quantify(
    ...     emissions_tco2e=Decimal("12.500"),
    ...     method="EQUIPMENT_BASED",
    ...     parameters={"charge_kg": 5.0, "leak_rate": 0.20, "gwp": 2088},
    ...     iterations=5000,
    ...     seed=42,
    ... )
    >>> print(result.confidence_intervals["95"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-SCOPE1-002 Refrigerants & F-Gas Agent
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
from decimal import Decimal, ROUND_HALF_UP
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
    from greenlang.refrigerants_fgas.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.refrigerants_fgas.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.refrigerants_fgas.metrics import (
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


# ===========================================================================
# Enumerations
# ===========================================================================


class UncertaintyMethod(str, Enum):
    """Calculation method types for uncertainty parameterisation.

    Each method has distinct uncertainty characteristics reflecting the
    quality and specificity of underlying data.

    EQUIPMENT_BASED: Emissions from charge * leak_rate * GWP.
        Uncertainty driven by charge accuracy and leak rate estimation.
        Typical combined uncertainty: +/-20-30%.
    MASS_BALANCE: Emissions from inventory change calculation.
        Uncertainty driven by scale accuracy and inventory completeness.
        Typical combined uncertainty: +/-5-15%.
    SCREENING: Simplified estimation using aggregate defaults.
        Highest uncertainty due to use of broad default factors.
        Typical combined uncertainty: +/-40-60%.
    DIRECT_MEASUREMENT: Emissions from direct gas measurement instruments.
        Lowest uncertainty when instruments are properly calibrated.
        Typical combined uncertainty: +/-5-10%.
    TOP_DOWN: Emissions from purchasing/recovery records at org level.
        Moderate uncertainty from record completeness and allocation.
        Typical combined uncertainty: +/-15-25%.
    """

    EQUIPMENT_BASED = "EQUIPMENT_BASED"
    MASS_BALANCE = "MASS_BALANCE"
    SCREENING = "SCREENING"
    DIRECT_MEASUREMENT = "DIRECT_MEASUREMENT"
    TOP_DOWN = "TOP_DOWN"


class DataQualityLevel(str, Enum):
    """Data Quality Indicator (DQI) levels on a 1-5 scale.

    Higher scores indicate better data quality and lower uncertainty.
    Aligns with GHG Protocol data quality guidance.

    LEVEL_5: Direct measurement, facility-specific factors, real-time
        monitoring. Best quality. Uncertainty multiplier: 0.5x.
    LEVEL_4: Supplier data, equipment specifications, recent calibration
        records. Good quality. Uncertainty multiplier: 0.75x.
    LEVEL_3: Published defaults, industry averages, national statistics.
        Moderate quality (baseline). Uncertainty multiplier: 1.0x.
    LEVEL_2: Estimated data, proxy from similar facilities, outdated
        records. Below average quality. Uncertainty multiplier: 1.5x.
    LEVEL_1: Expert judgment, no supporting data, complete gaps.
        Poorest quality. Uncertainty multiplier: 2.0x.
    """

    LEVEL_5 = "LEVEL_5"
    LEVEL_4 = "LEVEL_4"
    LEVEL_3 = "LEVEL_3"
    LEVEL_2 = "LEVEL_2"
    LEVEL_1 = "LEVEL_1"


# ===========================================================================
# Default Uncertainty Parameters
# ===========================================================================

# Method-specific base uncertainty ranges (half-width of 95% CI as fraction)
_METHOD_UNCERTAINTY_RANGES: Dict[str, Tuple[float, float]] = {
    UncertaintyMethod.EQUIPMENT_BASED.value: (0.20, 0.30),
    UncertaintyMethod.MASS_BALANCE.value: (0.05, 0.15),
    UncertaintyMethod.SCREENING.value: (0.40, 0.60),
    UncertaintyMethod.DIRECT_MEASUREMENT.value: (0.05, 0.10),
    UncertaintyMethod.TOP_DOWN.value: (0.15, 0.25),
}

# Method-specific default midpoint uncertainty (half-width as fraction)
_METHOD_DEFAULT_UNCERTAINTY: Dict[str, float] = {
    UncertaintyMethod.EQUIPMENT_BASED.value: 0.25,
    UncertaintyMethod.MASS_BALANCE.value: 0.10,
    UncertaintyMethod.SCREENING.value: 0.50,
    UncertaintyMethod.DIRECT_MEASUREMENT.value: 0.075,
    UncertaintyMethod.TOP_DOWN.value: 0.20,
}

# Component-level relative uncertainties for equipment-based method
_EQUIPMENT_COMPONENT_UNCERTAINTIES: Dict[str, float] = {
    "charge_kg": 0.05,        # +/-5% uncertainty on charge
    "leak_rate": 0.20,        # +/-20% uncertainty on leak rate
    "gwp": 0.05,              # +/-5% uncertainty on GWP
}

# Component-level relative uncertainties for mass balance method
_MASS_BALANCE_COMPONENT_UNCERTAINTIES: Dict[str, float] = {
    "beginning_inventory_kg": 0.02,   # +/-2% on scale reading
    "ending_inventory_kg": 0.02,
    "purchases_kg": 0.03,             # +/-3% on purchase records
    "sales_kg": 0.05,                 # +/-5% on sales records
    "acquisitions_kg": 0.05,
    "divestitures_kg": 0.05,
    "capacity_change_kg": 0.05,
}

# Component-level relative uncertainties for screening method
_SCREENING_COMPONENT_UNCERTAINTIES: Dict[str, float] = {
    "total_charge_kg": 0.10,    # +/-10% on total charge estimate
    "leak_rate": 0.30,          # +/-30% on default leak rate
    "gwp": 0.05,                # +/-5% on GWP
}

# Component-level relative uncertainties for direct measurement
_DIRECT_MEASUREMENT_COMPONENT_UNCERTAINTIES: Dict[str, float] = {
    "measured_loss_kg": 0.05,   # +/-5% instrument accuracy
    "gwp": 0.05,                # +/-5% on GWP
}

# Component-level relative uncertainties for top-down method
_TOP_DOWN_COMPONENT_UNCERTAINTIES: Dict[str, float] = {
    "total_purchased_kg": 0.05,    # +/-5% purchasing records
    "total_recovered_kg": 0.10,    # +/-10% recovery records
    "gwp": 0.05,                   # +/-5% on GWP
}

# Data quality level multipliers for uncertainty
_DQI_MULTIPLIERS: Dict[str, float] = {
    DataQualityLevel.LEVEL_5.value: 0.50,
    DataQualityLevel.LEVEL_4.value: 0.75,
    DataQualityLevel.LEVEL_3.value: 1.00,
    DataQualityLevel.LEVEL_2.value: 1.50,
    DataQualityLevel.LEVEL_1.value: 2.00,
}

# DQI scoring criteria mapping
_DQI_CRITERIA: Dict[str, Dict[str, int]] = {
    "data_source": {
        "direct_measurement": 5,
        "continuous_monitoring": 5,
        "calibrated_instrument": 5,
        "supplier_data": 4,
        "equipment_nameplate": 4,
        "recent_records": 4,
        "published_default": 3,
        "industry_average": 3,
        "national_statistics": 3,
        "estimated": 2,
        "proxy_data": 2,
        "outdated_records": 2,
        "expert_judgment": 1,
        "no_data": 1,
        "assumed": 1,
    },
    "measurement_method": {
        "weighed": 5,
        "flow_metered": 5,
        "ultrasonic": 4,
        "pressure_differential": 4,
        "infrared_detection": 4,
        "visual_inspection": 3,
        "bubble_test": 3,
        "estimated_from_recharge": 2,
        "estimated_from_age": 2,
        "not_measured": 1,
    },
    "data_age": {
        "current_year": 5,
        "within_2_years": 4,
        "within_5_years": 3,
        "within_10_years": 2,
        "older_than_10_years": 1,
    },
    "completeness": {
        "complete": 5,
        "above_90_pct": 4,
        "above_70_pct": 3,
        "above_50_pct": 2,
        "below_50_pct": 1,
    },
}

# Confidence interval z-scores (two-tailed)
_CONFIDENCE_Z_SCORES: Dict[str, float] = {
    "90": 1.6449,
    "95": 1.9600,
    "99": 2.5758,
}

# Default Monte Carlo iteration count
_DEFAULT_ITERATIONS: int = 5000

# Minimum Monte Carlo iteration count
_MIN_ITERATIONS: int = 100

# Maximum Monte Carlo iteration count
_MAX_ITERATIONS: int = 100000


# ===========================================================================
# Dataclasses for results
# ===========================================================================


@dataclass
class UncertaintyResult:
    """Complete uncertainty quantification result with provenance.

    Attributes:
        result_id: Unique identifier for this uncertainty assessment.
        emissions_tco2e: Central estimate of emissions in tCO2e.
        method: Calculation method used.
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
    emissions_tco2e: Decimal
    method: str
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
            "emissions_tco2e": str(self.emissions_tco2e),
            "method": self.method,
            "combined_uncertainty_pct": str(self.combined_uncertainty_pct),
            "confidence_intervals": {
                k: [str(v[0]), str(v[1])]
                for k, v in self.confidence_intervals.items()
            },
            "monte_carlo_mean": str(self.monte_carlo_mean) if self.monte_carlo_mean is not None else None,
            "monte_carlo_std": str(self.monte_carlo_std) if self.monte_carlo_std is not None else None,
            "monte_carlo_median": str(self.monte_carlo_median) if self.monte_carlo_median is not None else None,
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


# ===========================================================================
# UncertaintyQuantifierEngine
# ===========================================================================


class UncertaintyQuantifierEngine:
    """Monte Carlo and analytical uncertainty quantification engine for
    refrigerant and F-gas emission calculations.

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
        - EQUIPMENT_BASED: charge * leak_rate * GWP calculation
        - MASS_BALANCE: inventory change calculation
        - SCREENING: simplified default-factor estimation
        - DIRECT_MEASUREMENT: instrument-based measurement
        - TOP_DOWN: purchasing/recovery records

    Thread Safety:
        All mutable state (_assessment_history) is protected by a
        reentrant lock. Monte Carlo simulations create per-call Random
        instances so concurrent callers never interfere.

    Example:
        >>> engine = UncertaintyQuantifierEngine()
        >>> result = engine.quantify(
        ...     emissions_tco2e=Decimal("12.500"),
        ...     method="EQUIPMENT_BASED",
        ...     parameters={"charge_kg": 5.0, "leak_rate": 0.20, "gwp": 2088},
        ...     iterations=5000,
        ...     seed=42,
        ... )
        >>> print(result.combined_uncertainty_pct)
    """

    def __init__(self) -> None:
        """Initialize the UncertaintyQuantifierEngine."""
        self._assessment_history: List[UncertaintyResult] = []
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "UncertaintyQuantifierEngine initialized: "
            "%d methods supported, "
            "default iterations=%d, "
            "DQI levels=%d",
            len(_METHOD_UNCERTAINTY_RANGES),
            _DEFAULT_ITERATIONS,
            len(_DQI_MULTIPLIERS),
        )

    # ------------------------------------------------------------------
    # Public API: Main Entry Point
    # ------------------------------------------------------------------

    def quantify(
        self,
        emissions_tco2e: Decimal,
        method: str,
        parameters: Optional[Dict[str, Any]] = None,
        iterations: int = _DEFAULT_ITERATIONS,
        seed: Optional[int] = None,
        data_sources: Optional[Dict[str, str]] = None,
    ) -> UncertaintyResult:
        """Quantify uncertainty for a refrigerant/F-gas emission calculation.

        Runs both Monte Carlo simulation and analytical error propagation,
        then combines them with DQI-adjusted uncertainty to produce a
        comprehensive UncertaintyResult.

        Args:
            emissions_tco2e: Central estimate of emissions in tCO2e.
                Must be >= 0.
            method: Calculation method string matching an
                UncertaintyMethod enum value.
            parameters: Optional dictionary of calculation parameters
                used for Monte Carlo sampling. Keys depend on method:
                - EQUIPMENT_BASED: charge_kg, leak_rate, gwp
                - MASS_BALANCE: beginning_inventory_kg, ending_inventory_kg,
                  purchases_kg, sales_kg, etc.
                - SCREENING: total_charge_kg, leak_rate, gwp
                - DIRECT_MEASUREMENT: measured_loss_kg, gwp
                - TOP_DOWN: total_purchased_kg, total_recovered_kg, gwp
            iterations: Number of Monte Carlo iterations. Must be in
                [100, 100000]. Defaults to 5000.
            seed: Optional random seed for reproducibility. If None,
                a default seed of 12345 is used.
            data_sources: Optional dictionary mapping data source names
                to DQI criteria values for scoring. Keys can include
                "data_source", "measurement_method", "data_age",
                "completeness".

        Returns:
            UncertaintyResult with complete uncertainty characterization.

        Raises:
            ValueError: If method is not recognized, emissions_tco2e < 0,
                or iterations is out of range.
        """
        t_start = time.monotonic()

        # Validate inputs
        if not isinstance(emissions_tco2e, Decimal):
            emissions_tco2e = Decimal(str(emissions_tco2e))
        if emissions_tco2e < Decimal("0"):
            raise ValueError(
                f"emissions_tco2e must be >= 0, got {emissions_tco2e}"
            )

        self._validate_method(method)

        if iterations < _MIN_ITERATIONS or iterations > _MAX_ITERATIONS:
            raise ValueError(
                f"iterations must be in [{_MIN_ITERATIONS}, {_MAX_ITERATIONS}], "
                f"got {iterations}"
            )

        if seed is None:
            seed = 12345

        if parameters is None:
            parameters = {}

        # Data quality scoring
        dqi_score = self.score_data_quality(method, data_sources)
        dqi_level = self._dqi_score_to_level(dqi_score)
        dqi_multiplier = Decimal(str(_DQI_MULTIPLIERS[dqi_level]))

        # Run analytical propagation
        analytical_result = self.analytical_propagation(
            emissions_tco2e, method, parameters, dqi_multiplier
        )

        # Run Monte Carlo simulation
        mc_result = self.monte_carlo(
            emissions_tco2e, method, parameters, iterations, seed, dqi_multiplier
        )

        # Combined uncertainty: weighted average of analytical and MC
        analytical_unc = Decimal(str(analytical_result["combined_uncertainty_pct"]))
        mc_unc_pct = Decimal("0")
        if mc_result["std"] is not None and emissions_tco2e > Decimal("0"):
            mc_unc_pct = (
                Decimal(str(mc_result["std"])) / emissions_tco2e * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Use max of analytical and MC as the conservative combined estimate
        combined_uncertainty_pct = max(analytical_unc, mc_unc_pct)
        if combined_uncertainty_pct == Decimal("0") and emissions_tco2e > Decimal("0"):
            # Fallback to method default
            default_unc = _METHOD_DEFAULT_UNCERTAINTY[method]
            combined_uncertainty_pct = Decimal(str(default_unc * 100 * float(dqi_multiplier))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        # Build confidence intervals from combined uncertainty
        confidence_intervals: Dict[str, Tuple[Decimal, Decimal]] = {}
        for ci_label, z_score in _CONFIDENCE_Z_SCORES.items():
            # Scale the 95% CI half-width to other confidence levels
            half_width_95 = combined_uncertainty_pct / Decimal("100") * emissions_tco2e
            scaling = Decimal(str(z_score / 1.9600))
            half_width = (half_width_95 * scaling).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            lower = max(Decimal("0"), emissions_tco2e - half_width)
            upper = emissions_tco2e + half_width
            confidence_intervals[ci_label] = (
                lower.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                upper.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            )

        # If MC produced better CI estimates, use those for MC-derived CIs
        if mc_result["percentiles"]:
            mc_cis = mc_result.get("confidence_intervals", {})
            if mc_cis:
                for ci_label in mc_cis:
                    if ci_label in confidence_intervals:
                        mc_lower = Decimal(str(mc_cis[ci_label][0])).quantize(
                            Decimal("0.001"), rounding=ROUND_HALF_UP
                        )
                        mc_upper = Decimal(str(mc_cis[ci_label][1])).quantize(
                            Decimal("0.001"), rounding=ROUND_HALF_UP
                        )
                        # Use the wider of the two interval estimates
                        ana_lower, ana_upper = confidence_intervals[ci_label]
                        if (mc_upper - mc_lower) > (ana_upper - ana_lower):
                            confidence_intervals[ci_label] = (mc_lower, mc_upper)

        # Component uncertainties
        component_uncertainties = {
            k: Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            for k, v in analytical_result.get("component_uncertainties", {}).items()
        }

        # Contribution analysis
        contributions = self.contribution_analysis(parameters, method, dqi_multiplier)
        contribution_dec = {
            k: Decimal(str(v)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            for k, v in contributions.items()
        }

        # Monte Carlo percentiles as Decimal
        mc_percentiles = {
            k: Decimal(str(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            for k, v in mc_result.get("percentiles", {}).items()
        }

        # Build provenance hash
        provenance_data = {
            "emissions_tco2e": str(emissions_tco2e),
            "method": method,
            "parameters": {k: str(v) for k, v in parameters.items()},
            "iterations": iterations,
            "seed": seed,
            "combined_uncertainty_pct": str(combined_uncertainty_pct),
            "dqi_score": str(dqi_score),
            "dqi_multiplier": str(dqi_multiplier),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        timestamp = _utcnow().isoformat()

        result = UncertaintyResult(
            result_id=f"uq_{uuid4().hex[:12]}",
            emissions_tco2e=emissions_tco2e,
            method=method,
            combined_uncertainty_pct=combined_uncertainty_pct,
            confidence_intervals=confidence_intervals,
            monte_carlo_mean=(
                Decimal(str(mc_result["mean"])).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                if mc_result["mean"] is not None else None
            ),
            monte_carlo_std=(
                Decimal(str(mc_result["std"])).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                if mc_result["std"] is not None else None
            ),
            monte_carlo_median=(
                Decimal(str(mc_result["median"])).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                if mc_result["median"] is not None else None
            ),
            monte_carlo_percentiles=mc_percentiles,
            monte_carlo_iterations=iterations,
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
                "method_range_low": str(_METHOD_UNCERTAINTY_RANGES[method][0] * 100),
                "method_range_high": str(_METHOD_UNCERTAINTY_RANGES[method][1] * 100),
            },
        )

        # Record in history
        with self._lock:
            self._assessment_history.append(result)

        # Record provenance
        if _PROVENANCE_AVAILABLE and _get_provenance_tracker is not None:
            try:
                tracker = _get_provenance_tracker()
                tracker.record(
                    entity_type="uncertainty",
                    action="quantify_uncertainty",
                    entity_id=result.result_id,
                    data=provenance_data,
                    metadata={"method": method},
                )
            except Exception:
                logger.debug("Provenance recording skipped", exc_info=True)

        # Record metrics
        elapsed = time.monotonic() - t_start
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

        logger.debug(
            "Uncertainty quantified: method=%s emissions=%.3f tCO2e "
            "combined=%.2f%% analytical=%.2f%% MC_mean=%.3f "
            "DQI=%.1f in %.1fms",
            method,
            emissions_tco2e,
            combined_uncertainty_pct,
            analytical_unc,
            mc_result["mean"] if mc_result["mean"] is not None else 0,
            dqi_score,
            elapsed * 1000,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Monte Carlo Simulation
    # ------------------------------------------------------------------

    def monte_carlo(
        self,
        emissions_tco2e: Decimal,
        method: str,
        parameters: Dict[str, Any],
        iterations: int = _DEFAULT_ITERATIONS,
        seed: Optional[int] = None,
        dqi_multiplier: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for uncertainty estimation.

        Generates N random draws from parameterised distributions for
        each uncertain input, recomputes emissions for each draw, and
        derives percentile-based statistics.

        Distribution Models by Method:
            - EQUIPMENT_BASED: emissions = charge * leak_rate * gwp / 1000
              charge ~ Normal(charge_kg, charge_kg * 0.05 * dqi_mult)
              leak_rate ~ Normal(rate, rate * 0.20 * dqi_mult)
              gwp ~ Normal(gwp, gwp * 0.05 * dqi_mult)
            - MASS_BALANCE: emissions = delta * gwp / 1000
              each inventory item ~ Normal(val, val * unc * dqi_mult)
            - SCREENING: emissions = charge * rate * gwp / 1000
              charge ~ Normal(charge, charge * 0.10 * dqi_mult)
              rate ~ Normal(rate, rate * 0.30 * dqi_mult)
            - DIRECT_MEASUREMENT: emissions = loss * gwp / 1000
              loss ~ Normal(loss, loss * 0.05 * dqi_mult)
            - TOP_DOWN: emissions = (purchased - recovered) * gwp / 1000
              purchased ~ Normal(val, val * 0.05 * dqi_mult)
              recovered ~ Normal(val, val * 0.10 * dqi_mult)

        Args:
            emissions_tco2e: Central emissions estimate.
            method: Calculation method string.
            parameters: Calculation parameters for sampling.
            iterations: Number of Monte Carlo iterations.
            seed: Random seed for reproducibility.
            dqi_multiplier: Optional DQI-derived uncertainty multiplier.

        Returns:
            Dictionary with Monte Carlo results including mean, std,
            median, percentiles, and confidence intervals.
        """
        if not isinstance(emissions_tco2e, Decimal):
            emissions_tco2e = Decimal(str(emissions_tco2e))

        self._validate_method(method)

        if seed is None:
            seed = 12345
        if dqi_multiplier is None:
            dqi_multiplier = Decimal("1.0")

        dqi_mult_float = float(dqi_multiplier)

        # Create a dedicated Random instance (thread-safe)
        rng = random.Random(seed)

        # Generate simulation draws
        samples: List[float] = []

        if method == UncertaintyMethod.EQUIPMENT_BASED.value:
            samples = self._mc_equipment_based(
                parameters, iterations, rng, dqi_mult_float
            )
        elif method == UncertaintyMethod.MASS_BALANCE.value:
            samples = self._mc_mass_balance(
                parameters, iterations, rng, dqi_mult_float
            )
        elif method == UncertaintyMethod.SCREENING.value:
            samples = self._mc_screening(
                parameters, iterations, rng, dqi_mult_float
            )
        elif method == UncertaintyMethod.DIRECT_MEASUREMENT.value:
            samples = self._mc_direct_measurement(
                parameters, iterations, rng, dqi_mult_float
            )
        elif method == UncertaintyMethod.TOP_DOWN.value:
            samples = self._mc_top_down(
                parameters, iterations, rng, dqi_mult_float
            )

        # If no method-specific simulation could run, fall back to
        # parametric sampling around the central estimate
        if not samples:
            samples = self._mc_parametric_fallback(
                float(emissions_tco2e), method, iterations, rng, dqi_mult_float
            )

        # Compute statistics
        if samples:
            samples_sorted = sorted(samples)
            n = len(samples_sorted)
            mc_mean = sum(samples_sorted) / n
            mc_variance = sum((x - mc_mean) ** 2 for x in samples_sorted) / max(n - 1, 1)
            mc_std = math.sqrt(mc_variance)
            mc_median = samples_sorted[n // 2] if n % 2 == 1 else (
                (samples_sorted[n // 2 - 1] + samples_sorted[n // 2]) / 2
            )

            # Percentiles
            percentile_keys = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
            percentiles: Dict[str, float] = {}
            for p in percentile_keys:
                idx = max(0, min(n - 1, int(p / 100.0 * n)))
                percentiles[str(p)] = samples_sorted[idx]

            # Confidence intervals from percentiles
            mc_confidence_intervals: Dict[str, Tuple[float, float]] = {
                "90": (percentiles["5"], percentiles["95"]),
                "95": (percentiles["2.5"], percentiles["97.5"]),
                "99": (percentiles["1"], percentiles["99"]),
            }
        else:
            mc_mean = float(emissions_tco2e)
            mc_std = 0.0
            mc_median = float(emissions_tco2e)
            percentiles = {}
            mc_confidence_intervals = {}

        return {
            "mean": mc_mean,
            "std": mc_std,
            "median": mc_median,
            "percentiles": percentiles,
            "confidence_intervals": mc_confidence_intervals,
            "iterations": iterations,
            "seed": seed,
            "sample_count": len(samples),
        }

    # ------------------------------------------------------------------
    # Public API: Analytical Error Propagation
    # ------------------------------------------------------------------

    def analytical_propagation(
        self,
        emissions_tco2e: Decimal,
        method: str,
        parameters: Dict[str, Any],
        dqi_multiplier: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Calculate analytical uncertainty using IPCC Approach 1.

        For multiplicative relationships (emission = A * B * C * ...),
        the combined relative uncertainty is:

            U_combined = sqrt( U_A^2 + U_B^2 + U_C^2 + ... )

        where U_x is the relative uncertainty (half-width / value) of
        each component at the 95% confidence level.

        For additive relationships (mass_balance = A + B - C - D),
        the combined absolute uncertainty is:

            U_combined = sqrt( (U_A * A)^2 + (U_B * B)^2 + ... ) / |result|

        The DQI multiplier scales all component uncertainties.

        Args:
            emissions_tco2e: Central emissions estimate.
            method: Calculation method string.
            parameters: Calculation parameters with component values.
            dqi_multiplier: Optional DQI-derived uncertainty multiplier.

        Returns:
            Dictionary with analytical uncertainty results including
            combined uncertainty percentage and component-level detail.
        """
        if not isinstance(emissions_tco2e, Decimal):
            emissions_tco2e = Decimal(str(emissions_tco2e))

        self._validate_method(method)

        if dqi_multiplier is None:
            dqi_multiplier = Decimal("1.0")

        dqi_float = float(dqi_multiplier)

        # Get component uncertainties for the method
        component_unc_table = self._get_component_uncertainties(method)

        # Calculate relative uncertainty for each component
        component_results: Dict[str, float] = {}
        sum_sq_relative = 0.0

        if method == UncertaintyMethod.MASS_BALANCE.value:
            # Additive relationship: use absolute uncertainties
            abs_uncertainties_sq = 0.0
            total_value = 0.0

            for comp_name, base_unc in component_unc_table.items():
                comp_value = float(parameters.get(comp_name, 0))
                adjusted_unc = base_unc * dqi_float
                abs_unc = adjusted_unc * abs(comp_value)
                abs_uncertainties_sq += abs_unc ** 2
                component_results[comp_name] = adjusted_unc * 100  # as %
                total_value += abs(comp_value)

            if total_value > 0 and emissions_tco2e > Decimal("0"):
                combined_abs = math.sqrt(abs_uncertainties_sq)
                combined_relative = combined_abs / float(emissions_tco2e) * 100
            else:
                combined_relative = _METHOD_DEFAULT_UNCERTAINTY[method] * dqi_float * 100
        else:
            # Multiplicative relationship: use relative uncertainties
            for comp_name, base_unc in component_unc_table.items():
                adjusted_unc = base_unc * dqi_float
                sum_sq_relative += adjusted_unc ** 2
                component_results[comp_name] = adjusted_unc * 100  # as %

            combined_relative = math.sqrt(sum_sq_relative) * 100

        combined_uncertainty_pct = round(combined_relative, 2)

        return {
            "combined_uncertainty_pct": combined_uncertainty_pct,
            "component_uncertainties": component_results,
            "dqi_multiplier": float(dqi_multiplier),
            "method": method,
        }

    # ------------------------------------------------------------------
    # Public API: Data Quality Scoring
    # ------------------------------------------------------------------

    def score_data_quality(
        self,
        method: str,
        data_sources: Optional[Dict[str, str]] = None,
    ) -> float:
        """Score data quality on a 1-5 DQI scale.

        Evaluates data quality across four criteria dimensions:
            1. Data source quality (e.g. direct measurement vs estimated)
            2. Measurement method quality (e.g. weighed vs visual)
            3. Data age (e.g. current year vs older than 10 years)
            4. Data completeness (e.g. complete vs below 50%)

        Each dimension receives a score from 1-5 based on the provided
        data_sources criteria. The final DQI score is the weighted
        average across all available dimensions. If no data_sources
        are provided, a method-specific default is used.

        Method-specific defaults:
            - EQUIPMENT_BASED: 3.0 (published defaults)
            - MASS_BALANCE: 4.0 (supplier data, inventory records)
            - SCREENING: 2.0 (estimated, proxy)
            - DIRECT_MEASUREMENT: 5.0 (direct measurement)
            - TOP_DOWN: 3.5 (recent records)

        Args:
            method: Calculation method string.
            data_sources: Optional dictionary mapping DQI criteria
                category names to criteria value strings. Categories:
                "data_source", "measurement_method", "data_age",
                "completeness".

        Returns:
            Float DQI score in [1.0, 5.0].
        """
        self._validate_method(method)

        if data_sources is None or len(data_sources) == 0:
            # Method-specific defaults
            method_defaults: Dict[str, float] = {
                UncertaintyMethod.EQUIPMENT_BASED.value: 3.0,
                UncertaintyMethod.MASS_BALANCE.value: 4.0,
                UncertaintyMethod.SCREENING.value: 2.0,
                UncertaintyMethod.DIRECT_MEASUREMENT.value: 5.0,
                UncertaintyMethod.TOP_DOWN.value: 3.5,
            }
            return method_defaults.get(method, 3.0)

        scores: List[float] = []
        for category, criteria_value in data_sources.items():
            if category in _DQI_CRITERIA:
                criteria_map = _DQI_CRITERIA[category]
                score = criteria_map.get(criteria_value, 3)
                scores.append(float(score))

        if not scores:
            return 3.0

        return round(sum(scores) / len(scores), 2)

    def get_method_uncertainty_range(
        self, method: str,
    ) -> Tuple[float, float]:
        """Get the base uncertainty range for a calculation method.

        Returns the low and high bounds of the method's typical
        uncertainty (as percentages) before DQI adjustment.

        Args:
            method: Calculation method string.

        Returns:
            Tuple of (low_pct, high_pct) representing the 95% CI
            half-width range as percentages.

        Raises:
            ValueError: If method is not recognized.
        """
        self._validate_method(method)
        low, high = _METHOD_UNCERTAINTY_RANGES[method]
        return (low * 100, high * 100)

    def contribution_analysis(
        self,
        parameters: Dict[str, Any],
        method: Optional[str] = None,
        dqi_multiplier: Optional[Decimal] = None,
    ) -> Dict[str, float]:
        """Analyze which parameters contribute most to total uncertainty.

        Computes the fraction of total variance attributable to each
        parameter, based on the relative magnitudes of their individual
        uncertainty contributions.

        Formula:
            contribution_i = (U_i)^2 / sum_j((U_j)^2)

        Where U_i is the adjusted relative uncertainty for parameter i.

        Args:
            parameters: Calculation parameters (same as for quantify).
            method: Calculation method string. Defaults to EQUIPMENT_BASED.
            dqi_multiplier: Optional DQI multiplier. Defaults to 1.0.

        Returns:
            Dictionary mapping parameter names to their fractional
            contribution (0.0 to 1.0), summing to 1.0.
        """
        if method is None:
            method = UncertaintyMethod.EQUIPMENT_BASED.value
        if dqi_multiplier is None:
            dqi_multiplier = Decimal("1.0")

        self._validate_method(method)
        dqi_float = float(dqi_multiplier)

        component_unc_table = self._get_component_uncertainties(method)

        # Calculate squared uncertainties for each component
        squared_uncs: Dict[str, float] = {}
        total_sq = 0.0

        for comp_name, base_unc in component_unc_table.items():
            adjusted_unc = base_unc * dqi_float
            sq = adjusted_unc ** 2
            squared_uncs[comp_name] = sq
            total_sq += sq

        # Compute fractional contributions
        contributions: Dict[str, float] = {}
        if total_sq > 0:
            for comp_name, sq in squared_uncs.items():
                contributions[comp_name] = round(sq / total_sq, 4)
        else:
            # Equal contribution if all zero
            if squared_uncs:
                equal_share = round(1.0 / len(squared_uncs), 4)
                for comp_name in squared_uncs:
                    contributions[comp_name] = equal_share

        return contributions

    # ------------------------------------------------------------------
    # Public API: History and Stats
    # ------------------------------------------------------------------

    def get_history(
        self,
        method: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[UncertaintyResult]:
        """Return uncertainty assessment history.

        Args:
            method: Optional filter by calculation method.
            limit: Optional maximum number of recent entries to return.

        Returns:
            List of UncertaintyResult objects, oldest first.
        """
        with self._lock:
            entries = list(self._assessment_history)

        if method:
            entries = [e for e in entries if e.method == method]

        if limit is not None and limit > 0 and len(entries) > limit:
            entries = entries[-limit:]

        return entries

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics.

        Returns:
            Dictionary with counts and operational statistics.
        """
        with self._lock:
            history_count = len(self._assessment_history)
            by_method: Dict[str, int] = {}
            for entry in self._assessment_history:
                by_method[entry.method] = by_method.get(entry.method, 0) + 1

        return {
            "total_assessments": history_count,
            "methods_supported": len(_METHOD_UNCERTAINTY_RANGES),
            "dqi_levels": len(_DQI_MULTIPLIERS),
            "default_iterations": _DEFAULT_ITERATIONS,
            "assessments_by_method": by_method,
        }

    def clear(self) -> None:
        """Clear assessment history. Intended for testing."""
        with self._lock:
            self._assessment_history.clear()
        logger.info("UncertaintyQuantifierEngine cleared")

    # ------------------------------------------------------------------
    # Monte Carlo Simulation Helpers (Private)
    # ------------------------------------------------------------------

    def _mc_equipment_based(
        self,
        params: Dict[str, Any],
        iterations: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Monte Carlo for equipment-based: charge * leak_rate * gwp / 1000."""
        charge = float(params.get("charge_kg", 0))
        rate = float(params.get("leak_rate", 0))
        gwp = float(params.get("gwp", 0))

        if charge <= 0 or rate <= 0 or gwp <= 0:
            return []

        charge_std = charge * 0.05 * dqi_mult
        rate_std = rate * 0.20 * dqi_mult
        gwp_std = gwp * 0.05 * dqi_mult

        samples: List[float] = []
        for _ in range(iterations):
            c = max(0.0, rng.gauss(charge, charge_std))
            r = max(0.0, rng.gauss(rate, rate_std))
            g = max(0.0, rng.gauss(gwp, gwp_std))
            emission = c * r * g / 1000.0
            samples.append(emission)

        return samples

    def _mc_mass_balance(
        self,
        params: Dict[str, Any],
        iterations: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Monte Carlo for mass balance: (begin + purchases - end - sales + ...) * gwp / 1000."""
        begin = float(params.get("beginning_inventory_kg", 0))
        end = float(params.get("ending_inventory_kg", 0))
        purchases = float(params.get("purchases_kg", 0))
        sales = float(params.get("sales_kg", 0))
        acquisitions = float(params.get("acquisitions_kg", 0))
        divestitures = float(params.get("divestitures_kg", 0))
        capacity_change = float(params.get("capacity_change_kg", 0))
        gwp = float(params.get("gwp", 0))

        if gwp <= 0:
            return []

        unc = _MASS_BALANCE_COMPONENT_UNCERTAINTIES
        gwp_std = gwp * 0.05 * dqi_mult

        samples: List[float] = []
        for _ in range(iterations):
            b = rng.gauss(begin, abs(begin) * unc.get("beginning_inventory_kg", 0.02) * dqi_mult)
            e = rng.gauss(end, abs(end) * unc.get("ending_inventory_kg", 0.02) * dqi_mult)
            p = max(0.0, rng.gauss(purchases, abs(purchases) * unc.get("purchases_kg", 0.03) * dqi_mult))
            s = max(0.0, rng.gauss(sales, abs(sales) * unc.get("sales_kg", 0.05) * dqi_mult))
            a = max(0.0, rng.gauss(acquisitions, abs(acquisitions) * unc.get("acquisitions_kg", 0.05) * dqi_mult))
            d = max(0.0, rng.gauss(divestitures, abs(divestitures) * unc.get("divestitures_kg", 0.05) * dqi_mult))
            cc = rng.gauss(capacity_change, abs(capacity_change) * unc.get("capacity_change_kg", 0.05) * dqi_mult) if capacity_change != 0 else 0.0
            g = max(0.0, rng.gauss(gwp, gwp_std))

            loss_kg = b + p + a - e - s - d - cc
            emission = max(0.0, loss_kg * g / 1000.0)
            samples.append(emission)

        return samples

    def _mc_screening(
        self,
        params: Dict[str, Any],
        iterations: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Monte Carlo for screening: total_charge * leak_rate * gwp / 1000."""
        charge = float(params.get("total_charge_kg", params.get("charge_kg", 0)))
        rate = float(params.get("leak_rate", 0))
        gwp = float(params.get("gwp", 0))

        if charge <= 0 or rate <= 0 or gwp <= 0:
            return []

        charge_std = charge * 0.10 * dqi_mult
        rate_std = rate * 0.30 * dqi_mult
        gwp_std = gwp * 0.05 * dqi_mult

        samples: List[float] = []
        for _ in range(iterations):
            c = max(0.0, rng.gauss(charge, charge_std))
            r = max(0.0, rng.gauss(rate, rate_std))
            g = max(0.0, rng.gauss(gwp, gwp_std))
            emission = c * r * g / 1000.0
            samples.append(emission)

        return samples

    def _mc_direct_measurement(
        self,
        params: Dict[str, Any],
        iterations: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Monte Carlo for direct measurement: measured_loss * gwp / 1000."""
        loss = float(params.get("measured_loss_kg", 0))
        gwp = float(params.get("gwp", 0))

        if loss <= 0 or gwp <= 0:
            return []

        loss_std = loss * 0.05 * dqi_mult
        gwp_std = gwp * 0.05 * dqi_mult

        samples: List[float] = []
        for _ in range(iterations):
            l_draw = max(0.0, rng.gauss(loss, loss_std))
            g_draw = max(0.0, rng.gauss(gwp, gwp_std))
            emission = l_draw * g_draw / 1000.0
            samples.append(emission)

        return samples

    def _mc_top_down(
        self,
        params: Dict[str, Any],
        iterations: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Monte Carlo for top-down: (purchased - recovered) * gwp / 1000."""
        purchased = float(params.get("total_purchased_kg", 0))
        recovered = float(params.get("total_recovered_kg", 0))
        gwp = float(params.get("gwp", 0))

        if gwp <= 0:
            return []

        purchased_std = abs(purchased) * 0.05 * dqi_mult
        recovered_std = abs(recovered) * 0.10 * dqi_mult
        gwp_std = gwp * 0.05 * dqi_mult

        samples: List[float] = []
        for _ in range(iterations):
            p = max(0.0, rng.gauss(purchased, purchased_std))
            r = max(0.0, rng.gauss(recovered, recovered_std))
            g = max(0.0, rng.gauss(gwp, gwp_std))
            loss = max(0.0, p - r)
            emission = loss * g / 1000.0
            samples.append(emission)

        return samples

    def _mc_parametric_fallback(
        self,
        emissions: float,
        method: str,
        iterations: int,
        rng: random.Random,
        dqi_mult: float,
    ) -> List[float]:
        """Fallback Monte Carlo using method-level uncertainty around central estimate."""
        if emissions <= 0:
            return [0.0] * iterations

        default_unc = _METHOD_DEFAULT_UNCERTAINTY.get(method, 0.25)
        std = emissions * default_unc * dqi_mult

        samples: List[float] = []
        for _ in range(iterations):
            draw = max(0.0, rng.gauss(emissions, std))
            samples.append(draw)

        return samples

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_component_uncertainties(
        self, method: str,
    ) -> Dict[str, float]:
        """Get the component-level uncertainty table for a method."""
        tables: Dict[str, Dict[str, float]] = {
            UncertaintyMethod.EQUIPMENT_BASED.value: _EQUIPMENT_COMPONENT_UNCERTAINTIES,
            UncertaintyMethod.MASS_BALANCE.value: _MASS_BALANCE_COMPONENT_UNCERTAINTIES,
            UncertaintyMethod.SCREENING.value: _SCREENING_COMPONENT_UNCERTAINTIES,
            UncertaintyMethod.DIRECT_MEASUREMENT.value: _DIRECT_MEASUREMENT_COMPONENT_UNCERTAINTIES,
            UncertaintyMethod.TOP_DOWN.value: _TOP_DOWN_COMPONENT_UNCERTAINTIES,
        }
        return tables.get(method, {})

    @staticmethod
    def _dqi_score_to_level(score: float) -> str:
        """Map a numeric DQI score (1-5) to a DataQualityLevel enum value."""
        if score >= 4.5:
            return DataQualityLevel.LEVEL_5.value
        elif score >= 3.5:
            return DataQualityLevel.LEVEL_4.value
        elif score >= 2.5:
            return DataQualityLevel.LEVEL_3.value
        elif score >= 1.5:
            return DataQualityLevel.LEVEL_2.value
        else:
            return DataQualityLevel.LEVEL_1.value

    @staticmethod
    def _validate_method(method: str) -> None:
        """Validate that method is a recognized uncertainty method."""
        valid = {e.value for e in UncertaintyMethod}
        if method not in valid:
            raise ValueError(
                f"Unknown uncertainty method '{method}'. "
                f"Valid methods: {sorted(valid)}"
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        with self._lock:
            hist_count = len(self._assessment_history)
        return (
            f"UncertaintyQuantifierEngine("
            f"methods={len(_METHOD_UNCERTAINTY_RANGES)}, "
            f"assessments={hist_count}, "
            f"default_iterations={_DEFAULT_ITERATIONS})"
        )

    def __len__(self) -> int:
        """Return the number of assessments performed."""
        with self._lock:
            return len(self._assessment_history)
