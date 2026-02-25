# -*- coding: utf-8 -*-
"""
UncertaintyQuantifierEngine - Monte Carlo & Analytical Error Propagation (Engine 5 of 7)

AGENT-MRV-009: Scope 2 Location-Based Emissions Agent

Quantifies the uncertainty in Scope 2 location-based emission calculations
using Monte Carlo simulation, analytical error propagation (IPCC Approach 1),
data quality indicator (DQI) scoring, and sensitivity/tornado analysis.

Sources of Uncertainty in Scope 2 Location-Based Calculations:
    - Grid emission factor uncertainty: +/-5-50% depending on source quality,
      granularity (national vs. subregional vs. facility-specific), and data age.
    - Activity data uncertainty: +/-2-30% depending on metering accuracy,
      invoice reconciliation, or estimation method.
    - Transmission & Distribution (T&D) loss factor uncertainty: +/-10-30%
      depending on grid operator data quality.
    - GWP uncertainty: +/-5-15% per IPCC AR5/AR6 ranges for CH4 and N2O.
    - Steam/heat/cooling emission factor uncertainty: +/-10-30% depending
      on boiler efficiency, fuel mix, and conversion technology.

IPCC Default Uncertainty Ranges (95% CI half-widths):
    Grid EF (Tier 1 country): +/-15-30%
    Grid EF (Tier 2 subregional): +/-10-20%
    Grid EF (Tier 3 measured): +/-5-10%
    Activity data (metered): +/-2-5%
    Activity data (invoiced): +/-5-10%
    Activity data (estimated): +/-15-30%
    T&D loss factor: +/-10-30%
    Steam/heat EF: +/-10-25%
    Cooling EF: +/-15-30%

Monte Carlo Simulation:
    - Configurable iterations (default 10,000)
    - Normal distributions for activity data (bounded at zero)
    - Normal distributions for emission factors (bounded at zero)
    - Normal distributions for T&D loss factors (bounded at zero)
    - Lognormal distributions for CH4/N2O factors (non-negative)
    - Explicit seed support for full reproducibility

Analytical Propagation (IPCC Approach 1):
    Combined relative uncertainty for multiplicative chains:
    U_total = sqrt(sum(Ui^2)) for uncorrelated parameters.

DQI Scoring (4 dimensions, 0-1 scale):
    - EF source quality (custom=0.9, national=0.8, eGRID=0.85, IEA=0.7, IPCC=0.5)
    - EF age (current year=1.0, -1yr=0.9, -2yr=0.8, -3yr=0.7, older=0.5)
    - Activity data source (meter=0.95, invoice=0.85, estimate=0.6, benchmark=0.4)
    - Temporal representativeness (same year=1.0, -1yr=0.9, ...)

Zero-Hallucination Guarantees:
    - All formulas are deterministic mathematical operations.
    - No LLM involvement in any numeric path.
    - PRNG is seeded explicitly for full reproducibility.
    - Every result carries a SHA-256 provenance hash.

Thread Safety:
    Monte Carlo simulations create per-call Random instances so
    concurrent callers never interfere. Shared counters are
    protected by a reentrant lock.

Example:
    >>> from greenlang.scope2_location.uncertainty_quantifier import (
    ...     UncertaintyQuantifierEngine,
    ... )
    >>> engine = UncertaintyQuantifierEngine()
    >>> result = engine.run_monte_carlo(
    ...     base_emissions_kg=Decimal("150000"),
    ...     ef_uncertainty_pct=Decimal("0.10"),
    ...     activity_uncertainty_pct=Decimal("0.05"),
    ...     td_uncertainty_pct=Decimal("0.03"),
    ...     iterations=10000,
    ...     seed=42,
    ... )
    >>> print(result["ci_lower"], result["ci_upper"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-009 Scope 2 Location-Based Emissions (GL-MRV-SCOPE2-001)
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
    from greenlang.scope2_location.metrics import get_metrics as _get_metrics
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _get_metrics = None  # type: ignore[assignment]

try:
    from greenlang.scope2_location.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.scope2_location.provenance import (
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


# ---------------------------------------------------------------------------
# SHA-256 provenance helper
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Any JSON-serialisable object.

    Returns:
        Hex-encoded SHA-256 digest string (64 characters).
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Decimal helpers
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal.

    Args:
        value: Numeric or string value.

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert to Decimal with a fallback.

    Args:
        value: Value to convert.
        default: Fallback value if conversion fails.

    Returns:
        Decimal representation or *default*.
    """
    if value is None:
        return default
    try:
        return _D(value)
    except (InvalidOperation, ValueError, TypeError):
        return default


# ===========================================================================
# IPCC Default Uncertainty Ranges for Scope 2 Location-Based Calculations
# ===========================================================================

#: Default uncertainty ranges (95% CI half-widths as fractions) per parameter
#: type, data quality tier, and energy type.  Values are drawn from:
#:   - IPCC 2006 Guidelines Vol 1 Ch 3 Table 3.2
#:   - GHG Protocol Corporate Standard, Chapter 8 (Uncertainty Guidance)
#:   - EPA eGRID Technical Support Document
#:   - IEA Emission Factors Methodology Documentation

GRID_EF_UNCERTAINTY: Dict[str, Dict[str, float]] = {
    "tier_1": {
        "description": "National / country-level grid average",
        "lower_pct": 0.15,
        "upper_pct": 0.30,
        "default_pct": 0.20,
    },
    "tier_2": {
        "description": "Subregional / state-level grid average (e.g. eGRID subregion)",
        "lower_pct": 0.10,
        "upper_pct": 0.20,
        "default_pct": 0.15,
    },
    "tier_3": {
        "description": "Facility / utility-specific measured data",
        "lower_pct": 0.05,
        "upper_pct": 0.10,
        "default_pct": 0.07,
    },
}

ACTIVITY_DATA_UNCERTAINTY: Dict[str, Dict[str, float]] = {
    "meter": {
        "description": "Direct meter reading (calibrated revenue meters)",
        "lower_pct": 0.02,
        "upper_pct": 0.05,
        "default_pct": 0.02,
    },
    "invoice": {
        "description": "Utility invoices / billing records",
        "lower_pct": 0.05,
        "upper_pct": 0.10,
        "default_pct": 0.05,
    },
    "estimate": {
        "description": "Engineering estimate or pro-rata allocation",
        "lower_pct": 0.15,
        "upper_pct": 0.30,
        "default_pct": 0.20,
    },
    "benchmark": {
        "description": "Industry benchmark / intensity-based estimate",
        "lower_pct": 0.20,
        "upper_pct": 0.40,
        "default_pct": 0.30,
    },
}

TD_LOSS_UNCERTAINTY: Dict[str, float] = {
    "lower_pct": 0.10,
    "upper_pct": 0.30,
    "default_pct": 0.15,
}

STEAM_HEAT_UNCERTAINTY: Dict[str, float] = {
    "lower_pct": 0.10,
    "upper_pct": 0.25,
    "default_pct": 0.15,
}

COOLING_UNCERTAINTY: Dict[str, float] = {
    "lower_pct": 0.15,
    "upper_pct": 0.30,
    "default_pct": 0.20,
}

GWP_UNCERTAINTY: Dict[str, Dict[str, float]] = {
    "CO2": {"pct": 0.0},
    "CH4": {"pct": 0.30},
    "N2O": {"pct": 0.40},
}

#: Per-gas default EF uncertainty (95% CI half-width as fraction).
PER_GAS_EF_UNCERTAINTY: Dict[str, Dict[str, float]] = {
    "CO2": {
        "lower_pct": 0.05,
        "upper_pct": 0.15,
        "default_pct": 0.10,
    },
    "CH4": {
        "lower_pct": 0.50,
        "upper_pct": 1.50,
        "default_pct": 1.00,
    },
    "N2O": {
        "lower_pct": 1.00,
        "upper_pct": 3.00,
        "default_pct": 2.00,
    },
}


# ===========================================================================
# DQI Scoring Constants
# ===========================================================================

#: Emission factor source quality scores (0-1 scale, higher = better).
EF_SOURCE_SCORES: Dict[str, float] = {
    "custom": 0.90,
    "facility_specific": 0.90,
    "utility_specific": 0.88,
    "egrid": 0.85,
    "egrid_subregion": 0.85,
    "national": 0.80,
    "national_registry": 0.80,
    "defra": 0.78,
    "iea": 0.70,
    "aib": 0.72,
    "unfccc": 0.75,
    "ipcc": 0.50,
    "estimated": 0.30,
    "unknown": 0.20,
}

#: Activity data source quality scores (0-1 scale, higher = better).
ACTIVITY_SOURCE_SCORES: Dict[str, float] = {
    "meter": 0.95,
    "smart_meter": 0.95,
    "calibrated_meter": 0.95,
    "revenue_meter": 0.93,
    "invoice": 0.85,
    "utility_bill": 0.85,
    "estimate": 0.60,
    "engineering_estimate": 0.60,
    "benchmark": 0.40,
    "industry_average": 0.40,
    "unknown": 0.20,
}

#: Data quality tier to uncertainty percentage mapping.
DATA_QUALITY_TIER_UNCERTAINTY: Dict[str, float] = {
    "tier_1": 0.05,
    "tier_2": 0.15,
    "tier_3": 0.30,
}

#: Consumption source to uncertainty percentage mapping.
CONSUMPTION_SOURCE_UNCERTAINTY: Dict[str, float] = {
    "meter": 0.02,
    "smart_meter": 0.02,
    "calibrated_meter": 0.02,
    "revenue_meter": 0.03,
    "invoice": 0.05,
    "utility_bill": 0.05,
    "estimate": 0.20,
    "engineering_estimate": 0.20,
    "benchmark": 0.30,
    "industry_average": 0.30,
    "unknown": 0.40,
}


# ===========================================================================
# UncertaintyQuantifierEngine
# ===========================================================================


class UncertaintyQuantifierEngine:
    """Monte Carlo simulation and analytical error propagation for Scope 2
    location-based emission uncertainty quantification.

    Implements both IPCC Approach 1 (analytical root-sum-of-squares) and
    Approach 2 (Monte Carlo) for quantifying uncertainty in Scope 2
    location-based emission calculations covering electricity, steam,
    heating, cooling, and other purchased energy types.

    Configuration (iterations, seed, confidence levels) can be supplied
    either via the *config* parameter or the module-level ``_get_config()``
    singleton.

    Thread Safety:
        Monte Carlo uses per-call Random instances.  Shared counters
        are protected by a reentrant lock.

    Attributes:
        _config: Optional configuration object.
        _metrics: Optional Prometheus metrics recorder.
        _provenance: Optional provenance tracker.
        _default_iterations: Default Monte Carlo iterations.
        _default_confidence: Default confidence level for CIs.
        _rng: Lazy-initialised random generator (not used for MC).
        _lock: Reentrant lock for thread safety.
        _total_analyses: Running counter of analyses performed.
        _total_monte_carlo: Running counter of MC simulations.
        _total_analytical: Running counter of analytical propagations.
        _total_dqi: Running counter of DQI scoring operations.
        _total_sensitivity: Running counter of sensitivity analyses.
        _created_at: Engine creation timestamp.

    Example:
        >>> engine = UncertaintyQuantifierEngine()
        >>> result = engine.run_monte_carlo(
        ...     base_emissions_kg=Decimal("150000"),
        ...     ef_uncertainty_pct=Decimal("0.10"),
        ...     seed=42,
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Any = None,
        metrics: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialise the UncertaintyQuantifierEngine.

        Args:
            config: Optional Scope2LocationConfig object.  When ``None``,
                the singleton from ``_get_config()`` is attempted, with
                safe in-code defaults as the final fallback.
            metrics: Optional Scope2LocationMetrics object for Prometheus
                recording.  When ``None``, the module-level singleton is
                used if available.
            provenance: Optional provenance tracker for audit trails.
        """
        self._config = config
        self._metrics = metrics
        self._provenance = provenance

        # Defaults
        self._default_iterations: int = 10_000
        self._default_confidence: Decimal = Decimal("0.95")
        self._rng: Optional[random.Random] = None  # Lazy-init

        # Try to pull iteration count from config
        if config is not None:
            self._default_iterations = int(
                getattr(config, "monte_carlo_iterations", self._default_iterations)
            )
            conf = getattr(config, "default_confidence", None)
            if conf is not None:
                self._default_confidence = _safe_decimal(conf, self._default_confidence)
        elif _CONFIG_AVAILABLE and _get_config is not None:
            try:
                cfg = _get_config()
                self._default_iterations = int(
                    getattr(cfg, "monte_carlo_iterations", self._default_iterations)
                )
                conf = getattr(cfg, "default_confidence", None)
                if conf is not None:
                    self._default_confidence = _safe_decimal(
                        conf, self._default_confidence,
                    )
            except Exception:
                pass  # Fall back to in-code defaults

        # Resolve metrics
        if self._metrics is None and _METRICS_AVAILABLE and _get_metrics is not None:
            try:
                self._metrics = _get_metrics()
            except Exception:
                pass

        # Resolve provenance
        if self._provenance is None and _PROVENANCE_AVAILABLE and _get_provenance_tracker is not None:
            try:
                self._provenance = _get_provenance_tracker()
            except Exception:
                pass

        # Thread safety
        self._lock: threading.RLock = threading.RLock()

        # Counters
        self._total_analyses: int = 0
        self._total_monte_carlo: int = 0
        self._total_analytical: int = 0
        self._total_dqi: int = 0
        self._total_sensitivity: int = 0
        self._created_at: datetime = _utcnow()

        logger.info(
            "UncertaintyQuantifierEngine initialised: "
            "iterations=%d, confidence=%.2f",
            self._default_iterations,
            float(self._default_confidence),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _increment(self, counter_name: str) -> None:
        """Thread-safe increment of a named counter.

        Args:
            counter_name: Attribute name of the counter to increment.
        """
        with self._lock:
            current = getattr(self, counter_name, 0)
            setattr(self, counter_name, current + 1)

    def _record_metric(self, method: str) -> None:
        """Record an uncertainty run metric if Prometheus is available.

        Args:
            method: Statistical method name for the metric label.
        """
        if self._metrics is not None:
            try:
                self._metrics.record_uncertainty_run(method)
            except Exception as exc:
                logger.debug("Metrics recording skipped: %s", exc)

    def _record_provenance(self, entity_id: str, data: Dict[str, Any]) -> None:
        """Record provenance if tracker is available.

        Args:
            entity_id: Short identifier for the provenance entry.
            data: Provenance data dictionary.
        """
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="uncertainty_analysis",
                    action="compute_uncertainty",
                    entity_id=entity_id,
                    data=data,
                )
            except Exception as exc:
                logger.debug("Provenance recording skipped: %s", exc)

    # ==================================================================
    # Public API -- Monte Carlo Simulation
    # ==================================================================

    def run_monte_carlo(
        self,
        base_emissions_kg: Decimal,
        ef_uncertainty_pct: Decimal = Decimal("0.10"),
        activity_uncertainty_pct: Decimal = Decimal("0.05"),
        td_uncertainty_pct: Decimal = Decimal("0.03"),
        iterations: int = 10_000,
        confidence_level: Decimal = Decimal("0.95"),
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for Scope 2 emission uncertainty.

        Samples from normal distributions around the base emission value
        using multiplicative perturbation factors for emission factor,
        activity data, and T&D loss uncertainties.  Each iteration draws
        independent samples for all three uncertainty sources and computes
        the combined emission estimate.

        Sampling model (per iteration *i*):
            ef_sample_i   = gauss(1.0, ef_uncertainty_pct)
            act_sample_i  = gauss(1.0, activity_uncertainty_pct)
            td_sample_i   = gauss(1.0, td_uncertainty_pct)
            emission_i    = base_emissions_kg * ef_sample_i * act_sample_i * td_sample_i

        The resulting distribution is sorted and percentile-based confidence
        intervals are extracted.

        Args:
            base_emissions_kg: Central emission estimate in kg CO2e.
                Must be non-negative.
            ef_uncertainty_pct: Emission factor relative uncertainty as a
                fraction (e.g. 0.10 = +/-10%).  Default: 0.10.
            activity_uncertainty_pct: Activity data relative uncertainty
                as a fraction.  Default: 0.05.
            td_uncertainty_pct: T&D loss factor relative uncertainty
                as a fraction.  Default: 0.03.
            iterations: Number of Monte Carlo iterations.  Default: 10,000.
                Must be between 100 and 1,000,000.
            confidence_level: Confidence level as a fraction (e.g. 0.95).
                Default: 0.95.
            seed: Optional PRNG seed for reproducibility.  When ``None``,
                a non-deterministic seed is used.

        Returns:
            Dictionary with keys:
                - calculation_id: UUID string
                - status: "SUCCESS" or "VALIDATION_ERROR"
                - method: "MONTE_CARLO"
                - mean_co2e_kg: Decimal mean of simulated emissions
                - std_dev: Decimal standard deviation
                - cv_pct: Decimal coefficient of variation (%)
                - ci_lower: Decimal lower confidence interval bound
                - ci_upper: Decimal upper confidence interval bound
                - confidence_level: Decimal confidence level used
                - iterations: int number of iterations run
                - seed: int seed used (or -1 if non-deterministic)
                - percentiles: dict of {pct: Decimal value}
                - relative_uncertainty_pct: Decimal combined relative uncertainty
                - provenance_hash: SHA-256 hex digest
                - processing_time_ms: float wall-clock time in milliseconds
                - errors: list of validation errors (if any)

        Raises:
            No exceptions are raised; validation errors are returned
            in the result dictionary.
        """
        self._increment("_total_analyses")
        self._increment("_total_monte_carlo")
        start_time = time.monotonic()
        calc_id = str(uuid4())

        # -- Validate inputs -----------------------------------------------
        errors = self.validate_uncertainty_input(base_emissions_kg, ef_uncertainty_pct)
        errors.extend(self.validate_uncertainty_input(base_emissions_kg, activity_uncertainty_pct))
        errors.extend(self.validate_uncertainty_input(base_emissions_kg, td_uncertainty_pct))
        iter_errors = self.validate_iterations(iterations)
        errors.extend(iter_errors)

        # Deduplicate errors
        seen: set = set()
        unique_errors: List[str] = []
        for e in errors:
            if e not in seen:
                seen.add(e)
                unique_errors.append(e)

        if unique_errors:
            processing_time = round((time.monotonic() - start_time) * 1000, 3)
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "method": "MONTE_CARLO",
                "errors": unique_errors,
                "processing_time_ms": processing_time,
            }

        # -- Convert to floats for simulation performance ------------------
        base_float = float(base_emissions_kg)
        ef_unc = float(ef_uncertainty_pct)
        act_unc = float(activity_uncertainty_pct)
        td_unc = float(td_uncertainty_pct)
        conf = float(confidence_level)

        # -- Initialise PRNG -----------------------------------------------
        actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
        rng = random.Random(actual_seed)

        # -- Run simulation ------------------------------------------------
        results: List[float] = []
        iters = max(100, min(iterations, 1_000_000))

        for _ in range(iters):
            ef_sample = rng.gauss(1.0, ef_unc)
            act_sample = rng.gauss(1.0, act_unc)
            td_sample = rng.gauss(1.0, td_unc)
            sample = base_float * ef_sample * act_sample * td_sample
            # Clamp to zero (emissions cannot be negative)
            results.append(max(0.0, sample))

        results.sort()
        n = len(results)

        # -- Calculate statistics ------------------------------------------
        mean_val = sum(results) / n
        variance = sum((x - mean_val) ** 2 for x in results) / max(n - 1, 1)
        std_dev = math.sqrt(variance)
        cv_pct = (std_dev / abs(mean_val) * 100.0) if mean_val != 0.0 else 0.0

        # -- Confidence interval (percentile method) -----------------------
        alpha = (1.0 - conf) / 2.0
        ci_lower_idx = max(0, int(alpha * n))
        ci_upper_idx = min(n - 1, int((1.0 - alpha) * n))
        ci_lower = results[ci_lower_idx]
        ci_upper = results[ci_upper_idx]

        # -- Percentiles --------------------------------------------------
        percentile_points = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
        percentiles = self._compute_percentiles_float(results, percentile_points)

        # -- Combined relative uncertainty ---------------------------------
        rel_unc = math.sqrt(ef_unc**2 + act_unc**2 + td_unc**2)

        # -- Convert to Decimal for output ---------------------------------
        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "MONTE_CARLO",
            "mean_co2e_kg": Decimal(str(round(mean_val, 6))),
            "std_dev": Decimal(str(round(std_dev, 6))),
            "cv_pct": Decimal(str(round(cv_pct, 4))),
            "ci_lower": Decimal(str(round(ci_lower, 6))),
            "ci_upper": Decimal(str(round(ci_upper, 6))),
            "confidence_level": confidence_level,
            "iterations": iters,
            "seed": actual_seed,
            "percentiles": {
                str(k): Decimal(str(round(v, 6))) for k, v in percentiles.items()
            },
            "relative_uncertainty_pct": Decimal(str(round(rel_unc * 100, 4))),
            "input_uncertainties": {
                "ef_uncertainty_pct": ef_uncertainty_pct,
                "activity_uncertainty_pct": activity_uncertainty_pct,
                "td_uncertainty_pct": td_uncertainty_pct,
            },
            "base_emissions_kg": base_emissions_kg,
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        # -- Record metrics and provenance ---------------------------------
        self._record_metric("monte_carlo")
        self._record_provenance(
            entity_id=calc_id[:16],
            data={
                "method": "monte_carlo",
                "base_kg": str(base_emissions_kg),
                "mean_kg": str(result["mean_co2e_kg"]),
                "ci_lower": str(result["ci_lower"]),
                "ci_upper": str(result["ci_upper"]),
                "iterations": iters,
                "seed": actual_seed,
            },
        )

        logger.info(
            "Monte Carlo complete: id=%s, n=%d, base=%.2f, mean=%.2f, "
            "std=%.2f, cv=%.1f%%, 95%% CI=[%.2f, %.2f], time=%.3fms",
            calc_id, iters, base_float, mean_val, std_dev, cv_pct,
            ci_lower, ci_upper, processing_time,
        )
        return result

    # ==================================================================
    # Public API -- Per-Gas Monte Carlo
    # ==================================================================

    def run_monte_carlo_per_gas(
        self,
        co2_kg: Decimal,
        ch4_kg: Decimal,
        n2o_kg: Decimal,
        gwp_source: str = "AR5",
        ef_uncertainties: Optional[Dict[str, Decimal]] = None,
        iterations: int = 10_000,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run per-gas Monte Carlo simulation with individual uncertainty ranges.

        Each greenhouse gas has distinct uncertainty characteristics:
        - CO2 EF uncertainty: +/-5-15% (well-characterised combustion EFs)
        - CH4 EF uncertainty: +/-50-150% (highly variable, lognormal)
        - N2O EF uncertainty: +/-100-300% (very large, lognormal)

        Sampling model (per gas, per iteration):
            CO2: normal distribution (bounded at zero)
            CH4: lognormal distribution (non-negative)
            N2O: lognormal distribution (non-negative)

        Args:
            co2_kg: CO2 emissions in kg.
            ch4_kg: CH4 emissions in kg (CO2e-weighted).
            n2o_kg: N2O emissions in kg (CO2e-weighted).
            gwp_source: GWP source identifier (AR4, AR5, AR6).
                Used for GWP uncertainty ranges.
            ef_uncertainties: Optional per-gas uncertainty overrides.
                Keys: "CO2", "CH4", "N2O" with Decimal fraction values.
            iterations: Number of MC iterations.
            seed: Optional PRNG seed.

        Returns:
            Dictionary with per-gas and combined results including
            mean, std_dev, ci_lower, ci_upper, percentiles, and
            provenance_hash.
        """
        self._increment("_total_analyses")
        self._increment("_total_monte_carlo")
        start_time = time.monotonic()
        calc_id = str(uuid4())

        # -- Resolve per-gas uncertainties ---------------------------------
        default_ef_unc = {
            "CO2": Decimal(str(PER_GAS_EF_UNCERTAINTY["CO2"]["default_pct"])),
            "CH4": Decimal(str(PER_GAS_EF_UNCERTAINTY["CH4"]["default_pct"])),
            "N2O": Decimal(str(PER_GAS_EF_UNCERTAINTY["N2O"]["default_pct"])),
        }
        if ef_uncertainties:
            for gas in ("CO2", "CH4", "N2O"):
                if gas in ef_uncertainties:
                    default_ef_unc[gas] = _safe_decimal(
                        ef_uncertainties[gas], default_ef_unc[gas],
                    )
        ef_unc = default_ef_unc

        # -- Resolve GWP uncertainties -------------------------------------
        gwp_unc = dict(GWP_UNCERTAINTY)

        # -- Validate ------------------------------------------------------
        iters = max(100, min(iterations, 1_000_000))
        actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
        rng = random.Random(actual_seed)

        # -- Per-gas simulation --------------------------------------------
        co2_float = float(co2_kg)
        ch4_float = float(ch4_kg)
        n2o_float = float(n2o_kg)

        co2_samples: List[float] = []
        ch4_samples: List[float] = []
        n2o_samples: List[float] = []
        total_samples: List[float] = []

        co2_unc_float = float(ef_unc["CO2"])
        ch4_unc_float = float(ef_unc["CH4"])
        n2o_unc_float = float(ef_unc["N2O"])

        for _ in range(iters):
            # CO2: normal distribution
            co2_sample = max(0.0, co2_float * rng.gauss(1.0, co2_unc_float))

            # CH4: lognormal distribution
            if ch4_float > 0 and ch4_unc_float > 0:
                sigma2 = math.log(1 + ch4_unc_float**2)
                sigma = math.sqrt(sigma2)
                mu = math.log(ch4_float) - sigma2 / 2
                ch4_sample = rng.lognormvariate(mu, sigma)
            else:
                ch4_sample = ch4_float

            # N2O: lognormal distribution
            if n2o_float > 0 and n2o_unc_float > 0:
                sigma2 = math.log(1 + n2o_unc_float**2)
                sigma = math.sqrt(sigma2)
                mu = math.log(n2o_float) - sigma2 / 2
                n2o_sample = rng.lognormvariate(mu, sigma)
            else:
                n2o_sample = n2o_float

            co2_samples.append(co2_sample)
            ch4_samples.append(ch4_sample)
            n2o_samples.append(n2o_sample)
            total_samples.append(co2_sample + ch4_sample + n2o_sample)

        # -- Calculate per-gas statistics ----------------------------------
        def _gas_stats(
            samples: List[float], gas_name: str,
        ) -> Dict[str, Any]:
            """Compute statistics for a single gas."""
            sorted_s = sorted(samples)
            n = len(sorted_s)
            mean_v = sum(sorted_s) / n
            var_v = sum((x - mean_v) ** 2 for x in sorted_s) / max(n - 1, 1)
            std_v = math.sqrt(var_v)
            cv_v = (std_v / abs(mean_v) * 100.0) if mean_v != 0.0 else 0.0
            ci_l_idx = max(0, int(0.025 * n))
            ci_u_idx = min(n - 1, int(0.975 * n))
            return {
                "gas": gas_name,
                "mean_kg": Decimal(str(round(mean_v, 6))),
                "std_dev": Decimal(str(round(std_v, 6))),
                "cv_pct": Decimal(str(round(cv_v, 4))),
                "ci_lower_95": Decimal(str(round(sorted_s[ci_l_idx], 6))),
                "ci_upper_95": Decimal(str(round(sorted_s[ci_u_idx], 6))),
                "percentiles": {
                    str(p): Decimal(str(round(v, 6)))
                    for p, v in self._compute_percentiles_float(
                        sorted_s, [5, 25, 50, 75, 95],
                    ).items()
                },
            }

        co2_stats = _gas_stats(co2_samples, "CO2")
        ch4_stats = _gas_stats(ch4_samples, "CH4")
        n2o_stats = _gas_stats(n2o_samples, "N2O")
        total_stats = _gas_stats(total_samples, "CO2e_total")

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "MONTE_CARLO_PER_GAS",
            "gwp_source": gwp_source,
            "iterations": iters,
            "seed": actual_seed,
            "per_gas": {
                "CO2": co2_stats,
                "CH4": ch4_stats,
                "N2O": n2o_stats,
            },
            "total": total_stats,
            "ef_uncertainties_used": {
                k: str(v) for k, v in ef_unc.items()
            },
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        self._record_metric("monte_carlo")

        logger.info(
            "Per-gas Monte Carlo complete: id=%s, n=%d, "
            "CO2 mean=%.2f, CH4 mean=%.2f, N2O mean=%.2f, "
            "total mean=%.2f, time=%.3fms",
            calc_id, iters,
            float(co2_stats["mean_kg"]),
            float(ch4_stats["mean_kg"]),
            float(n2o_stats["mean_kg"]),
            float(total_stats["mean_kg"]),
            processing_time,
        )
        return result

    # ==================================================================
    # Public API -- Batch Monte Carlo
    # ==================================================================

    def run_monte_carlo_batch(
        self,
        calculations: List[Dict[str, Any]],
        iterations: int = 10_000,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for a batch of calculations.

        Supports correlations between calculations that share the same
        emission factor source (same-source factors receive correlated
        perturbations).

        Each entry in *calculations* should contain:
            - id: Unique identifier for the calculation
            - base_emissions_kg: Central emission estimate (Decimal or numeric)
            - ef_uncertainty_pct: EF relative uncertainty (Decimal or numeric)
            - activity_uncertainty_pct: Activity data uncertainty (Decimal or numeric)
            - td_uncertainty_pct: T&D loss uncertainty (Decimal or numeric)
            - ef_source: (optional) EF source identifier for correlation grouping

        Args:
            calculations: List of calculation dictionaries.
            iterations: Number of MC iterations.
            seed: Optional PRNG seed.

        Returns:
            Dictionary with per-calculation and aggregated results,
            including portfolio-level confidence intervals.
        """
        self._increment("_total_analyses")
        self._increment("_total_monte_carlo")
        start_time = time.monotonic()
        batch_id = str(uuid4())

        if not calculations:
            return {
                "batch_id": batch_id,
                "status": "VALIDATION_ERROR",
                "errors": ["At least one calculation is required"],
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3,
                ),
            }

        iters = max(100, min(iterations, 1_000_000))
        actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
        rng = random.Random(actual_seed)

        # -- Group by EF source for correlation ----------------------------
        source_groups: Dict[str, List[int]] = {}
        for idx, calc in enumerate(calculations):
            src = str(calc.get("ef_source", f"independent_{idx}"))
            if src not in source_groups:
                source_groups[src] = []
            source_groups[src].append(idx)

        # -- Pre-generate correlated EF perturbations ----------------------
        # For same-source groups, use the same EF perturbation per iteration
        ef_perturbations: Dict[str, List[float]] = {}
        for src in source_groups:
            ef_unc = 0.10  # Default
            # Use the first calculation in the group to determine uncertainty
            first_idx = source_groups[src][0]
            ef_unc = float(
                _safe_decimal(
                    calculations[first_idx].get("ef_uncertainty_pct", "0.10"),
                    Decimal("0.10"),
                )
            )
            ef_perturbations[src] = [
                rng.gauss(1.0, ef_unc) for _ in range(iters)
            ]

        # -- Simulate each calculation -------------------------------------
        per_calc_results: List[Dict[str, Any]] = []
        portfolio_samples: List[float] = [0.0] * iters

        for idx, calc in enumerate(calculations):
            calc_calc_id = str(calc.get("id", f"calc_{idx}"))
            base_kg = float(
                _safe_decimal(calc.get("base_emissions_kg", 0), _ZERO)
            )
            act_unc = float(
                _safe_decimal(
                    calc.get("activity_uncertainty_pct", "0.05"),
                    Decimal("0.05"),
                )
            )
            td_unc = float(
                _safe_decimal(
                    calc.get("td_uncertainty_pct", "0.03"),
                    Decimal("0.03"),
                )
            )
            src = str(calc.get("ef_source", f"independent_{idx}"))

            samples: List[float] = []
            for i in range(iters):
                ef_sample = ef_perturbations[src][i]
                act_sample = rng.gauss(1.0, act_unc)
                td_sample = rng.gauss(1.0, td_unc)
                sample = max(0.0, base_kg * ef_sample * act_sample * td_sample)
                samples.append(sample)
                portfolio_samples[i] += sample

            # Per-calculation stats
            sorted_s = sorted(samples)
            n = len(sorted_s)
            mean_v = sum(sorted_s) / n
            var_v = sum((x - mean_v) ** 2 for x in sorted_s) / max(n - 1, 1)
            std_v = math.sqrt(var_v)
            ci_l = sorted_s[max(0, int(0.025 * n))]
            ci_u = sorted_s[min(n - 1, int(0.975 * n))]

            per_calc_results.append({
                "id": calc_calc_id,
                "base_emissions_kg": Decimal(str(round(base_kg, 6))),
                "mean_co2e_kg": Decimal(str(round(mean_v, 6))),
                "std_dev": Decimal(str(round(std_v, 6))),
                "ci_lower_95": Decimal(str(round(ci_l, 6))),
                "ci_upper_95": Decimal(str(round(ci_u, 6))),
            })

        # -- Portfolio-level stats -----------------------------------------
        sorted_portfolio = sorted(portfolio_samples)
        n_p = len(sorted_portfolio)
        port_mean = sum(sorted_portfolio) / n_p
        port_var = sum((x - port_mean) ** 2 for x in sorted_portfolio) / max(n_p - 1, 1)
        port_std = math.sqrt(port_var)
        port_cv = (port_std / abs(port_mean) * 100.0) if port_mean != 0.0 else 0.0

        port_ci_l = sorted_portfolio[max(0, int(0.025 * n_p))]
        port_ci_u = sorted_portfolio[min(n_p - 1, int(0.975 * n_p))]

        port_percentiles = self._compute_percentiles_float(
            sorted_portfolio, [5, 10, 25, 50, 75, 90, 95],
        )

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result = {
            "batch_id": batch_id,
            "status": "SUCCESS",
            "method": "MONTE_CARLO_BATCH",
            "calculation_count": len(calculations),
            "iterations": iters,
            "seed": actual_seed,
            "per_calculation": per_calc_results,
            "portfolio": {
                "total_base_kg": Decimal(str(round(
                    sum(float(_safe_decimal(c.get("base_emissions_kg", 0), _ZERO))
                        for c in calculations), 6,
                ))),
                "mean_co2e_kg": Decimal(str(round(port_mean, 6))),
                "std_dev": Decimal(str(round(port_std, 6))),
                "cv_pct": Decimal(str(round(port_cv, 4))),
                "ci_lower_95": Decimal(str(round(port_ci_l, 6))),
                "ci_upper_95": Decimal(str(round(port_ci_u, 6))),
                "percentiles": {
                    str(k): Decimal(str(round(v, 6)))
                    for k, v in port_percentiles.items()
                },
            },
            "correlation_groups": {
                src: len(idxs) for src, idxs in source_groups.items()
            },
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        self._record_metric("monte_carlo")

        logger.info(
            "Batch Monte Carlo complete: id=%s, calcs=%d, n=%d, "
            "portfolio mean=%.2f, 95%% CI=[%.2f, %.2f], time=%.3fms",
            batch_id, len(calculations), iters,
            port_mean, port_ci_l, port_ci_u, processing_time,
        )
        return result

    # ==================================================================
    # Public API -- Analytical Error Propagation
    # ==================================================================

    def analytical_propagation(
        self,
        consumption: Decimal,
        consumption_uncertainty: Decimal,
        ef: Decimal,
        ef_uncertainty: Decimal,
        td_loss: Decimal = Decimal("0"),
        td_uncertainty: Decimal = Decimal("0"),
    ) -> Dict[str, Any]:
        """IPCC Approach 1 analytical error propagation for Scope 2.

        For the multiplicative model:
            Emissions = Consumption x EF x (1 + TD_loss)

        The combined relative uncertainty is:
            sigma_rel = sqrt(u_consumption^2 + u_ef^2 + u_td^2)

        where u_x is the relative uncertainty (sigma / value) for each
        parameter.

        Args:
            consumption: Energy consumption value (e.g. MWh).
            consumption_uncertainty: Absolute uncertainty in consumption.
            ef: Emission factor value (e.g. kg CO2e / MWh).
            ef_uncertainty: Absolute uncertainty in emission factor.
            td_loss: T&D loss factor as a fraction (e.g. 0.05 = 5%).
                Set to 0 to omit T&D from propagation.
            td_uncertainty: Absolute uncertainty in T&D loss factor.

        Returns:
            Dictionary with keys:
                - calculation_id: UUID string
                - status: "SUCCESS"
                - method: "ANALYTICAL_PROPAGATION"
                - emissions_kg: Decimal central emission estimate
                - combined_relative_uncertainty: Decimal fraction
                - combined_absolute_uncertainty: Decimal (kg)
                - ci_lower_95: Decimal lower bound (95% CI)
                - ci_upper_95: Decimal upper bound (95% CI)
                - parameter_contributions: dict of parameter -> fraction
                - formula: description string
                - provenance_hash: SHA-256 hex digest
        """
        self._increment("_total_analyses")
        self._increment("_total_analytical")
        start_time = time.monotonic()
        calc_id = str(uuid4())

        # -- Compute central estimate --------------------------------------
        td_multiplier = _ONE + td_loss
        emissions = consumption * ef * td_multiplier

        # -- Relative uncertainties ----------------------------------------
        u_consumption = (
            (consumption_uncertainty / consumption) if consumption != _ZERO else _ZERO
        )
        u_ef = (ef_uncertainty / ef) if ef != _ZERO else _ZERO
        u_td = (
            (td_uncertainty / td_multiplier) if td_multiplier != _ZERO else _ZERO
        )

        # -- Combined relative uncertainty (RSS) ---------------------------
        u_consumption_sq = u_consumption ** 2
        u_ef_sq = u_ef ** 2
        u_td_sq = u_td ** 2
        combined_sq = u_consumption_sq + u_ef_sq + u_td_sq
        combined_rel = _D(str(math.sqrt(float(combined_sq))))

        # -- Absolute uncertainty and 95% CI -------------------------------
        combined_abs = (emissions * combined_rel).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        z_95 = _D("1.96")
        ci_lower = (emissions - combined_abs * z_95).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        ci_upper = (emissions + combined_abs * z_95).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )

        # -- Parameter contributions --------------------------------------
        total_sq_float = float(combined_sq)
        contributions: Dict[str, Decimal] = {}
        if total_sq_float > 0:
            contributions["consumption"] = _D(str(round(
                float(u_consumption_sq) / total_sq_float, 6,
            )))
            contributions["emission_factor"] = _D(str(round(
                float(u_ef_sq) / total_sq_float, 6,
            )))
            contributions["td_loss"] = _D(str(round(
                float(u_td_sq) / total_sq_float, 6,
            )))
        else:
            contributions["consumption"] = _ZERO
            contributions["emission_factor"] = _ZERO
            contributions["td_loss"] = _ZERO

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "ANALYTICAL_PROPAGATION",
            "emissions_kg": emissions.quantize(_PRECISION, rounding=ROUND_HALF_UP),
            "combined_relative_uncertainty": combined_rel.quantize(
                _PRECISION, rounding=ROUND_HALF_UP,
            ),
            "combined_relative_uncertainty_pct": (combined_rel * _HUNDRED).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            ),
            "combined_absolute_uncertainty": combined_abs,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "parameter_contributions": contributions,
            "input_parameters": {
                "consumption": str(consumption),
                "consumption_uncertainty": str(consumption_uncertainty),
                "ef": str(ef),
                "ef_uncertainty": str(ef_uncertainty),
                "td_loss": str(td_loss),
                "td_uncertainty": str(td_uncertainty),
            },
            "formula": "sigma_rel = sqrt(u_consumption^2 + u_ef^2 + u_td^2)",
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        self._record_metric("analytical")

        logger.info(
            "Analytical propagation: id=%s, emissions=%.2f, "
            "combined_rel=%.4f (%.2f%%), 95%% CI=[%.2f, %.2f], time=%.3fms",
            calc_id, float(emissions), float(combined_rel),
            float(combined_rel * _HUNDRED),
            float(ci_lower), float(ci_upper), processing_time,
        )
        return result

    # ==================================================================
    # Public API -- Combined Uncertainty
    # ==================================================================

    def combined_uncertainty(self, uncertainties: List[Decimal]) -> Decimal:
        """Compute root-sum-of-squares for a list of independent uncertainties.

        For independent uncertainties u_1, u_2, ..., u_n (as fractions):
            u_combined = sqrt(u_1^2 + u_2^2 + ... + u_n^2)

        Args:
            uncertainties: List of relative uncertainty values as Decimal
                fractions (e.g. Decimal("0.05") for 5%).

        Returns:
            Combined relative uncertainty as Decimal.
        """
        if not uncertainties:
            return _ZERO

        sum_sq = sum(u ** 2 for u in uncertainties)
        return _D(str(round(math.sqrt(float(sum_sq)), 8)))

    # ==================================================================
    # Public API -- Propagate Multiplication
    # ==================================================================

    def propagate_multiplication(
        self,
        a: Decimal,
        u_a: Decimal,
        b: Decimal,
        u_b: Decimal,
    ) -> Tuple[Decimal, Decimal]:
        """Error propagation for multiplication: z = a * b.

        For z = a * b, the relative uncertainty is:
            u_z / z = sqrt((u_a / a)^2 + (u_b / b)^2)
        therefore:
            u_z = z * sqrt((u_a / a)^2 + (u_b / b)^2)

        Args:
            a: First operand value.
            u_a: Absolute uncertainty in *a*.
            b: Second operand value.
            u_b: Absolute uncertainty in *b*.

        Returns:
            Tuple of (z, u_z) where z = a * b and u_z is the
            absolute uncertainty in z.
        """
        z = a * b

        rel_a_sq = (u_a / a) ** 2 if a != _ZERO else _ZERO
        rel_b_sq = (u_b / b) ** 2 if b != _ZERO else _ZERO
        rel_z = _D(str(math.sqrt(float(rel_a_sq + rel_b_sq))))
        u_z = (abs(z) * rel_z).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        return (
            z.quantize(_PRECISION, rounding=ROUND_HALF_UP),
            u_z,
        )

    # ==================================================================
    # Public API -- Propagate Addition
    # ==================================================================

    def propagate_addition(
        self,
        values: List[Decimal],
        uncertainties: List[Decimal],
    ) -> Tuple[Decimal, Decimal]:
        """Error propagation for addition: z = sum(a_i).

        For z = a_1 + a_2 + ... + a_n, the absolute uncertainty is:
            sigma_z = sqrt(sigma_1^2 + sigma_2^2 + ... + sigma_n^2)

        Args:
            values: List of operand values.
            uncertainties: List of absolute uncertainties (same length
                as *values*).

        Returns:
            Tuple of (z, sigma_z) where z = sum(values) and sigma_z
            is the absolute uncertainty in z.

        Raises:
            ValueError: If *values* and *uncertainties* have different lengths.
        """
        if len(values) != len(uncertainties):
            raise ValueError(
                f"values ({len(values)}) and uncertainties ({len(uncertainties)}) "
                "must have the same length"
            )

        z = sum(values, _ZERO)
        sum_sq = sum(u ** 2 for u in uncertainties)
        sigma_z = _D(str(round(math.sqrt(float(sum_sq)), 8)))

        return (
            z.quantize(_PRECISION, rounding=ROUND_HALF_UP),
            sigma_z.quantize(_PRECISION, rounding=ROUND_HALF_UP),
        )

    # ==================================================================
    # Public API -- IPCC Default Uncertainty Ranges
    # ==================================================================

    def get_ipcc_default_uncertainties(self) -> Dict[str, Any]:
        """Return IPCC default uncertainty ranges for Scope 2 parameters.

        Returns a comprehensive dictionary of uncertainty ranges (95% CI
        half-widths as fractions) for all key parameters in Scope 2
        location-based emission calculations.

        The ranges are drawn from:
            - IPCC 2006 Guidelines Vol 1 Ch 3 Table 3.2
            - GHG Protocol Corporate Standard Chapter 8
            - EPA eGRID Technical Support Document
            - IEA Emission Factors Methodology

        Returns:
            Dictionary with nested structure:
                grid_ef -> {tier_1, tier_2, tier_3} -> {lower_pct, upper_pct, default_pct}
                activity_data -> {meter, invoice, estimate, benchmark} -> ...
                td_loss -> {lower_pct, upper_pct, default_pct}
                steam_heat_ef -> {lower_pct, upper_pct, default_pct}
                cooling_ef -> {lower_pct, upper_pct, default_pct}
                gwp -> {CO2, CH4, N2O} -> {pct}
                per_gas_ef -> {CO2, CH4, N2O} -> {lower_pct, upper_pct, default_pct}
        """
        return {
            "grid_ef": dict(GRID_EF_UNCERTAINTY),
            "activity_data": dict(ACTIVITY_DATA_UNCERTAINTY),
            "td_loss": dict(TD_LOSS_UNCERTAINTY),
            "steam_heat_ef": dict(STEAM_HEAT_UNCERTAINTY),
            "cooling_ef": dict(COOLING_UNCERTAINTY),
            "gwp": {k: dict(v) for k, v in GWP_UNCERTAINTY.items()},
            "per_gas_ef": {k: dict(v) for k, v in PER_GAS_EF_UNCERTAINTY.items()},
        }

    # ==================================================================
    # Public API -- Data Quality Tier Uncertainty
    # ==================================================================

    def get_data_quality_uncertainty(self, data_quality_tier: str) -> Decimal:
        """Map a data quality tier to an uncertainty percentage.

        Tier mapping (95% CI half-widths):
            tier_1: +/-5%  (high quality - measured/verified data)
            tier_2: +/-15% (medium quality - documented estimates)
            tier_3: +/-30% (low quality - rough estimates/benchmarks)

        Args:
            data_quality_tier: One of "tier_1", "tier_2", "tier_3".
                Case-insensitive.

        Returns:
            Decimal uncertainty as a fraction (e.g. Decimal("0.05")
            for tier_1).
        """
        tier = data_quality_tier.lower().strip()
        pct = DATA_QUALITY_TIER_UNCERTAINTY.get(tier)
        if pct is not None:
            return _D(str(pct))
        logger.warning(
            "Unknown data quality tier '%s'; returning tier_3 default (0.30)",
            data_quality_tier,
        )
        return Decimal("0.30")

    # ==================================================================
    # Public API -- Consumption Source Uncertainty
    # ==================================================================

    def get_consumption_source_uncertainty(self, source: str) -> Decimal:
        """Map a consumption data source to an uncertainty percentage.

        Source mapping (95% CI half-widths):
            meter / smart_meter / calibrated_meter: +/-2%
            revenue_meter: +/-3%
            invoice / utility_bill: +/-5%
            estimate / engineering_estimate: +/-20%
            benchmark / industry_average: +/-30%
            unknown: +/-40%

        Args:
            source: Consumption data source identifier.
                Case-insensitive.

        Returns:
            Decimal uncertainty as a fraction (e.g. Decimal("0.02")
            for "meter").
        """
        src = source.lower().strip()
        pct = CONSUMPTION_SOURCE_UNCERTAINTY.get(src)
        if pct is not None:
            return _D(str(pct))
        logger.warning(
            "Unknown consumption source '%s'; returning default (0.40)",
            source,
        )
        return Decimal("0.40")

    # ==================================================================
    # Public API -- DQI (Data Quality Indicator) Scoring
    # ==================================================================

    def calculate_dqi_score(
        self,
        ef_source: str,
        ef_year: int,
        activity_source: str,
        temporal_representativeness: int = 0,
    ) -> Decimal:
        """Calculate a composite Data Quality Indicator (DQI) score.

        Combines four quality dimensions into a single 0-1 score where
        higher values indicate better data quality.

        Scoring dimensions:
            1. EF source quality: custom=0.9, national=0.8, eGRID=0.85,
               IEA=0.7, IPCC=0.5, unknown=0.2
            2. EF age: current_year=1.0, -1yr=0.9, -2yr=0.8, -3yr=0.7,
               older=0.5
            3. Activity data source: meter=0.95, invoice=0.85,
               estimate=0.6, benchmark=0.4
            4. Temporal representativeness: same_year=1.0, -1yr=0.9,
               -2yr=0.8, -3yr=0.7, older=0.5

        The composite score is the weighted average:
            DQI = 0.35 * ef_source + 0.20 * ef_age + 0.30 * activity + 0.15 * temporal

        Args:
            ef_source: Emission factor source identifier (case-insensitive).
                See EF_SOURCE_SCORES for valid values.
            ef_year: Year the emission factor was published or last updated.
            activity_source: Activity data source identifier (case-insensitive).
                See ACTIVITY_SOURCE_SCORES for valid values.
            temporal_representativeness: Year gap between reporting period
                and data collection.  0 = same year, 1 = one year old, etc.

        Returns:
            Decimal DQI score between 0 and 1.
        """
        self._increment("_total_analyses")
        self._increment("_total_dqi")

        # -- Dimension 1: EF source quality --------------------------------
        ef_src_lower = ef_source.lower().strip()
        ef_src_score = EF_SOURCE_SCORES.get(ef_src_lower, 0.20)

        # -- Dimension 2: EF age -------------------------------------------
        current_year = _utcnow().year
        age = max(0, current_year - ef_year)
        if age == 0:
            ef_age_score = 1.0
        elif age == 1:
            ef_age_score = 0.9
        elif age == 2:
            ef_age_score = 0.8
        elif age == 3:
            ef_age_score = 0.7
        elif age <= 5:
            ef_age_score = 0.6
        else:
            ef_age_score = 0.5

        # -- Dimension 3: Activity data source -----------------------------
        act_src_lower = activity_source.lower().strip()
        act_src_score = ACTIVITY_SOURCE_SCORES.get(act_src_lower, 0.20)

        # -- Dimension 4: Temporal representativeness ----------------------
        temp_gap = max(0, temporal_representativeness)
        if temp_gap == 0:
            temp_score = 1.0
        elif temp_gap == 1:
            temp_score = 0.9
        elif temp_gap == 2:
            temp_score = 0.8
        elif temp_gap == 3:
            temp_score = 0.7
        else:
            temp_score = 0.5

        # -- Composite score (weighted average) ----------------------------
        composite = (
            0.35 * ef_src_score
            + 0.20 * ef_age_score
            + 0.30 * act_src_score
            + 0.15 * temp_score
        )

        # Clamp to [0, 1]
        composite = max(0.0, min(1.0, composite))

        result = _D(str(round(composite, 4)))

        self._record_metric("ipcc_default_uncertainty")

        logger.debug(
            "DQI score: ef_src=%s (%.2f), ef_age=%d (%.2f), "
            "act_src=%s (%.2f), temp=%d (%.2f) -> composite=%.4f",
            ef_source, ef_src_score, age, ef_age_score,
            activity_source, act_src_score, temp_gap, temp_score,
            float(result),
        )
        return result

    # ==================================================================
    # Public API -- DQI Score to Uncertainty
    # ==================================================================

    def score_to_uncertainty(self, dqi_score: Decimal) -> Decimal:
        """Map a DQI score (0-1) to an uncertainty percentage.

        The mapping uses an inverse linear relationship where higher
        DQI scores (better quality) produce lower uncertainty:

            uncertainty = 0.40 - 0.35 * dqi_score

        This maps:
            DQI 1.0 -> 5% uncertainty
            DQI 0.8 -> 12% uncertainty
            DQI 0.5 -> 22.5% uncertainty
            DQI 0.2 -> 33% uncertainty
            DQI 0.0 -> 40% uncertainty

        Args:
            dqi_score: DQI score between 0 and 1.

        Returns:
            Decimal uncertainty as a fraction (e.g. Decimal("0.05")
            for DQI score of 1.0).
        """
        score_float = max(0.0, min(1.0, float(dqi_score)))
        uncertainty = 0.40 - 0.35 * score_float
        # Clamp to reasonable range [0.02, 0.50]
        uncertainty = max(0.02, min(0.50, uncertainty))
        return _D(str(round(uncertainty, 4)))

    # ==================================================================
    # Public API -- Sensitivity Analysis
    # ==================================================================

    def sensitivity_analysis(
        self,
        base_result: Decimal,
        parameters: Dict[str, Decimal],
        variation_pct: Decimal = Decimal("0.10"),
    ) -> Dict[str, Any]:
        """One-at-a-time sensitivity analysis for Scope 2 calculations.

        Varies each parameter by +/- *variation_pct* while holding all
        others constant, and measures the impact on the emission result.
        The sensitivity coefficient is defined as:
            S_i = (result_high - result_low) / (2 * variation * base_result)

        Args:
            base_result: Central emission estimate (kg or tCO2e).
            parameters: Dictionary mapping parameter names to their
                central values.  Each parameter is assumed to enter
                the calculation multiplicatively.
            variation_pct: Fraction to vary each parameter (e.g. 0.10
                for +/-10%).

        Returns:
            Dictionary with keys:
                - calculation_id: UUID string
                - status: "SUCCESS"
                - method: "SENSITIVITY_ANALYSIS"
                - base_result: Decimal
                - variation_pct: Decimal
                - parameters: dict of parameter_name -> {base, low, high,
                  result_low, result_high, sensitivity_coefficient, impact_range}
                - provenance_hash: SHA-256 hex digest
        """
        self._increment("_total_analyses")
        self._increment("_total_sensitivity")
        start_time = time.monotonic()
        calc_id = str(uuid4())

        base_float = float(base_result)
        var_float = float(variation_pct)
        param_results: Dict[str, Dict[str, Any]] = {}

        for name, value in parameters.items():
            val_float = float(value)
            if val_float == 0.0:
                param_results[name] = {
                    "base": value,
                    "low": value,
                    "high": value,
                    "result_low": base_result,
                    "result_high": base_result,
                    "sensitivity_coefficient": _ZERO,
                    "impact_range": _ZERO,
                }
                continue

            low_val = val_float * (1.0 - var_float)
            high_val = val_float * (1.0 + var_float)

            # Multiplicative scaling
            scale_low = low_val / val_float
            scale_high = high_val / val_float
            result_low = base_float * scale_low
            result_high = base_float * scale_high

            impact_range = abs(result_high - result_low)
            sensitivity_coeff = (
                impact_range / (2.0 * var_float * base_float)
                if base_float != 0.0 and var_float != 0.0
                else 0.0
            )

            param_results[name] = {
                "base": _D(str(round(val_float, 6))),
                "low": _D(str(round(low_val, 6))),
                "high": _D(str(round(high_val, 6))),
                "result_low": _D(str(round(result_low, 6))),
                "result_high": _D(str(round(result_high, 6))),
                "sensitivity_coefficient": _D(str(round(sensitivity_coeff, 6))),
                "impact_range": _D(str(round(impact_range, 6))),
            }

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "SENSITIVITY_ANALYSIS",
            "base_result": base_result,
            "variation_pct": variation_pct,
            "parameters": param_results,
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        self._record_metric("error_propagation")

        logger.info(
            "Sensitivity analysis complete: id=%s, params=%d, time=%.3fms",
            calc_id, len(parameters), processing_time,
        )
        return result

    # ==================================================================
    # Public API -- Tornado Analysis
    # ==================================================================

    def tornado_analysis(
        self,
        base_result: Decimal,
        parameters: Dict[str, Decimal],
    ) -> List[Dict[str, Any]]:
        """Generate tornado chart data sorted by impact magnitude.

        Runs a sensitivity analysis with +/-10% variation on each
        parameter, then sorts the results by descending impact range
        for tornado chart visualisation.

        Args:
            base_result: Central emission estimate.
            parameters: Dictionary mapping parameter names to values.

        Returns:
            List of dictionaries sorted by descending impact_range,
            each containing:
                - parameter: name
                - result_low: Decimal emission at -10%
                - result_high: Decimal emission at +10%
                - impact_range: Decimal absolute range
                - sensitivity_coefficient: Decimal
                - contribution_pct: Decimal percentage of total variance
        """
        self._increment("_total_analyses")
        self._increment("_total_sensitivity")

        # Run base sensitivity analysis at +/-10%
        sa = self.sensitivity_analysis(
            base_result=base_result,
            parameters=parameters,
            variation_pct=Decimal("0.10"),
        )

        param_data = sa.get("parameters", {})

        # Collect impact ranges
        items: List[Dict[str, Any]] = []
        total_variance = Decimal("0")

        for name, data in param_data.items():
            impact = data.get("impact_range", _ZERO)
            variance_contrib = impact ** 2
            total_variance += variance_contrib
            items.append({
                "parameter": name,
                "result_low": data.get("result_low", _ZERO),
                "result_high": data.get("result_high", _ZERO),
                "impact_range": impact,
                "sensitivity_coefficient": data.get("sensitivity_coefficient", _ZERO),
                "variance_contribution": variance_contrib,
            })

        # Sort by descending impact range
        items.sort(key=lambda x: x["impact_range"], reverse=True)

        # Add contribution percentages
        for item in items:
            if total_variance > _ZERO:
                item["contribution_pct"] = (
                    item["variance_contribution"] / total_variance * _HUNDRED
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            else:
                item["contribution_pct"] = _ZERO
            # Remove intermediate field
            del item["variance_contribution"]

        return items

    # ==================================================================
    # Public API -- Statistical Helpers: Percentiles
    # ==================================================================

    def calculate_percentiles(
        self,
        values: List[Decimal],
        percentiles: Optional[List[int]] = None,
    ) -> Dict[str, Decimal]:
        """Calculate percentiles from a list of Decimal values.

        Uses linear interpolation between adjacent sorted values
        for non-integer index positions.

        Args:
            values: List of Decimal values.
            percentiles: List of percentile points (0-100).
                Default: [5, 25, 50, 75, 95].

        Returns:
            Dictionary mapping percentile labels to Decimal values.
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]

        if not values:
            return {str(p): _ZERO for p in percentiles}

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        result: Dict[str, Decimal] = {}

        for p in percentiles:
            idx = (float(p) / 100.0) * (n - 1)
            lower = int(math.floor(idx))
            upper = min(lower + 1, n - 1)
            frac = _D(str(idx - lower))
            val = sorted_vals[lower] * (_ONE - frac) + sorted_vals[upper] * frac
            result[str(p)] = val.quantize(_PRECISION, rounding=ROUND_HALF_UP)

        return result

    # ==================================================================
    # Public API -- Statistical Helpers: Confidence Interval
    # ==================================================================

    def calculate_confidence_interval(
        self,
        values: List[Decimal],
        confidence: Decimal = Decimal("0.95"),
    ) -> Tuple[Decimal, Decimal]:
        """Calculate a confidence interval from a list of Decimal values.

        Uses the percentile method on sorted values.

        Args:
            values: List of Decimal values.
            confidence: Confidence level as a fraction (e.g. 0.95).

        Returns:
            Tuple of (lower_bound, upper_bound) as Decimals.
        """
        if not values:
            return (_ZERO, _ZERO)

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        alpha = (float(_ONE - confidence)) / 2.0
        lower_idx = max(0, int(alpha * n))
        upper_idx = min(n - 1, int((1.0 - alpha) * n))
        return (
            sorted_vals[lower_idx].quantize(_PRECISION, rounding=ROUND_HALF_UP),
            sorted_vals[upper_idx].quantize(_PRECISION, rounding=ROUND_HALF_UP),
        )

    # ==================================================================
    # Public API -- Statistical Helpers: Full Statistics
    # ==================================================================

    def calculate_statistics(self, values: List[Decimal]) -> Dict[str, Any]:
        """Calculate descriptive statistics for a list of Decimal values.

        Computes mean, median, standard deviation, minimum, maximum,
        skewness, and kurtosis.

        Args:
            values: List of Decimal values.

        Returns:
            Dictionary with keys:
                - count: int
                - mean: Decimal
                - median: Decimal
                - std_dev: Decimal
                - min: Decimal
                - max: Decimal
                - skewness: Decimal (Fisher's definition)
                - kurtosis: Decimal (excess kurtosis)
                - cv_pct: Decimal coefficient of variation (%)
        """
        if not values:
            return {
                "count": 0,
                "mean": _ZERO,
                "median": _ZERO,
                "std_dev": _ZERO,
                "min": _ZERO,
                "max": _ZERO,
                "skewness": _ZERO,
                "kurtosis": _ZERO,
                "cv_pct": _ZERO,
            }

        n = len(values)
        sorted_vals = sorted(values)
        floats = [float(v) for v in sorted_vals]

        mean_val = sum(floats) / n

        # Median
        if n % 2 == 0:
            median_val = (floats[n // 2 - 1] + floats[n // 2]) / 2.0
        else:
            median_val = floats[n // 2]

        # Standard deviation (sample)
        if n > 1:
            variance = sum((x - mean_val) ** 2 for x in floats) / (n - 1)
            std_val = math.sqrt(variance)
        else:
            variance = 0.0
            std_val = 0.0

        # CV
        cv_pct = (std_val / abs(mean_val) * 100.0) if mean_val != 0.0 else 0.0

        # Skewness (Fisher's definition)
        if n > 2 and std_val > 0:
            skew = (
                n / ((n - 1) * (n - 2))
                * sum(((x - mean_val) / std_val) ** 3 for x in floats)
            )
        else:
            skew = 0.0

        # Excess kurtosis
        if n > 3 and std_val > 0:
            m4 = sum((x - mean_val) ** 4 for x in floats) / n
            m2 = sum((x - mean_val) ** 2 for x in floats) / n
            kurt = (m4 / (m2 ** 2)) - 3.0 if m2 != 0 else 0.0
        else:
            kurt = 0.0

        return {
            "count": n,
            "mean": _D(str(round(mean_val, 8))),
            "median": _D(str(round(median_val, 8))),
            "std_dev": _D(str(round(std_val, 8))),
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "skewness": _D(str(round(skew, 6))),
            "kurtosis": _D(str(round(kurt, 6))),
            "cv_pct": _D(str(round(cv_pct, 4))),
        }

    # ==================================================================
    # Public API -- Sampling Helpers
    # ==================================================================

    def normal_sample(
        self,
        mean: float,
        std: float,
        n: int,
        seed: Optional[int] = None,
    ) -> List[Decimal]:
        """Generate *n* samples from a normal distribution as Decimals.

        Samples are bounded at zero (emissions cannot be negative).

        Args:
            mean: Mean of the normal distribution.
            std: Standard deviation of the normal distribution.
            n: Number of samples to generate.
            seed: Optional PRNG seed for reproducibility.

        Returns:
            List of *n* Decimal samples.
        """
        actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
        rng = random.Random(actual_seed)
        samples: List[Decimal] = []
        for _ in range(n):
            val = max(0.0, rng.gauss(mean, std))
            samples.append(_D(str(round(val, 8))))
        return samples

    def lognormal_sample(
        self,
        mean: float,
        std: float,
        n: int,
        seed: Optional[int] = None,
    ) -> List[Decimal]:
        """Generate *n* samples from a lognormal distribution as Decimals.

        The lognormal is parameterised from the desired arithmetic mean
        and standard deviation using:
            sigma^2 = ln(1 + (std/mean)^2)
            mu = ln(mean) - sigma^2 / 2

        Args:
            mean: Desired arithmetic mean of the lognormal.  Must be > 0.
            std: Desired arithmetic standard deviation.  Must be > 0.
            n: Number of samples to generate.
            seed: Optional PRNG seed for reproducibility.

        Returns:
            List of *n* Decimal samples.
        """
        if mean <= 0 or std <= 0:
            return [_D(str(round(max(0.0, mean), 8)))] * n

        actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
        rng = random.Random(actual_seed)

        cv = std / mean
        sigma2 = math.log(1 + cv ** 2)
        sigma = math.sqrt(sigma2)
        mu = math.log(mean) - sigma2 / 2

        samples: List[Decimal] = []
        for _ in range(n):
            val = rng.lognormvariate(mu, sigma)
            samples.append(_D(str(round(val, 8))))
        return samples

    # ==================================================================
    # Public API -- Validation Helpers
    # ==================================================================

    def validate_uncertainty_input(
        self,
        base: Decimal,
        uncertainty_pct: Decimal,
    ) -> List[str]:
        """Validate Monte Carlo / analytical uncertainty inputs.

        Checks:
            - base must be non-negative
            - uncertainty_pct must be in [0, 10.0] (0% to 1000%)

        Args:
            base: Base emission value (kg or tCO2e).
            uncertainty_pct: Relative uncertainty as a fraction.

        Returns:
            List of validation error strings (empty if valid).
        """
        errors: List[str] = []

        base_d = _safe_decimal(base, Decimal("-1"))
        if base_d < _ZERO:
            errors.append("base_emissions must be non-negative")

        unc_d = _safe_decimal(uncertainty_pct, Decimal("-1"))
        if unc_d < _ZERO:
            errors.append("uncertainty_pct must be non-negative")
        elif unc_d > Decimal("10"):
            errors.append(
                "uncertainty_pct must be <= 10.0 (1000%); "
                f"got {uncertainty_pct}"
            )

        return errors

    def validate_iterations(self, n: int) -> List[str]:
        """Validate Monte Carlo iteration count.

        Args:
            n: Number of iterations.

        Returns:
            List of validation error strings (empty if valid).
        """
        errors: List[str] = []
        if n < 100:
            errors.append(
                f"iterations must be >= 100; got {n}"
            )
        if n > 1_000_000:
            errors.append(
                f"iterations must be <= 1,000,000; got {n}"
            )
        return errors

    # ==================================================================
    # Public API -- Engine Statistics
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine usage statistics.

        Returns:
            Dictionary with engine identification, creation timestamp,
            and running counters for all analysis types.
        """
        with self._lock:
            return {
                "engine": "UncertaintyQuantifierEngine",
                "agent": "AGENT-MRV-009",
                "component": "Scope 2 Location-Based Emissions",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "default_iterations": self._default_iterations,
                "default_confidence": str(self._default_confidence),
                "total_analyses": self._total_analyses,
                "total_monte_carlo": self._total_monte_carlo,
                "total_analytical": self._total_analytical,
                "total_dqi": self._total_dqi,
                "total_sensitivity": self._total_sensitivity,
            }

    # ==================================================================
    # Public API -- Reset
    # ==================================================================

    def reset(self) -> None:
        """Reset all engine counters to zero.

        Intended for testing only. Does not affect configuration.
        """
        with self._lock:
            self._total_analyses = 0
            self._total_monte_carlo = 0
            self._total_analytical = 0
            self._total_dqi = 0
            self._total_sensitivity = 0
        logger.info("UncertaintyQuantifierEngine counters reset")

    # ==================================================================
    # Internal -- Percentile Computation (float)
    # ==================================================================

    def _compute_percentiles_float(
        self,
        sorted_values: List[float],
        percentile_points: List[float],
    ) -> Dict[float, float]:
        """Compute percentiles from a pre-sorted list of floats.

        Uses linear interpolation between adjacent values for
        non-integer index positions.

        Args:
            sorted_values: Pre-sorted list of float values.
            percentile_points: List of percentile points (0-100).

        Returns:
            Dictionary mapping percentile points to values.
        """
        n = len(sorted_values)
        if n == 0:
            return {p: 0.0 for p in percentile_points}

        result: Dict[float, float] = {}
        for p in percentile_points:
            idx = (p / 100.0) * (n - 1)
            lower = int(math.floor(idx))
            upper = min(lower + 1, n - 1)
            frac = idx - lower
            val = sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac
            result[p] = val

        return result

    # ==================================================================
    # Internal -- Percentile Computation (Decimal)
    # ==================================================================

    def _compute_percentiles_decimal(
        self,
        sorted_values: List[Decimal],
        percentile_points: List[int],
    ) -> Dict[str, Decimal]:
        """Compute percentiles from a pre-sorted list of Decimals.

        Uses linear interpolation between adjacent values for
        non-integer index positions.

        Args:
            sorted_values: Pre-sorted list of Decimal values.
            percentile_points: List of percentile points (0-100).

        Returns:
            Dictionary mapping percentile labels to Decimal values.
        """
        n = len(sorted_values)
        if n == 0:
            return {str(p): _ZERO for p in percentile_points}

        result: Dict[str, Decimal] = {}
        for p in percentile_points:
            idx = (float(p) / 100.0) * (n - 1)
            lower = int(math.floor(idx))
            upper = min(lower + 1, n - 1)
            frac = _D(str(idx - lower))
            val = sorted_values[lower] * (_ONE - frac) + sorted_values[upper] * frac
            result[str(p)] = val.quantize(_PRECISION, rounding=ROUND_HALF_UP)

        return result


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    "UncertaintyQuantifierEngine",
    # Constants
    "GRID_EF_UNCERTAINTY",
    "ACTIVITY_DATA_UNCERTAINTY",
    "TD_LOSS_UNCERTAINTY",
    "STEAM_HEAT_UNCERTAINTY",
    "COOLING_UNCERTAINTY",
    "GWP_UNCERTAINTY",
    "PER_GAS_EF_UNCERTAINTY",
    "EF_SOURCE_SCORES",
    "ACTIVITY_SOURCE_SCORES",
    "DATA_QUALITY_TIER_UNCERTAINTY",
    "CONSUMPTION_SOURCE_UNCERTAINTY",
]
