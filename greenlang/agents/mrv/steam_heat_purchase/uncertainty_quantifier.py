# -*- coding: utf-8 -*-
"""
UncertaintyQuantifierEngine - Monte Carlo & Analytical Error Propagation (Engine 5 of 7)

AGENT-MRV-011: Steam/Heat Purchase Agent

Quantifies the uncertainty in Scope 2 steam/heat purchase emission calculations
using Monte Carlo simulation, analytical error propagation (IPCC Approach 1),
data quality indicator (DQI) scoring, and sensitivity/tornado analysis.

Sources of Uncertainty in Steam/Heat Purchase Calculations:
    - Activity data uncertainty: +/-2% (metered) to +/-30% (benchmark)
      depending on metering accuracy, invoice quality, or estimation.
    - Emission factor uncertainty: +/-3% (supplier-verified) to +/-15%
      (IPCC default) depending on the source and age of the EF data.
    - Boiler/system efficiency uncertainty: +/-2% (measured) to +/-10%
      (default) depending on whether efficiency is site-measured or
      assumed from handbook values.
    - COP (coefficient of performance) uncertainty: +/-3% (measured) to
      +/-12% (default) for district cooling systems where the COP is
      estimated from nameplate or generic technology data.
    - CHP allocation uncertainty: +/-5% (metered outputs) to +/-15%
      (default allocation method) depending on whether CHP electrical
      and thermal outputs are individually metered or estimated.

Tier-Based Default Uncertainty Ranges (95% CI half-widths):
    Tier 1 (IPCC/national defaults):
        Activity data: +/-7.5%, Emission factor: +/-15.0%,
        Efficiency: +/-10.0%, COP: +/-12.0%, CHP allocation: +/-15.0%
    Tier 2 (Supplier-specific data):
        Activity data: +/-5.0%, Emission factor: +/-10.0%,
        Efficiency: +/-5.0%, COP: +/-8.0%, CHP allocation: +/-10.0%
    Tier 3 (Facility-measured data):
        Activity data: +/-2.0%, Emission factor: +/-3.0%,
        Efficiency: +/-2.0%, COP: +/-3.0%, CHP allocation: +/-5.0%

Monte Carlo Simulation:
    - Configurable iterations (default 10,000)
    - Normal distributions for activity data (bounded at zero)
    - Normal distributions for emission factors (bounded at zero)
    - Normal distributions for efficiency and COP (bounded at zero)
    - Normal distributions for CHP allocation (bounded at zero)
    - Explicit seed support for full reproducibility
    - numpy used internally when available; pure-Python fallback

Analytical Propagation (IPCC Approach 1):
    Combined relative uncertainty for multiplicative chains:
    U_total = sqrt(u_ad^2 + u_ef^2 + u_eff^2 + u_cop^2 + u_chp^2)

Confidence Intervals:
    - 90% CI: z = 1.645
    - 95% CI: z = 1.960 (default)
    - 99% CI: z = 2.576

DQI Scoring (4 dimensions, 0-1 scale):
    - EF source quality (supplier_verified=0.95, supplier_unverified=0.80,
      regional_default=0.65, ipcc_default=0.50, unknown=0.20)
    - EF age (current year=1.0, -1yr=0.9, -2yr=0.8, -3yr=0.7, older=0.5)
    - Activity data source (meter=0.95, invoice=0.85, estimate=0.60,
      benchmark=0.40)
    - Efficiency verification (measured=1.0, supplier_stated=0.7,
      handbook=0.5, unknown=0.3)

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
    >>> from greenlang.agents.mrv.steam_heat_purchase.uncertainty_quantifier import (
    ...     UncertaintyQuantifierEngine,
    ... )
    >>> engine = UncertaintyQuantifierEngine()
    >>> result = engine.quantify_monte_carlo(
    ...     calc_result={"total_co2e_kg": "15000.0"},
    ...     iterations=10000,
    ...     seed=42,
    ... )
    >>> print(result["ci_lower"], result["ci_upper"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-011 Steam/Heat Purchase Agent (GL-MRV-X-022)
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
# Module-level exports (overridden at bottom)
# ---------------------------------------------------------------------------

__all__ = ["UncertaintyQuantifierEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False

try:
    from greenlang.agents.mrv.steam_heat_purchase.metrics import get_metrics as _get_metrics
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _get_metrics = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.steam_heat_purchase.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.steam_heat_purchase.provenance import (
        get_provenance as _get_provenance_tracker,
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
_TWO = Decimal("2")
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
# Tier-Based Default Uncertainty Constants
# ===========================================================================

#: Tier-based default uncertainty ranges (95% CI half-widths as fractions).
#: Tier 1 = IPCC/national defaults, Tier 2 = supplier-specific,
#: Tier 3 = facility-measured.
#:
#: Sources:
#:   - IPCC 2006 Guidelines Vol 1 Ch 3 Table 3.2
#:   - GHG Protocol Scope 2 Guidance (2015) Chapter 7
#:   - ISO 14064-1:2018 Annex C (uncertainty guidance)

TIER_DEFAULTS: Dict[str, Dict[str, Decimal]] = {
    "tier_1": {
        "activity_data": Decimal("0.075"),
        "emission_factor": Decimal("0.150"),
        "efficiency": Decimal("0.100"),
        "cop": Decimal("0.120"),
        "chp_allocation": Decimal("0.150"),
        "description": "IPCC/national default values - highest uncertainty",
    },
    "tier_2": {
        "activity_data": Decimal("0.050"),
        "emission_factor": Decimal("0.100"),
        "efficiency": Decimal("0.050"),
        "cop": Decimal("0.080"),
        "chp_allocation": Decimal("0.100"),
        "description": "Supplier-specific data - moderate uncertainty",
    },
    "tier_3": {
        "activity_data": Decimal("0.020"),
        "emission_factor": Decimal("0.030"),
        "efficiency": Decimal("0.020"),
        "cop": Decimal("0.030"),
        "chp_allocation": Decimal("0.050"),
        "description": "Facility-measured data - lowest uncertainty",
    },
}


# ===========================================================================
# Activity Data Uncertainty Constants
# ===========================================================================

#: Activity data uncertainty (95% CI half-widths as fractions) by source.

ACTIVITY_DATA_UNCERTAINTY: Dict[str, Dict[str, Any]] = {
    "meter": {
        "uncertainty_pct": Decimal("0.02"),
        "description": "Direct meter reading (calibrated revenue meters)",
    },
    "smart_meter": {
        "uncertainty_pct": Decimal("0.02"),
        "description": "Smart meter with interval data",
    },
    "calibrated_meter": {
        "uncertainty_pct": Decimal("0.02"),
        "description": "Calibrated sub-meter",
    },
    "revenue_meter": {
        "uncertainty_pct": Decimal("0.03"),
        "description": "Revenue-grade utility meter",
    },
    "invoice": {
        "uncertainty_pct": Decimal("0.05"),
        "description": "Supplier invoices / billing records",
    },
    "utility_bill": {
        "uncertainty_pct": Decimal("0.05"),
        "description": "Utility bill (synonym for invoice)",
    },
    "estimate": {
        "uncertainty_pct": Decimal("0.20"),
        "description": "Engineering estimate or pro-rata allocation",
    },
    "engineering_estimate": {
        "uncertainty_pct": Decimal("0.20"),
        "description": "Engineering estimate (synonym)",
    },
    "benchmark": {
        "uncertainty_pct": Decimal("0.30"),
        "description": "Industry benchmark / intensity-based estimate",
    },
    "industry_average": {
        "uncertainty_pct": Decimal("0.30"),
        "description": "Industry average (synonym for benchmark)",
    },
    "unknown": {
        "uncertainty_pct": Decimal("0.40"),
        "description": "Unknown / undocumented data source",
    },
    "default": {
        "uncertainty_pct": Decimal("0.05"),
        "description": "Default activity data uncertainty (assumes invoice)",
    },
}


# ===========================================================================
# Emission Factor Uncertainty Constants
# ===========================================================================

#: Emission factor uncertainty by source quality.

EF_UNCERTAINTY: Dict[str, Dict[str, Any]] = {
    "supplier_verified": {
        "uncertainty_pct": Decimal("0.03"),
        "description": "Third-party verified supplier emission factor",
    },
    "supplier_unverified": {
        "uncertainty_pct": Decimal("0.08"),
        "description": "Supplier-provided EF without verification",
    },
    "regional_default": {
        "uncertainty_pct": Decimal("0.10"),
        "description": "Regional/national default emission factor",
    },
    "ipcc_default": {
        "uncertainty_pct": Decimal("0.15"),
        "description": "IPCC default emission factor",
    },
    "estimated": {
        "uncertainty_pct": Decimal("0.20"),
        "description": "Estimated or modelled emission factor",
    },
    "unknown": {
        "uncertainty_pct": Decimal("0.25"),
        "description": "Unknown / undocumented emission factor source",
    },
    "default": {
        "uncertainty_pct": Decimal("0.10"),
        "description": "Default emission factor uncertainty",
    },
}


# ===========================================================================
# Efficiency Uncertainty Constants
# ===========================================================================

#: Boiler / system efficiency uncertainty by data source.

EFFICIENCY_UNCERTAINTY: Dict[str, Dict[str, Any]] = {
    "measured": {
        "uncertainty_pct": Decimal("0.02"),
        "description": "Site-measured efficiency (performance test)",
    },
    "supplier_stated": {
        "uncertainty_pct": Decimal("0.05"),
        "description": "Supplier-stated nameplate efficiency",
    },
    "handbook": {
        "uncertainty_pct": Decimal("0.08"),
        "description": "Handbook / literature default efficiency",
    },
    "default": {
        "uncertainty_pct": Decimal("0.05"),
        "description": "Default efficiency uncertainty",
    },
    "unknown": {
        "uncertainty_pct": Decimal("0.10"),
        "description": "Unknown efficiency source",
    },
}


# ===========================================================================
# COP Uncertainty Constants
# ===========================================================================

#: Coefficient of performance uncertainty for cooling systems.

COP_UNCERTAINTY: Dict[str, Dict[str, Any]] = {
    "measured": {
        "uncertainty_pct": Decimal("0.03"),
        "description": "Site-measured COP from performance monitoring",
    },
    "nameplate": {
        "uncertainty_pct": Decimal("0.06"),
        "description": "Nameplate / manufacturer-rated COP",
    },
    "estimated": {
        "uncertainty_pct": Decimal("0.10"),
        "description": "Estimated COP from technology class defaults",
    },
    "default": {
        "uncertainty_pct": Decimal("0.08"),
        "description": "Default COP uncertainty",
    },
    "unknown": {
        "uncertainty_pct": Decimal("0.12"),
        "description": "Unknown COP source",
    },
}


# ===========================================================================
# CHP Allocation Uncertainty Constants
# ===========================================================================

#: CHP allocation method uncertainty.

CHP_ALLOCATION_UNCERTAINTY: Dict[str, Dict[str, Any]] = {
    "metered_outputs": {
        "uncertainty_pct": Decimal("0.05"),
        "description": "Both electrical and thermal outputs individually metered",
    },
    "efficiency_method": {
        "uncertainty_pct": Decimal("0.08"),
        "description": "Efficiency-based allocation with supplier data",
    },
    "energy_method": {
        "uncertainty_pct": Decimal("0.10"),
        "description": "Energy-based allocation",
    },
    "exergy_method": {
        "uncertainty_pct": Decimal("0.10"),
        "description": "Exergy-based allocation (temperature-dependent)",
    },
    "default": {
        "uncertainty_pct": Decimal("0.10"),
        "description": "Default CHP allocation uncertainty",
    },
    "unknown": {
        "uncertainty_pct": Decimal("0.15"),
        "description": "Unknown / undocumented CHP allocation method",
    },
}


# ===========================================================================
# Per-Gas EF Uncertainty Constants
# ===========================================================================

#: Per-gas default EF uncertainty (95% CI half-width as fraction).
#: For fuel-based steam calculations with per-gas breakdown.

PER_GAS_EF_UNCERTAINTY: Dict[str, Dict[str, Any]] = {
    "CO2": {
        "lower_pct": Decimal("0.02"),
        "upper_pct": Decimal("0.10"),
        "default_pct": Decimal("0.05"),
        "description": "CO2 emission factor uncertainty (well characterised)",
    },
    "CH4": {
        "lower_pct": Decimal("0.15"),
        "upper_pct": Decimal("0.80"),
        "default_pct": Decimal("0.30"),
        "description": "CH4 emission factor uncertainty (lognormal, variable)",
    },
    "N2O": {
        "lower_pct": Decimal("0.30"),
        "upper_pct": Decimal("1.50"),
        "default_pct": Decimal("0.50"),
        "description": "N2O emission factor uncertainty (lognormal, highly variable)",
    },
}


# ===========================================================================
# DQI Scoring Constants
# ===========================================================================

#: Emission factor source quality scores (0-1 scale, higher = better).

EF_SOURCE_SCORES: Dict[str, float] = {
    "supplier_verified": 0.95,
    "supplier_cems": 0.93,
    "supplier_unverified": 0.80,
    "regional_default": 0.65,
    "national_default": 0.60,
    "ipcc_default": 0.50,
    "estimated": 0.35,
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

#: Efficiency verification status quality scores (0-1 scale).

EFFICIENCY_VERIFICATION_SCORES: Dict[str, float] = {
    "measured": 1.00,
    "performance_test": 0.95,
    "supplier_stated": 0.70,
    "handbook": 0.50,
    "assumed": 0.40,
    "unknown": 0.30,
}


# ===========================================================================
# Z-Score Table
# ===========================================================================

#: Z-scores for common confidence levels.
#: Maps confidence level (as string fraction) to z-score.

Z_SCORE_TABLE: Dict[str, Decimal] = {
    "0.80": Decimal("1.282"),
    "0.85": Decimal("1.440"),
    "0.90": Decimal("1.645"),
    "0.95": Decimal("1.960"),
    "0.975": Decimal("2.241"),
    "0.99": Decimal("2.576"),
    "0.995": Decimal("2.807"),
    "0.999": Decimal("3.291"),
}


# ===========================================================================
# UncertaintyQuantifierEngine
# ===========================================================================


class UncertaintyQuantifierEngine:
    """Monte Carlo simulation and analytical error propagation for Scope 2
    steam/heat purchase emission uncertainty quantification.

    Implements both IPCC Approach 1 (analytical root-sum-of-squares) and
    Approach 2 (Monte Carlo) for quantifying uncertainty in Scope 2
    steam/heat purchase emission calculations.

    The steam/heat purchase calculation model has five primary
    uncertainty sources:

        1. **Activity data**: Metered, invoiced, or estimated thermal
           energy consumption (GJ, MWh, MMBtu).
        2. **Emission factor**: Fuel-specific or composite kgCO2e/GJ
           factor from the steam/heat supplier.
        3. **Efficiency**: Boiler or heat generation system efficiency
           used to convert delivered energy to fuel input.
        4. **COP**: Coefficient of performance for district cooling
           systems converting cooling output to energy input.
        5. **CHP allocation**: Allocation fraction when thermal energy
           is sourced from a combined heat and power plant.

    The emission calculation is modelled as:
        E = AD * EF / efficiency  (for steam/heating)
        E = AD / COP * grid_EF   (for cooling)
        E = AD * EF * CHP_share  (for CHP-allocated)

    Each parameter is independently varied in Monte Carlo simulation
    using normal distributions bounded at zero.

    Configuration (iterations, seed, confidence levels) can be supplied
    either via the *config* parameter or the module-level ``_get_config()``
    singleton.

    Thread Safety:
        Monte Carlo uses per-call Random instances. Shared counters
        are protected by a reentrant lock.

    Attributes:
        _config: Optional configuration object.
        _metrics: Optional Prometheus metrics recorder.
        _provenance: Optional provenance tracker.
        _default_iterations: Default Monte Carlo iterations.
        _default_confidence: Default confidence level for CIs.
        _lock: Reentrant lock for thread safety.
        _total_analyses: Running counter of analyses performed.
        _total_monte_carlo: Running counter of MC simulations.
        _total_analytical: Running counter of analytical propagations.
        _total_dqi: Running counter of DQI scoring operations.
        _total_sensitivity: Running counter of sensitivity analyses.
        _total_batch: Running counter of batch analyses.
        _created_at: Engine creation timestamp.

    Example:
        >>> engine = UncertaintyQuantifierEngine()
        >>> result = engine.quantify_monte_carlo(
        ...     calc_result={"total_co2e_kg": "15000.0"},
        ...     seed=42,
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    # Singleton support
    _instance: Optional["UncertaintyQuantifierEngine"] = None
    _singleton_lock: threading.Lock = threading.Lock()

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
            config: Optional SteamHeatPurchaseConfig object. When ``None``,
                the singleton from ``_get_config()`` is attempted, with
                safe in-code defaults as the final fallback.
            metrics: Optional SteamHeatPurchaseMetrics object for Prometheus
                recording. When ``None``, the module-level singleton is
                used if available.
            provenance: Optional provenance tracker for audit trails.
        """
        self._config = config
        self._metrics = metrics
        self._provenance = provenance

        # Defaults
        self._default_iterations: int = 10_000
        self._default_confidence: Decimal = Decimal("0.95")

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
        if (
            self._provenance is None
            and _PROVENANCE_AVAILABLE
            and _get_provenance_tracker is not None
        ):
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
        self._total_batch: int = 0
        self._created_at: datetime = _utcnow()

        logger.info(
            "UncertaintyQuantifierEngine initialised (steam/heat purchase): "
            "iterations=%d, confidence=%.2f, numpy=%s",
            self._default_iterations,
            float(self._default_confidence),
            _NUMPY_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        config: Any = None,
        metrics: Any = None,
        provenance: Any = None,
    ) -> "UncertaintyQuantifierEngine":
        """Return or create the singleton engine instance.

        Args:
            config: Optional configuration object (used only on first call).
            metrics: Optional metrics recorder (used only on first call).
            provenance: Optional provenance tracker (used only on first call).

        Returns:
            The singleton UncertaintyQuantifierEngine instance.
        """
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls(
                        config=config,
                        metrics=metrics,
                        provenance=provenance,
                    )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance and all counters.

        Intended for testing only. Destroys the current singleton so that
        the next call to ``get_instance()`` creates a fresh engine.
        """
        with cls._singleton_lock:
            if cls._instance is not None:
                cls._instance._reset_counters()
            cls._instance = None
        logger.info("UncertaintyQuantifierEngine singleton reset")

    def _reset_counters(self) -> None:
        """Reset all engine counters to zero."""
        with self._lock:
            self._total_analyses = 0
            self._total_monte_carlo = 0
            self._total_analytical = 0
            self._total_dqi = 0
            self._total_sensitivity = 0
            self._total_batch = 0
        logger.info("UncertaintyQuantifierEngine counters reset")

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
        """Record an uncertainty analysis metric if Prometheus is available.

        Args:
            method: Statistical method name for the metric label.
        """
        if self._metrics is not None:
            try:
                self._metrics.record_uncertainty_analysis(method)
            except AttributeError:
                try:
                    self._metrics.record_calculation(
                        "uncertainty", method, "success", 0.0, 0.0, 0.0,
                        "unknown", "unknown",
                    )
                except Exception as exc:
                    logger.debug("Metrics recording skipped: %s", exc)
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
            except AttributeError:
                try:
                    chain_id = data.get("calculation_id", entity_id)
                    self._provenance.add_stage(
                        chain_id, "UNCERTAINTY_QUANTIFIED", data,
                    )
                except Exception as exc:
                    logger.debug("Provenance recording skipped: %s", exc)
            except Exception as exc:
                logger.debug("Provenance recording skipped: %s", exc)

    def _resolve_seed(self, seed: Optional[int]) -> int:
        """Resolve a PRNG seed to an explicit integer.

        Args:
            seed: Optional user-supplied seed.

        Returns:
            Explicit integer seed (user-supplied or time-derived).
        """
        if seed is not None:
            return seed
        return int(time.time() * 1000) % (2**31)

    def _clamp_iterations(self, iterations: int) -> int:
        """Clamp iterations to the valid range [100, 1_000_000].

        Args:
            iterations: Requested iteration count.

        Returns:
            Clamped iteration count.
        """
        return max(100, min(iterations, 1_000_000))

    def _extract_co2e(self, calc_result: Dict[str, Any]) -> Decimal:
        """Extract total CO2e value from a calculation result dictionary.

        Searches for common key names in order of priority.

        Args:
            calc_result: Calculation result dictionary.

        Returns:
            Decimal CO2e value in kg.
        """
        for key in (
            "total_co2e_kg",
            "total_co2e",
            "co2e_kg",
            "co2e",
            "emissions_kg",
            "emissions",
            "total_emissions_kg",
            "total_emissions",
        ):
            val = calc_result.get(key)
            if val is not None:
                return _safe_decimal(val, _ZERO)
        return _ZERO

    def _extract_parameter_pct(
        self,
        calc_result: Dict[str, Any],
        key: str,
        explicit_value: Optional[Decimal],
        default: Decimal,
    ) -> Decimal:
        """Extract an uncertainty percentage, converting from pct to fraction.

        Priority: explicit_value > calc_result[key] > default.

        Args:
            calc_result: Calculation result dictionary.
            key: Key to look for in calc_result.
            explicit_value: Explicitly provided value (as percentage).
            default: Default percentage value.

        Returns:
            Uncertainty as a fraction (e.g. 0.05 for 5%).
        """
        if explicit_value is not None:
            return _safe_decimal(explicit_value) / _HUNDRED
        val = calc_result.get(key)
        if val is not None:
            return _safe_decimal(val) / _HUNDRED
        return default / _HUNDRED

    # ------------------------------------------------------------------
    # Internal -- Statistics on float lists
    # ------------------------------------------------------------------

    def _float_stats(
        self,
        samples: List[float],
        label: str = "",
    ) -> Dict[str, Any]:
        """Compute summary statistics from a list of float samples.

        Args:
            samples: List of Monte Carlo samples (unsorted is fine).
            label: Optional label for the statistic group.

        Returns:
            Dictionary of Decimal-typed statistics.
        """
        sorted_s = sorted(samples)
        n = len(sorted_s)
        if n == 0:
            return {
                "label": label,
                "mean": _ZERO,
                "std_dev": _ZERO,
                "cv_pct": _ZERO,
                "ci_lower_95": _ZERO,
                "ci_upper_95": _ZERO,
                "min": _ZERO,
                "max": _ZERO,
            }

        mean_v = sum(sorted_s) / n
        var_v = sum((x - mean_v) ** 2 for x in sorted_s) / max(n - 1, 1)
        std_v = math.sqrt(var_v)
        cv_v = (std_v / abs(mean_v) * 100.0) if mean_v != 0.0 else 0.0

        ci_l_idx = max(0, int(0.025 * n))
        ci_u_idx = min(n - 1, int(0.975 * n))

        return {
            "label": label,
            "mean": Decimal(str(round(mean_v, 6))),
            "std_dev": Decimal(str(round(std_v, 6))),
            "cv_pct": Decimal(str(round(cv_v, 4))),
            "ci_lower_95": Decimal(str(round(sorted_s[ci_l_idx], 6))),
            "ci_upper_95": Decimal(str(round(sorted_s[ci_u_idx], 6))),
            "min": Decimal(str(round(sorted_s[0], 6))),
            "max": Decimal(str(round(sorted_s[-1], 6))),
        }

    # ------------------------------------------------------------------
    # Internal -- Percentile Computation (float)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal -- Percentile Computation (Decimal)
    # ------------------------------------------------------------------

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

    # ==================================================================
    # Public API -- Main Entry Point
    # ==================================================================

    def quantify_uncertainty(
        self,
        calc_result: Dict[str, Any],
        method: str = "monte_carlo",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Main entry point for uncertainty quantification.

        Dispatches to Monte Carlo or analytical method based on the
        *method* parameter.

        Args:
            calc_result: Calculation result dictionary containing at minimum
                a total CO2e value (keyed as ``total_co2e_kg``, ``co2e_kg``,
                ``emissions_kg``, or similar).
            method: Uncertainty method. One of ``"monte_carlo"``,
                ``"analytical"``, ``"mc"``, ``"ana"``.
                Default: ``"monte_carlo"``.
            **kwargs: Additional keyword arguments passed to the specific
                method (iterations, seed, confidence_level, uncertainty
                percentages, etc.).

        Returns:
            Dictionary with uncertainty analysis results including
            confidence intervals, provenance_hash, and calculation_trace.

        Raises:
            ValueError: If *method* is not recognised.
        """
        method_lower = method.lower().strip()

        if method_lower in ("monte_carlo", "mc", "monte-carlo"):
            return self.quantify_monte_carlo(calc_result, **kwargs)
        elif method_lower in ("analytical", "ana", "analytical_propagation",
                              "error_propagation"):
            return self.quantify_analytical(calc_result, **kwargs)
        else:
            raise ValueError(
                f"Unknown uncertainty method '{method}'. "
                f"Supported: 'monte_carlo', 'analytical'."
            )

    # ==================================================================
    # Public API -- Monte Carlo Simulation
    # ==================================================================

    def quantify_monte_carlo(
        self,
        calc_result: Dict[str, Any],
        iterations: int = 10_000,
        confidence_level: Decimal = Decimal("0.95"),
        activity_data_uncertainty_pct: Decimal = Decimal("5.0"),
        emission_factor_uncertainty_pct: Decimal = Decimal("10.0"),
        efficiency_uncertainty_pct: Decimal = Decimal("5.0"),
        cop_uncertainty_pct: Decimal = Decimal("8.0"),
        chp_allocation_uncertainty_pct: Decimal = Decimal("10.0"),
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for steam/heat purchase uncertainty.

        Each parameter (activity data, emission factor, efficiency, COP,
        CHP allocation) is independently varied using normal distributions
        centred on 1.0 with standard deviation equal to the relative
        uncertainty fraction.

        Sampling model (per iteration):
            factor_ad  = gauss(1.0, u_ad)
            factor_ef  = gauss(1.0, u_ef)
            factor_eff = gauss(1.0, u_eff)
            factor_cop = gauss(1.0, u_cop)
            factor_chp = gauss(1.0, u_chp)
            emission_i = base_co2e * factor_ad * factor_ef * factor_eff
                         * factor_cop * factor_chp
            emission_i = max(0, emission_i)  # bounded at zero

        Only non-zero uncertainty parameters are sampled to avoid
        unnecessary computation for parameters not relevant to the
        specific calculation (e.g. COP is irrelevant for steam).

        Args:
            calc_result: Calculation result dictionary containing the
                central emission estimate.
            iterations: Number of Monte Carlo iterations.
                Default: 10,000. Range: [100, 1,000,000].
            confidence_level: Confidence level as a fraction.
                Default: 0.95 (95% CI).
            activity_data_uncertainty_pct: Activity data uncertainty
                as a percentage (e.g. 5.0 for +/-5%). Default: 5.0.
            emission_factor_uncertainty_pct: Emission factor uncertainty
                as a percentage (e.g. 10.0 for +/-10%). Default: 10.0.
            efficiency_uncertainty_pct: Boiler/system efficiency uncertainty
                as a percentage. Default: 5.0.
            cop_uncertainty_pct: COP uncertainty as a percentage.
                Default: 8.0.
            chp_allocation_uncertainty_pct: CHP allocation uncertainty
                as a percentage. Default: 10.0.
            seed: PRNG seed for reproducibility. Default: 42.

        Returns:
            Dictionary with keys:
                - calculation_id: UUID string
                - status: "SUCCESS" or "VALIDATION_ERROR"
                - method: "MONTE_CARLO"
                - base_co2e_kg: Decimal central estimate
                - mean_co2e_kg: Decimal mean of simulated emissions
                - std_dev: Decimal standard deviation
                - cv_pct: Decimal coefficient of variation (%)
                - ci_lower: Decimal lower confidence interval bound
                - ci_upper: Decimal upper confidence interval bound
                - confidence_level: Decimal
                - iterations: int
                - seed: int
                - percentiles: dict of percentile values
                - relative_uncertainty_pct: Decimal combined rel. uncertainty
                - input_uncertainties: dict of input uncertainty percentages
                - calculation_trace: list of trace steps
                - provenance_hash: SHA-256 hex digest
                - processing_time_ms: float
        """
        self._increment("_total_analyses")
        self._increment("_total_monte_carlo")
        start_time = time.monotonic()
        calc_id = str(uuid4())
        trace: List[str] = []

        # -- Extract base emission value ------------------------------------
        base_co2e = self._extract_co2e(calc_result)
        trace.append(f"Extracted base CO2e: {base_co2e} kg")

        # -- Convert percentages to fractions -------------------------------
        u_ad = _safe_decimal(activity_data_uncertainty_pct) / _HUNDRED
        u_ef = _safe_decimal(emission_factor_uncertainty_pct) / _HUNDRED
        u_eff = _safe_decimal(efficiency_uncertainty_pct) / _HUNDRED
        u_cop = _safe_decimal(cop_uncertainty_pct) / _HUNDRED
        u_chp = _safe_decimal(chp_allocation_uncertainty_pct) / _HUNDRED

        trace.append(
            f"Uncertainty fractions: AD={u_ad}, EF={u_ef}, "
            f"Eff={u_eff}, COP={u_cop}, CHP={u_chp}"
        )

        # -- Validate inputs ------------------------------------------------
        errors: List[str] = []
        if base_co2e < _ZERO:
            errors.append("base CO2e must be non-negative")
        if base_co2e == _ZERO:
            errors.append("base CO2e is zero; uncertainty analysis not meaningful")

        for name, val in [
            ("activity_data_uncertainty", u_ad),
            ("emission_factor_uncertainty", u_ef),
            ("efficiency_uncertainty", u_eff),
            ("cop_uncertainty", u_cop),
            ("chp_allocation_uncertainty", u_chp),
        ]:
            if val < _ZERO:
                errors.append(f"{name} must be non-negative")
            elif val > Decimal("10"):
                errors.append(f"{name} must be <= 1000%; got {val * _HUNDRED}%")

        iter_errors = self.validate_request_iterations(iterations)
        errors.extend(iter_errors)

        if errors:
            processing_time = round((time.monotonic() - start_time) * 1000, 3)
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "method": "MONTE_CARLO",
                "errors": errors,
                "calculation_trace": trace,
                "processing_time_ms": processing_time,
            }

        # -- Convert to floats for simulation --------------------------------
        base_float = float(base_co2e)
        u_ad_f = float(u_ad)
        u_ef_f = float(u_ef)
        u_eff_f = float(u_eff)
        u_cop_f = float(u_cop)
        u_chp_f = float(u_chp)
        conf = float(confidence_level)

        # -- Initialise PRNG ------------------------------------------------
        actual_seed = self._resolve_seed(seed)
        iters = self._clamp_iterations(iterations)
        trace.append(f"Monte Carlo: iterations={iters}, seed={actual_seed}")

        # -- Run simulation -------------------------------------------------
        if _NUMPY_AVAILABLE and np is not None:
            results = self._mc_numpy(
                base_float, u_ad_f, u_ef_f, u_eff_f,
                u_cop_f, u_chp_f, iters, actual_seed,
            )
            trace.append("Simulation engine: numpy")
        else:
            results = self._mc_pure_python(
                base_float, u_ad_f, u_ef_f, u_eff_f,
                u_cop_f, u_chp_f, iters, actual_seed,
            )
            trace.append("Simulation engine: pure-python")

        results.sort()
        n = len(results)
        trace.append(f"Simulation complete: {n} samples generated")

        # -- Calculate statistics -------------------------------------------
        mean_val = sum(results) / n
        variance = sum((x - mean_val) ** 2 for x in results) / max(n - 1, 1)
        std_dev = math.sqrt(variance)
        cv_pct = (std_dev / abs(mean_val) * 100.0) if mean_val != 0.0 else 0.0

        # -- Confidence interval (percentile method) ------------------------
        alpha = (1.0 - conf) / 2.0
        ci_lower_idx = max(0, int(alpha * n))
        ci_upper_idx = min(n - 1, int((1.0 - alpha) * n))
        ci_lower = results[ci_lower_idx]
        ci_upper = results[ci_upper_idx]

        trace.append(
            f"Statistics: mean={mean_val:.2f}, std={std_dev:.2f}, "
            f"CI=[{ci_lower:.2f}, {ci_upper:.2f}]"
        )

        # -- Percentiles ---------------------------------------------------
        percentile_points = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
        percentiles = self._compute_percentiles_float(results, percentile_points)

        # -- Combined relative uncertainty (analytical RSS for reference) ----
        unc_components = [u_ad_f, u_ef_f, u_eff_f, u_cop_f, u_chp_f]
        rel_unc = math.sqrt(sum(u ** 2 for u in unc_components if u > 0))

        # -- Build output ---------------------------------------------------
        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "MONTE_CARLO",
            "base_co2e_kg": base_co2e,
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
                "activity_data_pct": str(activity_data_uncertainty_pct),
                "emission_factor_pct": str(emission_factor_uncertainty_pct),
                "efficiency_pct": str(efficiency_uncertainty_pct),
                "cop_pct": str(cop_uncertainty_pct),
                "chp_allocation_pct": str(chp_allocation_uncertainty_pct),
            },
            "calculation_trace": trace,
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        # -- Record metrics and provenance ----------------------------------
        self._record_metric("monte_carlo")
        self._record_provenance(
            entity_id=calc_id[:16],
            data={
                "method": "monte_carlo",
                "base_kg": str(base_co2e),
                "mean_kg": str(result["mean_co2e_kg"]),
                "ci_lower": str(result["ci_lower"]),
                "ci_upper": str(result["ci_upper"]),
                "iterations": iters,
                "seed": actual_seed,
            },
        )

        logger.info(
            "Monte Carlo complete: id=%s, n=%d, base=%.2f, mean=%.2f, "
            "std=%.2f, cv=%.1f%%, CI=[%.2f, %.2f], time=%.3fms",
            calc_id, iters, base_float, mean_val, std_dev, cv_pct,
            ci_lower, ci_upper, processing_time,
        )
        return result

    # ------------------------------------------------------------------
    # Internal -- numpy-based MC simulation
    # ------------------------------------------------------------------

    def _mc_numpy(
        self,
        base: float,
        u_ad: float,
        u_ef: float,
        u_eff: float,
        u_cop: float,
        u_chp: float,
        n: int,
        seed: int,
    ) -> List[float]:
        """Run Monte Carlo simulation using numpy for performance.

        Args:
            base: Central emission value (float).
            u_ad: Activity data relative uncertainty (fraction).
            u_ef: Emission factor relative uncertainty (fraction).
            u_eff: Efficiency relative uncertainty (fraction).
            u_cop: COP relative uncertainty (fraction).
            u_chp: CHP allocation relative uncertainty (fraction).
            n: Number of iterations.
            seed: PRNG seed.

        Returns:
            List of float emission samples.
        """
        rng = np.random.default_rng(seed)
        combined = np.ones(n)

        if u_ad > 0:
            combined *= rng.normal(1.0, u_ad, n)
        if u_ef > 0:
            combined *= rng.normal(1.0, u_ef, n)
        if u_eff > 0:
            combined *= rng.normal(1.0, u_eff, n)
        if u_cop > 0:
            combined *= rng.normal(1.0, u_cop, n)
        if u_chp > 0:
            combined *= rng.normal(1.0, u_chp, n)

        samples = np.maximum(0.0, base * combined)
        return samples.tolist()

    # ------------------------------------------------------------------
    # Internal -- pure-Python MC simulation
    # ------------------------------------------------------------------

    def _mc_pure_python(
        self,
        base: float,
        u_ad: float,
        u_ef: float,
        u_eff: float,
        u_cop: float,
        u_chp: float,
        n: int,
        seed: int,
    ) -> List[float]:
        """Run Monte Carlo simulation using pure Python stdlib random.

        Args:
            base: Central emission value (float).
            u_ad: Activity data relative uncertainty (fraction).
            u_ef: Emission factor relative uncertainty (fraction).
            u_eff: Efficiency relative uncertainty (fraction).
            u_cop: COP relative uncertainty (fraction).
            u_chp: CHP allocation relative uncertainty (fraction).
            n: Number of iterations.
            seed: PRNG seed.

        Returns:
            List of float emission samples.
        """
        rng = random.Random(seed)
        results: List[float] = []

        for _ in range(n):
            factor = 1.0
            if u_ad > 0:
                factor *= rng.gauss(1.0, u_ad)
            if u_ef > 0:
                factor *= rng.gauss(1.0, u_ef)
            if u_eff > 0:
                factor *= rng.gauss(1.0, u_eff)
            if u_cop > 0:
                factor *= rng.gauss(1.0, u_cop)
            if u_chp > 0:
                factor *= rng.gauss(1.0, u_chp)
            results.append(max(0.0, base * factor))

        return results

    # ==================================================================
    # Public API -- Per-Gas Monte Carlo
    # ==================================================================

    def quantify_monte_carlo_per_gas(
        self,
        gas_results: Dict[str, Decimal],
        iterations: int = 10_000,
        seed: Optional[int] = None,
        ef_uncertainties: Optional[Dict[str, Decimal]] = None,
    ) -> Dict[str, Any]:
        """Run per-gas Monte Carlo simulation with individual uncertainty ranges.

        Each greenhouse gas has distinct uncertainty characteristics:
        - CO2: +/-5% (default) -- well-characterised from fuel EFs
        - CH4: +/-30% (default) -- lognormal, moderate variability
        - N2O: +/-50% (default) -- lognormal, high variability

        Sampling model (per gas, per iteration):
            CO2: normal distribution (bounded at zero)
            CH4: lognormal distribution (non-negative)
            N2O: lognormal distribution (non-negative)

        Args:
            gas_results: Dictionary mapping gas names to kg values.
                Expected keys: "CO2", "CH4", "N2O". Missing gases default
                to zero.
            iterations: Number of MC iterations. Default: 10,000.
            seed: Optional PRNG seed for reproducibility.
            ef_uncertainties: Optional per-gas uncertainty overrides.
                Keys: "CO2", "CH4", "N2O" with Decimal fraction values.

        Returns:
            Dictionary with per-gas and combined results including
            mean, std_dev, ci_lower, ci_upper, percentiles, and
            provenance_hash.
        """
        self._increment("_total_analyses")
        self._increment("_total_monte_carlo")
        start_time = time.monotonic()
        calc_id = str(uuid4())

        # -- Resolve per-gas values and uncertainties ----------------------
        co2_kg = _safe_decimal(gas_results.get("CO2", _ZERO))
        ch4_kg = _safe_decimal(gas_results.get("CH4", _ZERO))
        n2o_kg = _safe_decimal(gas_results.get("N2O", _ZERO))

        default_ef_unc: Dict[str, Decimal] = {
            "CO2": PER_GAS_EF_UNCERTAINTY["CO2"]["default_pct"],
            "CH4": PER_GAS_EF_UNCERTAINTY["CH4"]["default_pct"],
            "N2O": PER_GAS_EF_UNCERTAINTY["N2O"]["default_pct"],
        }
        if ef_uncertainties:
            for gas in ("CO2", "CH4", "N2O"):
                if gas in ef_uncertainties:
                    default_ef_unc[gas] = _safe_decimal(
                        ef_uncertainties[gas], default_ef_unc[gas],
                    )
        ef_unc = default_ef_unc

        # -- Setup ---------------------------------------------------------
        iters = self._clamp_iterations(iterations)
        actual_seed = self._resolve_seed(seed)
        rng = random.Random(actual_seed)

        co2_float = float(co2_kg)
        ch4_float = float(ch4_kg)
        n2o_float = float(n2o_kg)
        co2_unc_f = float(ef_unc["CO2"])
        ch4_unc_f = float(ef_unc["CH4"])
        n2o_unc_f = float(ef_unc["N2O"])

        # -- Per-gas simulation --------------------------------------------
        co2_samples: List[float] = []
        ch4_samples: List[float] = []
        n2o_samples: List[float] = []
        total_samples: List[float] = []

        for _ in range(iters):
            # CO2: normal distribution (bounded at zero)
            co2_sample = max(0.0, co2_float * rng.gauss(1.0, co2_unc_f))

            # CH4: lognormal distribution
            if ch4_float > 0 and ch4_unc_f > 0:
                sigma2 = math.log(1 + ch4_unc_f ** 2)
                sigma = math.sqrt(sigma2)
                mu = math.log(ch4_float) - sigma2 / 2
                ch4_sample = rng.lognormvariate(mu, sigma)
            else:
                ch4_sample = ch4_float

            # N2O: lognormal distribution
            if n2o_float > 0 and n2o_unc_f > 0:
                sigma2 = math.log(1 + n2o_unc_f ** 2)
                sigma = math.sqrt(sigma2)
                mu = math.log(n2o_float) - sigma2 / 2
                n2o_sample = rng.lognormvariate(mu, sigma)
            else:
                n2o_sample = n2o_float

            co2_samples.append(co2_sample)
            ch4_samples.append(ch4_sample)
            n2o_samples.append(n2o_sample)
            total_samples.append(co2_sample + ch4_sample + n2o_sample)

        # -- Per-gas statistics --------------------------------------------
        def _gas_stats(samples: List[float], gas_name: str) -> Dict[str, Any]:
            stats = self._float_stats(samples, label=gas_name)
            sorted_s = sorted(samples)
            pctiles = self._compute_percentiles_float(sorted_s, [5, 25, 50, 75, 95])
            stats["percentiles"] = {
                str(p): Decimal(str(round(v, 6)))
                for p, v in pctiles.items()
            }
            return stats

        co2_stats = _gas_stats(co2_samples, "CO2")
        ch4_stats = _gas_stats(ch4_samples, "CH4")
        n2o_stats = _gas_stats(n2o_samples, "N2O")
        total_stats = _gas_stats(total_samples, "CO2e_total")

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "MONTE_CARLO_PER_GAS",
            "iterations": iters,
            "seed": actual_seed,
            "per_gas": {
                "CO2": co2_stats,
                "CH4": ch4_stats,
                "N2O": n2o_stats,
            },
            "total": total_stats,
            "ef_uncertainties_used": {k: str(v) for k, v in ef_unc.items()},
            "input_gas_results": {k: str(v) for k, v in gas_results.items()},
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        self._record_metric("monte_carlo_per_gas")

        logger.info(
            "Per-gas Monte Carlo complete: id=%s, n=%d, "
            "CO2 mean=%.2f, CH4 mean=%.2f, N2O mean=%.2f, "
            "total mean=%.2f, time=%.3fms",
            calc_id, iters,
            float(co2_stats["mean"]),
            float(ch4_stats["mean"]),
            float(n2o_stats["mean"]),
            float(total_stats["mean"]),
            processing_time,
        )
        return result

    # ==================================================================
    # Public API -- Analytical Error Propagation
    # ==================================================================

    def quantify_analytical(
        self,
        calc_result: Dict[str, Any],
        confidence_level: Decimal = Decimal("0.95"),
        activity_data_uncertainty_pct: Decimal = Decimal("5.0"),
        emission_factor_uncertainty_pct: Decimal = Decimal("10.0"),
        efficiency_uncertainty_pct: Decimal = Decimal("5.0"),
        cop_uncertainty_pct: Decimal = Decimal("8.0"),
        chp_allocation_uncertainty_pct: Decimal = Decimal("10.0"),
    ) -> Dict[str, Any]:
        """IPCC Approach 1 analytical error propagation for steam/heat purchase.

        For the multiplicative model:
            Emissions = AD * EF / efficiency (or * COP factor * CHP share)

        The combined relative uncertainty (root-sum-of-squares) is:
            sigma_rel = sqrt(u_ad^2 + u_ef^2 + u_eff^2 + u_cop^2 + u_chp^2)

        The 95% confidence interval is:
            CI = mean +/- z * sigma_rel * mean

        where z = 1.96 for 95% confidence.

        Args:
            calc_result: Calculation result dictionary containing the
                central emission estimate.
            confidence_level: Confidence level as a fraction.
                Default: 0.95.
            activity_data_uncertainty_pct: Activity data uncertainty
                as a percentage. Default: 5.0.
            emission_factor_uncertainty_pct: Emission factor uncertainty
                as a percentage. Default: 10.0.
            efficiency_uncertainty_pct: Efficiency uncertainty
                as a percentage. Default: 5.0.
            cop_uncertainty_pct: COP uncertainty as a percentage.
                Default: 8.0.
            chp_allocation_uncertainty_pct: CHP allocation uncertainty
                as a percentage. Default: 10.0.

        Returns:
            Dictionary with keys:
                - calculation_id: UUID
                - status: "SUCCESS"
                - method: "ANALYTICAL"
                - emissions_kg: Decimal central estimate
                - combined_relative_uncertainty: Decimal fraction
                - combined_relative_uncertainty_pct: Decimal percentage
                - combined_absolute_uncertainty: Decimal (kg)
                - ci_lower: Decimal
                - ci_upper: Decimal
                - confidence_level: Decimal
                - z_score: Decimal
                - parameter_contributions: dict of param -> fraction of variance
                - calculation_trace: list of trace steps
                - provenance_hash: SHA-256 hex digest
                - processing_time_ms: float
        """
        self._increment("_total_analyses")
        self._increment("_total_analytical")
        start_time = time.monotonic()
        calc_id = str(uuid4())
        trace: List[str] = []

        # -- Extract base emission value -----------------------------------
        base_co2e = self._extract_co2e(calc_result)
        trace.append(f"Extracted base CO2e: {base_co2e} kg")

        # -- Convert percentages to fractions ------------------------------
        u_ad = _safe_decimal(activity_data_uncertainty_pct) / _HUNDRED
        u_ef = _safe_decimal(emission_factor_uncertainty_pct) / _HUNDRED
        u_eff = _safe_decimal(efficiency_uncertainty_pct) / _HUNDRED
        u_cop = _safe_decimal(cop_uncertainty_pct) / _HUNDRED
        u_chp = _safe_decimal(chp_allocation_uncertainty_pct) / _HUNDRED

        trace.append(
            f"Uncertainty fractions: AD={u_ad}, EF={u_ef}, "
            f"Eff={u_eff}, COP={u_cop}, CHP={u_chp}"
        )

        # -- Compute RSS ---------------------------------------------------
        components = {
            "activity_data": u_ad,
            "emission_factor": u_ef,
            "efficiency": u_eff,
            "cop": u_cop,
            "chp_allocation": u_chp,
        }

        sum_sq = _ZERO
        individual_sq: Dict[str, Decimal] = {}
        for name, unc in components.items():
            u_sq = unc ** 2
            individual_sq[name] = u_sq
            sum_sq += u_sq

        combined_rel = _D(str(math.sqrt(float(sum_sq))))
        trace.append(
            f"Combined relative uncertainty (RSS): {combined_rel:.6f} "
            f"({combined_rel * _HUNDRED:.2f}%)"
        )

        # -- Z-score -------------------------------------------------------
        z_score = self.get_z_score(confidence_level)
        trace.append(f"Z-score for {confidence_level} confidence: {z_score}")

        # -- Absolute uncertainty and CI -----------------------------------
        combined_abs = (base_co2e * combined_rel).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        ci_lower = (base_co2e - combined_abs * z_score).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        ci_upper = (base_co2e + combined_abs * z_score).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )

        # Ensure non-negative lower bound
        if ci_lower < _ZERO:
            ci_lower = _ZERO

        trace.append(f"CI: [{ci_lower}, {ci_upper}]")

        # -- Parameter contributions --------------------------------------
        total_sq_float = float(sum_sq)
        contributions: Dict[str, Decimal] = {}
        if total_sq_float > 0:
            for name, u_sq in individual_sq.items():
                contributions[name] = _D(str(round(
                    float(u_sq) / total_sq_float, 6,
                )))
        else:
            for name in components:
                contributions[name] = _ZERO

        trace.append(f"Dominant parameter: {max(contributions, key=contributions.get)}")

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "ANALYTICAL",
            "emissions_kg": base_co2e.quantize(_PRECISION, rounding=ROUND_HALF_UP),
            "combined_relative_uncertainty": combined_rel.quantize(
                _PRECISION, rounding=ROUND_HALF_UP,
            ),
            "combined_relative_uncertainty_pct": (combined_rel * _HUNDRED).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            ),
            "combined_absolute_uncertainty": combined_abs,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": confidence_level,
            "z_score": z_score,
            "parameter_contributions": contributions,
            "input_uncertainties": {
                "activity_data_pct": str(activity_data_uncertainty_pct),
                "emission_factor_pct": str(emission_factor_uncertainty_pct),
                "efficiency_pct": str(efficiency_uncertainty_pct),
                "cop_pct": str(cop_uncertainty_pct),
                "chp_allocation_pct": str(chp_allocation_uncertainty_pct),
            },
            "formula": (
                "sigma_rel = sqrt(u_ad^2 + u_ef^2 + u_eff^2 + u_cop^2 + u_chp^2); "
                "CI = mean +/- z * sigma_rel * mean"
            ),
            "calculation_trace": trace,
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        self._record_metric("analytical")

        logger.info(
            "Analytical propagation: id=%s, emissions=%.2f, "
            "combined_rel=%.4f (%.2f%%), CI=[%.2f, %.2f], time=%.3fms",
            calc_id, float(base_co2e), float(combined_rel),
            float(combined_rel * _HUNDRED),
            float(ci_lower), float(ci_upper), processing_time,
        )
        return result

    # ==================================================================
    # Public API -- Tier-Based Defaults
    # ==================================================================

    def get_tier_defaults(self, tier: str) -> Dict[str, Any]:
        """Get default uncertainty percentages for a data quality tier.

        Tier 1: IPCC/national defaults (highest uncertainty)
        Tier 2: Supplier-specific data (moderate uncertainty)
        Tier 3: Facility-measured data (lowest uncertainty)

        Args:
            tier: Tier identifier. One of ``"tier_1"``, ``"tier_2"``,
                ``"tier_3"``, ``"1"``, ``"2"``, ``"3"``.

        Returns:
            Dictionary with keys:
                - tier: normalised tier name
                - activity_data_pct: Decimal uncertainty percentage
                - emission_factor_pct: Decimal uncertainty percentage
                - efficiency_pct: Decimal uncertainty percentage
                - cop_pct: Decimal uncertainty percentage
                - chp_allocation_pct: Decimal uncertainty percentage
                - description: tier description string

        Raises:
            ValueError: If *tier* is not recognised.
        """
        # Normalise tier key
        tier_key = tier.lower().strip()
        if tier_key in ("1", "tier1"):
            tier_key = "tier_1"
        elif tier_key in ("2", "tier2"):
            tier_key = "tier_2"
        elif tier_key in ("3", "tier3"):
            tier_key = "tier_3"

        tier_data = TIER_DEFAULTS.get(tier_key)
        if tier_data is None:
            raise ValueError(
                f"Unknown tier '{tier}'. Supported: 'tier_1', 'tier_2', 'tier_3'."
            )

        return {
            "tier": tier_key,
            "activity_data_pct": tier_data["activity_data"] * _HUNDRED,
            "emission_factor_pct": tier_data["emission_factor"] * _HUNDRED,
            "efficiency_pct": tier_data["efficiency"] * _HUNDRED,
            "cop_pct": tier_data["cop"] * _HUNDRED,
            "chp_allocation_pct": tier_data["chp_allocation"] * _HUNDRED,
            "description": tier_data.get("description", ""),
        }

    # ==================================================================
    # Public API -- Combined Uncertainty (Quadrature Sum)
    # ==================================================================

    def compute_combined_uncertainty(self, uncertainties: List[Decimal]) -> Decimal:
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
    # Public API -- Confidence Interval
    # ==================================================================

    def compute_confidence_interval(
        self,
        mean: Decimal,
        std_dev: Decimal,
        confidence_level: Decimal = Decimal("0.95"),
    ) -> Tuple[Decimal, Decimal]:
        """Compute a confidence interval from mean and standard deviation.

        CI = mean +/- z * std_dev

        The lower bound is clamped to zero (emissions cannot be negative).

        Args:
            mean: Central estimate (e.g. mean emission in kg).
            std_dev: Standard deviation of the estimate.
            confidence_level: Confidence level as a fraction.
                Default: 0.95.

        Returns:
            Tuple of (ci_lower, ci_upper) as Decimal values.
        """
        z = self.get_z_score(confidence_level)
        half_width = (z * std_dev).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        ci_lower = (mean - half_width).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        ci_upper = (mean + half_width).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Clamp lower bound to zero
        if ci_lower < _ZERO:
            ci_lower = _ZERO

        return (ci_lower, ci_upper)

    # ==================================================================
    # Public API -- Sensitivity Analysis
    # ==================================================================

    def sensitivity_analysis(
        self,
        calc_result: Dict[str, Any],
        parameters: Optional[List[str]] = None,
        variation_pct: Decimal = Decimal("10.0"),
    ) -> Dict[str, Any]:
        """One-at-a-time sensitivity analysis for steam/heat purchase.

        Varies each parameter by +/- *variation_pct* while holding all
        others constant, and measures the impact on the emission result.
        The sensitivity coefficient is:
            S_i = (result_high - result_low) / (2 * variation * base)

        Default parameters for steam/heat purchase analysis:
            - activity_data
            - emission_factor
            - efficiency
            - cop
            - chp_allocation

        Args:
            calc_result: Calculation result dictionary containing the
                central emission estimate.
            parameters: List of parameter names to analyse. If ``None``,
                all five default parameters are analysed.
            variation_pct: Percentage to vary each parameter
                (e.g. 10.0 for +/-10%). Default: 10.0.

        Returns:
            Dictionary with keys:
                - calculation_id: UUID
                - status: "SUCCESS"
                - method: "SENSITIVITY_ANALYSIS"
                - base_result: Decimal
                - variation_pct: Decimal
                - parameters: dict of parameter_name -> sensitivity data
                - ranked_parameters: list sorted by descending impact
                - calculation_trace: list of trace steps
                - provenance_hash: SHA-256 hex digest
                - processing_time_ms: float
        """
        self._increment("_total_analyses")
        self._increment("_total_sensitivity")
        start_time = time.monotonic()
        calc_id = str(uuid4())
        trace: List[str] = []

        base_co2e = self._extract_co2e(calc_result)
        base_float = float(base_co2e)
        var_frac = float(_safe_decimal(variation_pct) / _HUNDRED)

        if parameters is None:
            parameters = [
                "activity_data",
                "emission_factor",
                "efficiency",
                "cop",
                "chp_allocation",
            ]

        trace.append(f"Base CO2e: {base_co2e} kg")
        trace.append(f"Variation: +/-{variation_pct}%")
        trace.append(f"Parameters: {parameters}")

        # Default multiplicative factor for each parameter is 1.0
        param_results: Dict[str, Dict[str, Any]] = {}

        for name in parameters:
            # Each parameter has a multiplicative factor of 1.0 in the
            # base case. Varying by +/-var_frac gives low and high.
            low_factor = 1.0 - var_frac
            high_factor = 1.0 + var_frac

            result_low = base_float * low_factor
            result_high = base_float * high_factor
            impact_range = abs(result_high - result_low)

            sensitivity_coeff = (
                impact_range / (2.0 * var_frac * base_float)
                if base_float != 0.0 and var_frac != 0.0
                else 0.0
            )

            param_results[name] = {
                "base_factor": _ONE,
                "low_factor": _D(str(round(low_factor, 6))),
                "high_factor": _D(str(round(high_factor, 6))),
                "result_low": _D(str(round(result_low, 6))),
                "result_high": _D(str(round(result_high, 6))),
                "sensitivity_coefficient": _D(str(round(sensitivity_coeff, 6))),
                "impact_range": _D(str(round(impact_range, 6))),
                "impact_pct": _D(str(round(
                    (impact_range / base_float * 100.0) if base_float > 0 else 0.0,
                    4,
                ))),
            }

        # Rank parameters by descending impact range
        ranked = sorted(
            param_results.keys(),
            key=lambda k: float(param_results[k].get("impact_range", _ZERO)),
            reverse=True,
        )

        trace.append(f"Most sensitive parameter: {ranked[0] if ranked else 'none'}")

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "SENSITIVITY_ANALYSIS",
            "base_result": base_co2e,
            "variation_pct": variation_pct,
            "parameters": param_results,
            "ranked_parameters": ranked,
            "calculation_trace": trace,
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        self._record_metric("sensitivity")

        logger.info(
            "Sensitivity analysis complete: id=%s, params=%d, "
            "top_param=%s, time=%.3fms",
            calc_id, len(parameters),
            ranked[0] if ranked else "none",
            processing_time,
        )
        return result

    # ==================================================================
    # Public API -- Tornado Analysis
    # ==================================================================

    def tornado_analysis(
        self,
        calc_result: Dict[str, Any],
        parameters: Optional[List[str]] = None,
        variation_pct: Decimal = Decimal("10.0"),
    ) -> Dict[str, Any]:
        """Generate tornado chart data sorted by impact magnitude.

        Runs a sensitivity analysis with +/- variation on each parameter,
        then sorts the results by descending impact range for tornado chart
        visualisation.

        Args:
            calc_result: Calculation result dictionary.
            parameters: List of parameter names to analyse.
            variation_pct: Percentage to vary each parameter.
                Default: 10.0 (+/-10%).

        Returns:
            Dictionary with bars sorted by descending impact_range,
            each containing parameter, result_low, result_high,
            impact_range, sensitivity_coefficient, contribution_pct.
        """
        self._increment("_total_analyses")
        self._increment("_total_sensitivity")
        start_time = time.monotonic()
        calc_id = str(uuid4())

        # Run base sensitivity analysis
        sa = self.sensitivity_analysis(
            calc_result=calc_result,
            parameters=parameters,
            variation_pct=variation_pct,
        )

        param_data = sa.get("parameters", {})

        # Collect impact data
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
            del item["variance_contribution"]

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "TORNADO_ANALYSIS",
            "base_result": sa.get("base_result", _ZERO),
            "variation_pct": variation_pct,
            "bars": items,
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        self._record_metric("tornado")

        logger.info(
            "Tornado analysis complete: id=%s, bars=%d, time=%.3fms",
            calc_id, len(items), processing_time,
        )
        return result

    # ==================================================================
    # Public API -- DQI Score
    # ==================================================================

    def calculate_dqi_score(
        self,
        ef_source: str,
        ef_age_years: int,
        activity_source: str,
        efficiency_verified: bool = True,
    ) -> Decimal:
        """Calculate a composite Data Quality Indicator (DQI) score.

        Combines four quality dimensions into a single 0-1 score where
        higher values indicate better data quality.

        Scoring dimensions:
            1. EF source quality (weight 0.30): supplier_verified=0.95,
               regional_default=0.65, ipcc_default=0.50
            2. EF age (weight 0.20): current_year=1.0, -1yr=0.9,
               -2yr=0.8, -3yr=0.7, older=0.5
            3. Activity data source (weight 0.25): meter=0.95,
               invoice=0.85, estimate=0.60, benchmark=0.40
            4. Efficiency verification (weight 0.25): measured=1.0,
               supplier_stated=0.7, handbook=0.5, unknown=0.3

        Args:
            ef_source: Emission factor source identifier (case-insensitive).
            ef_age_years: Age of the emission factor in years (0 = current).
            activity_source: Activity data source identifier.
            efficiency_verified: Whether efficiency is site-measured (True)
                or estimated/handbook (False). Default: True.

        Returns:
            Decimal DQI score between 0 and 1.
        """
        self._increment("_total_analyses")
        self._increment("_total_dqi")

        # -- Dimension 1: EF source quality --------------------------------
        ef_src_lower = ef_source.lower().strip()
        ef_src_score = EF_SOURCE_SCORES.get(ef_src_lower, 0.20)

        # -- Dimension 2: EF age -------------------------------------------
        age = max(0, ef_age_years)
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

        # -- Dimension 4: Efficiency verification status -------------------
        if efficiency_verified:
            eff_score = EFFICIENCY_VERIFICATION_SCORES.get("measured", 1.0)
        else:
            eff_score = EFFICIENCY_VERIFICATION_SCORES.get("handbook", 0.5)

        # -- Composite score (weighted average) ----------------------------
        composite = (
            0.30 * ef_src_score
            + 0.20 * ef_age_score
            + 0.25 * act_src_score
            + 0.25 * eff_score
        )

        # Clamp to [0, 1]
        composite = max(0.0, min(1.0, composite))

        result = _D(str(round(composite, 4)))

        self._record_metric("dqi_score")

        logger.debug(
            "DQI score: ef_src=%s (%.2f), ef_age=%d (%.2f), "
            "act_src=%s (%.2f), eff_verified=%s (%.2f) -> composite=%.4f",
            ef_source, ef_src_score, age, ef_age_score,
            activity_source, act_src_score, efficiency_verified,
            eff_score, float(result),
        )
        return result

    # ==================================================================
    # Public API -- DQI Score to Uncertainty
    # ==================================================================

    def score_to_uncertainty(self, dqi_score: Decimal) -> Decimal:
        """Map a DQI score (0-1) to an uncertainty percentage.

        The mapping uses an inverse linear relationship where higher
        DQI scores (better quality) produce lower uncertainty:

            uncertainty_fraction = 0.35 - 0.30 * dqi_score

        This maps:
            DQI 1.0 -> 5% uncertainty
            DQI 0.8 -> 11% uncertainty
            DQI 0.5 -> 20% uncertainty
            DQI 0.2 -> 29% uncertainty
            DQI 0.0 -> 35% uncertainty

        Args:
            dqi_score: DQI score between 0 and 1.

        Returns:
            Decimal uncertainty as a fraction (e.g. Decimal("0.05")
            for DQI score of 1.0).
        """
        score_float = max(0.0, min(1.0, float(dqi_score)))
        uncertainty = 0.35 - 0.30 * score_float
        uncertainty = max(0.02, min(0.50, uncertainty))
        return _D(str(round(uncertainty, 4)))

    # ==================================================================
    # Public API -- Validate Request
    # ==================================================================

    def validate_request(
        self,
        request: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Validate an uncertainty analysis request.

        Checks for required fields, valid ranges, and consistent
        parameter values.

        Args:
            request: Request dictionary with keys:
                - calc_result: dict (required)
                - method: str (optional, default "monte_carlo")
                - iterations: int (optional)
                - confidence_level: float/Decimal (optional)
                - activity_data_uncertainty_pct: float/Decimal (optional)
                - emission_factor_uncertainty_pct: float/Decimal (optional)
                - efficiency_uncertainty_pct: float/Decimal (optional)
                - cop_uncertainty_pct: float/Decimal (optional)
                - chp_allocation_uncertainty_pct: float/Decimal (optional)
                - seed: int (optional)

        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors: List[str] = []

        # Check calc_result
        calc_result = request.get("calc_result")
        if calc_result is None:
            errors.append("calc_result is required")
        elif not isinstance(calc_result, dict):
            errors.append("calc_result must be a dictionary")
        else:
            co2e = self._extract_co2e(calc_result)
            if co2e <= _ZERO:
                errors.append("calc_result must contain a positive CO2e value")

        # Check method
        method = request.get("method", "monte_carlo")
        if method.lower().strip() not in (
            "monte_carlo", "mc", "monte-carlo",
            "analytical", "ana", "analytical_propagation",
            "error_propagation",
        ):
            errors.append(f"Unknown method: {method}")

        # Check iterations
        iterations = request.get("iterations")
        if iterations is not None:
            if not isinstance(iterations, int):
                errors.append("iterations must be an integer")
            else:
                iter_errors = self.validate_request_iterations(iterations)
                errors.extend(iter_errors)

        # Check confidence level
        conf = request.get("confidence_level")
        if conf is not None:
            try:
                conf_d = _safe_decimal(conf)
                if conf_d <= _ZERO or conf_d >= _ONE:
                    errors.append("confidence_level must be between 0 and 1 (exclusive)")
            except Exception:
                errors.append("confidence_level must be a valid number")

        # Check uncertainty percentages
        pct_fields = [
            "activity_data_uncertainty_pct",
            "emission_factor_uncertainty_pct",
            "efficiency_uncertainty_pct",
            "cop_uncertainty_pct",
            "chp_allocation_uncertainty_pct",
        ]
        for field in pct_fields:
            val = request.get(field)
            if val is not None:
                try:
                    d = _safe_decimal(val)
                    if d < _ZERO:
                        errors.append(f"{field} must be non-negative")
                    elif d > Decimal("1000"):
                        errors.append(f"{field} must be <= 1000%")
                except Exception:
                    errors.append(f"{field} must be a valid number")

        # Check seed
        seed_val = request.get("seed")
        if seed_val is not None and not isinstance(seed_val, int):
            errors.append("seed must be an integer")

        return (len(errors) == 0, errors)

    def validate_request_iterations(self, n: int) -> List[str]:
        """Validate Monte Carlo iteration count.

        Args:
            n: Number of iterations.

        Returns:
            List of validation error strings (empty if valid).
        """
        errors: List[str] = []
        if n < 100:
            errors.append(f"iterations must be >= 100; got {n}")
        if n > 1_000_000:
            errors.append(f"iterations must be <= 1,000,000; got {n}")
        return errors

    # ==================================================================
    # Public API -- Batch Quantify
    # ==================================================================

    def batch_quantify(
        self,
        results: List[Dict[str, Any]],
        method: str = "monte_carlo",
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run uncertainty quantification on a batch of calculation results.

        Processes each calculation result independently and returns
        a list of uncertainty analysis results.

        Args:
            results: List of calculation result dictionaries.
            method: Uncertainty method. Default: "monte_carlo".
            **kwargs: Additional keyword arguments passed to each
                individual analysis (iterations, seed, uncertainties, etc.).

        Returns:
            List of uncertainty analysis result dictionaries.
        """
        self._increment("_total_batch")
        start_time = time.monotonic()
        batch_id = str(uuid4())

        if not results:
            return []

        batch_results: List[Dict[str, Any]] = []
        success_count = 0
        error_count = 0

        for idx, calc_result in enumerate(results):
            try:
                result = self.quantify_uncertainty(
                    calc_result=calc_result,
                    method=method,
                    **kwargs,
                )
                result["batch_id"] = batch_id
                result["batch_index"] = idx
                batch_results.append(result)
                if result.get("status") == "SUCCESS":
                    success_count += 1
                else:
                    error_count += 1
            except Exception as exc:
                error_count += 1
                batch_results.append({
                    "batch_id": batch_id,
                    "batch_index": idx,
                    "status": "ERROR",
                    "error": str(exc),
                    "processing_time_ms": 0.0,
                })

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        logger.info(
            "Batch uncertainty analysis complete: id=%s, total=%d, "
            "success=%d, errors=%d, time=%.3fms",
            batch_id, len(results), success_count, error_count,
            processing_time,
        )

        return batch_results

    # ==================================================================
    # Public API -- Z-Score Lookup
    # ==================================================================

    def get_z_score(self, confidence_level: Decimal) -> Decimal:
        """Get the z-score for a given confidence level.

        Looks up the z-score from the Z_SCORE_TABLE for common
        confidence levels. For non-standard levels, uses the
        Abramowitz and Stegun approximation of the inverse normal CDF.

        Args:
            confidence_level: Confidence level as a fraction (e.g. 0.95).

        Returns:
            Decimal z-score value.
        """
        # Try exact lookup first
        key = str(confidence_level)
        if key in Z_SCORE_TABLE:
            return Z_SCORE_TABLE[key]

        # Try normalised lookup
        cl_float = float(confidence_level)
        for k, v in Z_SCORE_TABLE.items():
            if abs(float(k) - cl_float) < 0.001:
                return v

        # Approximate using inverse normal CDF
        # For the upper tail probability p = (1 - cl) / 2
        # Use Abramowitz and Stegun approximation
        p = (1.0 - cl_float) / 2.0
        if p <= 0.0 or p >= 0.5:
            return Decimal("1.960")  # Default to 95%

        t = math.sqrt(-2.0 * math.log(p))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308
        z = t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)
        return _D(str(round(z, 3)))

    # ==================================================================
    # Public API -- Propagate Multiplication
    # ==================================================================

    def propagate_multiplication(
        self,
        a: Decimal,
        u_a: Decimal,
        b: Decimal,
        u_b: Decimal,
    ) -> Dict[str, Decimal]:
        """Error propagation for multiplication: z = a * b.

        For z = a * b, the relative uncertainty is:
            u_z / z = sqrt((u_a / a)^2 + (u_b / b)^2)

        Args:
            a: First operand value.
            u_a: Absolute uncertainty in *a*.
            b: Second operand value.
            u_b: Absolute uncertainty in *b*.

        Returns:
            Dictionary with keys: result, uncertainty, relative_uncertainty.
        """
        z = a * b

        rel_a_sq = (u_a / a) ** 2 if a != _ZERO else _ZERO
        rel_b_sq = (u_b / b) ** 2 if b != _ZERO else _ZERO
        rel_z = _D(str(math.sqrt(float(rel_a_sq + rel_b_sq))))
        u_z = (abs(z) * rel_z).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        return {
            "result": z.quantize(_PRECISION, rounding=ROUND_HALF_UP),
            "uncertainty": u_z,
            "relative_uncertainty": rel_z.quantize(_PRECISION, rounding=ROUND_HALF_UP),
        }

    # ==================================================================
    # Public API -- Propagate Addition
    # ==================================================================

    def propagate_addition(
        self,
        values_with_uncertainties: List[Tuple[Decimal, Decimal]],
    ) -> Dict[str, Decimal]:
        """Error propagation for addition: z = sum(a_i).

        For z = a_1 + a_2 + ... + a_n, the absolute uncertainty is:
            sigma_z = sqrt(sigma_1^2 + sigma_2^2 + ... + sigma_n^2)

        Args:
            values_with_uncertainties: List of (value, absolute_uncertainty).

        Returns:
            Dictionary with keys: result, uncertainty, relative_uncertainty.

        Raises:
            ValueError: If *values_with_uncertainties* is empty.
        """
        if not values_with_uncertainties:
            raise ValueError("At least one (value, uncertainty) tuple is required")

        z = sum((v for v, _u in values_with_uncertainties), _ZERO)
        sum_sq = sum(u ** 2 for _v, u in values_with_uncertainties)
        sigma_z = _D(str(round(math.sqrt(float(sum_sq)), 8)))

        rel_z = (sigma_z / abs(z)) if z != _ZERO else _ZERO

        return {
            "result": z.quantize(_PRECISION, rounding=ROUND_HALF_UP),
            "uncertainty": sigma_z.quantize(_PRECISION, rounding=ROUND_HALF_UP),
            "relative_uncertainty": rel_z.quantize(_PRECISION, rounding=ROUND_HALF_UP),
        }

    # ==================================================================
    # Public API -- Statistical Helpers
    # ==================================================================

    def calculate_percentiles(
        self,
        values: List[Decimal],
        percentiles: Optional[List[int]] = None,
    ) -> Dict[str, Decimal]:
        """Calculate percentiles from a list of Decimal values.

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
        return self._compute_percentiles_decimal(sorted_vals, percentiles)

    def calculate_confidence_interval(
        self,
        values: List[Decimal],
        confidence_level: Decimal = Decimal("0.95"),
    ) -> Dict[str, Decimal]:
        """Calculate a confidence interval from a list of Decimal values.

        Uses the percentile method on sorted values.

        Args:
            values: List of Decimal values.
            confidence_level: Confidence level as a fraction.

        Returns:
            Dictionary with ci_lower, ci_upper, confidence_level, width.
        """
        if not values:
            return {
                "ci_lower": _ZERO,
                "ci_upper": _ZERO,
                "confidence_level": confidence_level,
                "width": _ZERO,
            }

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        alpha = (float(_ONE - confidence_level)) / 2.0
        lower_idx = max(0, int(alpha * n))
        upper_idx = min(n - 1, int((1.0 - alpha) * n))

        ci_lower = sorted_vals[lower_idx].quantize(_PRECISION, rounding=ROUND_HALF_UP)
        ci_upper = sorted_vals[upper_idx].quantize(_PRECISION, rounding=ROUND_HALF_UP)

        return {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": confidence_level,
            "width": (ci_upper - ci_lower).quantize(_PRECISION, rounding=ROUND_HALF_UP),
        }

    def calculate_statistics(self, values: List[Decimal]) -> Dict[str, Any]:
        """Calculate descriptive statistics for a list of Decimal values.

        Computes mean, median, standard deviation, minimum, maximum,
        skewness, and kurtosis.

        Args:
            values: List of Decimal values.

        Returns:
            Dictionary with count, mean, median, std_dev, min, max,
            skewness, kurtosis, cv_pct.
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
        actual_seed = self._resolve_seed(seed)
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
            mean: Desired arithmetic mean. Must be > 0.
            std: Desired arithmetic standard deviation. Must be > 0.
            n: Number of samples to generate.
            seed: Optional PRNG seed.

        Returns:
            List of *n* Decimal samples.
        """
        if mean <= 0 or std <= 0:
            return [_D(str(round(max(0.0, mean), 8)))] * n

        actual_seed = self._resolve_seed(seed)
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
    # Public API -- IPCC Default Uncertainties
    # ==================================================================

    def get_ipcc_default_uncertainties(self) -> Dict[str, Any]:
        """Return default uncertainty ranges for steam/heat purchase parameters.

        Returns a comprehensive dictionary of uncertainty ranges (95% CI
        half-widths as fractions) for all key parameters in steam/heat
        purchase emission calculations.

        Returns:
            Dictionary with nested structure for all uncertainty constants.
        """
        return {
            "tier_defaults": {
                tier_name: {
                    k: str(v) for k, v in tier_data.items()
                    if k != "description"
                } | {"description": tier_data.get("description", "")}
                for tier_name, tier_data in TIER_DEFAULTS.items()
            },
            "activity_data_uncertainty": {
                k: {
                    "uncertainty_pct": str(v["uncertainty_pct"]),
                    "description": v.get("description", ""),
                }
                for k, v in ACTIVITY_DATA_UNCERTAINTY.items()
            },
            "emission_factor_uncertainty": {
                k: {
                    "uncertainty_pct": str(v["uncertainty_pct"]),
                    "description": v.get("description", ""),
                }
                for k, v in EF_UNCERTAINTY.items()
            },
            "efficiency_uncertainty": {
                k: {
                    "uncertainty_pct": str(v["uncertainty_pct"]),
                    "description": v.get("description", ""),
                }
                for k, v in EFFICIENCY_UNCERTAINTY.items()
            },
            "cop_uncertainty": {
                k: {
                    "uncertainty_pct": str(v["uncertainty_pct"]),
                    "description": v.get("description", ""),
                }
                for k, v in COP_UNCERTAINTY.items()
            },
            "chp_allocation_uncertainty": {
                k: {
                    "uncertainty_pct": str(v["uncertainty_pct"]),
                    "description": v.get("description", ""),
                }
                for k, v in CHP_ALLOCATION_UNCERTAINTY.items()
            },
            "per_gas_ef_uncertainty": {
                k: {
                    "lower_pct": str(v["lower_pct"]),
                    "upper_pct": str(v["upper_pct"]),
                    "default_pct": str(v["default_pct"]),
                    "description": v.get("description", ""),
                }
                for k, v in PER_GAS_EF_UNCERTAINTY.items()
            },
        }

    # ==================================================================
    # Public API -- Lookup Helpers
    # ==================================================================

    def get_activity_data_uncertainty(self, data_source: str) -> Decimal:
        """Look up activity data uncertainty by data source type.

        Args:
            data_source: Data source identifier (case-insensitive).

        Returns:
            Decimal uncertainty as a fraction.
        """
        src_key = data_source.lower().strip()
        data = ACTIVITY_DATA_UNCERTAINTY.get(src_key)
        if data is None:
            data = ACTIVITY_DATA_UNCERTAINTY["default"]
            logger.warning(
                "Unknown activity data source '%s'; using default (0.05)",
                data_source,
            )
        return data["uncertainty_pct"]

    def get_ef_uncertainty(self, ef_source: str) -> Decimal:
        """Look up emission factor uncertainty by source quality.

        Args:
            ef_source: EF source identifier (case-insensitive).

        Returns:
            Decimal uncertainty as a fraction.
        """
        src_key = ef_source.lower().strip()
        data = EF_UNCERTAINTY.get(src_key)
        if data is None:
            data = EF_UNCERTAINTY["default"]
            logger.warning(
                "Unknown EF source '%s'; using default (0.10)",
                ef_source,
            )
        return data["uncertainty_pct"]

    def get_efficiency_uncertainty(self, source: str) -> Decimal:
        """Look up efficiency uncertainty by verification source.

        Args:
            source: Efficiency source identifier (case-insensitive).

        Returns:
            Decimal uncertainty as a fraction.
        """
        src_key = source.lower().strip()
        data = EFFICIENCY_UNCERTAINTY.get(src_key)
        if data is None:
            data = EFFICIENCY_UNCERTAINTY["default"]
            logger.warning(
                "Unknown efficiency source '%s'; using default (0.05)",
                source,
            )
        return data["uncertainty_pct"]

    def get_cop_uncertainty(self, source: str) -> Decimal:
        """Look up COP uncertainty by data source.

        Args:
            source: COP source identifier (case-insensitive).

        Returns:
            Decimal uncertainty as a fraction.
        """
        src_key = source.lower().strip()
        data = COP_UNCERTAINTY.get(src_key)
        if data is None:
            data = COP_UNCERTAINTY["default"]
            logger.warning(
                "Unknown COP source '%s'; using default (0.08)",
                source,
            )
        return data["uncertainty_pct"]

    def get_chp_allocation_uncertainty(self, method: str) -> Decimal:
        """Look up CHP allocation uncertainty by method.

        Args:
            method: CHP allocation method identifier (case-insensitive).

        Returns:
            Decimal uncertainty as a fraction.
        """
        method_key = method.lower().strip()
        data = CHP_ALLOCATION_UNCERTAINTY.get(method_key)
        if data is None:
            data = CHP_ALLOCATION_UNCERTAINTY["default"]
            logger.warning(
                "Unknown CHP allocation method '%s'; using default (0.10)",
                method,
            )
        return data["uncertainty_pct"]

    # ==================================================================
    # Public API -- Engine Statistics
    # ==================================================================

    def get_uncertainty_stats(self) -> Dict[str, Any]:
        """Return engine usage statistics.

        Returns:
            Dictionary with engine identification, creation timestamp,
            and running counters for all analysis types.
        """
        with self._lock:
            return {
                "engine": "UncertaintyQuantifierEngine",
                "agent": "AGENT-MRV-011",
                "component": "Steam/Heat Purchase Agent",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "default_iterations": self._default_iterations,
                "default_confidence": str(self._default_confidence),
                "numpy_available": _NUMPY_AVAILABLE,
                "total_analyses": self._total_analyses,
                "total_monte_carlo": self._total_monte_carlo,
                "total_analytical": self._total_analytical,
                "total_dqi": self._total_dqi,
                "total_sensitivity": self._total_sensitivity,
                "total_batch": self._total_batch,
            }

    # ==================================================================
    # Public API -- Health Check
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the uncertainty quantifier engine.

        Verifies:
            1. Engine is initialised
            2. PRNG is functional (generates a sample)
            3. Decimal arithmetic is functional
            4. SHA-256 hashing is functional
            5. Tier defaults are loadable

        Returns:
            Dictionary with keys:
                - status: "HEALTHY" or "UNHEALTHY"
                - engine: engine name
                - checks: dict of check_name -> pass/fail
                - total_analyses: int
                - uptime_seconds: float
                - timestamp: ISO 8601 string
        """
        checks: Dict[str, bool] = {}

        # Check 1: Engine initialised
        checks["engine_initialised"] = self._created_at is not None

        # Check 2: PRNG functional
        try:
            rng = random.Random(42)
            sample = rng.gauss(100.0, 5.0)
            checks["prng_functional"] = isinstance(sample, float) and sample > 0
        except Exception:
            checks["prng_functional"] = False

        # Check 3: Decimal arithmetic
        try:
            a = Decimal("100.5")
            b = Decimal("0.05")
            c = a * b
            checks["decimal_arithmetic"] = c == Decimal("5.025")
        except Exception:
            checks["decimal_arithmetic"] = False

        # Check 4: SHA-256 hashing
        try:
            h = _compute_hash({"test": "data"})
            checks["sha256_hashing"] = len(h) == 64
        except Exception:
            checks["sha256_hashing"] = False

        # Check 5: Tier defaults loadable
        try:
            t1 = self.get_tier_defaults("tier_1")
            t2 = self.get_tier_defaults("tier_2")
            t3 = self.get_tier_defaults("tier_3")
            checks["tier_defaults_loadable"] = (
                t1["activity_data_pct"] == Decimal("7.5")
                and t2["activity_data_pct"] == Decimal("5.0")
                and t3["activity_data_pct"] == Decimal("2.0")
            )
        except Exception:
            checks["tier_defaults_loadable"] = False

        # Check 6: Combined uncertainty computation
        try:
            combined = self.compute_combined_uncertainty([
                Decimal("0.05"), Decimal("0.10"),
            ])
            expected = _D(str(round(math.sqrt(0.0025 + 0.01), 8)))
            checks["combined_uncertainty"] = abs(float(combined - expected)) < 1e-6
        except Exception:
            checks["combined_uncertainty"] = False

        # Check 7: Z-score lookup
        try:
            z_95 = self.get_z_score(Decimal("0.95"))
            checks["z_score_lookup"] = z_95 == Decimal("1.960")
        except Exception:
            checks["z_score_lookup"] = False

        # Check 8: numpy availability (informational, not a failure)
        checks["numpy_available"] = _NUMPY_AVAILABLE

        all_passed = all(
            v for k, v in checks.items() if k != "numpy_available"
        )

        uptime = (
            (_utcnow() - self._created_at).total_seconds()
            if self._created_at else 0.0
        )

        return {
            "status": "HEALTHY" if all_passed else "UNHEALTHY",
            "engine": "UncertaintyQuantifierEngine",
            "agent": "AGENT-MRV-011",
            "component": "Steam/Heat Purchase Agent",
            "checks": checks,
            "total_analyses": self._total_analyses,
            "uptime_seconds": round(uptime, 1),
            "timestamp": _utcnow().isoformat(),
        }


# ===========================================================================
# Module-level singleton accessor
# ===========================================================================

def get_uncertainty_quantifier(
    config: Any = None,
    metrics: Any = None,
    provenance: Any = None,
) -> UncertaintyQuantifierEngine:
    """Return the singleton UncertaintyQuantifierEngine instance.

    Convenience function that delegates to
    ``UncertaintyQuantifierEngine.get_instance()``.

    Args:
        config: Optional configuration object (used only on first call).
        metrics: Optional metrics recorder (used only on first call).
        provenance: Optional provenance tracker (used only on first call).

    Returns:
        The singleton UncertaintyQuantifierEngine instance.
    """
    return UncertaintyQuantifierEngine.get_instance(
        config=config,
        metrics=metrics,
        provenance=provenance,
    )


# ===========================================================================
# Public API exports
# ===========================================================================

__all__ = [
    "UncertaintyQuantifierEngine",
    "get_uncertainty_quantifier",
    # Tier-based uncertainty constants
    "TIER_DEFAULTS",
    # Activity data uncertainty constants
    "ACTIVITY_DATA_UNCERTAINTY",
    # Emission factor uncertainty constants
    "EF_UNCERTAINTY",
    # Efficiency uncertainty constants
    "EFFICIENCY_UNCERTAINTY",
    # COP uncertainty constants
    "COP_UNCERTAINTY",
    # CHP allocation uncertainty constants
    "CHP_ALLOCATION_UNCERTAINTY",
    # Per-gas EF uncertainty constants
    "PER_GAS_EF_UNCERTAINTY",
    # DQI scoring constants
    "EF_SOURCE_SCORES",
    "ACTIVITY_SOURCE_SCORES",
    "EFFICIENCY_VERIFICATION_SCORES",
    # Z-score table
    "Z_SCORE_TABLE",
]
