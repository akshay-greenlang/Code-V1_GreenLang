# -*- coding: utf-8 -*-
"""
UncertaintyQuantifierEngine - Monte Carlo & Analytical Error Propagation (Engine 5 of 7)

AGENT-MRV-010: Scope 2 Market-Based Emissions Agent

Quantifies the uncertainty in Scope 2 market-based emission calculations
using Monte Carlo simulation, analytical error propagation (IPCC Approach 1),
data quality indicator (DQI) scoring, and sensitivity/tornado analysis.

Sources of Uncertainty in Scope 2 Market-Based Calculations:
    - Instrument EF uncertainty: +/-2% (verified) to +/-10% (unverified)
      depending on third-party verification status of the contractual
      instrument (EAC, PPA, REC, GO, I-REC, REGO, etc.).
    - Renewable instrument uncertainty: 0% (zero EF is certain when a
      100% renewable instrument is applied with full matching).
    - Supplier-specific EF uncertainty: +/-5-15% depending on disclosure
      quality, verification status, and reporting methodology.
    - Residual mix factor uncertainty: +/-10-20% depending on region,
      grid operator data quality, and residual mix methodology.
    - Activity data uncertainty: +/-2% (meter) to +/-30% (benchmark)
      depending on metering accuracy, invoice quality, or estimation.
    - Coverage allocation uncertainty: +/-1-5% for matching precision
      between instruments and consumption (temporal, geographic).

IPCC / GHG Protocol Default Uncertainty Ranges (95% CI half-widths):
    Verified contractual instrument EF: +/-2%
    Unverified contractual instrument EF: +/-10%
    Supplier-specific EF (high quality): +/-5%
    Supplier-specific EF (medium quality): +/-10%
    Supplier-specific EF (low quality): +/-15%
    Residual mix factor (Tier 1 national): +/-20%
    Residual mix factor (Tier 2 subnational): +/-15%
    Residual mix factor (Tier 3 grid operator): +/-10%
    Activity data (metered): +/-2%
    Activity data (invoiced): +/-5%
    Activity data (estimated): +/-20%
    Activity data (benchmark): +/-30%
    Coverage allocation (hourly matched): +/-1%
    Coverage allocation (annual matched): +/-3%
    Coverage allocation (unmatched): +/-5%

Monte Carlo Simulation:
    - Configurable iterations (default 5,000)
    - Normal distributions for activity data (bounded at zero)
    - Normal distributions for instrument EFs (bounded at zero)
    - Normal distributions for residual mix factors (bounded at zero)
    - Lognormal distributions for CH4/N2O factors (non-negative)
    - Dual-stream model: covered (instrument) + uncovered (residual)
    - Explicit seed support for full reproducibility

Analytical Propagation (IPCC Approach 1):
    Combined relative uncertainty for multiplicative chains:
    U_total = sqrt(sum(Ui^2)) for uncorrelated parameters.

DQI Scoring (4 dimensions, 0-1 scale):
    - EF source quality (verified_instrument=0.95, supplier_specific=0.85,
      residual_mix=0.65, national_grid=0.5, unknown=0.2)
    - EF age (current year=1.0, -1yr=0.9, -2yr=0.8, -3yr=0.7, older=0.5)
    - Activity data source (meter=0.95, invoice=0.85, estimate=0.6, benchmark=0.4)
    - Instrument verification status (verified=1.0, self_declared=0.6, none=0.3)

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
    >>> from greenlang.scope2_market.uncertainty_quantifier import (
    ...     UncertaintyQuantifierEngine,
    ... )
    >>> engine = UncertaintyQuantifierEngine()
    >>> result = engine.run_monte_carlo(
    ...     base_co2e=Decimal("150000"),
    ...     uncertainties={"instrument_ef": Decimal("0.02"), "activity": Decimal("0.05")},
    ...     iterations=5000,
    ...     seed=42,
    ... )
    >>> print(result["ci_lower"], result["ci_upper"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-010 Scope 2 Market-Based Emissions (GL-MRV-SCOPE2-002)
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
    from greenlang.scope2_market.metrics import get_metrics as _get_metrics
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _get_metrics = None  # type: ignore[assignment]

try:
    from greenlang.scope2_market.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.scope2_market.provenance import (
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
# Market-Based Instrument Uncertainty Constants
# ===========================================================================

#: Instrument EF uncertainty (95% CI half-widths as fractions) by instrument
#: type and verification status.  Values are drawn from:
#:   - GHG Protocol Scope 2 Guidance (2015) Chapter 6
#:   - RE100 Technical Criteria and Corporate Reporting
#:   - AIB European Residual Mix Documentation
#:   - Green-e Verification Protocol

INSTRUMENT_UNCERTAINTY: Dict[str, Dict[str, Decimal]] = {
    "eac": {
        "verified": Decimal("0.02"),
        "unverified": Decimal("0.10"),
        "description": "Energy Attribute Certificate (GO, REC, I-REC, REGO, etc.)",
    },
    "ppa": {
        "verified": Decimal("0.02"),
        "unverified": Decimal("0.08"),
        "description": "Power Purchase Agreement (physical or virtual)",
    },
    "rec": {
        "verified": Decimal("0.02"),
        "unverified": Decimal("0.10"),
        "description": "Renewable Energy Certificate (US/Canada)",
    },
    "go": {
        "verified": Decimal("0.02"),
        "unverified": Decimal("0.10"),
        "description": "Guarantee of Origin (EU)",
    },
    "i_rec": {
        "verified": Decimal("0.02"),
        "unverified": Decimal("0.10"),
        "description": "International REC (I-REC Standard)",
    },
    "rego": {
        "verified": Decimal("0.02"),
        "unverified": Decimal("0.10"),
        "description": "Renewable Energy Guarantee of Origin (UK)",
    },
    "direct_line": {
        "verified": Decimal("0.02"),
        "unverified": Decimal("0.05"),
        "description": "Direct line / on-site generation with grid connection",
    },
    "green_tariff": {
        "verified": Decimal("0.03"),
        "unverified": Decimal("0.10"),
        "description": "Utility green pricing / green tariff program",
    },
    "supplier_specific": {
        "verified": Decimal("0.05"),
        "unverified": Decimal("0.15"),
        "description": "Supplier-specific emission factor disclosure",
    },
    "renewable": {
        "verified": Decimal("0.00"),
        "unverified": Decimal("0.00"),
        "description": "100% renewable instrument (zero EF is certain)",
    },
    "default": {
        "verified": Decimal("0.05"),
        "unverified": Decimal("0.15"),
        "description": "Default / unspecified instrument type",
    },
}


# ===========================================================================
# Residual Mix Uncertainty Constants
# ===========================================================================

#: Residual mix factor uncertainty by source quality / methodology tier.
#: The residual mix represents untracked / unclaimed grid electricity.
#: Uncertainty depends on:
#:   - Tracking system completeness (GO/REC retirement rates)
#:   - Grid generation mix data quality
#:   - Cross-border electricity flow accounting
#:   - Temporal granularity (annual vs hourly)

RESIDUAL_MIX_UNCERTAINTY: Dict[str, Dict[str, Any]] = {
    "tier_1_national": {
        "uncertainty_pct": Decimal("0.20"),
        "description": "National average residual mix (lowest quality)",
    },
    "tier_2_subnational": {
        "uncertainty_pct": Decimal("0.15"),
        "description": "Subnational / regional residual mix",
    },
    "tier_3_grid_operator": {
        "uncertainty_pct": Decimal("0.10"),
        "description": "Grid operator certified residual mix (e.g. AIB)",
    },
    "aib_certified": {
        "uncertainty_pct": Decimal("0.10"),
        "description": "AIB European Residual Mix (certified methodology)",
    },
    "green_e_residual": {
        "uncertainty_pct": Decimal("0.12"),
        "description": "Green-e residual mix (US/Canada)",
    },
    "egrid_untracked": {
        "uncertainty_pct": Decimal("0.15"),
        "description": "EPA eGRID untracked / total output rate as proxy",
    },
    "ief_national": {
        "uncertainty_pct": Decimal("0.20"),
        "description": "IEA national emission factor as residual proxy",
    },
    "estimated": {
        "uncertainty_pct": Decimal("0.25"),
        "description": "Estimated / modelled residual mix (no tracking data)",
    },
    "unknown": {
        "uncertainty_pct": Decimal("0.30"),
        "description": "Unknown / undocumented residual mix source",
    },
    "default": {
        "uncertainty_pct": Decimal("0.20"),
        "description": "Default residual mix uncertainty",
    },
}


# ===========================================================================
# Activity Data Uncertainty Constants
# ===========================================================================

#: Activity data uncertainty (95% CI half-widths as fractions) by data source.
#: Same as location-based but market-based uses these for both covered and
#: uncovered consumption streams.

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
        "description": "Utility invoices / billing records",
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
# Coverage Allocation Uncertainty Constants
# ===========================================================================

#: Coverage allocation uncertainty for matching instruments to consumption.
#: Depends on temporal granularity and geographic matching precision.

COVERAGE_ALLOCATION_UNCERTAINTY: Dict[str, Dict[str, Any]] = {
    "hourly_matched": {
        "uncertainty_pct": Decimal("0.01"),
        "description": "Hourly 24/7 CFE matching (highest precision)",
    },
    "monthly_matched": {
        "uncertainty_pct": Decimal("0.02"),
        "description": "Monthly volume matching",
    },
    "quarterly_matched": {
        "uncertainty_pct": Decimal("0.03"),
        "description": "Quarterly volume matching",
    },
    "annual_matched": {
        "uncertainty_pct": Decimal("0.03"),
        "description": "Annual volume matching (standard GHG Protocol)",
    },
    "geographic_mismatch": {
        "uncertainty_pct": Decimal("0.04"),
        "description": "Instruments from different grid region than consumption",
    },
    "unmatched": {
        "uncertainty_pct": Decimal("0.05"),
        "description": "No temporal/geographic matching verification",
    },
    "default": {
        "uncertainty_pct": Decimal("0.03"),
        "description": "Default coverage allocation uncertainty",
    },
}


# ===========================================================================
# EF Source Quality Scores for DQI (Market-Based Specific)
# ===========================================================================

#: Emission factor source quality scores (0-1 scale, higher = better).
#: Market-based hierarchy follows GHG Protocol Scope 2 Guidance Table 6.1.

EF_SOURCE_SCORES: Dict[str, float] = {
    "verified_instrument": 0.95,
    "eac_verified": 0.95,
    "ppa_verified": 0.95,
    "direct_line": 0.93,
    "supplier_specific_verified": 0.90,
    "green_tariff_verified": 0.88,
    "supplier_specific_unverified": 0.80,
    "eac_unverified": 0.78,
    "residual_mix_certified": 0.75,
    "residual_mix_tier_2": 0.70,
    "residual_mix_tier_1": 0.60,
    "grid_average_proxy": 0.50,
    "national_grid": 0.50,
    "egrid_untracked": 0.55,
    "iea_proxy": 0.45,
    "ipcc_default": 0.35,
    "estimated": 0.25,
    "unknown": 0.15,
}


# ===========================================================================
# Per-Gas EF Uncertainty Constants
# ===========================================================================

#: Per-gas default EF uncertainty (95% CI half-width as fraction).
#: Market-based agents typically report CO2e directly from instrument EFs,
#: but per-gas breakdown is needed for supplier-specific and residual mix.

PER_GAS_EF_UNCERTAINTY: Dict[str, Dict[str, Any]] = {
    "CO2": {
        "lower_pct": Decimal("0.02"),
        "upper_pct": Decimal("0.10"),
        "default_pct": Decimal("0.03"),
        "description": "CO2 emission factor uncertainty (well characterised)",
    },
    "CH4": {
        "lower_pct": Decimal("0.15"),
        "upper_pct": Decimal("0.80"),
        "default_pct": Decimal("0.25"),
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
# Supplier-Specific Disclosure Quality Constants
# ===========================================================================

#: Supplier-specific EF uncertainty by disclosure quality level.

SUPPLIER_DISCLOSURE_UNCERTAINTY: Dict[str, Dict[str, Any]] = {
    "high": {
        "uncertainty_pct": Decimal("0.05"),
        "description": "Third-party verified, full methodology disclosure",
    },
    "medium": {
        "uncertainty_pct": Decimal("0.10"),
        "description": "Self-declared with documented methodology",
    },
    "low": {
        "uncertainty_pct": Decimal("0.15"),
        "description": "Self-declared, limited methodology documentation",
    },
    "unknown": {
        "uncertainty_pct": Decimal("0.20"),
        "description": "No disclosure quality information available",
    },
    "default": {
        "uncertainty_pct": Decimal("0.10"),
        "description": "Default supplier disclosure uncertainty",
    },
}


# ===========================================================================
# Activity Source Quality Scores for DQI
# ===========================================================================

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


# ===========================================================================
# Instrument Verification Scores for DQI
# ===========================================================================

#: Instrument verification status quality scores (0-1 scale).

INSTRUMENT_VERIFICATION_SCORES: Dict[str, float] = {
    "third_party_verified": 1.00,
    "verified": 1.00,
    "issuer_verified": 0.90,
    "registry_tracked": 0.85,
    "self_declared": 0.60,
    "unverified": 0.40,
    "none": 0.30,
    "unknown": 0.20,
}


# ===========================================================================
# UncertaintyQuantifierEngine
# ===========================================================================


class UncertaintyQuantifierEngine:
    """Monte Carlo simulation and analytical error propagation for Scope 2
    market-based emission uncertainty quantification.

    Implements both IPCC Approach 1 (analytical root-sum-of-squares) and
    Approach 2 (Monte Carlo) for quantifying uncertainty in Scope 2
    market-based emission calculations.  The market-based method uses a
    dual-stream model:

        1. **Covered stream**: Consumption matched to contractual instruments
           (EACs, PPAs, RECs, GOs, supplier-specific EFs). Uncertainty is
           driven by instrument EF quality and verification status.

        2. **Uncovered stream**: Remaining consumption assigned the residual
           mix factor.  Uncertainty is driven by residual mix data quality.

    The total market-based emission is:
        E_total = E_covered + E_uncovered
        E_covered = sum(consumption_i * instrument_ef_i)
        E_uncovered = uncovered_consumption * residual_mix_ef

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
        ...     base_co2e=Decimal("150000"),
        ...     uncertainties={"instrument_ef": Decimal("0.02")},
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
            config: Optional Scope2MarketConfig object.  When ``None``,
                the singleton from ``_get_config()`` is attempted, with
                safe in-code defaults as the final fallback.
            metrics: Optional Scope2MarketMetrics object for Prometheus
                recording.  When ``None``, the module-level singleton is
                used if available.
            provenance: Optional provenance tracker for audit trails.
        """
        self._config = config
        self._metrics = metrics
        self._provenance = provenance

        # Defaults -- market-based uses 5,000 as default (fewer iterations
        # than location-based because dual-stream model converges faster)
        self._default_iterations: int = 5_000
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
        self._created_at: datetime = _utcnow()

        logger.info(
            "UncertaintyQuantifierEngine initialised (market-based): "
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
    # Public API -- Monte Carlo Simulation (Generic)
    # ==================================================================

    def run_monte_carlo(
        self,
        base_co2e: Decimal,
        uncertainties: Dict[str, Decimal],
        iterations: int = 5_000,
        seed: Optional[int] = None,
        confidence_level: Decimal = Decimal("0.95"),
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for generic market-based uncertainty.

        Accepts a base CO2e value and a dictionary of named uncertainty
        components.  Each component is treated as an independent
        multiplicative perturbation factor sampled from N(1, sigma_i).

        Sampling model (per iteration):
            factor_i = gauss(1.0, uncertainties[key_i])
            emission = base_co2e * product(factor_i)

        This method supports any combination of uncertainty sources:
        instrument_ef, residual_mix, activity_data, coverage_allocation, etc.

        Args:
            base_co2e: Central emission estimate in kg CO2e.
                Must be non-negative.
            uncertainties: Dictionary mapping uncertainty source names to
                relative uncertainty fractions (e.g. {"instrument_ef": 0.02}).
            iterations: Number of Monte Carlo iterations. Default: 5,000.
                Must be between 100 and 1,000,000.
            seed: Optional PRNG seed for reproducibility.
            confidence_level: Confidence level as a fraction. Default: 0.95.

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
                - confidence_level: Decimal
                - iterations: int
                - seed: int
                - percentiles: dict
                - relative_uncertainty_pct: Decimal combined relative uncertainty
                - input_uncertainties: dict
                - provenance_hash: SHA-256 hex digest
                - processing_time_ms: float
                - errors: list (if validation failed)
        """
        self._increment("_total_analyses")
        self._increment("_total_monte_carlo")
        start_time = time.monotonic()
        calc_id = str(uuid4())

        # -- Validate inputs -----------------------------------------------
        errors: List[str] = []
        base_errors = self.validate_uncertainty_input(base_co2e, _ZERO)
        errors.extend(base_errors)

        for name, unc_val in uncertainties.items():
            unc_errors = self.validate_uncertainty_input(_ZERO, unc_val)
            # Filter: base check is redundant for uncertainty-only validation
            for e in unc_errors:
                if "base_emissions" not in e and e not in errors:
                    errors.append(e)

        iter_errors = self.validate_iterations(iterations)
        errors.extend(iter_errors)

        if errors:
            processing_time = round((time.monotonic() - start_time) * 1000, 3)
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "method": "MONTE_CARLO",
                "errors": errors,
                "processing_time_ms": processing_time,
            }

        # -- Convert to floats for simulation performance ------------------
        base_float = float(base_co2e)
        unc_floats: Dict[str, float] = {
            k: float(v) for k, v in uncertainties.items()
        }
        conf = float(confidence_level)

        # -- Initialise PRNG -----------------------------------------------
        actual_seed = self._resolve_seed(seed)
        rng = random.Random(actual_seed)

        # -- Run simulation ------------------------------------------------
        iters = self._clamp_iterations(iterations)
        results: List[float] = []

        for _ in range(iters):
            combined_factor = 1.0
            for _name, unc in unc_floats.items():
                if unc > 0.0:
                    combined_factor *= rng.gauss(1.0, unc)
            sample = base_float * combined_factor
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

        # -- Combined relative uncertainty (analytical RSS for reference) ---
        rel_unc = math.sqrt(sum(u ** 2 for u in unc_floats.values()))

        # -- Build output --------------------------------------------------
        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
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
            "input_uncertainties": {k: str(v) for k, v in uncertainties.items()},
            "base_co2e_kg": base_co2e,
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        # -- Record metrics and provenance ---------------------------------
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

    # ==================================================================
    # Public API -- Per-Gas Monte Carlo
    # ==================================================================

    def run_monte_carlo_per_gas(
        self,
        gas_results: Dict[str, Decimal],
        iterations: int = 5_000,
        seed: Optional[int] = None,
        ef_uncertainties: Optional[Dict[str, Decimal]] = None,
    ) -> Dict[str, Any]:
        """Run per-gas Monte Carlo simulation with individual uncertainty ranges.

        Each greenhouse gas has distinct uncertainty characteristics for
        market-based calculations:
        - CO2: +/-3% (default) -- well-characterised from instrument EFs
        - CH4: +/-25% (default) -- lognormal, moderate variability
        - N2O: +/-50% (default) -- lognormal, high variability

        Sampling model (per gas, per iteration):
            CO2: normal distribution (bounded at zero)
            CH4: lognormal distribution (non-negative)
            N2O: lognormal distribution (non-negative)

        Args:
            gas_results: Dictionary mapping gas names to kg CO2e values.
                Expected keys: "CO2", "CH4", "N2O".  Missing gases default
                to zero.
            iterations: Number of MC iterations. Default: 5,000.
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
            """Compute statistics for a single gas."""
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
    # Public API -- Market-Based Monte Carlo (Dual-Stream)
    # ==================================================================

    def run_monte_carlo_market_based(
        self,
        covered_co2e: Decimal,
        uncovered_co2e: Decimal,
        instrument_unc: Decimal,
        residual_unc: Decimal,
        activity_unc: Decimal = Decimal("0.05"),
        coverage_unc: Decimal = Decimal("0.03"),
        iterations: int = 5_000,
        seed: Optional[int] = None,
        confidence_level: Decimal = Decimal("0.95"),
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for dual-stream market-based emissions.

        The market-based method decomposes total emissions into two streams:

            E_total = E_covered + E_uncovered

        where:
            E_covered = covered_consumption * instrument_ef
            E_uncovered = uncovered_consumption * residual_mix_ef

        Each stream has independent uncertainty sources that are sampled
        separately and combined additively.

        Sampling model (per iteration):
            covered_i = covered_co2e * gauss(1, instrument_unc)
                        * gauss(1, activity_unc) * gauss(1, coverage_unc)
            uncovered_i = uncovered_co2e * gauss(1, residual_unc)
                          * gauss(1, activity_unc)
            total_i = covered_i + uncovered_i

        Args:
            covered_co2e: Emissions from instrument-covered consumption (kg CO2e).
            uncovered_co2e: Emissions from residual-mix-covered consumption (kg CO2e).
            instrument_unc: Relative uncertainty for instrument EFs (fraction).
            residual_unc: Relative uncertainty for residual mix factor (fraction).
            activity_unc: Relative uncertainty for activity data (fraction).
                Default: 0.05 (5%, assumes invoice-quality data).
            coverage_unc: Relative uncertainty for coverage allocation (fraction).
                Default: 0.03 (3%, assumes annual matching).
            iterations: Number of MC iterations. Default: 5,000.
            seed: Optional PRNG seed for reproducibility.
            confidence_level: Confidence level as a fraction. Default: 0.95.

        Returns:
            Dictionary with covered, uncovered, and total stream results,
            including mean, std_dev, ci_lower, ci_upper, percentiles,
            relative_uncertainty_pct, and provenance_hash.
        """
        self._increment("_total_analyses")
        self._increment("_total_monte_carlo")
        start_time = time.monotonic()
        calc_id = str(uuid4())

        # -- Validate inputs -----------------------------------------------
        errors: List[str] = []
        for name, val in [
            ("covered_co2e", covered_co2e),
            ("uncovered_co2e", uncovered_co2e),
        ]:
            d = _safe_decimal(val, Decimal("-1"))
            if d < _ZERO:
                errors.append(f"{name} must be non-negative")

        for name, val in [
            ("instrument_unc", instrument_unc),
            ("residual_unc", residual_unc),
            ("activity_unc", activity_unc),
            ("coverage_unc", coverage_unc),
        ]:
            d = _safe_decimal(val, Decimal("-1"))
            if d < _ZERO:
                errors.append(f"{name} must be non-negative")
            elif d > Decimal("10"):
                errors.append(f"{name} must be <= 10.0 (1000%); got {val}")

        iter_errors = self.validate_iterations(iterations)
        errors.extend(iter_errors)

        if errors:
            processing_time = round((time.monotonic() - start_time) * 1000, 3)
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "method": "MONTE_CARLO_MARKET_BASED",
                "errors": errors,
                "processing_time_ms": processing_time,
            }

        # -- Convert to floats ---------------------------------------------
        covered_f = float(covered_co2e)
        uncovered_f = float(uncovered_co2e)
        inst_unc_f = float(instrument_unc)
        resid_unc_f = float(residual_unc)
        act_unc_f = float(activity_unc)
        cov_unc_f = float(coverage_unc)
        conf = float(confidence_level)

        # -- Initialise PRNG -----------------------------------------------
        actual_seed = self._resolve_seed(seed)
        rng = random.Random(actual_seed)
        iters = self._clamp_iterations(iterations)

        # -- Run simulation ------------------------------------------------
        covered_samples: List[float] = []
        uncovered_samples: List[float] = []
        total_samples: List[float] = []

        for _ in range(iters):
            # Covered stream: instrument EF + activity + coverage allocation
            cov_ef = rng.gauss(1.0, inst_unc_f) if inst_unc_f > 0 else 1.0
            cov_act = rng.gauss(1.0, act_unc_f) if act_unc_f > 0 else 1.0
            cov_alloc = rng.gauss(1.0, cov_unc_f) if cov_unc_f > 0 else 1.0
            covered_sample = max(0.0, covered_f * cov_ef * cov_act * cov_alloc)

            # Uncovered stream: residual mix + activity
            unc_ef = rng.gauss(1.0, resid_unc_f) if resid_unc_f > 0 else 1.0
            unc_act = rng.gauss(1.0, act_unc_f) if act_unc_f > 0 else 1.0
            uncovered_sample = max(0.0, uncovered_f * unc_ef * unc_act)

            total_sample = covered_sample + uncovered_sample

            covered_samples.append(covered_sample)
            uncovered_samples.append(uncovered_sample)
            total_samples.append(total_sample)

        # -- Compute statistics for each stream ----------------------------
        covered_stats = self._float_stats(covered_samples, "covered")
        uncovered_stats = self._float_stats(uncovered_samples, "uncovered")
        total_stats = self._float_stats(total_samples, "total")

        # -- Percentiles for total -----------------------------------------
        sorted_total = sorted(total_samples)
        pctile_points = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
        total_percentiles = self._compute_percentiles_float(
            sorted_total, pctile_points,
        )

        # -- Confidence interval for total ---------------------------------
        alpha = (1.0 - conf) / 2.0
        n_t = len(sorted_total)
        ci_lower = sorted_total[max(0, int(alpha * n_t))]
        ci_upper = sorted_total[min(n_t - 1, int((1.0 - alpha) * n_t))]

        # -- Analytical RSS for reference ----------------------------------
        covered_rel = math.sqrt(inst_unc_f ** 2 + act_unc_f ** 2 + cov_unc_f ** 2)
        uncovered_rel = math.sqrt(resid_unc_f ** 2 + act_unc_f ** 2)

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "MONTE_CARLO_MARKET_BASED",
            "covered_stream": {
                "base_co2e_kg": covered_co2e,
                "mean_co2e_kg": covered_stats["mean"],
                "std_dev": covered_stats["std_dev"],
                "cv_pct": covered_stats["cv_pct"],
                "ci_lower_95": covered_stats["ci_lower_95"],
                "ci_upper_95": covered_stats["ci_upper_95"],
                "analytical_rel_uncertainty": Decimal(str(round(covered_rel, 6))),
            },
            "uncovered_stream": {
                "base_co2e_kg": uncovered_co2e,
                "mean_co2e_kg": uncovered_stats["mean"],
                "std_dev": uncovered_stats["std_dev"],
                "cv_pct": uncovered_stats["cv_pct"],
                "ci_lower_95": uncovered_stats["ci_lower_95"],
                "ci_upper_95": uncovered_stats["ci_upper_95"],
                "analytical_rel_uncertainty": Decimal(str(round(uncovered_rel, 6))),
            },
            "total": {
                "base_co2e_kg": covered_co2e + uncovered_co2e,
                "mean_co2e_kg": total_stats["mean"],
                "std_dev": total_stats["std_dev"],
                "cv_pct": total_stats["cv_pct"],
                "ci_lower": Decimal(str(round(ci_lower, 6))),
                "ci_upper": Decimal(str(round(ci_upper, 6))),
            },
            "confidence_level": confidence_level,
            "iterations": iters,
            "seed": actual_seed,
            "percentiles": {
                str(k): Decimal(str(round(v, 6)))
                for k, v in total_percentiles.items()
            },
            "input_uncertainties": {
                "instrument_unc": str(instrument_unc),
                "residual_unc": str(residual_unc),
                "activity_unc": str(activity_unc),
                "coverage_unc": str(coverage_unc),
            },
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        self._record_metric("monte_carlo_market_based")
        self._record_provenance(
            entity_id=calc_id[:16],
            data={
                "method": "monte_carlo_market_based",
                "covered_kg": str(covered_co2e),
                "uncovered_kg": str(uncovered_co2e),
                "total_mean_kg": str(total_stats["mean"]),
                "ci_lower": str(result["total"]["ci_lower"]),
                "ci_upper": str(result["total"]["ci_upper"]),
                "iterations": iters,
                "seed": actual_seed,
            },
        )

        logger.info(
            "Market-based Monte Carlo complete: id=%s, n=%d, "
            "covered=%.2f, uncovered=%.2f, total_mean=%.2f, "
            "CI=[%.2f, %.2f], time=%.3fms",
            calc_id, iters, covered_f, uncovered_f,
            float(total_stats["mean"]), ci_lower, ci_upper,
            processing_time,
        )
        return result

    # ==================================================================
    # Public API -- Batch Monte Carlo
    # ==================================================================

    def run_monte_carlo_batch(
        self,
        calculations: List[Dict[str, Any]],
        iterations: int = 5_000,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for a batch of market-based calculations.

        Supports correlations between calculations that share the same
        instrument source or residual mix region (same-source factors
        receive correlated perturbations).

        Each entry in *calculations* should contain:
            - id: Unique identifier
            - base_co2e_kg: Central emission estimate (Decimal or numeric)
            - instrument_unc: Instrument EF relative uncertainty (Decimal)
            - residual_unc: Residual mix relative uncertainty (Decimal)
            - activity_unc: Activity data uncertainty (Decimal, optional)
            - instrument_source: (optional) Instrument source for correlation
            - residual_source: (optional) Residual mix source for correlation

        Args:
            calculations: List of calculation dictionaries.
            iterations: Number of MC iterations. Default: 5,000.
            seed: Optional PRNG seed.

        Returns:
            Dictionary with per-calculation and aggregated portfolio results,
            including portfolio-level confidence intervals and provenance_hash.
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

        iters = self._clamp_iterations(iterations)
        actual_seed = self._resolve_seed(seed)
        rng = random.Random(actual_seed)

        # -- Group by instrument source for correlation --------------------
        inst_groups: Dict[str, List[int]] = {}
        resid_groups: Dict[str, List[int]] = {}
        for idx, calc in enumerate(calculations):
            inst_src = str(calc.get("instrument_source", f"inst_indep_{idx}"))
            resid_src = str(calc.get("residual_source", f"resid_indep_{idx}"))
            inst_groups.setdefault(inst_src, []).append(idx)
            resid_groups.setdefault(resid_src, []).append(idx)

        # -- Pre-generate correlated perturbations -------------------------
        inst_perturbations: Dict[str, List[float]] = {}
        for src, indices in inst_groups.items():
            first_idx = indices[0]
            unc = float(_safe_decimal(
                calculations[first_idx].get("instrument_unc", "0.05"),
                Decimal("0.05"),
            ))
            inst_perturbations[src] = [rng.gauss(1.0, unc) for _ in range(iters)]

        resid_perturbations: Dict[str, List[float]] = {}
        for src, indices in resid_groups.items():
            first_idx = indices[0]
            unc = float(_safe_decimal(
                calculations[first_idx].get("residual_unc", "0.15"),
                Decimal("0.15"),
            ))
            resid_perturbations[src] = [rng.gauss(1.0, unc) for _ in range(iters)]

        # -- Simulate each calculation -------------------------------------
        per_calc_results: List[Dict[str, Any]] = []
        portfolio_samples: List[float] = [0.0] * iters

        for idx, calc in enumerate(calculations):
            calc_id = str(calc.get("id", f"calc_{idx}"))
            base_kg = float(_safe_decimal(
                calc.get("base_co2e_kg", 0), _ZERO,
            ))
            act_unc = float(_safe_decimal(
                calc.get("activity_unc", "0.05"), Decimal("0.05"),
            ))

            inst_src = str(calc.get("instrument_source", f"inst_indep_{idx}"))
            resid_src = str(calc.get("residual_source", f"resid_indep_{idx}"))

            # Determine which perturbation to use
            is_instrument = calc.get("is_instrument_covered", True)

            samples: List[float] = []
            for i in range(iters):
                if is_instrument:
                    ef_sample = inst_perturbations[inst_src][i]
                else:
                    ef_sample = resid_perturbations[resid_src][i]
                act_sample = rng.gauss(1.0, act_unc)
                sample = max(0.0, base_kg * ef_sample * act_sample)
                samples.append(sample)
                portfolio_samples[i] += sample

            # Per-calculation stats
            stats = self._float_stats(samples, label=calc_id)
            per_calc_results.append({
                "id": calc_id,
                "base_co2e_kg": Decimal(str(round(base_kg, 6))),
                "mean_co2e_kg": stats["mean"],
                "std_dev": stats["std_dev"],
                "ci_lower_95": stats["ci_lower_95"],
                "ci_upper_95": stats["ci_upper_95"],
            })

        # -- Portfolio-level stats -----------------------------------------
        portfolio_stats = self._float_stats(portfolio_samples, label="portfolio")
        sorted_portfolio = sorted(portfolio_samples)
        n_p = len(sorted_portfolio)
        port_percentiles = self._compute_percentiles_float(
            sorted_portfolio, [5, 10, 25, 50, 75, 90, 95],
        )

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "batch_id": batch_id,
            "status": "SUCCESS",
            "method": "MONTE_CARLO_BATCH",
            "calculation_count": len(calculations),
            "iterations": iters,
            "seed": actual_seed,
            "per_calculation": per_calc_results,
            "portfolio": {
                "total_base_kg": Decimal(str(round(
                    sum(
                        float(_safe_decimal(c.get("base_co2e_kg", 0), _ZERO))
                        for c in calculations
                    ), 6,
                ))),
                "mean_co2e_kg": portfolio_stats["mean"],
                "std_dev": portfolio_stats["std_dev"],
                "cv_pct": portfolio_stats["cv_pct"],
                "ci_lower_95": portfolio_stats["ci_lower_95"],
                "ci_upper_95": portfolio_stats["ci_upper_95"],
                "percentiles": {
                    str(k): Decimal(str(round(v, 6)))
                    for k, v in port_percentiles.items()
                },
            },
            "correlation_groups": {
                "instrument_sources": {
                    src: len(idxs) for src, idxs in inst_groups.items()
                },
                "residual_sources": {
                    src: len(idxs) for src, idxs in resid_groups.items()
                },
            },
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        self._record_metric("monte_carlo_batch")

        logger.info(
            "Batch Monte Carlo complete: id=%s, calcs=%d, n=%d, "
            "portfolio mean=%.2f, CI=[%.2f, %.2f], time=%.3fms",
            batch_id, len(calculations), iters,
            float(portfolio_stats["mean"]),
            float(portfolio_stats["ci_lower_95"]),
            float(portfolio_stats["ci_upper_95"]),
            processing_time,
        )
        return result

    # ==================================================================
    # Public API -- Analytical Error Propagation
    # ==================================================================

    def analytical_propagation(
        self,
        base: Decimal,
        component_uncertainties: Dict[str, Decimal],
    ) -> Dict[str, Any]:
        """IPCC Approach 1 analytical error propagation for market-based emissions.

        For the multiplicative model:
            Emissions = base * product_of_factors

        The combined relative uncertainty (root-sum-of-squares) is:
            sigma_rel = sqrt(sum(u_i^2))

        where u_i is the relative uncertainty for each component.

        This method is appropriate when all uncertainty sources are
        independent and enter the calculation multiplicatively.

        Args:
            base: Central emission estimate (kg CO2e).
            component_uncertainties: Dictionary mapping uncertainty source
                names to relative uncertainty fractions.
                Example: {"instrument_ef": 0.02, "activity": 0.05, "coverage": 0.03}

        Returns:
            Dictionary with keys:
                - calculation_id: UUID
                - status: "SUCCESS"
                - method: "ANALYTICAL_PROPAGATION"
                - emissions_kg: Decimal central estimate
                - combined_relative_uncertainty: Decimal fraction
                - combined_relative_uncertainty_pct: Decimal percentage
                - combined_absolute_uncertainty: Decimal (kg)
                - ci_lower_95: Decimal
                - ci_upper_95: Decimal
                - parameter_contributions: dict of param -> fraction of variance
                - formula: description string
                - provenance_hash: SHA-256 hex digest
        """
        self._increment("_total_analyses")
        self._increment("_total_analytical")
        start_time = time.monotonic()
        calc_id = str(uuid4())

        # -- Compute relative uncertainties squared ------------------------
        sum_sq = _ZERO
        individual_sq: Dict[str, Decimal] = {}
        for name, unc in component_uncertainties.items():
            u_sq = _safe_decimal(unc) ** 2
            individual_sq[name] = u_sq
            sum_sq += u_sq

        # -- Combined relative uncertainty (RSS) ---------------------------
        combined_rel = _D(str(math.sqrt(float(sum_sq))))

        # -- Absolute uncertainty and 95% CI -------------------------------
        combined_abs = (base * combined_rel).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        z_95 = _D("1.96")
        ci_lower = (base - combined_abs * z_95).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        ci_upper = (base + combined_abs * z_95).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )

        # -- Parameter contributions --------------------------------------
        total_sq_float = float(sum_sq)
        contributions: Dict[str, Decimal] = {}
        if total_sq_float > 0:
            for name, u_sq in individual_sq.items():
                contributions[name] = _D(str(round(
                    float(u_sq) / total_sq_float, 6,
                )))
        else:
            for name in component_uncertainties:
                contributions[name] = _ZERO

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "ANALYTICAL_PROPAGATION",
            "emissions_kg": base.quantize(_PRECISION, rounding=ROUND_HALF_UP),
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
            "input_parameters": {k: str(v) for k, v in component_uncertainties.items()},
            "formula": "sigma_rel = sqrt(sum(u_i^2)) for independent multiplicative factors",
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        self._record_metric("analytical")

        logger.info(
            "Analytical propagation: id=%s, emissions=%.2f, "
            "combined_rel=%.4f (%.2f%%), CI=[%.2f, %.2f], time=%.3fms",
            calc_id, float(base), float(combined_rel),
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
    ) -> Dict[str, Decimal]:
        """Error propagation for multiplication: z = a * b.

        For z = a * b, the relative uncertainty is:
            u_z / z = sqrt((u_a / a)^2 + (u_b / b)^2)
        therefore:
            u_z = z * sqrt((u_a / a)^2 + (u_b / b)^2)

        Args:
            a: First operand value (e.g. consumption).
            u_a: Absolute uncertainty in *a*.
            b: Second operand value (e.g. emission factor).
            u_b: Absolute uncertainty in *b*.

        Returns:
            Dictionary with keys:
                - result: z = a * b
                - uncertainty: u_z (absolute uncertainty in z)
                - relative_uncertainty: u_z / |z| (relative)
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

        This is the standard formula for combining covered and uncovered
        emission streams in market-based calculations.

        Args:
            values_with_uncertainties: List of (value, absolute_uncertainty)
                tuples.  For example:
                [(covered_emissions, covered_unc), (uncovered_emissions, uncovered_unc)]

        Returns:
            Dictionary with keys:
                - result: z = sum of values
                - uncertainty: sigma_z (absolute uncertainty)
                - relative_uncertainty: sigma_z / |z| if z != 0

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
    # Public API -- IPCC Default Uncertainties
    # ==================================================================

    def get_ipcc_default_uncertainties(self) -> Dict[str, Any]:
        """Return default uncertainty ranges for Scope 2 market-based parameters.

        Returns a comprehensive dictionary of uncertainty ranges (95% CI
        half-widths as fractions) for all key parameters in Scope 2
        market-based emission calculations.

        The ranges are drawn from:
            - GHG Protocol Scope 2 Guidance (2015) Chapter 6
            - IPCC 2006 Guidelines Vol 1 Ch 3 Table 3.2
            - RE100 / CDP Technical Criteria
            - AIB European Residual Mix Documentation
            - EPA eGRID Technical Support Document

        Returns:
            Dictionary with nested structure for all market-based
            uncertainty constants:
                instrument_uncertainty -> by instrument type
                residual_mix_uncertainty -> by source quality
                activity_data_uncertainty -> by data source
                coverage_allocation_uncertainty -> by matching precision
                supplier_disclosure_uncertainty -> by disclosure quality
                per_gas_ef_uncertainty -> by gas
        """
        return {
            "instrument_uncertainty": {
                k: {
                    "verified": str(v["verified"]),
                    "unverified": str(v["unverified"]),
                    "description": v.get("description", ""),
                }
                for k, v in INSTRUMENT_UNCERTAINTY.items()
            },
            "residual_mix_uncertainty": {
                k: {
                    "uncertainty_pct": str(v["uncertainty_pct"]),
                    "description": v.get("description", ""),
                }
                for k, v in RESIDUAL_MIX_UNCERTAINTY.items()
            },
            "activity_data_uncertainty": {
                k: {
                    "uncertainty_pct": str(v["uncertainty_pct"]),
                    "description": v.get("description", ""),
                }
                for k, v in ACTIVITY_DATA_UNCERTAINTY.items()
            },
            "coverage_allocation_uncertainty": {
                k: {
                    "uncertainty_pct": str(v["uncertainty_pct"]),
                    "description": v.get("description", ""),
                }
                for k, v in COVERAGE_ALLOCATION_UNCERTAINTY.items()
            },
            "supplier_disclosure_uncertainty": {
                k: {
                    "uncertainty_pct": str(v["uncertainty_pct"]),
                    "description": v.get("description", ""),
                }
                for k, v in SUPPLIER_DISCLOSURE_UNCERTAINTY.items()
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
    # Public API -- Instrument Uncertainty Lookup
    # ==================================================================

    def get_instrument_uncertainty(
        self,
        instrument_type: str,
        verified: bool = True,
    ) -> Decimal:
        """Look up the uncertainty for a specific contractual instrument type.

        The GHG Protocol Scope 2 Guidance defines a hierarchy of
        contractual instruments.  Verified instruments carry lower
        uncertainty because their emission factors are subject to
        third-party audit.

        Args:
            instrument_type: Instrument type identifier (case-insensitive).
                Valid values: eac, ppa, rec, go, i_rec, rego, direct_line,
                green_tariff, supplier_specific, renewable.
            verified: Whether the instrument is third-party verified.
                Default: True.

        Returns:
            Decimal uncertainty as a fraction (e.g. Decimal("0.02")
            for a verified EAC).
        """
        inst_key = instrument_type.lower().strip()
        inst_data = INSTRUMENT_UNCERTAINTY.get(inst_key)

        if inst_data is None:
            inst_data = INSTRUMENT_UNCERTAINTY["default"]
            logger.warning(
                "Unknown instrument type '%s'; using default uncertainty",
                instrument_type,
            )

        status_key = "verified" if verified else "unverified"
        return inst_data[status_key]

    # ==================================================================
    # Public API -- Residual Mix Uncertainty Lookup
    # ==================================================================

    def get_residual_mix_uncertainty(self, source_quality: str) -> Decimal:
        """Look up the uncertainty for a residual mix factor by source quality.

        The residual mix represents untracked / unclaimed grid electricity.
        Its uncertainty depends on the tracking system completeness,
        cross-border flow accounting, and methodology quality.

        Args:
            source_quality: Source quality identifier (case-insensitive).
                Valid values: tier_1_national, tier_2_subnational,
                tier_3_grid_operator, aib_certified, green_e_residual,
                egrid_untracked, ief_national, estimated, unknown.

        Returns:
            Decimal uncertainty as a fraction (e.g. Decimal("0.10")
            for tier_3_grid_operator).
        """
        quality_key = source_quality.lower().strip()
        data = RESIDUAL_MIX_UNCERTAINTY.get(quality_key)

        if data is None:
            data = RESIDUAL_MIX_UNCERTAINTY["default"]
            logger.warning(
                "Unknown residual mix source quality '%s'; using default (0.20)",
                source_quality,
            )

        return data["uncertainty_pct"]

    # ==================================================================
    # Public API -- Activity Data Uncertainty Lookup
    # ==================================================================

    def get_activity_data_uncertainty(self, data_source: str) -> Decimal:
        """Look up the uncertainty for activity data by data source type.

        Activity data uncertainty depends on metering accuracy,
        invoice quality, or estimation methodology.

        Args:
            data_source: Data source identifier (case-insensitive).
                Valid values: meter, smart_meter, calibrated_meter,
                revenue_meter, invoice, utility_bill, estimate,
                engineering_estimate, benchmark, industry_average, unknown.

        Returns:
            Decimal uncertainty as a fraction (e.g. Decimal("0.02")
            for meter, Decimal("0.30") for benchmark).
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

    # ==================================================================
    # Public API -- DQI (Data Quality Indicator) Scoring
    # ==================================================================

    def calculate_dqi_score(
        self,
        ef_source: str,
        ef_age_years: int,
        activity_source: str,
        instrument_verified: bool = True,
    ) -> Decimal:
        """Calculate a composite Data Quality Indicator (DQI) score.

        Combines four quality dimensions into a single 0-1 score where
        higher values indicate better data quality.  This market-based
        version replaces the temporal representativeness dimension from
        location-based with instrument verification status.

        Scoring dimensions:
            1. EF source quality (weight 0.30): verified_instrument=0.95,
               supplier_specific=0.85, residual_mix=0.65, grid_average=0.50
            2. EF age (weight 0.20): current_year=1.0, -1yr=0.9, -2yr=0.8,
               -3yr=0.7, older=0.5
            3. Activity data source (weight 0.25): meter=0.95, invoice=0.85,
               estimate=0.60, benchmark=0.40
            4. Instrument verification (weight 0.25): verified=1.0,
               self_declared=0.6, none=0.3

        The composite score is the weighted average:
            DQI = 0.30*ef_source + 0.20*ef_age + 0.25*activity + 0.25*verification

        Args:
            ef_source: Emission factor source identifier (case-insensitive).
                See EF_SOURCE_SCORES for valid values.
            ef_age_years: Age of the emission factor in years (0 = current).
            activity_source: Activity data source identifier (case-insensitive).
                See ACTIVITY_SOURCE_SCORES for valid values.
            instrument_verified: Whether the contractual instrument has
                third-party verification. Default: True.

        Returns:
            Decimal DQI score between 0 and 1.
        """
        self._increment("_total_analyses")
        self._increment("_total_dqi")

        # -- Dimension 1: EF source quality --------------------------------
        ef_src_lower = ef_source.lower().strip()
        ef_src_score = EF_SOURCE_SCORES.get(ef_src_lower, 0.15)

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

        # -- Dimension 4: Instrument verification status -------------------
        if instrument_verified:
            verify_score = INSTRUMENT_VERIFICATION_SCORES.get("verified", 1.0)
        else:
            verify_score = INSTRUMENT_VERIFICATION_SCORES.get("unverified", 0.4)

        # -- Composite score (weighted average) ----------------------------
        composite = (
            0.30 * ef_src_score
            + 0.20 * ef_age_score
            + 0.25 * act_src_score
            + 0.25 * verify_score
        )

        # Clamp to [0, 1]
        composite = max(0.0, min(1.0, composite))

        result = _D(str(round(composite, 4)))

        self._record_metric("dqi_score")

        logger.debug(
            "DQI score: ef_src=%s (%.2f), ef_age=%d (%.2f), "
            "act_src=%s (%.2f), verified=%s (%.2f) -> composite=%.4f",
            ef_source, ef_src_score, age, ef_age_score,
            activity_source, act_src_score, instrument_verified,
            verify_score, float(result),
        )
        return result

    # ==================================================================
    # Public API -- DQI Score to Uncertainty
    # ==================================================================

    def score_to_uncertainty(self, dqi_score: Decimal) -> Decimal:
        """Map a DQI score (0-1) to an uncertainty percentage.

        The mapping uses an inverse linear relationship where higher
        DQI scores (better quality) produce lower uncertainty:

            uncertainty = 0.35 - 0.30 * dqi_score

        This maps (market-based calibration):
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
        # Clamp to reasonable range [0.02, 0.50]
        uncertainty = max(0.02, min(0.50, uncertainty))
        return _D(str(round(uncertainty, 4)))

    # ==================================================================
    # Public API -- Sensitivity Analysis
    # ==================================================================

    def sensitivity_analysis(
        self,
        base: Decimal,
        parameters: Dict[str, Decimal],
        variation_pct: Decimal = Decimal("0.10"),
    ) -> Dict[str, Any]:
        """One-at-a-time sensitivity analysis for market-based calculations.

        Varies each parameter by +/- *variation_pct* while holding all
        others constant, and measures the impact on the emission result.
        The sensitivity coefficient is defined as:
            S_i = (result_high - result_low) / (2 * variation * base)

        Typical parameters for market-based analysis:
            - instrument_ef: Contractual instrument emission factor
            - residual_mix_ef: Residual mix emission factor
            - consumption_covered: Instrument-covered consumption
            - consumption_uncovered: Residual-mix-covered consumption
            - coverage_ratio: Instrument coverage ratio

        Args:
            base: Central emission estimate (kg or tCO2e).
            parameters: Dictionary mapping parameter names to their
                central values.  Each parameter is assumed to enter
                the calculation multiplicatively.
            variation_pct: Fraction to vary each parameter (e.g. 0.10
                for +/-10%).

        Returns:
            Dictionary with keys:
                - calculation_id: UUID
                - status: "SUCCESS"
                - method: "SENSITIVITY_ANALYSIS"
                - base_result: Decimal
                - variation_pct: Decimal
                - parameters: dict of parameter_name -> {base, low, high,
                  result_low, result_high, sensitivity_coefficient, impact_range}
                - ranked_parameters: list sorted by descending impact
                - provenance_hash: SHA-256 hex digest
        """
        self._increment("_total_analyses")
        self._increment("_total_sensitivity")
        start_time = time.monotonic()
        calc_id = str(uuid4())

        base_float = float(base)
        var_float = float(variation_pct)
        param_results: Dict[str, Dict[str, Any]] = {}

        for name, value in parameters.items():
            val_float = float(value)
            if val_float == 0.0:
                param_results[name] = {
                    "base": value,
                    "low": value,
                    "high": value,
                    "result_low": base,
                    "result_high": base,
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

        # Rank parameters by descending impact range
        ranked = sorted(
            param_results.keys(),
            key=lambda k: float(param_results[k].get("impact_range", _ZERO)),
            reverse=True,
        )

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "SENSITIVITY_ANALYSIS",
            "base_result": base,
            "variation_pct": variation_pct,
            "parameters": param_results,
            "ranked_parameters": ranked,
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
        base: Decimal,
        parameters: Dict[str, Decimal],
        variation_pct: Decimal = Decimal("0.10"),
    ) -> Dict[str, Any]:
        """Generate tornado chart data sorted by impact magnitude.

        Runs a sensitivity analysis with +/- variation on each
        parameter, then sorts the results by descending impact range
        for tornado chart visualisation.

        Args:
            base: Central emission estimate.
            parameters: Dictionary mapping parameter names to values.
            variation_pct: Fraction to vary each parameter.
                Default: 0.10 (+/-10%).

        Returns:
            Dictionary with keys:
                - calculation_id: UUID
                - status: "SUCCESS"
                - method: "TORNADO_ANALYSIS"
                - base_result: Decimal
                - bars: List of dicts sorted by descending impact_range,
                  each containing parameter, result_low, result_high,
                  impact_range, sensitivity_coefficient, contribution_pct
                - provenance_hash: SHA-256 hex digest
        """
        self._increment("_total_analyses")
        self._increment("_total_sensitivity")
        start_time = time.monotonic()
        calc_id = str(uuid4())

        # Run base sensitivity analysis
        sa = self.sensitivity_analysis(
            base=base,
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
            # Remove intermediate field
            del item["variance_contribution"]

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "TORNADO_ANALYSIS",
            "base_result": base,
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
        confidence_level: Decimal = Decimal("0.95"),
    ) -> Dict[str, Decimal]:
        """Calculate a confidence interval from a list of Decimal values.

        Uses the percentile method on sorted values.

        Args:
            values: List of Decimal values.
            confidence_level: Confidence level as a fraction (e.g. 0.95).
                Default: 0.95.

        Returns:
            Dictionary with keys:
                - ci_lower: Decimal lower bound
                - ci_upper: Decimal upper bound
                - confidence_level: Decimal
                - width: Decimal (ci_upper - ci_lower)
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
            mean: Desired arithmetic mean of the lognormal.  Must be > 0.
            std: Desired arithmetic standard deviation.  Must be > 0.
            n: Number of samples to generate.
            seed: Optional PRNG seed for reproducibility.

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
    # Public API -- Validation Helpers
    # ==================================================================

    def validate_uncertainty_input(
        self,
        base: Decimal,
        uncertainty: Decimal,
    ) -> List[str]:
        """Validate Monte Carlo / analytical uncertainty inputs.

        Checks:
            - base must be non-negative
            - uncertainty must be in [0, 10.0] (0% to 1000%)

        Args:
            base: Base emission value (kg or tCO2e).
            uncertainty: Relative uncertainty as a fraction.

        Returns:
            List of validation error strings (empty if valid).
        """
        errors: List[str] = []

        base_d = _safe_decimal(base, Decimal("-1"))
        if base_d < _ZERO:
            errors.append("base_emissions must be non-negative")

        unc_d = _safe_decimal(uncertainty, Decimal("-1"))
        if unc_d < _ZERO:
            errors.append("uncertainty must be non-negative")
        elif unc_d > Decimal("10"):
            errors.append(
                "uncertainty must be <= 10.0 (1000%); "
                f"got {uncertainty}"
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
            errors.append(f"iterations must be >= 100; got {n}")
        if n > 1_000_000:
            errors.append(f"iterations must be <= 1,000,000; got {n}")
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
                "agent": "AGENT-MRV-010",
                "component": "Scope 2 Market-Based Emissions",
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
        logger.info("UncertaintyQuantifierEngine counters reset (market-based)")


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    "UncertaintyQuantifierEngine",
    # Instrument uncertainty constants
    "INSTRUMENT_UNCERTAINTY",
    # Residual mix uncertainty constants
    "RESIDUAL_MIX_UNCERTAINTY",
    # Activity data uncertainty constants
    "ACTIVITY_DATA_UNCERTAINTY",
    # Coverage allocation uncertainty constants
    "COVERAGE_ALLOCATION_UNCERTAINTY",
    # Supplier disclosure uncertainty constants
    "SUPPLIER_DISCLOSURE_UNCERTAINTY",
    # Per-gas EF uncertainty constants
    "PER_GAS_EF_UNCERTAINTY",
    # DQI scoring constants
    "EF_SOURCE_SCORES",
    "ACTIVITY_SOURCE_SCORES",
    "INSTRUMENT_VERIFICATION_SCORES",
]
