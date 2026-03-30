# -*- coding: utf-8 -*-
"""
UncertaintyQuantifierEngine - Monte Carlo & DQI Scoring (Engine 5 of 7)

AGENT-MRV-008: Agricultural Emissions Agent

Quantifies the uncertainty of agricultural emission calculations using
Monte Carlo simulation, analytical error propagation (IPCC Approach 1),
data quality indicator (DQI) scoring, and sensitivity analysis per
IPCC 2006 Guidelines Volume 4 (Agriculture, Forestry and Other Land Use)
uncertainty guidance.

Agriculture-Specific Uncertainty Ranges (coefficient of variation):
    - Enteric CH4 EF:           30% (Tier 1), 20% (Tier 2), 10% (Tier 3)
    - Ym factor (% GE as CH4):  20% (Tier 1), 15% (Tier 2),  8% (Tier 3)
    - Gross energy intake:      15% (Tier 1), 10% (Tier 2),  5% (Tier 3)
    - Manure VS excretion:      25% (Tier 1), 15% (Tier 2),  8% (Tier 3)
    - Manure Bo (CH4 max):      30% (Tier 1), 20% (Tier 2), 10% (Tier 3)
    - Manure MCF:               40% (Tier 1), 25% (Tier 2), 15% (Tier 3)
    - N excretion rate:         30% (Tier 1), 20% (Tier 2), 10% (Tier 3)
    - Manure N2O EF:            50% (Tier 1), 30% (Tier 2), 15% (Tier 3)
    - Soil direct N2O EF1:      50% (Tier 1), 30% (Tier 2), 15% (Tier 3)
    - Volatilization fraction:  40% (Tier 1), 25% (Tier 2), 15% (Tier 3)
    - Leaching fraction:        50% (Tier 1), 30% (Tier 2), 20% (Tier 3)
    - Rice baseline EF:         45% (Tier 1), 30% (Tier 2), 15% (Tier 3)
    - Rice scaling factors:     30% (Tier 1), 20% (Tier 2), 10% (Tier 3)
    - Field burning EF:         40% (Tier 1), 25% (Tier 2), 15% (Tier 3)
    - Head count / area:         5% (Tier 1),  3% (Tier 2),  1% (Tier 3)

DQI Scoring (5 dimensions, 1-5 scale):
    - Reliability: direct measurement (1) ... global default (5)
    - Completeness: >95% coverage (1) ... <40% coverage (5)
    - Temporal correlation: same year (1) ... >10 years old (5)
    - Geographical correlation: same region (1) ... global default (5)
    - Technological correlation: breed-specific (1) ... generic (5)

    Composite score: geometric mean of 5 dimension scores.

Monte Carlo Simulation:
    - Configurable iterations (default 5000)
    - Lognormal distributions for emission factors (non-negative, right-skewed)
    - Normal distributions for activity data (bounded at zero)
    - Uniform distributions for fractions (volatilization, leaching)
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
    >>> from greenlang.agents.mrv.agricultural_emissions.uncertainty_quantifier import (
    ...     UncertaintyQuantifierEngine,
    ... )
    >>> engine = UncertaintyQuantifierEngine()
    >>> result = engine.quantify_uncertainty(
    ...     calculation_input={
    ...         "total_co2e_kg": 125000,
    ...         "emission_source": "enteric_fermentation",
    ...         "tier": "TIER_1",
    ...         "parameters": [
    ...             {"name": "enteric_ef", "value": 66.0},
    ...             {"name": "head_count", "value": 500},
    ...         ],
    ...     },
    ...     method="monte_carlo",
    ...     n_iterations=5000,
    ... )
    >>> print(result["confidence_intervals"]["95"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 Agricultural Emissions (GL-MRV-SCOPE1-008)
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
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["UncertaintyQuantifierEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.agricultural_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.agricultural_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.agricultural_emissions.metrics import (
        record_uncertainty_run as _record_uncertainty_run,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_uncertainty_run = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash. If a Pydantic model, uses ``model_dump``.

    Returns:
        SHA-256 hex digest string.
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
    """Convert a value to Decimal.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation of the value.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))

def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert to Decimal, returning *default* on failure.

    Args:
        value: Value to convert.
        default: Fallback Decimal value.

    Returns:
        Converted Decimal or *default*.
    """
    if value is None:
        return default
    try:
        return _D(value)
    except (InvalidOperation, ValueError, TypeError):
        return default

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
    if value < _ZERO:
        raise ValueError(f"Cannot take sqrt of negative value: {value}")
    if value == _ZERO:
        return _ZERO
    return _D(str(math.sqrt(float(value))))

# ===========================================================================
# Enumerations
# ===========================================================================

class AgriculturalSourceCategory(str, Enum):
    """Agricultural emission source categories for uncertainty lookup."""

    ENTERIC_FERMENTATION = "ENTERIC_FERMENTATION"
    MANURE_MANAGEMENT = "MANURE_MANAGEMENT"
    RICE_CULTIVATION = "RICE_CULTIVATION"
    AGRICULTURAL_SOILS = "AGRICULTURAL_SOILS"
    FIELD_BURNING = "FIELD_BURNING"
    LIMING = "LIMING"
    UREA_APPLICATION = "UREA_APPLICATION"

class AgriculturalCalculationTier(str, Enum):
    """IPCC calculation tier for emission factor specificity.

    TIER_3: Facility-/farm-specific, measured emission factors.
    TIER_2: Country- or region-specific emission factors.
    TIER_1: IPCC default emission factors.
    """

    TIER_3 = "TIER_3"
    TIER_2 = "TIER_2"
    TIER_1 = "TIER_1"

class AgriculturalCalculationMethod(str, Enum):
    """Agricultural emission calculation method types."""

    IPCC_TIER1 = "IPCC_TIER1"
    IPCC_TIER2 = "IPCC_TIER2"
    IPCC_TIER3 = "IPCC_TIER3"
    COUNTRY_SPECIFIC = "COUNTRY_SPECIFIC"
    DIRECT_MEASUREMENT = "DIRECT_MEASUREMENT"
    EMISSION_FACTOR = "EMISSION_FACTOR"
    MASS_BALANCE = "MASS_BALANCE"

class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions (ISO 14044 / GHG Protocol)."""

    RELIABILITY = "reliability"
    COMPLETENESS = "completeness"
    TEMPORAL_CORRELATION = "temporal_correlation"
    GEOGRAPHICAL_CORRELATION = "geographical_correlation"
    TECHNOLOGICAL_CORRELATION = "technological_correlation"

# ===========================================================================
# Z-Scores for Common Confidence Levels
# ===========================================================================

#: Standard normal z-scores for commonly requested confidence levels.
#: Used in analytical error propagation and CI calculations.
Z_SCORES: Dict[int, float] = {
    90: 1.645,
    95: 1.960,
    99: 2.576,
}

# ===========================================================================
# PARAMETER_DISTRIBUTIONS - Agriculture-Specific (15 entries)
# ===========================================================================

#: Per-parameter distribution specifications for Monte Carlo simulation.
#: Each entry maps a parameter name to its distribution type and
#: coefficient of variation (CV) by IPCC tier level.
#:
#: Sources:
#:   - IPCC 2006 Guidelines Vol 4, Ch 10 (Enteric Fermentation)
#:   - IPCC 2006 Guidelines Vol 4, Ch 10 (Manure Management)
#:   - IPCC 2006 Guidelines Vol 4, Ch 11 (N2O from Managed Soils)
#:   - IPCC 2006 Guidelines Vol 4, Ch 5 (Rice Cultivation)
#:   - IPCC 2006 Guidelines Vol 4, Ch 2 (Field Burning)
#:   - IPCC 2006 GPG, Table 3.2 (Uncertainty ranges)
PARAMETER_DISTRIBUTIONS: Dict[str, Dict[str, Any]] = {
    "enteric_ef": {
        "distribution": "lognormal",
        "description": "Enteric fermentation CH4 emission factor (kg CH4/head/yr)",
        "cv_by_tier": {"TIER_1": 0.30, "TIER_2": 0.20, "TIER_3": 0.10},
        "unit": "kg CH4/head/yr",
        "source": "IPCC 2006 Vol 4 Ch 10 Table 10.11",
    },
    "ym_factor": {
        "distribution": "normal",
        "description": "Methane conversion factor Ym (fraction of GE lost as CH4)",
        "cv_by_tier": {"TIER_1": 0.20, "TIER_2": 0.15, "TIER_3": 0.08},
        "unit": "fraction",
        "source": "IPCC 2006 Vol 4 Ch 10 Table 10.12",
    },
    "gross_energy": {
        "distribution": "normal",
        "description": "Gross energy intake (MJ/head/day)",
        "cv_by_tier": {"TIER_1": 0.15, "TIER_2": 0.10, "TIER_3": 0.05},
        "unit": "MJ/head/day",
        "source": "IPCC 2006 Vol 4 Ch 10 Eq 10.16",
    },
    "manure_vs": {
        "distribution": "normal",
        "description": "Volatile solids excretion rate (kg VS/head/day)",
        "cv_by_tier": {"TIER_1": 0.25, "TIER_2": 0.15, "TIER_3": 0.08},
        "unit": "kg VS/head/day",
        "source": "IPCC 2006 Vol 4 Ch 10 Table 10.13A",
    },
    "manure_bo": {
        "distribution": "lognormal",
        "description": "Maximum methane producing capacity Bo (m3 CH4/kg VS)",
        "cv_by_tier": {"TIER_1": 0.30, "TIER_2": 0.20, "TIER_3": 0.10},
        "unit": "m3 CH4/kg VS",
        "source": "IPCC 2006 Vol 4 Ch 10 Table 10.16",
    },
    "manure_mcf": {
        "distribution": "lognormal",
        "description": "Methane conversion factor for manure management system",
        "cv_by_tier": {"TIER_1": 0.40, "TIER_2": 0.25, "TIER_3": 0.15},
        "unit": "fraction",
        "source": "IPCC 2006 Vol 4 Ch 10 Table 10.17",
    },
    "n_excretion": {
        "distribution": "normal",
        "description": "Nitrogen excretion rate (kg N/head/yr)",
        "cv_by_tier": {"TIER_1": 0.30, "TIER_2": 0.20, "TIER_3": 0.10},
        "unit": "kg N/head/yr",
        "source": "IPCC 2006 Vol 4 Ch 10 Table 10.19",
    },
    "n2o_ef_manure": {
        "distribution": "lognormal",
        "description": "N2O emission factor for manure management (kg N2O-N/kg N)",
        "cv_by_tier": {"TIER_1": 0.50, "TIER_2": 0.30, "TIER_3": 0.15},
        "unit": "kg N2O-N/kg N",
        "source": "IPCC 2006 Vol 4 Ch 10 Table 10.21",
    },
    "soil_n2o_ef1": {
        "distribution": "lognormal",
        "description": "Direct N2O emission factor EF1 for managed soils",
        "cv_by_tier": {"TIER_1": 0.50, "TIER_2": 0.30, "TIER_3": 0.15},
        "unit": "kg N2O-N/kg N input",
        "source": "IPCC 2006 Vol 4 Ch 11 Table 11.1",
    },
    "volatilization_frac": {
        "distribution": "uniform",
        "description": "Fraction of applied N that volatilises as NH3 and NOx",
        "cv_by_tier": {"TIER_1": 0.40, "TIER_2": 0.25, "TIER_3": 0.15},
        "unit": "fraction",
        "source": "IPCC 2006 Vol 4 Ch 11 Table 11.3",
    },
    "leaching_frac": {
        "distribution": "uniform",
        "description": "Fraction of applied N lost through leaching and runoff",
        "cv_by_tier": {"TIER_1": 0.50, "TIER_2": 0.30, "TIER_3": 0.20},
        "unit": "fraction",
        "source": "IPCC 2006 Vol 4 Ch 11 Table 11.3",
    },
    "rice_baseline_ef": {
        "distribution": "lognormal",
        "description": "Baseline CH4 emission factor for rice cultivation",
        "cv_by_tier": {"TIER_1": 0.45, "TIER_2": 0.30, "TIER_3": 0.15},
        "unit": "kg CH4/ha/day",
        "source": "IPCC 2006 Vol 4 Ch 5 Table 5.11",
    },
    "rice_scaling_factor": {
        "distribution": "normal",
        "description": "Rice cultivation scaling factor (water regime, organic amendment)",
        "cv_by_tier": {"TIER_1": 0.30, "TIER_2": 0.20, "TIER_3": 0.10},
        "unit": "dimensionless",
        "source": "IPCC 2006 Vol 4 Ch 5 Tables 5.12-5.14",
    },
    "field_burning_ef": {
        "distribution": "lognormal",
        "description": "Emission factor for field burning of agricultural residues",
        "cv_by_tier": {"TIER_1": 0.40, "TIER_2": 0.25, "TIER_3": 0.15},
        "unit": "g gas/kg dry matter burned",
        "source": "IPCC 2006 Vol 4 Ch 2 Table 2.5",
    },
    "head_count": {
        "distribution": "normal",
        "description": "Livestock head count or cultivated area (ha)",
        "cv_by_tier": {"TIER_1": 0.05, "TIER_2": 0.03, "TIER_3": 0.01},
        "unit": "head or ha",
        "source": "IPCC 2006 Vol 4 Ch 10 / national statistics",
    },
}

# ===========================================================================
# DEFAULT_CV - Flattened CV percentages (18 parameters x 3 tiers)
# ===========================================================================

#: Default coefficient of variation (CV%) by parameter name and tier.
#: Values expressed as percentages (e.g. 30.0 means 30% CV).
#: Includes the 15 core PARAMETER_DISTRIBUTIONS entries plus 3 additional
#: commonly referenced agricultural parameters.
DEFAULT_CV: Dict[str, Dict[str, float]] = {
    # --- Enteric fermentation ---
    "ENTERIC_EF": {"TIER_1": 30.0, "TIER_2": 20.0, "TIER_3": 10.0},
    "YM_FACTOR": {"TIER_1": 20.0, "TIER_2": 15.0, "TIER_3": 8.0},
    "GROSS_ENERGY": {"TIER_1": 15.0, "TIER_2": 10.0, "TIER_3": 5.0},
    # --- Manure management ---
    "MANURE_VS": {"TIER_1": 25.0, "TIER_2": 15.0, "TIER_3": 8.0},
    "MANURE_BO": {"TIER_1": 30.0, "TIER_2": 20.0, "TIER_3": 10.0},
    "MANURE_MCF": {"TIER_1": 40.0, "TIER_2": 25.0, "TIER_3": 15.0},
    "N_EXCRETION": {"TIER_1": 30.0, "TIER_2": 20.0, "TIER_3": 10.0},
    "N2O_EF_MANURE": {"TIER_1": 50.0, "TIER_2": 30.0, "TIER_3": 15.0},
    # --- Agricultural soils ---
    "SOIL_N2O_EF1": {"TIER_1": 50.0, "TIER_2": 30.0, "TIER_3": 15.0},
    "VOLATILIZATION_FRAC": {"TIER_1": 40.0, "TIER_2": 25.0, "TIER_3": 15.0},
    "LEACHING_FRAC": {"TIER_1": 50.0, "TIER_2": 30.0, "TIER_3": 20.0},
    # --- Rice cultivation ---
    "RICE_BASELINE_EF": {"TIER_1": 45.0, "TIER_2": 30.0, "TIER_3": 15.0},
    "RICE_SCALING_FACTOR": {"TIER_1": 30.0, "TIER_2": 20.0, "TIER_3": 10.0},
    # --- Field burning ---
    "FIELD_BURNING_EF": {"TIER_1": 40.0, "TIER_2": 25.0, "TIER_3": 15.0},
    # --- Activity data ---
    "HEAD_COUNT": {"TIER_1": 5.0, "TIER_2": 3.0, "TIER_3": 1.0},
    # --- Additional commonly used parameters ---
    "GWP": {"TIER_1": 10.0, "TIER_2": 10.0, "TIER_3": 10.0},
    "LIMING_EF": {"TIER_1": 25.0, "TIER_2": 15.0, "TIER_3": 8.0},
    "UREA_EF": {"TIER_1": 20.0, "TIER_2": 12.0, "TIER_3": 5.0},
}

# ===========================================================================
# DQI Scoring Reference
# ===========================================================================

#: DQI scoring matrix: {dimension: [(label, score), ...]}
#: Lower score = better quality (1 = best, 5 = worst).
DQI_SCORING_MATRIX: Dict[str, List[Tuple[str, int]]] = {
    "reliability": [
        ("verified_direct_measurement", 1),
        ("direct_measurement", 2),
        ("calculated_from_measurements", 3),
        ("literature_estimate", 4),
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
        ("within_3_years", 2),
        ("within_5_years", 3),
        ("within_10_years", 4),
        ("older_than_10_years", 5),
    ],
    "geographical_correlation": [
        ("same_farm", 1),
        ("same_region", 2),
        ("same_country", 3),
        ("similar_climate", 4),
        ("global_default", 5),
    ],
    "technological_correlation": [
        ("same_breed_crop", 1),
        ("same_species_type", 2),
        ("similar_species_type", 3),
        ("related_category", 4),
        ("generic", 5),
    ],
}

#: Map DQI quality category to approximate uncertainty multiplier.
#: Applied to base uncertainty ranges based on data quality assessment.
DQI_UNCERTAINTY_MULTIPLIERS: Dict[str, float] = {
    "EXCELLENT": 0.8,
    "GOOD": 1.0,
    "FAIR": 1.3,
    "POOR": 1.8,
    "VERY_POOR": 2.5,
}

# ===========================================================================
# Source-specific uncertainty ranges
# ===========================================================================

#: Combined uncertainty ranges (half-width of 95% CI as percentage) by
#: agricultural source category and calculation tier. Sourced from IPCC
#: 2006 Guidelines Vol 4 and IPCC GPG for LULUCF.
SOURCE_UNCERTAINTY_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "ENTERIC_FERMENTATION": {
        "TIER_1": (20.0, 50.0),
        "TIER_2": (10.0, 30.0),
        "TIER_3": (5.0, 15.0),
    },
    "MANURE_MANAGEMENT": {
        "TIER_1": (30.0, 100.0),
        "TIER_2": (15.0, 50.0),
        "TIER_3": (8.0, 25.0),
    },
    "RICE_CULTIVATION": {
        "TIER_1": (30.0, 80.0),
        "TIER_2": (20.0, 50.0),
        "TIER_3": (10.0, 25.0),
    },
    "AGRICULTURAL_SOILS": {
        "TIER_1": (50.0, 150.0),
        "TIER_2": (25.0, 75.0),
        "TIER_3": (10.0, 30.0),
    },
    "FIELD_BURNING": {
        "TIER_1": (30.0, 80.0),
        "TIER_2": (20.0, 50.0),
        "TIER_3": (10.0, 25.0),
    },
    "LIMING": {
        "TIER_1": (15.0, 40.0),
        "TIER_2": (10.0, 25.0),
        "TIER_3": (5.0, 15.0),
    },
    "UREA_APPLICATION": {
        "TIER_1": (15.0, 35.0),
        "TIER_2": (10.0, 20.0),
        "TIER_3": (5.0, 12.0),
    },
}

# ===========================================================================
# Default parameter uncertainties (flat lookup for analytical propagation)
# ===========================================================================

#: Default uncertainty percentages for individual parameters when
#: agriculture-specific values are not provided in calculation_input.
DEFAULT_PARAMETER_UNCERTAINTIES: Dict[str, float] = {
    "emission_factor": 50.0,
    "activity_data": 10.0,
    "head_count": 5.0,
    "area_ha": 5.0,
    "gwp": 10.0,
    "feed_intake": 15.0,
    "ym_factor": 20.0,
    "volatile_solids": 25.0,
    "bo_factor": 30.0,
    "mcf_factor": 40.0,
    "n_excretion": 30.0,
    "n2o_emission_factor": 50.0,
    "volatilization_fraction": 40.0,
    "leaching_fraction": 50.0,
    "rice_ef": 45.0,
    "burning_ef": 40.0,
    "cultivation_period": 10.0,
    "residue_mass": 20.0,
    "combustion_factor": 25.0,
}

# ===========================================================================
# UncertaintyQuantifierEngine
# ===========================================================================

class UncertaintyQuantifierEngine:
    """Uncertainty quantification engine for agricultural emissions.

    Provides Monte Carlo simulation, analytical error propagation (IPCC
    Approach 1), DQI scoring, sensitivity analysis, confidence interval
    calculations, and parameter distribution lookup for agricultural
    emission estimates per IPCC 2006 Vol 4 uncertainty guidance.

    All numeric calculations are deterministic when seeded (zero-hallucination).
    Thread-safe via reentrant lock with per-call Random instances.

    Attributes:
        config: Configuration dictionary.

    Example:
        >>> engine = UncertaintyQuantifierEngine()
        >>> result = engine.quantify_uncertainty({
        ...     "total_co2e_kg": 125000,
        ...     "emission_source": "enteric_fermentation",
        ...     "tier": "TIER_1",
        ... })
        >>> assert result["status"] == "SUCCESS"
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the UncertaintyQuantifierEngine.

        Args:
            config: Optional configuration dictionary. Recognised keys:
                - monte_carlo_iterations (int): Default iterations.
                - monte_carlo_seed (int): Default random seed.
                - confidence_levels (str): Comma-separated levels.
        """
        self._config = config or {}
        self._lock = threading.RLock()
        self._created_at = utcnow()

        self._default_iterations: int = int(
            self._config.get("monte_carlo_iterations", 5000),
        )
        self._default_seed: int = int(
            self._config.get("monte_carlo_seed", 42),
        )
        self._default_confidence_levels: List[float] = [90.0, 95.0, 99.0]

        # Parse confidence levels from config string
        cl_str = self._config.get("confidence_levels", "90,95,99")
        if isinstance(cl_str, str):
            try:
                self._default_confidence_levels = [
                    float(x.strip()) for x in cl_str.split(",")
                    if x.strip()
                ]
            except ValueError:
                pass  # Keep defaults

        # Statistics counters (thread-safe via _lock)
        self._total_mc_runs: int = 0
        self._total_analytical_runs: int = 0
        self._total_dqi_scores: int = 0
        self._total_sensitivity_runs: int = 0
        self._total_parameter_lookups: int = 0
        self._total_combine_runs: int = 0

        logger.info(
            "UncertaintyQuantifierEngine initialized: "
            "iterations=%d, seed=%d, confidence_levels=%s",
            self._default_iterations,
            self._default_seed,
            self._default_confidence_levels,
        )

    # ------------------------------------------------------------------
    # Public API 1: Unified quantification dispatch
    # ------------------------------------------------------------------

    def quantify_uncertainty(
        self,
        calculation_input: Dict[str, Any],
        method: str = "monte_carlo",
        n_iterations: Optional[int] = None,
        seed: Optional[int] = None,
        confidence_level: int = 95,
    ) -> Dict[str, Any]:
        """Quantify uncertainty of an agricultural emission calculation.

        Dispatches to Monte Carlo, analytical, or DQI method based on
        the *method* parameter.

        Args:
            calculation_input: Calculation data dictionary containing:
                - total_co2e_kg (float): Point estimate in kg CO2e.
                - emission_source (str): AgriculturalSourceCategory value.
                - tier (str): AgriculturalCalculationTier value.
                - parameters (list): Parameter specifications for MC.
                - Optional: DQI dimension scores, parameter uncertainties.
            method: "monte_carlo", "analytical", or "dqi".
            n_iterations: Monte Carlo iterations (default from config).
            seed: Random seed for reproducibility (default from config).
            confidence_level: Confidence level percentage (default 95).

        Returns:
            Dictionary with uncertainty characterisation including
            provenance_hash and processing_time_ms.
        """
        start_time = time.monotonic()
        calc_id = str(uuid4())

        # Validate method
        valid_methods = ("monte_carlo", "analytical", "dqi")
        method_lower = method.lower().strip()

        if method_lower not in valid_methods:
            return self._error_result(
                calc_id,
                start_time,
                [f"Invalid method '{method}'. Must be one of: {valid_methods}"],
            )

        try:
            if method_lower == "analytical":
                return self.analytical_uncertainty(
                    calculation_input=calculation_input,
                    confidence_level=confidence_level,
                )
            elif method_lower == "dqi":
                return self.calculate_dqi(calculation_input)
            else:
                return self.monte_carlo_simulation(
                    calculation_input=calculation_input,
                    n_iterations=n_iterations,
                    seed=seed,
                    confidence_levels=[
                        float(confidence_level),
                    ] + [
                        cl for cl in self._default_confidence_levels
                        if cl != float(confidence_level)
                    ],
                )
        except Exception as exc:
            logger.error(
                "quantify_uncertainty failed: %s", exc, exc_info=True,
            )
            return self._error_result(
                calc_id,
                start_time,
                [f"Processing error: {str(exc)}"],
            )

    # ------------------------------------------------------------------
    # Public API 2: Monte Carlo Simulation
    # ------------------------------------------------------------------

    def monte_carlo_simulation(
        self,
        calculation_input: Dict[str, Any],
        n_iterations: Optional[int] = None,
        seed: Optional[int] = None,
        confidence_levels: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for uncertainty quantification.

        Draws from parameterised distributions for each uncertain parameter
        (emission factors, activity data, livestock/crop parameters) and
        produces percentile tables and confidence intervals.

        Distributions (agriculture-specific):
            - Emission factors (enteric_ef, manure_bo, manure_mcf, n2o_ef_manure,
              soil_n2o_ef1, rice_baseline_ef, field_burning_ef): lognormal
            - Activity data (ym_factor, gross_energy, manure_vs, n_excretion,
              rice_scaling_factor, head_count): normal (bounded at zero)
            - Fractions (volatilization_frac, leaching_frac): uniform

        Args:
            calculation_input: Calculation data with total_co2e_kg,
                emission_source, tier, and parameter details.
            n_iterations: Number of MC iterations.
            seed: Random seed for reproducibility.
            confidence_levels: Confidence levels for intervals.

        Returns:
            Dictionary with mean, std_dev, cv, percentiles (p5/p10/p25/
            p50/p75/p90/p95), confidence intervals, parameter statistics,
            and provenance_hash.
        """
        t0 = time.monotonic()
        calc_id = str(uuid4())

        iterations = n_iterations or self._default_iterations
        rng_seed = seed if seed is not None else self._default_seed
        cl = confidence_levels or self._default_confidence_levels

        # Extract inputs
        total_co2e = float(calculation_input.get("total_co2e_kg", 0))
        emission_source = str(
            calculation_input.get("emission_source", "enteric_fermentation"),
        ).upper()
        tier = str(calculation_input.get("tier", "TIER_1")).upper()
        parameters = calculation_input.get("parameters", [])

        # -- Validation ---------------------------------------------------
        errors: List[str] = []
        if total_co2e == 0:
            errors.append("total_co2e_kg must be non-zero")
        if iterations <= 0:
            errors.append("n_iterations must be > 0")
        if iterations > 1_000_000:
            errors.append("n_iterations must be <= 1000000")

        if errors:
            return self._error_result(calc_id, t0, errors)

        # -- Build parameter distributions --------------------------------
        params = self._build_parameter_distributions(
            parameters, emission_source, tier,
        )

        if not params:
            # No explicit parameters: use source-level uncertainty
            source_range = self._get_source_uncertainty_range(
                emission_source, tier,
            )
            params = self._build_source_level_distributions(
                emission_source, tier, source_range,
            )

        # -- Generate per-parameter samples -------------------------------
        rng = random.Random(rng_seed)
        param_samples: Dict[str, List[float]] = {}

        for pname, pinfo in params.items():
            value = float(pinfo.get("value", 1.0))
            cv = float(pinfo.get("cv", 0.30))
            dist = str(pinfo.get("distribution", "normal")).lower()
            samples = self._generate_samples(rng, value, cv, dist, iterations)
            param_samples[pname] = samples

        # -- Simulate total results (multiplicative model) ----------------
        total_results: List[float] = []

        for i in range(iterations):
            scale = 1.0
            for pname, pinfo in params.items():
                value = float(pinfo.get("value", 1.0))
                if value != 0 and pname in param_samples:
                    scale *= param_samples[pname][i] / value
            total_results.append(total_co2e * scale)

        # -- Compute statistics -------------------------------------------
        stats = self._compute_statistics(total_results)

        # -- Percentiles --------------------------------------------------
        percentile_points = [5, 10, 25, 50, 75, 90, 95]
        percentiles = self._calculate_percentiles(
            total_results, percentile_points,
        )

        # -- Confidence intervals -----------------------------------------
        ci = self._compute_confidence_intervals(total_results, cl, stats["mean"])

        # -- Per-parameter statistics -------------------------------------
        param_stats = self._compute_parameter_statistics(param_samples)

        # -- Record metrics -----------------------------------------------
        with self._lock:
            self._total_mc_runs += 1

        if _METRICS_AVAILABLE and _record_uncertainty_run is not None:
            try:
                _record_uncertainty_run("monte_carlo")
            except Exception:
                pass  # Non-critical

        elapsed_ms = round((time.monotonic() - t0) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "MONTE_CARLO",
            "emission_source": emission_source,
            "tier": tier,
            "n_iterations": iterations,
            "seed": rng_seed,
            "central_estimate": total_co2e,
            "statistics": {
                "mean": stats["mean"],
                "std_dev": stats["std_dev"],
                "cv": stats["cv"],
                "cv_pct": stats["cv_pct"],
                "min": stats["min"],
                "max": stats["max"],
                "median": percentiles.get("50", 0),
            },
            "percentiles": percentiles,
            "confidence_intervals": ci,
            "parameter_statistics": param_stats,
            "parameter_count": len(params),
            "processing_time_ms": elapsed_ms,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Monte Carlo complete: id=%s, source=%s, n=%d, mean=%.2f, "
            "cv=%.1f%%, 95%% CI=[%.2f, %.2f], time=%.3fms",
            calc_id, emission_source, iterations, stats["mean"],
            stats["cv_pct"],
            ci.get("95", {}).get("lower", 0),
            ci.get("95", {}).get("upper", 0),
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public API 3: Analytical Uncertainty (IPCC Approach 1)
    # ------------------------------------------------------------------

    def analytical_uncertainty(
        self,
        calculation_input: Optional[Dict[str, Any]] = None,
        base_emissions: Optional[float] = None,
        activity_uncertainty_pct: Optional[float] = None,
        ef_uncertainty_pct: Optional[float] = None,
        additional_uncertainties: Optional[List[float]] = None,
        confidence_level: int = 95,
    ) -> Dict[str, Any]:
        """Compute combined uncertainty using IPCC Approach 1 error propagation.

        For multiplicative chains (emission = EF x AD x ...):
            U_total = sqrt(U_activity^2 + U_ef^2 + U_other1^2 + ...)

        The result is a combined relative uncertainty (half-width of CI
        as percentage of the central estimate).

        Args:
            calculation_input: Calculation data with total_co2e_kg and
                parameter uncertainties. Used if explicit values not given.
            base_emissions: Central emission estimate (overrides input dict).
            activity_uncertainty_pct: Activity data uncertainty (%).
            ef_uncertainty_pct: Emission factor uncertainty (%).
            additional_uncertainties: Additional uncertainty terms (%).
            confidence_level: Confidence level (90, 95, or 99).

        Returns:
            Dictionary with combined uncertainty, CI, and provenance_hash.
        """
        t0 = time.monotonic()
        calc_id = str(uuid4())

        # Resolve values from calculation_input or explicit args
        input_data = calculation_input or {}

        total_co2e = _safe_decimal(
            base_emissions
            or input_data.get("total_co2e_kg")
            or input_data.get("base_emissions", 0),
        )

        emission_source = str(
            input_data.get("emission_source", "unknown"),
        ).upper()
        tier = str(input_data.get("tier", "TIER_1")).upper()

        # Gather parameter uncertainties
        u_activity = _safe_decimal(
            activity_uncertainty_pct
            or input_data.get("activity_uncertainty_pct")
            or DEFAULT_PARAMETER_UNCERTAINTIES.get("activity_data", 10.0),
        )
        u_ef = _safe_decimal(
            ef_uncertainty_pct
            or input_data.get("ef_uncertainty_pct")
            or DEFAULT_PARAMETER_UNCERTAINTIES.get("emission_factor", 50.0),
        )

        # Additional parameter uncertainties from input
        param_uncertainties = self._extract_parameter_uncertainties(input_data)

        # Include any additional explicit uncertainties
        all_uncertainties: List[Decimal] = [u_activity, u_ef]
        for u_val in (additional_uncertainties or []):
            all_uncertainties.append(_safe_decimal(u_val))

        # Add extracted parameter-level uncertainties
        for _pname, u_val in param_uncertainties.items():
            if u_val not in (float(u_activity), float(u_ef)):
                all_uncertainties.append(_safe_decimal(u_val))

        # Root-sum-of-squares (multiplicative model)
        sum_u_sq = _ZERO
        for u in all_uncertainties:
            sum_u_sq += (u / _HUNDRED) ** 2

        combined_pct = (
            _D(str(math.sqrt(float(sum_u_sq)))) * _HUNDRED
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        combined_abs = (
            total_co2e * combined_pct / _HUNDRED
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Confidence interval using z-score
        z = _D(str(Z_SCORES.get(confidence_level, 1.960)))
        ci_lower = (total_co2e - combined_abs * z).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        ci_upper = (total_co2e + combined_abs * z).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )

        # Record metrics
        with self._lock:
            self._total_analytical_runs += 1

        if _METRICS_AVAILABLE and _record_uncertainty_run is not None:
            try:
                _record_uncertainty_run("analytical")
            except Exception:
                pass

        elapsed_ms = round((time.monotonic() - t0) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "ANALYTICAL_ERROR_PROPAGATION",
            "emission_source": emission_source,
            "tier": tier,
            "central_estimate": str(total_co2e),
            "combined_uncertainty_pct": str(combined_pct),
            "combined_uncertainty_abs": str(combined_abs),
            "confidence_level": confidence_level,
            "confidence_interval": {
                "lower": str(ci_lower),
                "upper": str(ci_upper),
            },
            "parameter_uncertainties": {
                "activity_data_pct": str(u_activity),
                "emission_factor_pct": str(u_ef),
                "additional_count": len(all_uncertainties) - 2,
            },
            "component_uncertainties_pct": [
                str(u) for u in all_uncertainties
            ],
            "parameter_count": len(all_uncertainties),
            "processing_time_ms": elapsed_ms,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Analytical uncertainty: id=%s, combined=%.1f%%, "
            "%d%% CI=[%s, %s], time=%.3fms",
            calc_id, float(combined_pct), confidence_level,
            ci_lower, ci_upper, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public API 4: DQI Scoring
    # ------------------------------------------------------------------

    def calculate_dqi(
        self,
        scores_or_input: Dict[str, Any],
        reliability: Optional[int] = None,
        completeness: Optional[int] = None,
        temporal: Optional[int] = None,
        geographical: Optional[int] = None,
        technological: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculate Data Quality Indicator (DQI) composite score.

        Uses the geometric mean of 5 dimension scores (1-5 scale) to
        produce a single composite quality indicator per the GHG Protocol
        guidance on data quality assessment.

        Scores can be provided either as explicit keyword arguments or
        within the *scores_or_input* dictionary (the dictionary approach
        also accepts a ``dqi_scores`` sub-key).

        Args:
            scores_or_input: Dictionary with dimension scores or a
                calculation_input dict containing ``dqi_scores``.
            reliability: Reliability score 1-5 (optional override).
            completeness: Completeness score 1-5 (optional override).
            temporal: Temporal correlation score 1-5 (optional override).
            geographical: Geographical correlation score 1-5 (optional).
            technological: Technological correlation score 1-5 (optional).

        Returns:
            Dictionary with dimension_scores, composite_score,
            quality_category, uncertainty_multiplier, and provenance_hash.
        """
        t0 = time.monotonic()
        calc_id = str(uuid4())

        # Resolve scores from input
        dqi_data = scores_or_input.get("dqi_scores", scores_or_input)

        dimensions = [
            ("reliability", reliability),
            ("completeness", completeness),
            ("temporal_correlation", temporal),
            ("geographical_correlation", geographical),
            ("technological_correlation", technological),
        ]

        dimension_scores: Dict[str, int] = {}
        errors: List[str] = []

        for dim_name, explicit_val in dimensions:
            raw_value = explicit_val or dqi_data.get(dim_name, 0)

            if isinstance(raw_value, (int, float)):
                score = int(raw_value)
            elif isinstance(raw_value, str):
                score = self._label_to_score(dim_name, raw_value)
            else:
                score = 0

            if score < 1 or score > 5:
                errors.append(
                    f"{dim_name} must be 1-5, got {raw_value}"
                )
            else:
                dimension_scores[dim_name] = score

        if errors:
            return self._error_result(calc_id, t0, errors)

        # Geometric mean
        product = 1.0
        for score in dimension_scores.values():
            product *= score
        composite = product ** (1.0 / len(dimension_scores))

        # Quality category
        quality = self._composite_to_quality(composite)

        # Uncertainty multiplier
        uncertainty_multiplier = DQI_UNCERTAINTY_MULTIPLIERS.get(quality, 1.0)

        # Suggested uncertainty adjustment
        suggested_cv_adjustment = self._dqi_to_cv_adjustment(composite)

        # Record metrics
        with self._lock:
            self._total_dqi_scores += 1

        if _METRICS_AVAILABLE and _record_uncertainty_run is not None:
            try:
                _record_uncertainty_run("ipcc_default_uncertainty")
            except Exception:
                pass

        elapsed_ms = round((time.monotonic() - t0) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "DQI_SCORING",
            "dimension_scores": dimension_scores,
            "composite_score": round(composite, 4),
            "quality_category": quality,
            "uncertainty_multiplier": uncertainty_multiplier,
            "suggested_cv_adjustment": round(suggested_cv_adjustment, 4),
            "processing_time_ms": elapsed_ms,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "DQI calculated: id=%s, composite=%.2f, quality=%s, "
            "multiplier=%.2f, time=%.3fms",
            calc_id, composite, quality, uncertainty_multiplier, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public API 5: Parameter Uncertainty Lookup
    # ------------------------------------------------------------------

    def get_parameter_uncertainty(
        self,
        parameter_name: str,
        tier: str = "TIER_1",
    ) -> Dict[str, Any]:
        """Look up uncertainty distribution for an agricultural parameter.

        Returns the distribution type, coefficient of variation, and
        IPCC source reference from the PARAMETER_DISTRIBUTIONS registry.

        Args:
            parameter_name: Parameter name (e.g. "enteric_ef", "manure_mcf").
            tier: Calculation tier ("TIER_1", "TIER_2", or "TIER_3").

        Returns:
            Dictionary with distribution, cv, cv_pct, description, unit,
            source, and provenance_hash. Returns status VALIDATION_ERROR
            if parameter_name is not found.
        """
        t0 = time.monotonic()
        calc_id = str(uuid4())

        tier_upper = tier.upper().strip()
        param_lower = parameter_name.lower().strip()

        # Look up in PARAMETER_DISTRIBUTIONS
        param_info = PARAMETER_DISTRIBUTIONS.get(param_lower)

        if param_info is None:
            # Try uppercase lookup in DEFAULT_CV
            param_upper = parameter_name.upper().strip()
            cv_data = DEFAULT_CV.get(param_upper)
            if cv_data is not None:
                cv_value = cv_data.get(tier_upper, cv_data.get("TIER_1", 0.30))
                with self._lock:
                    self._total_parameter_lookups += 1

                elapsed_ms = round((time.monotonic() - t0) * 1000, 3)
                result: Dict[str, Any] = {
                    "calculation_id": calc_id,
                    "status": "SUCCESS",
                    "parameter_name": parameter_name,
                    "tier": tier_upper,
                    "distribution": "normal",
                    "cv": cv_value / 100.0,
                    "cv_pct": cv_value,
                    "description": f"Default CV for {parameter_name}",
                    "unit": "varies",
                    "source": "DEFAULT_CV lookup table",
                    "processing_time_ms": elapsed_ms,
                }
                result["provenance_hash"] = _compute_hash(result)
                return result

            return self._error_result(
                calc_id, t0,
                [f"Unknown parameter: '{parameter_name}'. Available: "
                 f"{list(PARAMETER_DISTRIBUTIONS.keys())}"],
            )

        cv_by_tier = param_info.get("cv_by_tier", {})
        cv_value = cv_by_tier.get(tier_upper, cv_by_tier.get("TIER_1", 0.30))

        with self._lock:
            self._total_parameter_lookups += 1

        elapsed_ms = round((time.monotonic() - t0) * 1000, 3)

        result = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "parameter_name": param_lower,
            "tier": tier_upper,
            "distribution": param_info.get("distribution", "normal"),
            "cv": cv_value,
            "cv_pct": round(cv_value * 100, 2),
            "description": param_info.get("description", ""),
            "unit": param_info.get("unit", ""),
            "source": param_info.get("source", ""),
            "processing_time_ms": elapsed_ms,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.debug(
            "Parameter lookup: %s (tier=%s) -> cv=%.2f (%s)",
            param_lower, tier_upper, cv_value,
            param_info.get("distribution", "normal"),
        )
        return result

    # ------------------------------------------------------------------
    # Public API 6: Combine Uncertainties
    # ------------------------------------------------------------------

    def combine_uncertainties(
        self,
        uncertainties: List[Dict[str, Any]],
        combination: str = "multiplicative",
    ) -> Dict[str, Any]:
        """Combine multiple uncertainty estimates into a single result.

        For multiplicative combination (e.g. emission = EF x AD):
            U_combined = sqrt(sum(Ui^2))
        where Ui are relative uncertainties (percentages).

        For additive combination (e.g. total = source1 + source2):
            U_combined_abs = sqrt(sum((Ui * Xi)^2))
            U_combined_pct = U_combined_abs / |sum(Xi)| * 100

        Args:
            uncertainties: List of uncertainty dictionaries, each with:
                - value (float): Central estimate of the component.
                - uncertainty_pct (float): Relative uncertainty (%).
                - name (str, optional): Component name.
            combination: "multiplicative" or "additive".

        Returns:
            Dictionary with combined uncertainty and provenance_hash.
        """
        t0 = time.monotonic()
        calc_id = str(uuid4())

        if not uncertainties:
            return self._error_result(
                calc_id, t0,
                ["At least one uncertainty entry is required"],
            )

        combination_lower = combination.lower().strip()
        valid_combinations = ("multiplicative", "additive")
        if combination_lower not in valid_combinations:
            return self._error_result(
                calc_id, t0,
                [f"Invalid combination '{combination}'. "
                 f"Must be one of: {valid_combinations}"],
            )

        component_details: List[Dict[str, Any]] = []

        if combination_lower == "additive":
            sum_x = _ZERO
            sum_ux_sq = _ZERO

            for entry in uncertainties:
                x = _safe_decimal(entry.get("value", 0))
                u_pct = _safe_decimal(entry.get("uncertainty_pct", 0))
                name = str(entry.get("name", "unnamed"))
                u_abs = (x * u_pct / _HUNDRED)
                sum_x += x
                sum_ux_sq += u_abs ** 2
                component_details.append({
                    "name": name,
                    "value": str(x),
                    "uncertainty_pct": str(u_pct),
                    "uncertainty_abs": str(u_abs.quantize(
                        _PRECISION, rounding=ROUND_HALF_UP,
                    )),
                })

            if sum_x != _ZERO:
                combined_abs = _D(str(math.sqrt(float(sum_ux_sq))))
                combined_pct = (
                    combined_abs / abs(sum_x) * _HUNDRED
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            else:
                combined_abs = _ZERO
                combined_pct = _ZERO

            total_value = sum_x

        else:
            # Multiplicative: U = sqrt(sum(Ui^2))
            sum_u_sq = _ZERO
            total_value = _ONE

            for entry in uncertainties:
                x = _safe_decimal(entry.get("value", 0))
                u_pct = _safe_decimal(entry.get("uncertainty_pct", 0))
                name = str(entry.get("name", "unnamed"))
                sum_u_sq += (u_pct / _HUNDRED) ** 2
                total_value *= x if x != _ZERO else _ONE
                component_details.append({
                    "name": name,
                    "value": str(x),
                    "uncertainty_pct": str(u_pct),
                })

            combined_pct = (
                _D(str(math.sqrt(float(sum_u_sq)))) * _HUNDRED
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            combined_abs = (
                total_value * combined_pct / _HUNDRED
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # 95% CI
        z_95 = _D(str(Z_SCORES[95]))
        ci_lower = (total_value - combined_abs * z_95).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        ci_upper = (total_value + combined_abs * z_95).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )

        with self._lock:
            self._total_combine_runs += 1

        elapsed_ms = round((time.monotonic() - t0) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "COMBINED_UNCERTAINTY",
            "combination": combination_lower,
            "total_value": str(total_value),
            "combined_uncertainty_pct": str(combined_pct),
            "combined_uncertainty_abs": str(combined_abs),
            "confidence_interval_95": {
                "lower": str(ci_lower),
                "upper": str(ci_upper),
            },
            "component_count": len(uncertainties),
            "component_details": component_details,
            "processing_time_ms": elapsed_ms,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Combine uncertainties: id=%s, mode=%s, combined=%.1f%%, "
            "95%% CI=[%s, %s], n=%d, time=%.3fms",
            calc_id, combination_lower, float(combined_pct),
            ci_lower, ci_upper, len(uncertainties), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public: Sensitivity Analysis
    # ------------------------------------------------------------------

    def run_sensitivity_analysis(
        self,
        calculation_input: Dict[str, Any],
        n_iterations: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform one-at-a-time sensitivity analysis.

        Varies each uncertain parameter individually while holding
        others at their mean values to determine each parameter's
        contribution to total output variance. Results are ranked
        by contribution (tornado chart data).

        Args:
            calculation_input: Calculation data with total_co2e_kg,
                emission_source, tier, and parameters.
            n_iterations: Iterations per parameter (default 1000).
            seed: Random seed for reproducibility.

        Returns:
            Dictionary with parameter rankings, tornado chart data,
            and provenance_hash.
        """
        t0 = time.monotonic()
        calc_id = str(uuid4())

        total_co2e = float(calculation_input.get("total_co2e_kg", 0))
        emission_source = str(
            calculation_input.get("emission_source", "enteric_fermentation"),
        ).upper()
        tier = str(calculation_input.get("tier", "TIER_1")).upper()
        parameters = calculation_input.get("parameters", [])
        rng_seed = seed if seed is not None else self._default_seed

        if total_co2e == 0:
            return self._error_result(
                calc_id, t0,
                ["total_co2e_kg must be non-zero"],
            )

        # Build parameter distributions
        params = self._build_parameter_distributions(
            parameters, emission_source, tier,
        )
        if not params:
            source_range = self._get_source_uncertainty_range(
                emission_source, tier,
            )
            params = self._build_source_level_distributions(
                emission_source, tier, source_range,
            )

        # Baseline MC (all parameters varying)
        rng_base = random.Random(rng_seed)
        baseline_samples = [
            self._simulate_single_iteration(rng_base, total_co2e, params)
            for _ in range(min(n_iterations, 2000))
        ]
        baseline_var = self._variance(baseline_samples)

        # One-at-a-time: fix each parameter, re-run MC
        tornado_data: List[Dict[str, Any]] = []
        total_variance_contribution = 0.0

        for param_name, param_info in params.items():
            rng_oat = random.Random(rng_seed)
            fixed_params = dict(params)
            fixed_params[param_name] = {
                **param_info,
                "cv": 0.0,
            }

            oat_samples = [
                self._simulate_single_iteration(
                    rng_oat, total_co2e, fixed_params,
                )
                for _ in range(min(n_iterations, 2000))
            ]
            oat_var = self._variance(oat_samples)

            # Contribution = reduction in variance when parameter is fixed
            contribution_frac = (
                (baseline_var - oat_var) / baseline_var
                if baseline_var > 0 else 0.0
            )

            # +/- 1 std_dev impact for tornado chart
            cv = float(param_info.get("cv", 0.0))
            low_impact = total_co2e * (1.0 - cv)
            high_impact = total_co2e * (1.0 + cv)
            impact_range = abs(high_impact - low_impact)
            variance_contribution = (impact_range / 2) ** 2
            total_variance_contribution += variance_contribution

            tornado_data.append({
                "parameter": param_name,
                "central_value": float(param_info.get("value", 1.0)),
                "cv_pct": round(cv * 100, 2),
                "distribution": str(param_info.get("distribution", "normal")),
                "result_low": round(low_impact, 8),
                "result_high": round(high_impact, 8),
                "impact_range": round(impact_range, 8),
                "contribution_frac": round(max(0.0, contribution_frac), 4),
                "variance_contribution": round(variance_contribution, 8),
            })

        # Rank by variance contribution descending
        tornado_data.sort(
            key=lambda x: x["variance_contribution"], reverse=True,
        )

        # Add percentage contribution
        for entry in tornado_data:
            if total_variance_contribution > 0:
                entry["contribution_pct"] = round(
                    entry["variance_contribution"]
                    / total_variance_contribution * 100, 2,
                )
            else:
                entry["contribution_pct"] = 0.0

        with self._lock:
            self._total_sensitivity_runs += 1

        elapsed_ms = round((time.monotonic() - t0) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "SENSITIVITY_ANALYSIS",
            "emission_source": emission_source,
            "tier": tier,
            "central_estimate": total_co2e,
            "baseline_variance": round(baseline_var, 8),
            "iterations_per_parameter": min(n_iterations, 2000),
            "sensitivities": tornado_data,
            "total_variance": round(total_variance_contribution, 8),
            "top_driver": (
                tornado_data[0]["parameter"] if tornado_data else None
            ),
            "parameter_count": len(tornado_data),
            "processing_time_ms": elapsed_ms,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Sensitivity analysis: id=%s, source=%s, params=%d, "
            "top_driver=%s, time=%.3fms",
            calc_id, emission_source, len(tornado_data),
            result["top_driver"], elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public: Confidence Intervals
    # ------------------------------------------------------------------

    def get_confidence_interval(
        self,
        mean: float,
        std_dev: float,
        confidence_level: float = 95.0,
        n_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Calculate a confidence interval from mean and standard deviation.

        Uses z-scores for large samples (n >= 30) and a simple
        t-approximation for small samples.

        Args:
            mean: Sample mean.
            std_dev: Sample standard deviation.
            confidence_level: Confidence level (e.g. 95.0).
            n_samples: Number of samples (for t-distribution correction).

        Returns:
            Dictionary with lower, upper, half_width, and confidence_level.
        """
        z = Z_SCORES.get(int(confidence_level), 1.960)

        if n_samples and n_samples < 30:
            # Approximate t-correction for small samples
            z = z * (1.0 + 1.0 / (4.0 * max(n_samples - 1, 1)))

        if n_samples and n_samples > 0:
            half_width = z * std_dev / math.sqrt(n_samples)
        else:
            half_width = z * std_dev

        return {
            "lower": round(mean - half_width, 8),
            "upper": round(mean + half_width, 8),
            "half_width": round(half_width, 8),
            "confidence_level": confidence_level,
        }

    # ------------------------------------------------------------------
    # Public: Percentiles
    # ------------------------------------------------------------------

    def get_percentiles(
        self,
        values: List[float],
        percentile_points: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Calculate percentiles from a list of values.

        Args:
            values: List of numeric values.
            percentile_points: Percentile points 0-100
                (default [5, 25, 50, 75, 95]).

        Returns:
            Dictionary mapping percentile labels to values.
        """
        if percentile_points is None:
            percentile_points = [5, 25, 50, 75, 95]

        return self._calculate_percentiles(values, percentile_points)

    # ------------------------------------------------------------------
    # Public: Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine usage statistics.

        Returns:
            Dictionary with operation counts, default configuration,
            and creation timestamp.
        """
        with self._lock:
            return {
                "engine": "UncertaintyQuantifierEngine",
                "agent": "AGENT-MRV-008",
                "domain": "agricultural_emissions",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_mc_runs": self._total_mc_runs,
                "total_analytical_runs": self._total_analytical_runs,
                "total_dqi_scores": self._total_dqi_scores,
                "total_sensitivity_runs": self._total_sensitivity_runs,
                "total_parameter_lookups": self._total_parameter_lookups,
                "total_combine_runs": self._total_combine_runs,
                "total_analyses": (
                    self._total_mc_runs
                    + self._total_analytical_runs
                    + self._total_dqi_scores
                    + self._total_sensitivity_runs
                    + self._total_parameter_lookups
                    + self._total_combine_runs
                ),
                "default_iterations": self._default_iterations,
                "default_seed": self._default_seed,
                "default_confidence_levels": self._default_confidence_levels,
                "parameter_count": len(PARAMETER_DISTRIBUTIONS),
                "default_cv_count": len(DEFAULT_CV),
                "source_category_count": len(SOURCE_UNCERTAINTY_RANGES),
            }

    # ------------------------------------------------------------------
    # Public: Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset engine counters to zero.

        Thread-safe. Does not reset configuration or creation time.
        """
        with self._lock:
            self._total_mc_runs = 0
            self._total_analytical_runs = 0
            self._total_dqi_scores = 0
            self._total_sensitivity_runs = 0
            self._total_parameter_lookups = 0
            self._total_combine_runs = 0
        logger.info("UncertaintyQuantifierEngine reset")

    # ==================================================================
    # Private helpers: Monte Carlo
    # ==================================================================

    def _generate_samples(
        self,
        rng: random.Random,
        value: float,
        cv: float,
        distribution: str,
        n: int,
    ) -> List[float]:
        """Generate random samples for a parameter.

        Creates *n* samples from the specified distribution centered on
        *value* with coefficient of variation *cv* (as a fraction, e.g. 0.30).

        Args:
            rng: Random number generator (per-call instance).
            value: Central/mean value.
            cv: Coefficient of variation as fraction (0.0-1.0+).
            distribution: Distribution type (normal, lognormal, uniform,
                triangular).
            n: Number of samples.

        Returns:
            List of sampled values.
        """
        if cv <= 0 or n <= 0:
            return [value] * max(n, 0)

        std_dev = abs(value) * cv

        if distribution == "lognormal":
            if value <= 0:
                return [0.0] * n
            # Parameterize lognormal from mean and CV
            sigma2 = math.log(1 + cv ** 2)
            sigma = math.sqrt(sigma2)
            mu = math.log(value) - sigma2 / 2
            return [rng.lognormvariate(mu, sigma) for _ in range(n)]

        elif distribution == "uniform":
            low = value * (1 - cv)
            high = value * (1 + cv)
            return [rng.uniform(low, high) for _ in range(n)]

        elif distribution == "triangular":
            low = value * max(0, 1 - cv)
            high = value * (1 + cv)
            return [rng.triangular(low, high, value) for _ in range(n)]

        else:
            # Normal distribution, bounded at zero for physical quantities
            samples: List[float] = []
            for _ in range(n):
                sample = rng.gauss(value, std_dev)
                samples.append(max(0.0, sample))
            return samples

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
        centered at 1.0 with spread determined by CV.

        Args:
            rng: Random number generator (per-call instance).
            base_value: Point estimate (center value).
            params: Parameter distribution specifications.

        Returns:
            Simulated emission value (kg CO2e).
        """
        combined_multiplier = 1.0

        for param_info in params.values():
            cv = float(param_info.get("cv", 0.0))
            dist = str(param_info.get("distribution", "normal")).lower()

            if cv <= 0.0:
                continue

            if dist == "lognormal":
                sigma2 = math.log(1.0 + cv ** 2)
                sigma = math.sqrt(sigma2)
                mu = -0.5 * sigma2  # mean of 1.0
                multiplier = rng.lognormvariate(mu, sigma)
            elif dist == "uniform":
                low = max(0.0, 1.0 - cv)
                high = 1.0 + cv
                multiplier = rng.uniform(low, high)
            elif dist == "triangular":
                low = max(0.0, 1.0 - cv)
                high = 1.0 + cv
                multiplier = rng.triangular(low, high, 1.0)
            else:
                # Normal, bounded at zero
                multiplier = max(0.0, rng.gauss(1.0, cv))

            combined_multiplier *= multiplier

        return base_value * combined_multiplier

    def _build_parameter_distributions(
        self,
        parameters: List[Dict[str, Any]],
        emission_source: str,
        tier: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Build parameter distribution specs from user-supplied parameters.

        For each parameter, looks up the distribution type and default CV
        from PARAMETER_DISTRIBUTIONS. User can override CV via cv_pct or
        cv keys in the parameter dict.

        Args:
            parameters: User-supplied parameter list.
            emission_source: Source category for context.
            tier: Calculation tier for CV lookup.

        Returns:
            Dictionary of parameter_name -> distribution specification.
        """
        params: Dict[str, Dict[str, Any]] = {}

        for param in parameters:
            name = str(param.get("name", "")).lower().strip()
            value = float(param.get("value", 1.0))

            if not name:
                continue

            # Look up from PARAMETER_DISTRIBUTIONS registry
            registry_info = PARAMETER_DISTRIBUTIONS.get(name, {})
            default_dist = registry_info.get("distribution", "normal")
            cv_by_tier = registry_info.get("cv_by_tier", {})
            default_cv = cv_by_tier.get(tier, cv_by_tier.get("TIER_1", 0.30))

            # User can override CV
            user_cv_pct = param.get("cv_pct")
            user_cv = param.get("cv")
            user_dist = param.get("dist") or param.get("distribution")

            if user_cv_pct is not None:
                cv = float(user_cv_pct) / 100.0
            elif user_cv is not None:
                cv = float(user_cv)
            else:
                cv = default_cv

            dist = str(user_dist or default_dist).lower()

            params[name] = {
                "value": value,
                "cv": cv,
                "distribution": dist,
                "tier": tier,
                "source": registry_info.get("source", "user-supplied"),
            }

        return params

    def _build_source_level_distributions(
        self,
        emission_source: str,
        tier: str,
        source_range: Tuple[float, float],
    ) -> Dict[str, Dict[str, Any]]:
        """Build default distributions when no explicit parameters given.

        Creates emission_factor and activity_data parameters using
        the source-level uncertainty range as a guide.

        Args:
            emission_source: Source category.
            tier: Calculation tier.
            source_range: (lower_pct, upper_pct) from SOURCE_UNCERTAINTY_RANGES.

        Returns:
            Dictionary of default parameter distributions.
        """
        midpoint_pct = (source_range[0] + source_range[1]) / 2.0
        ef_cv = midpoint_pct / 100.0

        # Determine appropriate distributions by source type
        source_params: Dict[str, Dict[str, str]] = {
            "ENTERIC_FERMENTATION": {
                "ef_dist": "lognormal",
                "ef_name": "enteric_ef",
                "ad_name": "head_count",
            },
            "MANURE_MANAGEMENT": {
                "ef_dist": "lognormal",
                "ef_name": "manure_mcf",
                "ad_name": "head_count",
            },
            "RICE_CULTIVATION": {
                "ef_dist": "lognormal",
                "ef_name": "rice_baseline_ef",
                "ad_name": "head_count",
            },
            "AGRICULTURAL_SOILS": {
                "ef_dist": "lognormal",
                "ef_name": "soil_n2o_ef1",
                "ad_name": "head_count",
            },
            "FIELD_BURNING": {
                "ef_dist": "lognormal",
                "ef_name": "field_burning_ef",
                "ad_name": "head_count",
            },
        }

        source_cfg = source_params.get(emission_source, {
            "ef_dist": "lognormal",
            "ef_name": "emission_factor",
            "ad_name": "activity_data",
        })

        # Activity data CV from DEFAULT_CV
        ad_cv_data = DEFAULT_CV.get("HEAD_COUNT", {})
        ad_cv = ad_cv_data.get(tier, 5.0) / 100.0

        # GWP uncertainty
        gwp_cv = DEFAULT_CV.get("GWP", {}).get(tier, 10.0) / 100.0

        params: Dict[str, Dict[str, Any]] = {
            source_cfg["ef_name"]: {
                "value": 1.0,
                "cv": ef_cv,
                "distribution": source_cfg["ef_dist"],
                "tier": tier,
                "source": "SOURCE_UNCERTAINTY_RANGES default",
            },
            source_cfg["ad_name"]: {
                "value": 1.0,
                "cv": ad_cv,
                "distribution": "normal",
                "tier": tier,
                "source": "DEFAULT_CV HEAD_COUNT",
            },
            "gwp": {
                "value": 1.0,
                "cv": gwp_cv,
                "distribution": "normal",
                "tier": tier,
                "source": "DEFAULT_CV GWP",
            },
        }

        return params

    def _get_source_uncertainty_range(
        self,
        emission_source: str,
        tier: str,
    ) -> Tuple[float, float]:
        """Get the applicable uncertainty range for a source/tier combo.

        Args:
            emission_source: Agricultural source category.
            tier: Calculation tier.

        Returns:
            (lower_pct, upper_pct) tuple.
        """
        source_ranges = SOURCE_UNCERTAINTY_RANGES.get(
            emission_source,
            SOURCE_UNCERTAINTY_RANGES.get(
                "ENTERIC_FERMENTATION", {},
            ),
        )
        return source_ranges.get(
            tier,
            source_ranges.get("TIER_1", (20.0, 50.0)),
        )

    # ==================================================================
    # Private helpers: Statistics
    # ==================================================================

    def _compute_statistics(
        self,
        values: List[float],
    ) -> Dict[str, Any]:
        """Compute descriptive statistics from a list of values.

        Args:
            values: List of numeric values.

        Returns:
            Dictionary with mean, std_dev, cv, cv_pct, min, max.
        """
        n = len(values)
        if n == 0:
            return {
                "mean": 0.0,
                "std_dev": 0.0,
                "cv": 0.0,
                "cv_pct": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        sorted_vals = sorted(values)
        mean_val = sum(sorted_vals) / n
        variance = sum(
            (x - mean_val) ** 2 for x in sorted_vals
        ) / max(n - 1, 1)
        std_dev = math.sqrt(variance)
        cv = (std_dev / abs(mean_val)) if mean_val != 0 else 0.0
        cv_pct = cv * 100.0

        return {
            "mean": round(mean_val, 8),
            "std_dev": round(std_dev, 8),
            "cv": round(cv, 6),
            "cv_pct": round(cv_pct, 2),
            "min": round(sorted_vals[0], 8),
            "max": round(sorted_vals[-1], 8),
        }

    def _calculate_percentiles(
        self,
        values: List[float],
        percentiles: List[float],
    ) -> Dict[str, float]:
        """Calculate percentiles from a list of values using interpolation.

        Args:
            values: List of numeric values (need not be sorted).
            percentiles: Percentile points (0-100).

        Returns:
            Dictionary mapping percentile labels (e.g. "5", "50") to values.
        """
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        result: Dict[str, float] = {}

        if n == 0:
            for p in percentiles:
                key = str(int(p)) if p == int(p) else str(p)
                result[key] = 0.0
            return result

        for p in percentiles:
            idx = (p / 100.0) * (n - 1)
            lower = int(math.floor(idx))
            upper = min(lower + 1, n - 1)
            frac = idx - lower
            val = sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac
            key = str(int(p)) if p == int(p) else str(p)
            result[key] = round(val, 8)

        return result

    def _compute_confidence_intervals(
        self,
        values: List[float],
        confidence_levels: List[float],
        mean_val: float,
    ) -> Dict[str, Dict[str, float]]:
        """Compute confidence intervals from Monte Carlo samples.

        Args:
            values: Sample values.
            confidence_levels: Confidence levels (e.g. [90, 95, 99]).
            mean_val: Sample mean for relative percentage calculation.

        Returns:
            Dictionary of {level: {lower, upper, half_width, relative_pct}}.
        """
        if not values:
            return {}

        sorted_vals = sorted(values)
        ci: Dict[str, Dict[str, float]] = {}

        for level in confidence_levels:
            alpha = (100.0 - level) / 2.0
            lower_vals = self._calculate_percentiles(sorted_vals, [alpha])
            upper_vals = self._calculate_percentiles(
                sorted_vals, [100.0 - alpha],
            )

            lower_key = (
                str(int(alpha)) if alpha == int(alpha) else str(alpha)
            )
            upper_key = (
                str(int(100 - alpha))
                if (100 - alpha) == int(100 - alpha)
                else str(100 - alpha)
            )

            lower_val = lower_vals.get(lower_key, 0.0)
            upper_val = upper_vals.get(upper_key, 0.0)
            half_width = (upper_val - lower_val) / 2.0

            relative_pct = (
                (upper_val - lower_val) / (2.0 * abs(mean_val)) * 100.0
                if mean_val != 0 else 0.0
            )

            level_key = str(int(level))
            ci[level_key] = {
                "lower": round(lower_val, 8),
                "upper": round(upper_val, 8),
                "half_width": round(half_width, 8),
                "relative_pct": round(relative_pct, 2),
            }

        return ci

    def _compute_parameter_statistics(
        self,
        param_samples: Dict[str, List[float]],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-parameter statistics from MC samples.

        Args:
            param_samples: Dictionary of param_name -> sample list.

        Returns:
            Dictionary of param_name -> {mean, std_dev, cv_pct, min, max}.
        """
        param_stats: Dict[str, Dict[str, Any]] = {}

        for name, samples in param_samples.items():
            if not samples:
                continue
            n = len(samples)
            p_mean = sum(samples) / n
            p_var = sum(
                (x - p_mean) ** 2 for x in samples
            ) / max(n - 1, 1)
            p_std = math.sqrt(p_var)

            param_stats[name] = {
                "mean": round(p_mean, 8),
                "std_dev": round(p_std, 8),
                "cv_pct": round(
                    (p_std / abs(p_mean) * 100.0) if p_mean != 0 else 0.0,
                    2,
                ),
                "min": round(min(samples), 8),
                "max": round(max(samples), 8),
            }

        return param_stats

    # ==================================================================
    # Private helpers: DQI
    # ==================================================================

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

    def _composite_to_quality(self, composite: float) -> str:
        """Map a composite DQI score to a quality category.

        Args:
            composite: Composite DQI score (1.0-5.0).

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

    def _dqi_to_cv_adjustment(self, composite: float) -> float:
        """Derive a CV adjustment factor from composite DQI score.

        Better data quality (lower score) yields lower CV multiplier.
        Linear interpolation: 1.0 -> 0.5, 5.0 -> 2.0.

        Args:
            composite: Composite DQI score (1.0-5.0).

        Returns:
            CV adjustment multiplier (0.5 to 2.0).
        """
        return 0.5 + (composite - 1.0) * (2.0 - 0.5) / (5.0 - 1.0)

    # ==================================================================
    # Private helpers: Analytical
    # ==================================================================

    def _extract_parameter_uncertainties(
        self,
        calculation_input: Dict[str, Any],
    ) -> Dict[str, float]:
        """Extract parameter-level uncertainties from input dict.

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
                # Only include defaults relevant to the emission source
                source = str(
                    calculation_input.get("emission_source", ""),
                ).upper()
                if self._is_param_relevant(param_name, source):
                    uncertainties[param_name] = default_pct

        return uncertainties

    def _is_param_relevant(
        self,
        param_name: str,
        emission_source: str,
    ) -> bool:
        """Check if a parameter is relevant to the given emission source.

        Args:
            param_name: Parameter name from DEFAULT_PARAMETER_UNCERTAINTIES.
            emission_source: Source category (uppercase).

        Returns:
            True if the parameter is relevant.
        """
        relevance_map: Dict[str, List[str]] = {
            "ENTERIC_FERMENTATION": [
                "emission_factor", "activity_data", "head_count",
                "gwp", "feed_intake", "ym_factor",
            ],
            "MANURE_MANAGEMENT": [
                "emission_factor", "activity_data", "head_count",
                "gwp", "volatile_solids", "bo_factor", "mcf_factor",
                "n_excretion", "n2o_emission_factor",
            ],
            "RICE_CULTIVATION": [
                "emission_factor", "activity_data", "area_ha",
                "gwp", "rice_ef", "cultivation_period",
            ],
            "AGRICULTURAL_SOILS": [
                "emission_factor", "activity_data", "area_ha",
                "gwp", "n2o_emission_factor", "volatilization_fraction",
                "leaching_fraction", "n_excretion", "residue_mass",
            ],
            "FIELD_BURNING": [
                "emission_factor", "activity_data", "area_ha",
                "gwp", "burning_ef", "residue_mass", "combustion_factor",
            ],
            "LIMING": [
                "emission_factor", "activity_data", "area_ha",
            ],
            "UREA_APPLICATION": [
                "emission_factor", "activity_data", "area_ha",
            ],
        }

        # If source is not recognised, include common parameters
        relevant_params = relevance_map.get(
            emission_source,
            ["emission_factor", "activity_data", "gwp"],
        )
        return param_name in relevant_params

    # ==================================================================
    # Private helpers: Math utilities
    # ==================================================================

    @staticmethod
    def _variance(samples: List[float]) -> float:
        """Compute sample variance (unbiased, N-1 denominator).

        Args:
            samples: List of numeric values.

        Returns:
            Sample variance. Returns 0.0 if fewer than 2 samples.
        """
        n = len(samples)
        if n < 2:
            return 0.0
        mean = sum(samples) / n
        return sum((x - mean) ** 2 for x in samples) / (n - 1)

    # ==================================================================
    # Private helpers: Error result builder
    # ==================================================================

    def _error_result(
        self,
        calc_id: str,
        start_time: float,
        errors: List[str],
    ) -> Dict[str, Any]:
        """Build a standardised error result dictionary.

        Args:
            calc_id: Calculation UUID.
            start_time: Monotonic start time for elapsed calculation.
            errors: List of error messages.

        Returns:
            Error result dictionary with status VALIDATION_ERROR.
        """
        elapsed_ms = round((time.monotonic() - start_time) * 1000, 3)
        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "VALIDATION_ERROR",
            "errors": errors,
            "processing_time_ms": elapsed_ms,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result
