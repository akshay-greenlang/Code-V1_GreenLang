# -*- coding: utf-8 -*-
"""
CroplandEmissionsEngine - Agricultural Soils N2O, Liming/Urea CO2, Rice CH4,
Field Burning CH4/N2O (Engine 4 of 7)

AGENT-MRV-008: Agricultural Emissions Agent

Core calculation engine implementing IPCC 2006 Guidelines Volume 4 Chapters 5
and 11 methods for estimating greenhouse gas emissions from cropland management
activities including:

    1. Direct N2O from managed agricultural soils (Eq 11.1)
    2. Indirect N2O from atmospheric deposition of volatilized N (Eq 11.9)
    3. Indirect N2O from leaching and runoff (Eq 11.10)
    4. CO2 from liming of agricultural soils (Eq 11.12)
    5. CO2 from urea application (Eq 11.13)
    6. CH4 from rice cultivation (Eq 5.1)
    7. CH4 and N2O from field burning of agricultural residues (Eq 2.27)

Key Formulae (IPCC 2006 Vol 4):

    Direct Soil N2O (Eq 11.1):
        N2O_direct = [(F_SN + F_ON + F_CR + F_SOM) * EF1
                      + F_OS_CG * EF2_CG + F_OS_F * EF2_F
                      + F_PRP * EF3_PRP] * 44/28

    Indirect N2O - Volatilization (Eq 11.9):
        N2O_ATD = [(F_SN * Frac_GASF) + ((F_ON + F_PRP) * Frac_GASM)]
                  * EF4 * 44/28

    Indirect N2O - Leaching (Eq 11.10):
        N2O_L = (F_SN + F_ON + F_PRP + F_CR + F_SOM)
                * Frac_LEACH * EF5 * 44/28

    Liming CO2 (Eq 11.12):
        CO2_liming = (M_limestone * 0.12 + M_dolomite * 0.13) * 44/12

    Urea CO2 (Eq 11.13):
        CO2_urea = M_urea * 0.20 * 44/12

    Rice CH4 (Eq 5.1):
        CH4_rice = EF_ijk * t_ijk * A_ijk * 10^-6  (Gg/yr)
        EF_ijk = EF_c * SF_w * SF_p * SF_o * SF_s

    Field Burning (Eq 2.27):
        L_fire = A * M_B * C_f * G_ef * 10^-3  (tonnes gas/yr)

Zero-Hallucination Guarantees:
    - All numeric calculations use Python Decimal with 8-decimal-place quantization.
    - No LLM calls in any calculation path.
    - Every calculation step is logged and traceable.
    - SHA-256 provenance hash for every result.
    - Identical inputs always produce identical outputs.

Thread Safety:
    All reference data is immutable after class initialization. The mutable
    custom factor registry is protected by a reentrant lock (threading.RLock).
    Concurrent calls to calculate_* methods are safe.

Example:
    >>> from greenlang.agents.mrv.agricultural_emissions.cropland_emissions import (
    ...     CroplandEmissionsEngine,
    ... )
    >>> engine = CroplandEmissionsEngine()
    >>> result = engine.calculate_direct_n2o(synthetic_n_kg=10000)
    >>> assert result["status"] == "SUCCESS"
    >>> assert float(result["n2o_kg"]) > 0

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 Agricultural Emissions (GL-MRV-SCOPE1-008)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["CroplandEmissionsEngine"]

# ---------------------------------------------------------------------------
# Conditional imports -- graceful fallback when peer modules are absent
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
        record_calculation as _record_calculation,
        record_emissions as _record_emissions,
        record_cropland_calculation as _record_cropland_calculation,
        record_rice_calculation as _record_rice_calculation,
        record_field_burning_calculation as _record_field_burning_calculation,
        record_calculation_error as _record_calculation_error,
        observe_calculation_duration as _observe_duration,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_calculation = None  # type: ignore[assignment]
    _record_emissions = None  # type: ignore[assignment]
    _record_cropland_calculation = None  # type: ignore[assignment]
    _record_rice_calculation = None  # type: ignore[assignment]
    _record_field_burning_calculation = None  # type: ignore[assignment]
    _record_calculation_error = None  # type: ignore[assignment]
    _observe_duration = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Decimal precision constants
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_THOUSAND = Decimal("1000")

#: N2O molecular weight ratio: 44/28 (N2O-N to N2O conversion)
N2O_MOLECULAR_RATIO = Decimal("1.57142857")

#: CO2/C molecular weight ratio: 44/12
CO2_C_RATIO = Decimal("3.66666667")

#: kg to tonnes conversion factor
KG_TO_TONNES = Decimal("0.001")

#: Gg to tonnes conversion: 1 Gg = 1000 tonnes
GG_TO_TONNES = Decimal("1000")


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal with controlled precision.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted to Decimal.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert a value to Decimal, returning default on failure.

    Args:
        value: Value to convert.
        default: Default if conversion fails.

    Returns:
        Decimal value.
    """
    if value is None:
        return default
    try:
        return _D(value)
    except (InvalidOperation, ValueError, TypeError):
        return default


def _Q(value: Decimal) -> Decimal:
    """Quantize a Decimal to the standard 8-decimal-place precision.

    Args:
        value: Decimal value to quantize.

    Returns:
        Quantized Decimal.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash. Accepts dict, list, str, Decimal, or numeric types.

    Returns:
        Lowercase hexadecimal SHA-256 digest string (64 characters).
    """
    if isinstance(data, dict):
        canonical = json.dumps(data, sort_keys=True, default=str)
    elif isinstance(data, (list, tuple)):
        canonical = json.dumps(data, sort_keys=True, default=str)
    else:
        canonical = str(data)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ===========================================================================
# GWP Values (fallback when models module is not available)
# ===========================================================================

GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR4": {
        "CO2": Decimal("1"),
        "CH4": Decimal("25"),
        "N2O": Decimal("298"),
    },
    "AR5": {
        "CO2": Decimal("1"),
        "CH4": Decimal("28"),
        "N2O": Decimal("265"),
    },
    "AR6": {
        "CO2": Decimal("1"),
        "CH4": Decimal("29.8"),
        "N2O": Decimal("273"),
    },
    "AR6_20YR": {
        "CO2": Decimal("1"),
        "CH4": Decimal("82.5"),
        "N2O": Decimal("273"),
    },
}


# ===========================================================================
# Enumerations
# ===========================================================================


class WaterRegime(str, Enum):
    """Rice paddy water regime types with IPCC scaling factors."""

    CONTINUOUSLY_FLOODED = "continuously_flooded"
    INTERMITTENT_SINGLE = "intermittent_single"
    INTERMITTENT_MULTIPLE = "intermittent_multiple"
    RAINFED_REGULAR = "rainfed_regular"
    RAINFED_DROUGHT = "rainfed_drought"
    DEEP_WATER = "deep_water"
    UPLAND = "upland"


class PreSeasonFlooding(str, Enum):
    """Pre-season water status for rice paddies."""

    NOT_FLOODED_SHORT = "not_flooded_short"
    NOT_FLOODED_LONG = "not_flooded_long"
    FLOODED_SHORT = "flooded_short"
    FLOODED_LONG = "flooded_long"


class OrganicAmendmentType(str, Enum):
    """Types of organic amendments applied to rice paddies."""

    STRAW_SHORT = "straw_short"
    STRAW_LONG = "straw_long"
    COMPOST = "compost"
    FARM_YARD_MANURE = "farm_yard_manure"
    GREEN_MANURE = "green_manure"


class PRPAnimalType(str, Enum):
    """Animal type categories for PRP (pasture, range, paddock) N inputs."""

    CATTLE = "cattle"
    POULTRY = "poultry"
    OTHER = "other"


class CropType(str, Enum):
    """Crop types for field burning and residue estimation."""

    RICE = "rice"
    WHEAT = "wheat"
    MAIZE = "maize"
    SUGARCANE = "sugarcane"
    COTTON = "cotton"
    BARLEY = "barley"
    OATS = "oats"
    SORGHUM = "sorghum"
    MILLET = "millet"
    OTHER = "other"


class CalculationStatus(str, Enum):
    """Result status codes."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"


# ===========================================================================
# IPCC Default Emission Factors -- Soil N2O (Vol 4 Ch 11)
# ===========================================================================

#: EF1 -- Direct N2O from synthetic/organic N inputs (kg N2O-N / kg N applied)
#: IPCC 2006 default = 0.01 (1% of applied N emitted as N2O-N)
EF1_DEFAULT = Decimal("0.01")

#: EF2_CG -- N2O from drained organic soils, cropland/grassland (kg N2O-N/ha/yr)
EF2_CG_DEFAULT = Decimal("8")

#: EF2_F -- N2O from drained organic soils, forest (kg N2O-N/ha/yr)
EF2_F_DEFAULT = Decimal("2.5")

#: EF3_PRP -- N2O from PRP deposits: cattle/poultry
EF3_PRP_CATTLE_POULTRY = Decimal("0.02")

#: EF3_PRP -- N2O from PRP deposits: other animals
EF3_PRP_OTHER = Decimal("0.01")

#: Frac_GASF -- Fraction of synthetic N that volatilizes as NH3 + NOx
FRAC_GASF_DEFAULT = Decimal("0.10")

#: Frac_GASM -- Fraction of organic/PRP N that volatilizes as NH3 + NOx
FRAC_GASM_DEFAULT = Decimal("0.20")

#: EF4 -- N2O from atmospheric deposition of volatilized N (kg N2O-N/kg NH3-N+NOx-N)
EF4_DEFAULT = Decimal("0.01")

#: Frac_LEACH -- Fraction of N that leaches/runs off in wet climates
FRAC_LEACH_DEFAULT = Decimal("0.30")

#: EF5 -- N2O from leaching/runoff (kg N2O-N / kg N leached)
EF5_DEFAULT = Decimal("0.0075")


# ===========================================================================
# IPCC Liming and Urea Emission Factors (Vol 4 Ch 11)
# ===========================================================================

#: Limestone (CaCO3) emission factor -- fraction of C released as CO2
LIMESTONE_EF = Decimal("0.12")

#: Dolomite (CaMg(CO3)2) emission factor -- fraction of C released as CO2
DOLOMITE_EF = Decimal("0.13")

#: Urea CO(NH2)2 emission factor -- fraction of C released as CO2
UREA_EF = Decimal("0.20")


# ===========================================================================
# IPCC Rice CH4 Parameters (Vol 4 Ch 5)
# ===========================================================================

#: Baseline emission factor EF_c for continuously flooded fields
#: without organic amendments (kg CH4/ha/day) -- IPCC 2006 Table 5.11
EF_C_RICE_DEFAULT = Decimal("1.30")

#: Water regime scaling factors (SF_w) -- IPCC 2006 Table 5.12
WATER_REGIME_SF: Dict[str, Decimal] = {
    WaterRegime.CONTINUOUSLY_FLOODED.value: Decimal("1.0"),
    WaterRegime.INTERMITTENT_SINGLE.value: Decimal("0.60"),
    WaterRegime.INTERMITTENT_MULTIPLE.value: Decimal("0.52"),
    WaterRegime.RAINFED_REGULAR.value: Decimal("0.28"),
    WaterRegime.RAINFED_DROUGHT.value: Decimal("0.25"),
    WaterRegime.DEEP_WATER.value: Decimal("0.31"),
    WaterRegime.UPLAND.value: Decimal("0.0"),
}

#: Pre-season flooding scaling factors (SF_p) -- IPCC 2006 Table 5.13
PRE_SEASON_SF: Dict[str, Decimal] = {
    PreSeasonFlooding.NOT_FLOODED_SHORT.value: Decimal("1.00"),
    PreSeasonFlooding.NOT_FLOODED_LONG.value: Decimal("0.68"),
    PreSeasonFlooding.FLOODED_SHORT.value: Decimal("1.90"),
    PreSeasonFlooding.FLOODED_LONG.value: Decimal("2.40"),
}

#: Organic amendment conversion factor for aerobic decay (CFOA) values
#: IPCC 2006 Table 5.14
ORGANIC_AMENDMENT_CFOA: Dict[str, Decimal] = {
    OrganicAmendmentType.STRAW_SHORT.value: Decimal("1.0"),
    OrganicAmendmentType.STRAW_LONG.value: Decimal("0.29"),
    OrganicAmendmentType.COMPOST.value: Decimal("0.05"),
    OrganicAmendmentType.FARM_YARD_MANURE.value: Decimal("0.14"),
    OrganicAmendmentType.GREEN_MANURE.value: Decimal("0.50"),
}

#: Exponent for organic amendment scaling factor (IPCC 2006 Eq 5.3)
ORGANIC_AMENDMENT_EXPONENT = Decimal("0.59")


# ===========================================================================
# Field Burning Parameters (Vol 4 Ch 2, Table 2.6 / Vol 4 Ch 5)
# ===========================================================================

#: Crop residue parameters: residue_to_crop_ratio, dry_matter_fraction,
#: carbon_fraction, nitrogen_content (kg N/kg DM)
CROP_RESIDUE_PARAMS: Dict[str, Dict[str, Decimal]] = {
    CropType.RICE.value: {
        "residue_to_crop_ratio": Decimal("1.40"),
        "dry_matter_fraction": Decimal("0.86"),
        "carbon_fraction": Decimal("0.41"),
        "nitrogen_content": Decimal("0.0067"),
        "combustion_factor": Decimal("0.80"),
        "above_ground_fraction": Decimal("0.50"),
    },
    CropType.WHEAT.value: {
        "residue_to_crop_ratio": Decimal("1.30"),
        "dry_matter_fraction": Decimal("0.88"),
        "carbon_fraction": Decimal("0.43"),
        "nitrogen_content": Decimal("0.0060"),
        "combustion_factor": Decimal("0.90"),
        "above_ground_fraction": Decimal("0.55"),
    },
    CropType.MAIZE.value: {
        "residue_to_crop_ratio": Decimal("1.00"),
        "dry_matter_fraction": Decimal("0.87"),
        "carbon_fraction": Decimal("0.47"),
        "nitrogen_content": Decimal("0.0081"),
        "combustion_factor": Decimal("0.80"),
        "above_ground_fraction": Decimal("0.53"),
    },
    CropType.SUGARCANE.value: {
        "residue_to_crop_ratio": Decimal("0.30"),
        "dry_matter_fraction": Decimal("0.30"),
        "carbon_fraction": Decimal("0.42"),
        "nitrogen_content": Decimal("0.0040"),
        "combustion_factor": Decimal("0.80"),
        "above_ground_fraction": Decimal("0.80"),
    },
    CropType.COTTON.value: {
        "residue_to_crop_ratio": Decimal("2.10"),
        "dry_matter_fraction": Decimal("0.91"),
        "carbon_fraction": Decimal("0.45"),
        "nitrogen_content": Decimal("0.0070"),
        "combustion_factor": Decimal("0.85"),
        "above_ground_fraction": Decimal("0.50"),
    },
    CropType.BARLEY.value: {
        "residue_to_crop_ratio": Decimal("1.20"),
        "dry_matter_fraction": Decimal("0.88"),
        "carbon_fraction": Decimal("0.45"),
        "nitrogen_content": Decimal("0.0070"),
        "combustion_factor": Decimal("0.90"),
        "above_ground_fraction": Decimal("0.55"),
    },
    CropType.OATS.value: {
        "residue_to_crop_ratio": Decimal("1.30"),
        "dry_matter_fraction": Decimal("0.89"),
        "carbon_fraction": Decimal("0.44"),
        "nitrogen_content": Decimal("0.0070"),
        "combustion_factor": Decimal("0.90"),
        "above_ground_fraction": Decimal("0.55"),
    },
    CropType.SORGHUM.value: {
        "residue_to_crop_ratio": Decimal("1.40"),
        "dry_matter_fraction": Decimal("0.88"),
        "carbon_fraction": Decimal("0.45"),
        "nitrogen_content": Decimal("0.0075"),
        "combustion_factor": Decimal("0.80"),
        "above_ground_fraction": Decimal("0.50"),
    },
    CropType.MILLET.value: {
        "residue_to_crop_ratio": Decimal("1.40"),
        "dry_matter_fraction": Decimal("0.88"),
        "carbon_fraction": Decimal("0.44"),
        "nitrogen_content": Decimal("0.0070"),
        "combustion_factor": Decimal("0.80"),
        "above_ground_fraction": Decimal("0.50"),
    },
    CropType.OTHER.value: {
        "residue_to_crop_ratio": Decimal("1.20"),
        "dry_matter_fraction": Decimal("0.85"),
        "carbon_fraction": Decimal("0.43"),
        "nitrogen_content": Decimal("0.0070"),
        "combustion_factor": Decimal("0.80"),
        "above_ground_fraction": Decimal("0.50"),
    },
}

#: Field burning emission factors (g/kg dry matter burned)
#: IPCC 2006 Vol 4 Table 2.5
FIELD_BURNING_EF: Dict[str, Dict[str, Decimal]] = {
    CropType.RICE.value: {
        "CH4": Decimal("2.7"),
        "N2O": Decimal("0.07"),
    },
    CropType.WHEAT.value: {
        "CH4": Decimal("2.7"),
        "N2O": Decimal("0.07"),
    },
    CropType.MAIZE.value: {
        "CH4": Decimal("2.7"),
        "N2O": Decimal("0.07"),
    },
    CropType.SUGARCANE.value: {
        "CH4": Decimal("2.7"),
        "N2O": Decimal("0.07"),
    },
    CropType.COTTON.value: {
        "CH4": Decimal("2.7"),
        "N2O": Decimal("0.07"),
    },
    CropType.BARLEY.value: {
        "CH4": Decimal("2.7"),
        "N2O": Decimal("0.07"),
    },
    CropType.OATS.value: {
        "CH4": Decimal("2.7"),
        "N2O": Decimal("0.07"),
    },
    CropType.SORGHUM.value: {
        "CH4": Decimal("2.7"),
        "N2O": Decimal("0.07"),
    },
    CropType.MILLET.value: {
        "CH4": Decimal("2.7"),
        "N2O": Decimal("0.07"),
    },
    CropType.OTHER.value: {
        "CH4": Decimal("2.7"),
        "N2O": Decimal("0.07"),
    },
}


# ===========================================================================
# CroplandEmissionsEngine
# ===========================================================================


class CroplandEmissionsEngine:
    """Engine 4: Cropland emissions calculations covering agricultural soils
    N2O, liming/urea CO2, rice cultivation CH4, and field burning emissions.

    Implements IPCC 2006 Volume 4 Chapters 5 and 11 methodology with
    deterministic Decimal arithmetic, full audit trails, and SHA-256
    provenance hashing.

    All calculation methods return a result dictionary containing at minimum:
        - ``status``: ``"SUCCESS"`` or ``"FAILED"``
        - Per-gas emission values in kg and tonnes
        - ``total_co2e_tonnes``: Total CO2-equivalent
        - ``calculation_trace``: Step-by-step audit trail
        - ``provenance_hash``: SHA-256 hash of inputs + outputs
        - ``processing_time_ms``: Wall-clock calculation time

    Attributes:
        _config: Optional configuration dictionary.
        _lock: Reentrant lock for thread-safe custom factor mutations.
        _custom_ef: User-registered custom emission factors.
        _default_gwp_source: Default GWP assessment report.

    Example:
        >>> engine = CroplandEmissionsEngine()
        >>> result = engine.calculate_direct_n2o(synthetic_n_kg=10000)
        >>> assert result["status"] == "SUCCESS"
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize CroplandEmissionsEngine.

        Args:
            config: Optional configuration dictionary. Supports:
                - ``default_gwp_source`` (str): Default GWP AR. Default ``"AR6"``.
                - ``decimal_precision`` (int): Decimal places. Default 8.
                - ``enable_provenance`` (bool): Track provenance. Default True.
                - ``enable_metrics`` (bool): Record Prometheus metrics. Default True.
        """
        self._config: Dict[str, Any] = config or {}
        self._lock = threading.RLock()
        self._total_calculations: int = 0
        self._total_errors: int = 0
        self._created_at = _utcnow()

        self._default_gwp_source: str = str(
            self._config.get("default_gwp_source", "AR6")
        ).upper()
        self._enable_provenance: bool = self._config.get("enable_provenance", True)
        self._enable_metrics: bool = self._config.get("enable_metrics", True)

        # Mutable custom factor registries (guarded by _lock)
        self._custom_ef: Dict[str, Dict[str, Decimal]] = {}
        self._custom_crop_params: Dict[str, Dict[str, Decimal]] = {}
        self._custom_burning_ef: Dict[str, Dict[str, Decimal]] = {}

        # Provenance tracker
        self._provenance: Any = None
        if self._enable_provenance and _PROVENANCE_AVAILABLE:
            try:
                self._provenance = _get_provenance_tracker()
            except Exception:
                logger.warning(
                    "Provenance tracker initialization failed; continuing without"
                )

        logger.info(
            "CroplandEmissionsEngine initialized (gwp=%s, provenance=%s)",
            self._default_gwp_source,
            self._provenance is not None,
        )

    # ------------------------------------------------------------------
    # Thread-safe counters
    # ------------------------------------------------------------------

    def _increment_calculations(self) -> None:
        """Thread-safe increment of the calculation counter."""
        with self._lock:
            self._total_calculations += 1

    def _increment_errors(self) -> None:
        """Thread-safe increment of the error counter."""
        with self._lock:
            self._total_errors += 1

    # ------------------------------------------------------------------
    # GWP resolution
    # ------------------------------------------------------------------

    def _resolve_gwp(
        self,
        gas: str,
        gwp_source: Optional[str] = None,
    ) -> Decimal:
        """Look up GWP value for a gas species.

        Args:
            gas: Gas species (CO2, CH4, N2O).
            gwp_source: IPCC assessment report. Defaults to engine default.

        Returns:
            GWP value as Decimal. Falls back to 1 for unknown gases.
        """
        source = (gwp_source or self._default_gwp_source).upper()
        gas_key = gas.upper()
        if source in GWP_VALUES and gas_key in GWP_VALUES[source]:
            return GWP_VALUES[source][gas_key]
        logger.warning("GWP lookup failed for %s/%s, using 1.0", gas, source)
        return _ONE

    # ------------------------------------------------------------------
    # Custom factor registration (thread-safe)
    # ------------------------------------------------------------------

    def register_custom_soil_ef(
        self,
        ef_name: str,
        value: Any,
    ) -> None:
        """Register a custom soil N2O emission factor override.

        Thread-safe. Overrides built-in IPCC defaults for subsequent
        calculations.

        Args:
            ef_name: Emission factor name (e.g. ``"EF1"``, ``"EF4"``).
            value: Factor value (must be non-negative).

        Raises:
            ValueError: If value is negative.
        """
        dec_val = _D(value)
        if dec_val < _ZERO:
            raise ValueError(f"Emission factor {ef_name} must be >= 0, got {dec_val}")
        key = ef_name.upper().strip()
        with self._lock:
            self._custom_ef[key] = {"value": dec_val}
        logger.info("Registered custom soil EF: %s = %s", key, dec_val)

    def register_custom_crop_params(
        self,
        crop_type: str,
        residue_to_crop_ratio: Any,
        dry_matter_fraction: Any,
        carbon_fraction: Any,
        nitrogen_content: Any,
        combustion_factor: Any = "0.80",
        above_ground_fraction: Any = "0.50",
    ) -> None:
        """Register custom crop residue parameters.

        Thread-safe. Overrides built-in IPCC defaults for the specified
        crop type in subsequent calculations.

        Args:
            crop_type: Crop type key (will be normalized to lowercase).
            residue_to_crop_ratio: Ratio of residue to crop yield.
            dry_matter_fraction: Dry matter fraction (0-1).
            carbon_fraction: Carbon fraction of dry matter (0-1).
            nitrogen_content: N content of residue (kg N/kg DM).
            combustion_factor: Combustion completeness factor (0-1).
            above_ground_fraction: Fraction of residue above ground (0-1).

        Raises:
            ValueError: If any fraction is outside valid range.
        """
        params = {
            "residue_to_crop_ratio": _D(residue_to_crop_ratio),
            "dry_matter_fraction": _D(dry_matter_fraction),
            "carbon_fraction": _D(carbon_fraction),
            "nitrogen_content": _D(nitrogen_content),
            "combustion_factor": _D(combustion_factor),
            "above_ground_fraction": _D(above_ground_fraction),
        }
        for name in ["dry_matter_fraction", "carbon_fraction",
                      "combustion_factor", "above_ground_fraction"]:
            if params[name] < _ZERO or params[name] > _ONE:
                raise ValueError(f"{name} must be in [0, 1], got {params[name]}")

        key = crop_type.strip().lower().replace(" ", "_").replace("-", "_")
        with self._lock:
            self._custom_crop_params[key] = params
        logger.info("Registered custom crop params for %s", key)

    def register_custom_burning_ef(
        self,
        crop_type: str,
        ch4_g_per_kg_dm: Any,
        n2o_g_per_kg_dm: Any,
    ) -> None:
        """Register custom field burning emission factors for a crop type.

        Thread-safe. Overrides built-in IPCC Table 2.5 defaults.

        Args:
            crop_type: Crop type key (normalized to lowercase).
            ch4_g_per_kg_dm: CH4 emission factor (g/kg dry matter burned).
            n2o_g_per_kg_dm: N2O emission factor (g/kg dry matter burned).

        Raises:
            ValueError: If any factor is negative.
        """
        ch4 = _D(ch4_g_per_kg_dm)
        n2o = _D(n2o_g_per_kg_dm)
        if ch4 < _ZERO:
            raise ValueError(f"CH4 EF must be >= 0, got {ch4}")
        if n2o < _ZERO:
            raise ValueError(f"N2O EF must be >= 0, got {n2o}")

        key = crop_type.strip().lower().replace(" ", "_").replace("-", "_")
        with self._lock:
            self._custom_burning_ef[key] = {"CH4": ch4, "N2O": n2o}
        logger.info("Registered custom burning EF for %s", key)

    # ------------------------------------------------------------------
    # Reference data lookups (custom overrides -> built-in fallback)
    # ------------------------------------------------------------------

    def _get_soil_ef(self, ef_name: str, default: Decimal) -> Decimal:
        """Look up a soil N2O emission factor with custom override support.

        Args:
            ef_name: Uppercase emission factor name.
            default: Built-in IPCC default value.

        Returns:
            Emission factor value as Decimal.
        """
        with self._lock:
            if ef_name in self._custom_ef:
                return self._custom_ef[ef_name]["value"]
        return default

    def _get_crop_params(self, crop_type: str) -> Dict[str, Decimal]:
        """Look up crop residue parameters with custom override support.

        Args:
            crop_type: Normalized crop type key.

        Returns:
            Dict with residue_to_crop_ratio, dry_matter_fraction, etc.

        Raises:
            ValueError: If crop type is not found.
        """
        key = crop_type.strip().lower().replace(" ", "_").replace("-", "_")
        with self._lock:
            if key in self._custom_crop_params:
                return dict(self._custom_crop_params[key])

        if key in CROP_RESIDUE_PARAMS:
            return dict(CROP_RESIDUE_PARAMS[key])

        raise ValueError(
            f"Unknown crop_type '{crop_type}'. Available: "
            f"{sorted(CROP_RESIDUE_PARAMS.keys())}"
        )

    def _get_burning_ef(self, crop_type: str) -> Dict[str, Decimal]:
        """Look up field burning emission factors with custom override support.

        Args:
            crop_type: Normalized crop type key.

        Returns:
            Dict with CH4 and N2O in g/kg dry matter burned.

        Raises:
            ValueError: If crop type is not found.
        """
        key = crop_type.strip().lower().replace(" ", "_").replace("-", "_")
        with self._lock:
            if key in self._custom_burning_ef:
                return dict(self._custom_burning_ef[key])

        if key in FIELD_BURNING_EF:
            return dict(FIELD_BURNING_EF[key])

        raise ValueError(
            f"Unknown crop_type for burning '{crop_type}'. Available: "
            f"{sorted(FIELD_BURNING_EF.keys())}"
        )

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def _emit_metrics(
        self,
        emission_source: str,
        calc_method: str,
        crop_or_input: str,
        duration_s: float,
    ) -> None:
        """Record Prometheus metrics for a cropland calculation.

        Args:
            emission_source: Source label (agricultural_soils, rice, etc.).
            calc_method: Calculation method label.
            crop_or_input: Crop type or input type label.
            duration_s: Wall-clock time in seconds.
        """
        if not self._enable_metrics or not _METRICS_AVAILABLE:
            return
        try:
            if _record_calculation is not None:
                _record_calculation(emission_source, calc_method, crop_or_input)
            if _observe_duration is not None:
                _observe_duration(emission_source, calc_method, duration_s)
        except Exception:
            logger.debug("Metrics recording failed (non-critical)", exc_info=True)

    def _emit_error_metric(self, error_type: str) -> None:
        """Record an error metric.

        Args:
            error_type: Error classification string.
        """
        if not self._enable_metrics or not _METRICS_AVAILABLE:
            return
        try:
            if _record_calculation_error is not None:
                _record_calculation_error(error_type)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Provenance recording
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        action: str,
        entity_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Record a provenance entry if tracker is available.

        Args:
            action: Action identifier.
            entity_id: Unique entity identifier.
            data: Data dictionary to record.
        """
        if self._provenance is None:
            return
        try:
            self._provenance.record(
                entity_type="cropland_emissions",
                action=action,
                entity_id=entity_id,
                data=data,
            )
        except Exception:
            logger.debug("Provenance recording failed (non-critical)", exc_info=True)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_non_negative(self, value: Decimal, name: str) -> None:
        """Validate that a value is non-negative.

        Args:
            value: Decimal value to validate.
            name: Parameter name for error messages.

        Raises:
            ValueError: If value is negative.
        """
        if value < _ZERO:
            raise ValueError(f"{name} must be >= 0, got {value}")

    def _validate_positive(self, value: Decimal, name: str) -> None:
        """Validate that a value is strictly positive.

        Args:
            value: Decimal value to validate.
            name: Parameter name for error messages.

        Raises:
            ValueError: If value is <= 0.
        """
        if value <= _ZERO:
            raise ValueError(f"{name} must be > 0, got {value}")

    def _validate_fraction(self, value: Decimal, name: str) -> None:
        """Validate that a value is in [0, 1].

        Args:
            value: Decimal value to validate.
            name: Parameter name for error messages.

        Raises:
            ValueError: If value is outside [0, 1].
        """
        if value < _ZERO or value > _ONE:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _build_result(
        self,
        calc_id: str,
        status: str,
        emission_source: str,
        gases: Dict[str, Dict[str, Decimal]],
        total_co2e_tonnes: Decimal,
        gwp_source: str,
        trace: List[str],
        elapsed_ms: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a standardized calculation result dictionary.

        Args:
            calc_id: Unique calculation identifier.
            status: Result status (SUCCESS/FAILED).
            emission_source: Emission source category.
            gases: Per-gas breakdown: {gas: {kg, tonnes, co2e_kg, co2e_tonnes}}.
            total_co2e_tonnes: Total CO2e in tonnes.
            gwp_source: GWP assessment report used.
            trace: Calculation trace steps.
            elapsed_ms: Wall-clock time in milliseconds.
            extra: Additional key-value pairs to include.

        Returns:
            Complete result dictionary.
        """
        provenance_input = {
            "calculation_id": calc_id,
            "emission_source": emission_source,
            "total_co2e_tonnes": str(total_co2e_tonnes),
            "gwp_source": gwp_source,
        }
        for gas_name, gas_data in sorted(gases.items()):
            provenance_input[f"{gas_name}_tonnes"] = str(gas_data.get("tonnes", _ZERO))

        provenance_hash = _compute_hash(provenance_input)
        trace.append(f"[P] Provenance hash: {provenance_hash[:16]}...")
        trace.append(f"[T] Processing time: {elapsed_ms:.3f} ms")

        self._record_provenance(
            action=f"calculate_{emission_source}",
            entity_id=calc_id,
            data={
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "provenance_hash": provenance_hash,
            },
        )

        # Build per-gas fields
        result: Dict[str, Any] = {
            "status": status,
            "calculation_id": calc_id,
            "emission_source": emission_source,
            "gwp_source": gwp_source,
        }

        for gas_name, gas_data in sorted(gases.items()):
            gas_lower = gas_name.lower()
            result[f"{gas_lower}_kg"] = str(gas_data.get("kg", _ZERO))
            result[f"{gas_lower}_tonnes"] = str(gas_data.get("tonnes", _ZERO))
            result[f"{gas_lower}_co2e_tonnes"] = str(
                gas_data.get("co2e_tonnes", _ZERO)
            )

        result["total_co2e_tonnes"] = str(_Q(total_co2e_tonnes))
        result["calculation_trace"] = trace
        result["provenance_hash"] = provenance_hash
        result["processing_time_ms"] = round(elapsed_ms, 3)
        result["calculated_at"] = _utcnow().isoformat()

        if extra:
            result.update(extra)

        self._increment_calculations()
        return result

    # ------------------------------------------------------------------
    # Error handler
    # ------------------------------------------------------------------

    def _handle_error(
        self,
        calc_id: str,
        emission_source: str,
        exc: Exception,
        trace: List[str],
        start_time: float,
    ) -> Dict[str, Any]:
        """Build a standardized error result dictionary.

        Args:
            calc_id: Calculation identifier.
            emission_source: Emission source that failed.
            exc: Exception that was raised.
            trace: Calculation trace up to the point of failure.
            start_time: time.monotonic() start value.

        Returns:
            Error result dictionary with FAILED status.
        """
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            "%s calculation %s failed: %s",
            emission_source, calc_id, exc, exc_info=True,
        )
        self._emit_error_metric("calculation_error")
        self._increment_errors()

        trace.append(f"[E] Error: {exc}")
        trace.append(f"[T] Processing time: {elapsed_ms:.3f} ms")

        return {
            "status": "FAILED",
            "calculation_id": calc_id,
            "emission_source": emission_source,
            "error_message": str(exc),
            "n2o_kg": str(_ZERO),
            "n2o_tonnes": str(_ZERO),
            "n2o_co2e_tonnes": str(_ZERO),
            "co2_kg": str(_ZERO),
            "co2_tonnes": str(_ZERO),
            "co2_co2e_tonnes": str(_ZERO),
            "ch4_kg": str(_ZERO),
            "ch4_tonnes": str(_ZERO),
            "ch4_co2e_tonnes": str(_ZERO),
            "total_co2e_tonnes": str(_ZERO),
            "gwp_source": "",
            "calculation_trace": trace,
            "provenance_hash": "",
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": _utcnow().isoformat(),
        }

    # ==================================================================
    # PUBLIC API 1: Direct Soil N2O (IPCC Eq 11.1)
    # ==================================================================

    def calculate_direct_n2o(
        self,
        synthetic_n_kg: Any,
        organic_n_kg: Any = 0,
        crop_residue_n_kg: Any = 0,
        som_n_kg: Any = 0,
        organic_soil_area_ha: Any = 0,
        organic_soil_area_forest_ha: Any = 0,
        prp_n_kg: Any = 0,
        prp_animal_type: str = "cattle",
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate direct N2O emissions from managed agricultural soils.

        Implements IPCC 2006 Vol 4 Ch 11 Equation 11.1:
            N2O_direct = [(F_SN + F_ON + F_CR + F_SOM) * EF1
                          + F_OS_CG * EF2_CG + F_OS_F * EF2_F
                          + F_PRP * EF3_PRP] * 44/28

        Args:
            synthetic_n_kg: Synthetic fertilizer N applied (kg N/yr).
            organic_n_kg: Organic fertilizer N applied (kg N/yr).
            crop_residue_n_kg: N from crop residues returned to soil (kg N/yr).
            som_n_kg: N from soil organic matter mineralization (kg N/yr).
            organic_soil_area_ha: Area of drained organic soils,
                cropland/grassland (ha).
            organic_soil_area_forest_ha: Area of drained organic soils,
                forest land (ha).
            prp_n_kg: N deposited by grazing animals on PRP (kg N/yr).
            prp_animal_type: Animal type for EF3 selection. One of
                ``"cattle"``, ``"poultry"``, or ``"other"``.
            gwp_source: GWP assessment report override.

        Returns:
            Result dict with N2O emissions breakdown and CO2e.
        """
        start = time.monotonic()
        calc_id = f"direct_n2o_{uuid4().hex[:12]}"
        trace: List[str] = []
        gwp_src = (gwp_source or self._default_gwp_source).upper()

        try:
            # Convert and validate inputs
            f_sn = _D(synthetic_n_kg)
            f_on = _D(organic_n_kg)
            f_cr = _D(crop_residue_n_kg)
            f_som = _D(som_n_kg)
            f_os_cg = _D(organic_soil_area_ha)
            f_os_f = _D(organic_soil_area_forest_ha)
            f_prp = _D(prp_n_kg)

            self._validate_non_negative(f_sn, "synthetic_n_kg")
            self._validate_non_negative(f_on, "organic_n_kg")
            self._validate_non_negative(f_cr, "crop_residue_n_kg")
            self._validate_non_negative(f_som, "som_n_kg")
            self._validate_non_negative(f_os_cg, "organic_soil_area_ha")
            self._validate_non_negative(f_os_f, "organic_soil_area_forest_ha")
            self._validate_non_negative(f_prp, "prp_n_kg")

            # Resolve emission factors
            ef1 = self._get_soil_ef("EF1", EF1_DEFAULT)
            ef2_cg = self._get_soil_ef("EF2_CG", EF2_CG_DEFAULT)
            ef2_f = self._get_soil_ef("EF2_F", EF2_F_DEFAULT)

            # Resolve EF3_PRP based on animal type
            animal = prp_animal_type.strip().lower()
            if animal in ("cattle", "poultry"):
                ef3_prp = self._get_soil_ef("EF3_PRP_CP", EF3_PRP_CATTLE_POULTRY)
            else:
                ef3_prp = self._get_soil_ef("EF3_PRP_OTHER", EF3_PRP_OTHER)

            trace.append(
                f"[1] Inputs: F_SN={f_sn}, F_ON={f_on}, F_CR={f_cr}, "
                f"F_SOM={f_som}, F_OS_CG={f_os_cg} ha, F_OS_F={f_os_f} ha, "
                f"F_PRP={f_prp}, animal={animal}"
            )
            trace.append(
                f"[2] EFs: EF1={ef1}, EF2_CG={ef2_cg}, EF2_F={ef2_f}, "
                f"EF3_PRP={ef3_prp}"
            )

            # Step 1: N inputs pathway (F_SN + F_ON + F_CR + F_SOM) * EF1
            n_inputs_total = _Q(f_sn + f_on + f_cr + f_som)
            n2o_n_inputs = _Q(n_inputs_total * ef1)
            trace.append(
                f"[3] N inputs total = {n_inputs_total} kg N -> "
                f"N2O-N from inputs = {n2o_n_inputs} kg N2O-N"
            )

            # Step 2: Organic soils pathway
            n2o_n_os_cg = _Q(f_os_cg * ef2_cg)
            n2o_n_os_f = _Q(f_os_f * ef2_f)
            trace.append(
                f"[4] Organic soils: CG={f_os_cg}ha * EF2_CG={ef2_cg} = "
                f"{n2o_n_os_cg} kg N2O-N; F={f_os_f}ha * EF2_F={ef2_f} = "
                f"{n2o_n_os_f} kg N2O-N"
            )

            # Step 3: PRP pathway
            n2o_n_prp = _Q(f_prp * ef3_prp)
            trace.append(
                f"[5] PRP: {f_prp} kg N * EF3={ef3_prp} = "
                f"{n2o_n_prp} kg N2O-N"
            )

            # Step 4: Total N2O-N and convert to N2O
            total_n2o_n = _Q(n2o_n_inputs + n2o_n_os_cg + n2o_n_os_f + n2o_n_prp)
            total_n2o_kg = _Q(total_n2o_n * N2O_MOLECULAR_RATIO)
            total_n2o_tonnes = _Q(total_n2o_kg * KG_TO_TONNES)
            trace.append(
                f"[6] Total N2O-N = {total_n2o_n} kg -> "
                f"N2O = {total_n2o_kg} kg (* 44/28) = "
                f"{total_n2o_tonnes} tonnes"
            )

            # Step 5: CO2e conversion
            gwp_n2o = self._resolve_gwp("N2O", gwp_src)
            co2e_kg = _Q(total_n2o_kg * gwp_n2o)
            co2e_tonnes = _Q(co2e_kg * KG_TO_TONNES)
            trace.append(
                f"[7] CO2e: {total_n2o_kg} kg N2O * GWP({gwp_src})={gwp_n2o} = "
                f"{co2e_tonnes} t CO2e"
            )

            elapsed_ms = (time.monotonic() - start) * 1000

            gases = {
                "N2O": {
                    "kg": _Q(total_n2o_kg),
                    "tonnes": total_n2o_tonnes,
                    "co2e_tonnes": co2e_tonnes,
                },
            }

            result = self._build_result(
                calc_id=calc_id,
                status="SUCCESS",
                emission_source="direct_soil_n2o",
                gases=gases,
                total_co2e_tonnes=co2e_tonnes,
                gwp_source=gwp_src,
                trace=trace,
                elapsed_ms=elapsed_ms,
                extra={
                    "n2o_n_from_inputs_kg": str(n2o_n_inputs),
                    "n2o_n_from_organic_soils_cg_kg": str(n2o_n_os_cg),
                    "n2o_n_from_organic_soils_f_kg": str(n2o_n_os_f),
                    "n2o_n_from_prp_kg": str(n2o_n_prp),
                    "total_n2o_n_kg": str(total_n2o_n),
                    "ef1": str(ef1),
                    "ef2_cg": str(ef2_cg),
                    "ef2_f": str(ef2_f),
                    "ef3_prp": str(ef3_prp),
                    "prp_animal_type": animal,
                },
            )

            self._emit_metrics(
                "agricultural_soils", "ipcc_tier1", "fertilizer",
                elapsed_ms / 1000,
            )
            return result

        except Exception as exc:
            return self._handle_error(
                calc_id, "direct_soil_n2o", exc, trace, start,
            )

    # ==================================================================
    # PUBLIC API 2: Indirect N2O (Volatilization + Leaching)
    # ==================================================================

    def calculate_indirect_n2o(
        self,
        synthetic_n_kg: Any,
        organic_n_kg: Any = 0,
        prp_n_kg: Any = 0,
        crop_residue_n_kg: Any = 0,
        som_n_kg: Any = 0,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate indirect N2O emissions from volatilization and leaching.

        Implements IPCC 2006 Vol 4 Ch 11:
            Eq 11.9 (Volatilization):
                N2O_ATD = [(F_SN * Frac_GASF) + ((F_ON + F_PRP) * Frac_GASM)]
                          * EF4 * 44/28

            Eq 11.10 (Leaching):
                N2O_L = (F_SN + F_ON + F_PRP + F_CR + F_SOM)
                        * Frac_LEACH * EF5 * 44/28

        Args:
            synthetic_n_kg: Synthetic fertilizer N applied (kg N/yr).
            organic_n_kg: Organic fertilizer N applied (kg N/yr).
            prp_n_kg: N deposited by grazing animals (kg N/yr).
            crop_residue_n_kg: N from crop residues returned to soil (kg N/yr).
            som_n_kg: N from soil organic matter mineralization (kg N/yr).
            gwp_source: GWP assessment report override.

        Returns:
            Result dict with volatilization and leaching N2O breakdown.
        """
        start = time.monotonic()
        calc_id = f"indirect_n2o_{uuid4().hex[:12]}"
        trace: List[str] = []
        gwp_src = (gwp_source or self._default_gwp_source).upper()

        try:
            f_sn = _D(synthetic_n_kg)
            f_on = _D(organic_n_kg)
            f_prp = _D(prp_n_kg)
            f_cr = _D(crop_residue_n_kg)
            f_som = _D(som_n_kg)

            self._validate_non_negative(f_sn, "synthetic_n_kg")
            self._validate_non_negative(f_on, "organic_n_kg")
            self._validate_non_negative(f_prp, "prp_n_kg")
            self._validate_non_negative(f_cr, "crop_residue_n_kg")
            self._validate_non_negative(f_som, "som_n_kg")

            # Resolve factors
            frac_gasf = self._get_soil_ef("FRAC_GASF", FRAC_GASF_DEFAULT)
            frac_gasm = self._get_soil_ef("FRAC_GASM", FRAC_GASM_DEFAULT)
            ef4 = self._get_soil_ef("EF4", EF4_DEFAULT)
            frac_leach = self._get_soil_ef("FRAC_LEACH", FRAC_LEACH_DEFAULT)
            ef5 = self._get_soil_ef("EF5", EF5_DEFAULT)

            trace.append(
                f"[1] Inputs: F_SN={f_sn}, F_ON={f_on}, F_PRP={f_prp}, "
                f"F_CR={f_cr}, F_SOM={f_som}"
            )
            trace.append(
                f"[2] Fractions: Frac_GASF={frac_gasf}, Frac_GASM={frac_gasm}, "
                f"EF4={ef4}, Frac_LEACH={frac_leach}, EF5={ef5}"
            )

            # Volatilization (Eq 11.9)
            n_vol_synthetic = _Q(f_sn * frac_gasf)
            n_vol_organic = _Q((f_on + f_prp) * frac_gasm)
            n_volatilized = _Q(n_vol_synthetic + n_vol_organic)
            n2o_n_vol = _Q(n_volatilized * ef4)
            n2o_vol_kg = _Q(n2o_n_vol * N2O_MOLECULAR_RATIO)
            trace.append(
                f"[3] Volatilization: synthetic={n_vol_synthetic}, "
                f"organic+PRP={n_vol_organic}, total_vol={n_volatilized}, "
                f"N2O-N={n2o_n_vol}, N2O={n2o_vol_kg} kg"
            )

            # Leaching (Eq 11.10)
            n_total_leach = _Q(f_sn + f_on + f_prp + f_cr + f_som)
            n_leached = _Q(n_total_leach * frac_leach)
            n2o_n_leach = _Q(n_leached * ef5)
            n2o_leach_kg = _Q(n2o_n_leach * N2O_MOLECULAR_RATIO)
            trace.append(
                f"[4] Leaching: total_N={n_total_leach}, "
                f"leached={n_leached}, N2O-N={n2o_n_leach}, "
                f"N2O={n2o_leach_kg} kg"
            )

            # Total indirect N2O
            total_n2o_kg = _Q(n2o_vol_kg + n2o_leach_kg)
            total_n2o_tonnes = _Q(total_n2o_kg * KG_TO_TONNES)
            trace.append(
                f"[5] Total indirect N2O: vol={n2o_vol_kg} + leach={n2o_leach_kg} "
                f"= {total_n2o_kg} kg = {total_n2o_tonnes} tonnes"
            )

            # CO2e
            gwp_n2o = self._resolve_gwp("N2O", gwp_src)
            co2e_tonnes = _Q(total_n2o_tonnes * gwp_n2o)
            trace.append(
                f"[6] CO2e: {total_n2o_tonnes} t * GWP={gwp_n2o} = "
                f"{co2e_tonnes} t CO2e"
            )

            elapsed_ms = (time.monotonic() - start) * 1000

            gases = {
                "N2O": {
                    "kg": total_n2o_kg,
                    "tonnes": total_n2o_tonnes,
                    "co2e_tonnes": co2e_tonnes,
                },
            }

            result = self._build_result(
                calc_id=calc_id,
                status="SUCCESS",
                emission_source="indirect_soil_n2o",
                gases=gases,
                total_co2e_tonnes=co2e_tonnes,
                gwp_source=gwp_src,
                trace=trace,
                elapsed_ms=elapsed_ms,
                extra={
                    "n2o_volatilization_kg": str(n2o_vol_kg),
                    "n2o_volatilization_tonnes": str(_Q(n2o_vol_kg * KG_TO_TONNES)),
                    "n2o_leaching_kg": str(n2o_leach_kg),
                    "n2o_leaching_tonnes": str(_Q(n2o_leach_kg * KG_TO_TONNES)),
                    "n_volatilized_kg": str(n_volatilized),
                    "n_leached_kg": str(n_leached),
                    "frac_gasf": str(frac_gasf),
                    "frac_gasm": str(frac_gasm),
                    "ef4": str(ef4),
                    "frac_leach": str(frac_leach),
                    "ef5": str(ef5),
                },
            )

            self._emit_metrics(
                "agricultural_soils", "ipcc_tier1", "fertilizer",
                elapsed_ms / 1000,
            )
            return result

        except Exception as exc:
            return self._handle_error(
                calc_id, "indirect_soil_n2o", exc, trace, start,
            )

    # ==================================================================
    # PUBLIC API 3: Total Soil N2O (Direct + Indirect)
    # ==================================================================

    def calculate_total_soil_n2o(
        self,
        synthetic_n_kg: Any,
        organic_n_kg: Any = 0,
        crop_residue_n_kg: Any = 0,
        som_n_kg: Any = 0,
        organic_soil_area_ha: Any = 0,
        organic_soil_area_forest_ha: Any = 0,
        prp_n_kg: Any = 0,
        prp_animal_type: str = "cattle",
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate combined direct and indirect N2O from agricultural soils.

        Combines calculate_direct_n2o and calculate_indirect_n2o results into
        a single comprehensive output.

        Args:
            synthetic_n_kg: Synthetic fertilizer N applied (kg N/yr).
            organic_n_kg: Organic fertilizer N applied (kg N/yr).
            crop_residue_n_kg: N from crop residues returned to soil (kg N/yr).
            som_n_kg: N from soil organic matter mineralization (kg N/yr).
            organic_soil_area_ha: Area of drained organic soils,
                cropland/grassland (ha).
            organic_soil_area_forest_ha: Area of drained organic soils,
                forest (ha).
            prp_n_kg: N deposited by grazing animals (kg N/yr).
            prp_animal_type: Animal type for EF3 selection.
            gwp_source: GWP assessment report override.

        Returns:
            Result dict with combined direct + indirect N2O and CO2e.
        """
        start = time.monotonic()
        calc_id = f"total_n2o_{uuid4().hex[:12]}"
        trace: List[str] = []
        gwp_src = (gwp_source or self._default_gwp_source).upper()

        try:
            trace.append("[1] Calculating direct soil N2O...")

            direct_result = self.calculate_direct_n2o(
                synthetic_n_kg=synthetic_n_kg,
                organic_n_kg=organic_n_kg,
                crop_residue_n_kg=crop_residue_n_kg,
                som_n_kg=som_n_kg,
                organic_soil_area_ha=organic_soil_area_ha,
                organic_soil_area_forest_ha=organic_soil_area_forest_ha,
                prp_n_kg=prp_n_kg,
                prp_animal_type=prp_animal_type,
                gwp_source=gwp_src,
            )

            if direct_result["status"] != "SUCCESS":
                raise ValueError(
                    f"Direct N2O calculation failed: "
                    f"{direct_result.get('error_message', 'unknown')}"
                )

            trace.append("[2] Calculating indirect soil N2O...")

            indirect_result = self.calculate_indirect_n2o(
                synthetic_n_kg=synthetic_n_kg,
                organic_n_kg=organic_n_kg,
                prp_n_kg=prp_n_kg,
                crop_residue_n_kg=crop_residue_n_kg,
                som_n_kg=som_n_kg,
                gwp_source=gwp_src,
            )

            if indirect_result["status"] != "SUCCESS":
                raise ValueError(
                    f"Indirect N2O calculation failed: "
                    f"{indirect_result.get('error_message', 'unknown')}"
                )

            # Aggregate
            direct_n2o_kg = _D(direct_result["n2o_kg"])
            indirect_n2o_kg = _D(indirect_result["n2o_kg"])
            total_n2o_kg = _Q(direct_n2o_kg + indirect_n2o_kg)
            total_n2o_tonnes = _Q(total_n2o_kg * KG_TO_TONNES)

            direct_co2e = _D(direct_result["total_co2e_tonnes"])
            indirect_co2e = _D(indirect_result["total_co2e_tonnes"])
            total_co2e = _Q(direct_co2e + indirect_co2e)

            trace.append(
                f"[3] Direct N2O: {direct_n2o_kg} kg, "
                f"Indirect N2O: {indirect_n2o_kg} kg, "
                f"Total N2O: {total_n2o_kg} kg = {total_n2o_tonnes} t"
            )
            trace.append(
                f"[4] Total CO2e: direct={direct_co2e} + "
                f"indirect={indirect_co2e} = {total_co2e} t"
            )

            elapsed_ms = (time.monotonic() - start) * 1000

            gases = {
                "N2O": {
                    "kg": total_n2o_kg,
                    "tonnes": total_n2o_tonnes,
                    "co2e_tonnes": total_co2e,
                },
            }

            result = self._build_result(
                calc_id=calc_id,
                status="SUCCESS",
                emission_source="total_soil_n2o",
                gases=gases,
                total_co2e_tonnes=total_co2e,
                gwp_source=gwp_src,
                trace=trace,
                elapsed_ms=elapsed_ms,
                extra={
                    "direct_n2o_kg": str(direct_n2o_kg),
                    "direct_n2o_tonnes": direct_result["n2o_tonnes"],
                    "direct_co2e_tonnes": str(direct_co2e),
                    "indirect_n2o_kg": str(indirect_n2o_kg),
                    "indirect_n2o_tonnes": indirect_result["n2o_tonnes"],
                    "indirect_co2e_tonnes": str(indirect_co2e),
                    "indirect_volatilization_kg": indirect_result.get(
                        "n2o_volatilization_kg", str(_ZERO)
                    ),
                    "indirect_leaching_kg": indirect_result.get(
                        "n2o_leaching_kg", str(_ZERO)
                    ),
                    "direct_calculation_id": direct_result["calculation_id"],
                    "indirect_calculation_id": indirect_result["calculation_id"],
                },
            )
            return result

        except Exception as exc:
            return self._handle_error(
                calc_id, "total_soil_n2o", exc, trace, start,
            )

    # ==================================================================
    # PUBLIC API 4: Liming CO2 (IPCC Eq 11.12)
    # ==================================================================

    def calculate_liming_co2(
        self,
        limestone_tonnes: Any = 0,
        dolomite_tonnes: Any = 0,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate CO2 emissions from liming of agricultural soils.

        Implements IPCC 2006 Vol 4 Ch 11 Equation 11.12:
            CO2_liming = (M_limestone * 0.12 + M_dolomite * 0.13) * 44/12

        Args:
            limestone_tonnes: Mass of limestone (CaCO3) applied (tonnes).
            dolomite_tonnes: Mass of dolomite (CaMg(CO3)2) applied (tonnes).
            gwp_source: GWP assessment report override (CO2 GWP is always 1).

        Returns:
            Result dict with CO2 from liming.
        """
        start = time.monotonic()
        calc_id = f"liming_co2_{uuid4().hex[:12]}"
        trace: List[str] = []
        gwp_src = (gwp_source or self._default_gwp_source).upper()

        try:
            m_ls = _D(limestone_tonnes)
            m_dol = _D(dolomite_tonnes)

            self._validate_non_negative(m_ls, "limestone_tonnes")
            self._validate_non_negative(m_dol, "dolomite_tonnes")

            if m_ls == _ZERO and m_dol == _ZERO:
                raise ValueError(
                    "At least one of limestone_tonnes or dolomite_tonnes "
                    "must be > 0"
                )

            trace.append(
                f"[1] Inputs: limestone={m_ls} t, dolomite={m_dol} t"
            )

            # CO2 from limestone: M * 0.12 * 44/12
            co2_limestone = _Q(m_ls * LIMESTONE_EF * CO2_C_RATIO)
            trace.append(
                f"[2] CO2 from limestone: {m_ls} * {LIMESTONE_EF} * "
                f"{CO2_C_RATIO} = {co2_limestone} t CO2"
            )

            # CO2 from dolomite: M * 0.13 * 44/12
            co2_dolomite = _Q(m_dol * DOLOMITE_EF * CO2_C_RATIO)
            trace.append(
                f"[3] CO2 from dolomite: {m_dol} * {DOLOMITE_EF} * "
                f"{CO2_C_RATIO} = {co2_dolomite} t CO2"
            )

            # Total CO2
            total_co2_tonnes = _Q(co2_limestone + co2_dolomite)
            total_co2_kg = _Q(total_co2_tonnes * _THOUSAND)
            trace.append(
                f"[4] Total CO2: {co2_limestone} + {co2_dolomite} = "
                f"{total_co2_tonnes} t"
            )

            # CO2e (GWP_CO2 = 1, so co2e = co2)
            co2e_tonnes = total_co2_tonnes

            elapsed_ms = (time.monotonic() - start) * 1000

            gases = {
                "CO2": {
                    "kg": total_co2_kg,
                    "tonnes": total_co2_tonnes,
                    "co2e_tonnes": co2e_tonnes,
                },
            }

            result = self._build_result(
                calc_id=calc_id,
                status="SUCCESS",
                emission_source="liming",
                gases=gases,
                total_co2e_tonnes=co2e_tonnes,
                gwp_source=gwp_src,
                trace=trace,
                elapsed_ms=elapsed_ms,
                extra={
                    "co2_from_limestone_tonnes": str(co2_limestone),
                    "co2_from_dolomite_tonnes": str(co2_dolomite),
                    "limestone_tonnes": str(m_ls),
                    "dolomite_tonnes": str(m_dol),
                    "limestone_ef": str(LIMESTONE_EF),
                    "dolomite_ef": str(DOLOMITE_EF),
                },
            )

            self._emit_metrics(
                "liming", "ipcc_tier1", "liming",
                elapsed_ms / 1000,
            )
            return result

        except Exception as exc:
            return self._handle_error(
                calc_id, "liming", exc, trace, start,
            )

    # ==================================================================
    # PUBLIC API 5: Urea CO2 (IPCC Eq 11.13)
    # ==================================================================

    def calculate_urea_co2(
        self,
        urea_tonnes: Any,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate CO2 emissions from urea application to soils.

        Implements IPCC 2006 Vol 4 Ch 11 Equation 11.13:
            CO2_urea = M_urea * 0.20 * 44/12

        Args:
            urea_tonnes: Mass of urea (CO(NH2)2) applied (tonnes).
            gwp_source: GWP assessment report override.

        Returns:
            Result dict with CO2 from urea.
        """
        start = time.monotonic()
        calc_id = f"urea_co2_{uuid4().hex[:12]}"
        trace: List[str] = []
        gwp_src = (gwp_source or self._default_gwp_source).upper()

        try:
            m_urea = _D(urea_tonnes)
            self._validate_positive(m_urea, "urea_tonnes")

            trace.append(f"[1] Input: urea={m_urea} t")

            # CO2 = M_urea * 0.20 * 44/12
            co2_tonnes = _Q(m_urea * UREA_EF * CO2_C_RATIO)
            co2_kg = _Q(co2_tonnes * _THOUSAND)
            trace.append(
                f"[2] CO2: {m_urea} * {UREA_EF} * {CO2_C_RATIO} = "
                f"{co2_tonnes} t CO2"
            )

            co2e_tonnes = co2_tonnes

            elapsed_ms = (time.monotonic() - start) * 1000

            gases = {
                "CO2": {
                    "kg": co2_kg,
                    "tonnes": co2_tonnes,
                    "co2e_tonnes": co2e_tonnes,
                },
            }

            result = self._build_result(
                calc_id=calc_id,
                status="SUCCESS",
                emission_source="urea_application",
                gases=gases,
                total_co2e_tonnes=co2e_tonnes,
                gwp_source=gwp_src,
                trace=trace,
                elapsed_ms=elapsed_ms,
                extra={
                    "urea_tonnes": str(m_urea),
                    "urea_ef": str(UREA_EF),
                },
            )

            self._emit_metrics(
                "urea_application", "ipcc_tier1", "urea",
                elapsed_ms / 1000,
            )
            return result

        except Exception as exc:
            return self._handle_error(
                calc_id, "urea_application", exc, trace, start,
            )

    # ==================================================================
    # PUBLIC API 6: Rice CH4 (IPCC Eq 5.1)
    # ==================================================================

    def calculate_rice_ch4(
        self,
        area_ha: Any,
        cultivation_days: Any,
        water_regime: str = "continuously_flooded",
        pre_season_flooding: str = "not_flooded_short",
        organic_amendments: Optional[List[Dict[str, Any]]] = None,
        soil_type_scaling: Any = "1.0",
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate CH4 emissions from rice cultivation.

        Implements IPCC 2006 Vol 4 Ch 5 Equation 5.1:
            CH4_rice = EF_ijk * t_ijk * A_ijk * 10^-6  (Gg/yr)
            EF_ijk = EF_c * SF_w * SF_p * SF_o * SF_s
            SF_o = (1 + SUM[ROA_i * CFOA_i])^0.59

        Args:
            area_ha: Harvested rice area (hectares).
            cultivation_days: Duration of rice cultivation period (days).
            water_regime: Water management regime. One of:
                ``"continuously_flooded"``, ``"intermittent_single"``,
                ``"intermittent_multiple"``, ``"rainfed_regular"``,
                ``"rainfed_drought"``, ``"deep_water"``, ``"upland"``.
            pre_season_flooding: Pre-season water status. One of:
                ``"not_flooded_short"`` (<180 days), ``"not_flooded_long"``
                (>180 days), ``"flooded_short"`` (<30 days),
                ``"flooded_long"`` (>30 days).
            organic_amendments: List of amendment dicts, each with:
                - ``type`` (str): Amendment type (see OrganicAmendmentType).
                - ``rate_tonnes_ha`` (float): Application rate (t/ha).
            soil_type_scaling: Soil type scaling factor SF_s (default 1.0).
            gwp_source: GWP assessment report override.

        Returns:
            Result dict with daily EF, scaling factors, total CH4, and CO2e.
        """
        start = time.monotonic()
        calc_id = f"rice_ch4_{uuid4().hex[:12]}"
        trace: List[str] = []
        gwp_src = (gwp_source or self._default_gwp_source).upper()

        try:
            a = _D(area_ha)
            t_days = _D(cultivation_days)
            sf_s = _D(soil_type_scaling)

            self._validate_positive(a, "area_ha")
            self._validate_positive(t_days, "cultivation_days")
            self._validate_non_negative(sf_s, "soil_type_scaling")

            # Resolve water regime scaling factor
            wr_key = water_regime.strip().lower().replace(" ", "_").replace("-", "_")
            if wr_key not in WATER_REGIME_SF:
                raise ValueError(
                    f"Unknown water_regime '{water_regime}'. Valid: "
                    f"{list(WATER_REGIME_SF.keys())}"
                )
            sf_w = WATER_REGIME_SF[wr_key]

            # Resolve pre-season flooding scaling factor
            ps_key = pre_season_flooding.strip().lower().replace(
                " ", "_"
            ).replace("-", "_")
            if ps_key not in PRE_SEASON_SF:
                raise ValueError(
                    f"Unknown pre_season_flooding '{pre_season_flooding}'. Valid: "
                    f"{list(PRE_SEASON_SF.keys())}"
                )
            sf_p = PRE_SEASON_SF[ps_key]

            trace.append(
                f"[1] Inputs: area={a} ha, days={t_days}, "
                f"water_regime={wr_key} (SF_w={sf_w}), "
                f"pre_season={ps_key} (SF_p={sf_p}), SF_s={sf_s}"
            )

            # Calculate organic amendment scaling factor SF_o
            # SF_o = (1 + SUM[ROA_i * CFOA_i])^0.59
            amendments_used = organic_amendments or []
            roa_sum = _ZERO
            amendment_details: List[Dict[str, str]] = []

            for amend in amendments_used:
                amend_type = str(amend.get("type", "")).strip().lower().replace(
                    " ", "_"
                ).replace("-", "_")
                rate = _D(amend.get("rate_tonnes_ha", 0))
                self._validate_non_negative(rate, f"amendment rate ({amend_type})")

                if amend_type not in ORGANIC_AMENDMENT_CFOA:
                    raise ValueError(
                        f"Unknown amendment type '{amend_type}'. Valid: "
                        f"{list(ORGANIC_AMENDMENT_CFOA.keys())}"
                    )

                cfoa = ORGANIC_AMENDMENT_CFOA[amend_type]
                roa_contribution = _Q(rate * cfoa)
                roa_sum += roa_contribution

                amendment_details.append({
                    "type": amend_type,
                    "rate_tonnes_ha": str(rate),
                    "cfoa": str(cfoa),
                    "contribution": str(roa_contribution),
                })

            # SF_o = (1 + roa_sum)^0.59
            sf_o_base = _Q(_ONE + roa_sum)
            # Use float for fractional exponent, then convert back
            sf_o_float = float(sf_o_base) ** float(ORGANIC_AMENDMENT_EXPONENT)
            sf_o = _Q(_D(sf_o_float))

            trace.append(
                f"[2] Organic amendments: ROA_sum={roa_sum}, "
                f"SF_o_base={sf_o_base}, SF_o={sf_o} "
                f"(n_amendments={len(amendments_used)})"
            )

            # Baseline emission factor
            ef_c = self._get_soil_ef("EF_C_RICE", EF_C_RICE_DEFAULT)

            # Adjusted emission factor: EF_ijk = EF_c * SF_w * SF_p * SF_o * SF_s
            ef_adjusted = _Q(ef_c * sf_w * sf_p * sf_o * sf_s)
            trace.append(
                f"[3] EF_adjusted = EF_c({ef_c}) * SF_w({sf_w}) * SF_p({sf_p}) "
                f"* SF_o({sf_o}) * SF_s({sf_s}) = {ef_adjusted} kg CH4/ha/day"
            )

            # Total CH4: EF * days * area (kg CH4)
            ch4_kg = _Q(ef_adjusted * t_days * a)
            ch4_tonnes = _Q(ch4_kg * KG_TO_TONNES)
            # Also compute Gg: * 10^-6 (kg to Gg)
            ch4_gg = _Q(ch4_kg * _D("0.000001"))
            trace.append(
                f"[4] CH4 = {ef_adjusted} * {t_days} days * {a} ha = "
                f"{ch4_kg} kg = {ch4_tonnes} t = {ch4_gg} Gg"
            )

            # CO2e conversion
            gwp_ch4 = self._resolve_gwp("CH4", gwp_src)
            co2e_tonnes = _Q(ch4_tonnes * gwp_ch4)
            trace.append(
                f"[5] CO2e: {ch4_tonnes} t * GWP({gwp_src})={gwp_ch4} = "
                f"{co2e_tonnes} t CO2e"
            )

            elapsed_ms = (time.monotonic() - start) * 1000

            gases = {
                "CH4": {
                    "kg": ch4_kg,
                    "tonnes": ch4_tonnes,
                    "co2e_tonnes": co2e_tonnes,
                },
            }

            result = self._build_result(
                calc_id=calc_id,
                status="SUCCESS",
                emission_source="rice_cultivation",
                gases=gases,
                total_co2e_tonnes=co2e_tonnes,
                gwp_source=gwp_src,
                trace=trace,
                elapsed_ms=elapsed_ms,
                extra={
                    "area_ha": str(a),
                    "cultivation_days": str(t_days),
                    "water_regime": wr_key,
                    "pre_season_flooding": ps_key,
                    "ef_c_baseline": str(ef_c),
                    "daily_ef_adjusted": str(ef_adjusted),
                    "scaling_factors": {
                        "sf_w": str(sf_w),
                        "sf_p": str(sf_p),
                        "sf_o": str(sf_o),
                        "sf_s": str(sf_s),
                    },
                    "organic_amendments": amendment_details,
                    "ch4_gg": str(ch4_gg),
                },
            )

            self._emit_metrics(
                "rice_cultivation", "ipcc_tier1", "rice",
                elapsed_ms / 1000,
            )
            return result

        except Exception as exc:
            return self._handle_error(
                calc_id, "rice_cultivation", exc, trace, start,
            )

    # ==================================================================
    # PUBLIC API 7: Field Burning (IPCC Eq 2.27)
    # ==================================================================

    def calculate_field_burning(
        self,
        crop_type: str,
        area_burned_ha: Any,
        crop_yield_tonnes_ha: Any,
        burn_fraction: Any = "0.80",
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate CH4 and N2O emissions from field burning of crop residues.

        Implements IPCC 2006 Vol 4 Ch 2 Equation 2.27:
            L_fire = A * M_B * C_f * G_ef * 10^-3  (tonnes gas/yr)

        Where M_B is the mass of fuel (crop residue dry matter) available
        for burning, C_f is the combustion factor, and G_ef is the gas-
        specific emission factor.

        Args:
            crop_type: Crop type (see CropType enum values).
            area_burned_ha: Area of cropland burned (ha).
            crop_yield_tonnes_ha: Crop yield (tonnes fresh weight/ha).
            burn_fraction: Fraction of total residue that is burned (0-1).
                Default 0.80.
            gwp_source: GWP assessment report override.

        Returns:
            Result dict with CH4, N2O, and total CO2e from burning.
        """
        start = time.monotonic()
        calc_id = f"burn_{uuid4().hex[:12]}"
        trace: List[str] = []
        gwp_src = (gwp_source or self._default_gwp_source).upper()

        try:
            ct_key = crop_type.strip().lower().replace(" ", "_").replace("-", "_")
            a_burn = _D(area_burned_ha)
            yield_tha = _D(crop_yield_tonnes_ha)
            bf = _D(burn_fraction)

            self._validate_positive(a_burn, "area_burned_ha")
            self._validate_positive(yield_tha, "crop_yield_tonnes_ha")
            self._validate_fraction(bf, "burn_fraction")

            # Look up crop and burning parameters
            crop_params = self._get_crop_params(ct_key)
            burn_ef = self._get_burning_ef(ct_key)

            rpr = crop_params["residue_to_crop_ratio"]
            dm_frac = crop_params["dry_matter_fraction"]
            cf = crop_params["combustion_factor"]

            trace.append(
                f"[1] Inputs: crop={ct_key}, area={a_burn} ha, "
                f"yield={yield_tha} t/ha, burn_fraction={bf}"
            )
            trace.append(
                f"[2] Crop params: RPR={rpr}, DM={dm_frac}, CF={cf}"
            )
            trace.append(
                f"[3] Burning EFs: CH4={burn_ef['CH4']} g/kg DM, "
                f"N2O={burn_ef['N2O']} g/kg DM"
            )

            # Step 1: Calculate mass of residue dry matter
            # Total residue = yield * RPR * area
            total_residue_fresh = _Q(yield_tha * rpr * a_burn)
            total_residue_dm = _Q(total_residue_fresh * dm_frac)
            trace.append(
                f"[4] Residue: fresh={total_residue_fresh} t, "
                f"DM={total_residue_dm} t"
            )

            # Step 2: Mass actually burned
            mass_burned_dm = _Q(total_residue_dm * bf * cf)
            # Convert to kg for EF application (EFs are g/kg DM)
            mass_burned_kg = _Q(mass_burned_dm * _THOUSAND)
            trace.append(
                f"[5] Mass burned: {total_residue_dm} * {bf} * {cf} = "
                f"{mass_burned_dm} t DM = {mass_burned_kg} kg DM"
            )

            # Step 3: CH4 emissions (g/kg DM -> tonnes)
            ch4_g = _Q(mass_burned_kg * burn_ef["CH4"])
            ch4_kg = _Q(ch4_g * _D("0.001"))
            ch4_tonnes = _Q(ch4_kg * KG_TO_TONNES)
            trace.append(
                f"[6] CH4: {mass_burned_kg} kg * {burn_ef['CH4']} g/kg = "
                f"{ch4_g} g = {ch4_kg} kg = {ch4_tonnes} t"
            )

            # Step 4: N2O emissions (g/kg DM -> tonnes)
            n2o_g = _Q(mass_burned_kg * burn_ef["N2O"])
            n2o_kg = _Q(n2o_g * _D("0.001"))
            n2o_tonnes = _Q(n2o_kg * KG_TO_TONNES)
            trace.append(
                f"[7] N2O: {mass_burned_kg} kg * {burn_ef['N2O']} g/kg = "
                f"{n2o_g} g = {n2o_kg} kg = {n2o_tonnes} t"
            )

            # Step 5: CO2e
            gwp_ch4 = self._resolve_gwp("CH4", gwp_src)
            gwp_n2o = self._resolve_gwp("N2O", gwp_src)
            co2e_ch4 = _Q(ch4_tonnes * gwp_ch4)
            co2e_n2o = _Q(n2o_tonnes * gwp_n2o)
            total_co2e = _Q(co2e_ch4 + co2e_n2o)
            trace.append(
                f"[8] CO2e: CH4={co2e_ch4} t + N2O={co2e_n2o} t = "
                f"{total_co2e} t CO2e"
            )

            elapsed_ms = (time.monotonic() - start) * 1000

            gases = {
                "CH4": {
                    "kg": ch4_kg,
                    "tonnes": ch4_tonnes,
                    "co2e_tonnes": co2e_ch4,
                },
                "N2O": {
                    "kg": n2o_kg,
                    "tonnes": n2o_tonnes,
                    "co2e_tonnes": co2e_n2o,
                },
            }

            result = self._build_result(
                calc_id=calc_id,
                status="SUCCESS",
                emission_source="field_burning",
                gases=gases,
                total_co2e_tonnes=total_co2e,
                gwp_source=gwp_src,
                trace=trace,
                elapsed_ms=elapsed_ms,
                extra={
                    "crop_type": ct_key,
                    "area_burned_ha": str(a_burn),
                    "crop_yield_tonnes_ha": str(yield_tha),
                    "burn_fraction": str(bf),
                    "residue_to_crop_ratio": str(rpr),
                    "mass_burned_dm_tonnes": str(mass_burned_dm),
                    "co2e_from_ch4": str(co2e_ch4),
                    "co2e_from_n2o": str(co2e_n2o),
                },
            )

            self._emit_metrics(
                "field_burning", "ipcc_tier1", ct_key,
                elapsed_ms / 1000,
            )
            return result

        except Exception as exc:
            return self._handle_error(
                calc_id, "field_burning", exc, trace, start,
            )

    # ==================================================================
    # PUBLIC API 8: Cropland Total (combined calculation)
    # ==================================================================

    def calculate_cropland_total(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate combined emissions from all cropland sources.

        Aggregates results from soil N2O, liming, urea, rice cultivation,
        and field burning into a single comprehensive result.

        Args:
            inputs: Dictionary containing parameters for each source:
                - ``soil_n2o`` (dict, optional): Parameters for
                  calculate_total_soil_n2o.
                - ``liming`` (dict, optional): Parameters for
                  calculate_liming_co2.
                - ``urea`` (dict, optional): Parameters for
                  calculate_urea_co2.
                - ``rice`` (dict, optional): Parameters for
                  calculate_rice_ch4.
                - ``burning`` (dict, optional): Parameters for
                  calculate_field_burning.
                - ``gwp_source`` (str, optional): GWP source for all
                  sub-calculations.

        Returns:
            Aggregated result dict with per-source breakdown and totals.
        """
        start = time.monotonic()
        calc_id = f"cropland_total_{uuid4().hex[:12]}"
        trace: List[str] = []
        gwp_src = str(inputs.get("gwp_source", self._default_gwp_source)).upper()

        try:
            sub_results: Dict[str, Dict[str, Any]] = {}
            total_n2o_kg = _ZERO
            total_co2_kg = _ZERO
            total_ch4_kg = _ZERO
            total_co2e = _ZERO
            sources_calculated: List[str] = []

            # --- Soil N2O ---
            soil_params = inputs.get("soil_n2o")
            if soil_params is not None:
                trace.append("[1] Calculating soil N2O emissions...")
                soil_result = self.calculate_total_soil_n2o(
                    gwp_source=gwp_src, **soil_params,
                )
                sub_results["soil_n2o"] = soil_result
                if soil_result["status"] == "SUCCESS":
                    total_n2o_kg += _D(soil_result["n2o_kg"])
                    total_co2e += _D(soil_result["total_co2e_tonnes"])
                    sources_calculated.append("soil_n2o")
                    trace.append(
                        f"  -> Soil N2O: {soil_result['n2o_tonnes']} t N2O, "
                        f"{soil_result['total_co2e_tonnes']} t CO2e"
                    )
                else:
                    trace.append(
                        f"  -> Soil N2O FAILED: "
                        f"{soil_result.get('error_message', 'unknown')}"
                    )

            # --- Liming ---
            liming_params = inputs.get("liming")
            if liming_params is not None:
                trace.append("[2] Calculating liming CO2 emissions...")
                liming_result = self.calculate_liming_co2(
                    gwp_source=gwp_src, **liming_params,
                )
                sub_results["liming"] = liming_result
                if liming_result["status"] == "SUCCESS":
                    total_co2_kg += _D(liming_result["co2_kg"])
                    total_co2e += _D(liming_result["total_co2e_tonnes"])
                    sources_calculated.append("liming")
                    trace.append(
                        f"  -> Liming: {liming_result['co2_tonnes']} t CO2"
                    )
                else:
                    trace.append(
                        f"  -> Liming FAILED: "
                        f"{liming_result.get('error_message', 'unknown')}"
                    )

            # --- Urea ---
            urea_params = inputs.get("urea")
            if urea_params is not None:
                trace.append("[3] Calculating urea CO2 emissions...")
                urea_result = self.calculate_urea_co2(
                    gwp_source=gwp_src, **urea_params,
                )
                sub_results["urea"] = urea_result
                if urea_result["status"] == "SUCCESS":
                    total_co2_kg += _D(urea_result["co2_kg"])
                    total_co2e += _D(urea_result["total_co2e_tonnes"])
                    sources_calculated.append("urea")
                    trace.append(
                        f"  -> Urea: {urea_result['co2_tonnes']} t CO2"
                    )
                else:
                    trace.append(
                        f"  -> Urea FAILED: "
                        f"{urea_result.get('error_message', 'unknown')}"
                    )

            # --- Rice ---
            rice_params = inputs.get("rice")
            if rice_params is not None:
                trace.append("[4] Calculating rice CH4 emissions...")
                rice_result = self.calculate_rice_ch4(
                    gwp_source=gwp_src, **rice_params,
                )
                sub_results["rice"] = rice_result
                if rice_result["status"] == "SUCCESS":
                    total_ch4_kg += _D(rice_result["ch4_kg"])
                    total_co2e += _D(rice_result["total_co2e_tonnes"])
                    sources_calculated.append("rice")
                    trace.append(
                        f"  -> Rice: {rice_result['ch4_tonnes']} t CH4, "
                        f"{rice_result['total_co2e_tonnes']} t CO2e"
                    )
                else:
                    trace.append(
                        f"  -> Rice FAILED: "
                        f"{rice_result.get('error_message', 'unknown')}"
                    )

            # --- Field Burning ---
            burning_params = inputs.get("burning")
            if burning_params is not None:
                trace.append("[5] Calculating field burning emissions...")
                burn_result = self.calculate_field_burning(
                    gwp_source=gwp_src, **burning_params,
                )
                sub_results["burning"] = burn_result
                if burn_result["status"] == "SUCCESS":
                    total_ch4_kg += _D(burn_result["ch4_kg"])
                    total_n2o_kg += _D(burn_result["n2o_kg"])
                    total_co2e += _D(burn_result["total_co2e_tonnes"])
                    sources_calculated.append("burning")
                    trace.append(
                        f"  -> Burning: {burn_result['ch4_tonnes']} t CH4, "
                        f"{burn_result['n2o_tonnes']} t N2O, "
                        f"{burn_result['total_co2e_tonnes']} t CO2e"
                    )
                else:
                    trace.append(
                        f"  -> Burning FAILED: "
                        f"{burn_result.get('error_message', 'unknown')}"
                    )

            if not sources_calculated:
                raise ValueError(
                    "No valid emission sources provided. Supply at least one of: "
                    "soil_n2o, liming, urea, rice, burning"
                )

            # Compute totals
            total_n2o_tonnes = _Q(total_n2o_kg * KG_TO_TONNES)
            total_co2_tonnes = _Q(total_co2_kg * KG_TO_TONNES)
            total_ch4_tonnes = _Q(total_ch4_kg * KG_TO_TONNES)
            total_co2e = _Q(total_co2e)

            trace.append(
                f"[6] Totals: N2O={total_n2o_tonnes} t, CO2={total_co2_tonnes} t, "
                f"CH4={total_ch4_tonnes} t, CO2e={total_co2e} t"
            )

            elapsed_ms = (time.monotonic() - start) * 1000

            gases = {
                "N2O": {
                    "kg": _Q(total_n2o_kg),
                    "tonnes": total_n2o_tonnes,
                    "co2e_tonnes": _Q(
                        total_n2o_tonnes * self._resolve_gwp("N2O", gwp_src)
                    ),
                },
                "CO2": {
                    "kg": _Q(total_co2_kg),
                    "tonnes": total_co2_tonnes,
                    "co2e_tonnes": total_co2_tonnes,
                },
                "CH4": {
                    "kg": _Q(total_ch4_kg),
                    "tonnes": total_ch4_tonnes,
                    "co2e_tonnes": _Q(
                        total_ch4_tonnes * self._resolve_gwp("CH4", gwp_src)
                    ),
                },
            }

            result = self._build_result(
                calc_id=calc_id,
                status="SUCCESS",
                emission_source="cropland_total",
                gases=gases,
                total_co2e_tonnes=total_co2e,
                gwp_source=gwp_src,
                trace=trace,
                elapsed_ms=elapsed_ms,
                extra={
                    "sources_calculated": sources_calculated,
                    "sub_results": {
                        k: {
                            "status": v.get("status"),
                            "calculation_id": v.get("calculation_id"),
                            "total_co2e_tonnes": v.get("total_co2e_tonnes"),
                        }
                        for k, v in sub_results.items()
                    },
                },
            )
            return result

        except Exception as exc:
            return self._handle_error(
                calc_id, "cropland_total", exc, trace, start,
            )

    # ==================================================================
    # PUBLIC API 9: Batch Processing
    # ==================================================================

    def calculate_cropland_batch(
        self,
        requests: List[Dict[str, Any]],
        continue_on_error: bool = True,
    ) -> Dict[str, Any]:
        """Process multiple cropland emission calculations in a batch.

        Each request must contain a ``method`` key to select the calculation:
            - ``"DIRECT_N2O"``: calls calculate_direct_n2o
            - ``"INDIRECT_N2O"``: calls calculate_indirect_n2o
            - ``"TOTAL_N2O"``: calls calculate_total_soil_n2o
            - ``"LIMING"``: calls calculate_liming_co2
            - ``"UREA"``: calls calculate_urea_co2
            - ``"RICE_CH4"``: calls calculate_rice_ch4
            - ``"FIELD_BURNING"``: calls calculate_field_burning
            - ``"CROPLAND_TOTAL"``: calls calculate_cropland_total

        Args:
            requests: List of calculation request dictionaries.
            continue_on_error: If True, skip failed records. If False, stop.

        Returns:
            Batch result with individual results, summary, and totals.
        """
        t0 = time.monotonic()
        batch_id = f"crop_batch_{uuid4().hex[:12]}"
        results: List[Dict[str, Any]] = []
        successful = 0
        failed = 0
        total_co2e = _ZERO

        method_dispatch: Dict[str, Any] = {
            "DIRECT_N2O": self._dispatch_direct_n2o,
            "INDIRECT_N2O": self._dispatch_indirect_n2o,
            "TOTAL_N2O": self._dispatch_total_n2o,
            "LIMING": self._dispatch_liming,
            "UREA": self._dispatch_urea,
            "RICE_CH4": self._dispatch_rice_ch4,
            "FIELD_BURNING": self._dispatch_field_burning,
            "CROPLAND_TOTAL": self._dispatch_cropland_total,
        }

        for idx, record in enumerate(requests):
            try:
                method_key = str(record.get("method", "")).upper().strip()
                handler = method_dispatch.get(method_key)

                if handler is None:
                    raise ValueError(
                        f"Unknown batch method: {method_key}. "
                        f"Valid: {list(method_dispatch.keys())}"
                    )

                calc_result = handler(record)
                calc_result["batch_index"] = idx
                results.append(calc_result)

                if calc_result.get("status") == "SUCCESS":
                    successful += 1
                    co2e = _safe_decimal(
                        calc_result.get("total_co2e_tonnes", "0"),
                    )
                    total_co2e += co2e
                else:
                    failed += 1

            except Exception as exc:
                failed += 1
                error_entry: Dict[str, Any] = {
                    "batch_index": idx,
                    "status": CalculationStatus.ERROR.value,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                results.append(error_entry)

                if not continue_on_error:
                    logger.error(
                        "Batch %s stopped at record %d: %s",
                        batch_id, idx, exc,
                    )
                    break

                logger.warning(
                    "Batch %s record %d failed (continuing): %s",
                    batch_id, idx, exc,
                )

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        batch_result: Dict[str, Any] = {
            "batch_id": batch_id,
            "results": results,
            "summary": {
                "total_co2e_tonnes": str(_Q(total_co2e)),
            },
            "total_records": len(requests),
            "successful": successful,
            "failed": failed,
            "continue_on_error": continue_on_error,
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": _utcnow().isoformat(),
        }
        batch_result["provenance_hash"] = _compute_hash({
            k: v for k, v in batch_result.items()
            if k != "results"
        })

        logger.info(
            "Batch %s: %d/%d successful, %s t CO2e in %.1fms",
            batch_id, successful, len(requests),
            _Q(total_co2e), elapsed_ms,
        )
        return batch_result

    # ------------------------------------------------------------------
    # Batch dispatch helpers
    # ------------------------------------------------------------------

    def _dispatch_direct_n2o(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch direct N2O calculation from batch record."""
        return self.calculate_direct_n2o(
            synthetic_n_kg=record["synthetic_n_kg"],
            organic_n_kg=record.get("organic_n_kg", 0),
            crop_residue_n_kg=record.get("crop_residue_n_kg", 0),
            som_n_kg=record.get("som_n_kg", 0),
            organic_soil_area_ha=record.get("organic_soil_area_ha", 0),
            organic_soil_area_forest_ha=record.get(
                "organic_soil_area_forest_ha", 0
            ),
            prp_n_kg=record.get("prp_n_kg", 0),
            prp_animal_type=record.get("prp_animal_type", "cattle"),
            gwp_source=record.get("gwp_source"),
        )

    def _dispatch_indirect_n2o(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch indirect N2O calculation from batch record."""
        return self.calculate_indirect_n2o(
            synthetic_n_kg=record["synthetic_n_kg"],
            organic_n_kg=record.get("organic_n_kg", 0),
            prp_n_kg=record.get("prp_n_kg", 0),
            crop_residue_n_kg=record.get("crop_residue_n_kg", 0),
            som_n_kg=record.get("som_n_kg", 0),
            gwp_source=record.get("gwp_source"),
        )

    def _dispatch_total_n2o(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch total soil N2O calculation from batch record."""
        return self.calculate_total_soil_n2o(
            synthetic_n_kg=record["synthetic_n_kg"],
            organic_n_kg=record.get("organic_n_kg", 0),
            crop_residue_n_kg=record.get("crop_residue_n_kg", 0),
            som_n_kg=record.get("som_n_kg", 0),
            organic_soil_area_ha=record.get("organic_soil_area_ha", 0),
            organic_soil_area_forest_ha=record.get(
                "organic_soil_area_forest_ha", 0
            ),
            prp_n_kg=record.get("prp_n_kg", 0),
            prp_animal_type=record.get("prp_animal_type", "cattle"),
            gwp_source=record.get("gwp_source"),
        )

    def _dispatch_liming(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch liming CO2 calculation from batch record."""
        return self.calculate_liming_co2(
            limestone_tonnes=record.get("limestone_tonnes", 0),
            dolomite_tonnes=record.get("dolomite_tonnes", 0),
            gwp_source=record.get("gwp_source"),
        )

    def _dispatch_urea(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch urea CO2 calculation from batch record."""
        return self.calculate_urea_co2(
            urea_tonnes=record["urea_tonnes"],
            gwp_source=record.get("gwp_source"),
        )

    def _dispatch_rice_ch4(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch rice CH4 calculation from batch record."""
        return self.calculate_rice_ch4(
            area_ha=record["area_ha"],
            cultivation_days=record["cultivation_days"],
            water_regime=record.get("water_regime", "continuously_flooded"),
            pre_season_flooding=record.get(
                "pre_season_flooding", "not_flooded_short"
            ),
            organic_amendments=record.get("organic_amendments"),
            soil_type_scaling=record.get("soil_type_scaling", "1.0"),
            gwp_source=record.get("gwp_source"),
        )

    def _dispatch_field_burning(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch field burning calculation from batch record."""
        return self.calculate_field_burning(
            crop_type=record["crop_type"],
            area_burned_ha=record["area_burned_ha"],
            crop_yield_tonnes_ha=record["crop_yield_tonnes_ha"],
            burn_fraction=record.get("burn_fraction", "0.80"),
            gwp_source=record.get("gwp_source"),
        )

    def _dispatch_cropland_total(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch cropland total calculation from batch record."""
        return self.calculate_cropland_total(inputs=record.get("inputs", record))

    # ==================================================================
    # PUBLIC API 10: Estimate Crop Residue N
    # ==================================================================

    def estimate_crop_residue_n(
        self,
        crop_type: str,
        area_ha: Any,
        crop_yield_tonnes_ha: Any,
        above_ground_fraction: Optional[Any] = None,
    ) -> Decimal:
        """Estimate nitrogen from crop residues returned to soil.

        Calculates the N content of above-ground crop residues that are
        returned to soil (not burned or removed), for use as the F_CR
        input to soil N2O calculations.

        Formula:
            F_CR = area * yield * RPR * DM * above_ground_frac * N_content

        Args:
            crop_type: Crop type (see CropType enum values).
            area_ha: Harvested area (hectares).
            crop_yield_tonnes_ha: Crop yield (tonnes fresh weight/ha).
            above_ground_fraction: Override fraction of residue above ground
                (0-1). Defaults to crop-specific value.

        Returns:
            Estimated crop residue N in kg N/yr as Decimal.

        Raises:
            ValueError: If crop_type is unknown or inputs are invalid.
        """
        ct_key = crop_type.strip().lower().replace(" ", "_").replace("-", "_")
        a = _D(area_ha)
        y = _D(crop_yield_tonnes_ha)

        self._validate_positive(a, "area_ha")
        self._validate_positive(y, "crop_yield_tonnes_ha")

        params = self._get_crop_params(ct_key)

        ag_frac = (
            _D(above_ground_fraction)
            if above_ground_fraction is not None
            else params["above_ground_fraction"]
        )
        self._validate_fraction(ag_frac, "above_ground_fraction")

        rpr = params["residue_to_crop_ratio"]
        dm = params["dry_matter_fraction"]
        n_content = params["nitrogen_content"]

        # F_CR = area * yield_per_ha * RPR * DM_fraction * AG_fraction * N_content
        # Result unit: tonnes * kg_N/kg_DM = tonnes * (dimensionless) = kg N
        # (since DM fraction and N content handle unit conversion)
        total_residue_dm_tonnes = _Q(a * y * rpr * dm)
        # Convert to kg for N calculation
        total_residue_dm_kg = _Q(total_residue_dm_tonnes * _THOUSAND)
        residue_n_kg = _Q(total_residue_dm_kg * ag_frac * n_content)

        logger.debug(
            "Crop residue N estimate: crop=%s, area=%s ha, yield=%s t/ha -> "
            "residue_dm=%s t, AG_frac=%s, N=%s kg",
            ct_key, a, y, total_residue_dm_tonnes, ag_frac, residue_n_kg,
        )
        return residue_n_kg

    # ==================================================================
    # PUBLIC API: Diagnostics
    # ==================================================================

    def get_engine_info(self) -> Dict[str, Any]:
        """Return engine diagnostic information.

        Returns:
            Dictionary with engine status, counters, and configuration.
        """
        with self._lock:
            return {
                "engine": "CroplandEmissionsEngine",
                "agent": "AGENT-MRV-008",
                "engine_number": "4 of 7",
                "ipcc_reference": "IPCC 2006 Vol 4 Ch 5, Ch 11",
                "default_gwp_source": self._default_gwp_source,
                "total_calculations": self._total_calculations,
                "total_errors": self._total_errors,
                "created_at": self._created_at.isoformat(),
                "custom_soil_efs": len(self._custom_ef),
                "custom_crop_params": len(self._custom_crop_params),
                "custom_burning_efs": len(self._custom_burning_ef),
                "provenance_enabled": self._provenance is not None,
                "metrics_enabled": self._enable_metrics and _METRICS_AVAILABLE,
                "supported_methods": [
                    "calculate_direct_n2o",
                    "calculate_indirect_n2o",
                    "calculate_total_soil_n2o",
                    "calculate_liming_co2",
                    "calculate_urea_co2",
                    "calculate_rice_ch4",
                    "calculate_field_burning",
                    "calculate_cropland_total",
                    "calculate_cropland_batch",
                    "estimate_crop_residue_n",
                ],
                "supported_crop_types": sorted(CROP_RESIDUE_PARAMS.keys()),
                "supported_water_regimes": sorted(WATER_REGIME_SF.keys()),
                "supported_amendment_types": sorted(
                    ORGANIC_AMENDMENT_CFOA.keys()
                ),
            }

    def get_default_emission_factors(self) -> Dict[str, str]:
        """Return all IPCC default emission factors used by this engine.

        Returns:
            Dictionary mapping factor names to their default Decimal values
            (as strings for JSON serialization).
        """
        return {
            "EF1": str(EF1_DEFAULT),
            "EF2_CG": str(EF2_CG_DEFAULT),
            "EF2_F": str(EF2_F_DEFAULT),
            "EF3_PRP_CATTLE_POULTRY": str(EF3_PRP_CATTLE_POULTRY),
            "EF3_PRP_OTHER": str(EF3_PRP_OTHER),
            "EF4": str(EF4_DEFAULT),
            "EF5": str(EF5_DEFAULT),
            "FRAC_GASF": str(FRAC_GASF_DEFAULT),
            "FRAC_GASM": str(FRAC_GASM_DEFAULT),
            "FRAC_LEACH": str(FRAC_LEACH_DEFAULT),
            "LIMESTONE_EF": str(LIMESTONE_EF),
            "DOLOMITE_EF": str(DOLOMITE_EF),
            "UREA_EF": str(UREA_EF),
            "EF_C_RICE": str(EF_C_RICE_DEFAULT),
        }

    def get_gwp_values(self, gwp_source: Optional[str] = None) -> Dict[str, str]:
        """Return GWP values for a given assessment report.

        Args:
            gwp_source: IPCC AR (default uses engine default).

        Returns:
            Dictionary mapping gas names to GWP values (as strings).
        """
        source = (gwp_source or self._default_gwp_source).upper()
        if source not in GWP_VALUES:
            return {"error": f"Unknown GWP source: {source}"}
        return {k: str(v) for k, v in GWP_VALUES[source].items()}
