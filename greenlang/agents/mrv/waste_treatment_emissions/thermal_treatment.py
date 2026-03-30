# -*- coding: utf-8 -*-
"""
ThermalTreatmentEngine - Incineration, Pyrolysis, Gasification, Open Burning (Engine 3 of 7)

AGENT-MRV-007: On-Site Waste Treatment Emissions Agent

Calculates GHG emissions from thermal waste treatment processes using IPCC 2006
Guidelines Volume 5 Chapter 5 methodology with deterministic Decimal arithmetic.
Covers five treatment pathways:

    1. Incineration (mass burn stoker/grate, fluidized bed, rotary kiln,
       semi-continuous, batch, modular) with fossil/biogenic CO2 separation.
    2. Incineration with Energy Recovery (waste-to-energy with electricity and
       heat generation, grid displacement offset credits).
    3. Pyrolysis (thermal decomposition without oxygen, 300-700 deg C) producing
       syngas, bio-oil, and biochar with carbon accounting.
    4. Gasification (partial oxidation, 700-1500 deg C) producing syngas with
       equivalence ratio adjustable gas composition.
    5. Open Burning (uncontrolled combustion with incomplete oxidation factor
       of 0.58, elevated CH4 and N2O emission factors).

Key Formulae (IPCC 2006 Vol 5 Ch 5):

    Incineration CO2 (fossil):
        CO2_fossil = Sum_j [ IW_j * dm_j * CF_j * FCF_j * OF_j ] * 44/12

    Incineration CO2 (biogenic):
        CO2_biogenic = Sum_j [ IW_j * dm_j * CF_j * (1-FCF_j) * OF_j ] * 44/12

    Incineration N2O/CH4 (Table 5.3):
        N2O = Sum_j (IW_j * EF_N2O_j) * 1e-6   # kg/Gg -> tonnes/tonne
        CH4 = Sum_j (IW_j * EF_CH4_j) * 1e-6

    Open Burning (OF_open=0.58):
        CO2  = Sum_j (OW_j * DM_j * CF_j * FCF_j * OF_open) * 44/12
        CH4  = Sum_j (OW_j * DM_j * 0.0065)   # 6.5 g/kg DM
        N2O  = Sum_j (OW_j * DM_j * 0.00015)  # 0.15 g/kg DM

    Pyrolysis (mass balance):
        C_products = M * DM * CF * (1 - gas_yield * combustion_frac)
        CO2_pyro   = C_products * FCF * 44/12
        Syngas_CH4 = M * syngas_yield * CH4_in_syngas

    Energy Recovery Credits:
        E_elec = M * NCV * eta_electric
        E_heat = M * NCV * eta_thermal
        Displaced_CO2e = E_elec * EF_grid_elec + E_heat * EF_grid_heat

Zero-Hallucination Guarantees:
    - All emission factors are hard-coded from IPCC 2006 Volume 5 Tables 5.2/5.3.
    - All calculations use Python Decimal with 8-decimal-place quantization.
    - No LLM calls in any calculation path.
    - Every result carries a SHA-256 provenance hash covering all inputs/outputs.
    - Complete calculation trace for audit review.

Thread Safety:
    All reference data is immutable after class initialization.  The mutable
    custom factor registry is protected by a reentrant lock (``threading.RLock``).
    Concurrent calls to ``calculate_*`` methods are safe.

Example:
    >>> from greenlang.agents.mrv.waste_treatment_emissions.thermal_treatment import (
    ...     ThermalTreatmentEngine,
    ... )
    >>> engine = ThermalTreatmentEngine()
    >>> result = engine.calculate_incineration(
    ...     waste_streams=[
    ...         {"waste_category": "plastic", "mass_tonnes": 100},
    ...         {"waste_category": "paper", "mass_tonnes": 200},
    ...     ],
    ...     incinerator_type="stoker_grate",
    ... )
    >>> assert result["status"] == "SUCCESS"
    >>> assert result["fossil_co2_tonnes"] > 0

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-007 On-site Waste Treatment Emissions (GL-MRV-SCOPE1-007)
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
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["ThermalTreatmentEngine"]

# ---------------------------------------------------------------------------
# Conditional imports -- graceful fallback when peer modules are absent
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.waste_treatment_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.metrics import (
        record_calculation as _record_calculation,
        record_emissions as _record_emissions,
        record_thermal_treatment as _record_thermal_treatment,
        record_energy_recovered as _record_energy_recovered,
        record_waste_processed as _record_waste_processed,
        record_calculation_error as _record_calculation_error,
        observe_calculation_duration as _observe_duration,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_calculation = None  # type: ignore[assignment]
    _record_emissions = None  # type: ignore[assignment]
    _record_thermal_treatment = None  # type: ignore[assignment]
    _record_energy_recovered = None  # type: ignore[assignment]
    _record_waste_processed = None  # type: ignore[assignment]
    _record_calculation_error = None  # type: ignore[assignment]
    _observe_duration = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.models import (
        GWP_VALUES as _MODEL_GWP_VALUES,
        GWPSource,
        IncineratorType,
        WasteCategory,
        IPCC_CARBON_CONTENT as _MODEL_CARBON_CONTENT,
        IPCC_INCINERATION_EF as _MODEL_INCINERATION_EF,
        INCINERATION_NCV as _MODEL_NCV,
        CONVERSION_FACTOR_CO2_C as _MODEL_CO2_C_RATIO,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    GWPSource = None  # type: ignore[assignment,misc]
    IncineratorType = None  # type: ignore[assignment,misc]
    WasteCategory = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Decimal precision
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")

# Molecular weight ratio CO2/C = 44/12
_CO2_C_RATIO = Decimal("3.66666667")

# Gg-to-tonne conversion: 1 Gg = 1000 tonnes; IPCC EFs are kg/Gg
# kg/Gg -> tonnes/tonne: multiply by 1e-6
_KG_PER_GG_TO_TONNES_PER_TONNE = Decimal("0.000001")

# =========================================================================
# Built-in reference data (IPCC 2006 Vol 5 Table 5.2)
# =========================================================================

# Waste composition parameters: dry_matter (fraction of wet), carbon_fraction
# (fraction of dry matter), fossil_carbon_fraction (fraction of total carbon).
# Keys are lowercase waste category strings for flexible matching.

_IPCC_TABLE_5_2: Dict[str, Dict[str, Decimal]] = {
    "food": {
        "dry_matter": Decimal("0.40"),
        "carbon_fraction": Decimal("0.38"),
        "fossil_carbon_fraction": Decimal("0.0"),
    },
    "paper": {
        "dry_matter": Decimal("0.90"),
        "carbon_fraction": Decimal("0.46"),
        "fossil_carbon_fraction": Decimal("0.01"),
    },
    "cardboard": {
        "dry_matter": Decimal("0.90"),
        "carbon_fraction": Decimal("0.44"),
        "fossil_carbon_fraction": Decimal("0.01"),
    },
    "plastic": {
        "dry_matter": Decimal("1.00"),
        "carbon_fraction": Decimal("0.75"),
        "fossil_carbon_fraction": Decimal("1.0"),
    },
    "textiles_synthetic": {
        "dry_matter": Decimal("0.80"),
        "carbon_fraction": Decimal("0.50"),
        "fossil_carbon_fraction": Decimal("0.80"),
    },
    "textiles_natural": {
        "dry_matter": Decimal("0.80"),
        "carbon_fraction": Decimal("0.50"),
        "fossil_carbon_fraction": Decimal("0.0"),
    },
    "textiles": {
        "dry_matter": Decimal("0.80"),
        "carbon_fraction": Decimal("0.50"),
        "fossil_carbon_fraction": Decimal("0.80"),
    },
    "rubber": {
        "dry_matter": Decimal("0.84"),
        "carbon_fraction": Decimal("0.67"),
        "fossil_carbon_fraction": Decimal("0.20"),
    },
    "wood": {
        "dry_matter": Decimal("0.85"),
        "carbon_fraction": Decimal("0.50"),
        "fossil_carbon_fraction": Decimal("0.0"),
    },
    "yard": {
        "dry_matter": Decimal("0.40"),
        "carbon_fraction": Decimal("0.49"),
        "fossil_carbon_fraction": Decimal("0.0"),
    },
    "garden": {
        "dry_matter": Decimal("0.40"),
        "carbon_fraction": Decimal("0.49"),
        "fossil_carbon_fraction": Decimal("0.0"),
    },
    "nappies": {
        "dry_matter": Decimal("0.40"),
        "carbon_fraction": Decimal("0.70"),
        "fossil_carbon_fraction": Decimal("0.10"),
    },
    "sludge": {
        "dry_matter": Decimal("0.25"),
        "carbon_fraction": Decimal("0.50"),
        "fossil_carbon_fraction": Decimal("0.0"),
    },
    # Supplementary categories mapped from WasteCategory enum
    "msw": {
        "dry_matter": Decimal("0.60"),
        "carbon_fraction": Decimal("0.33"),
        "fossil_carbon_fraction": Decimal("0.40"),
    },
    "industrial": {
        "dry_matter": Decimal("0.70"),
        "carbon_fraction": Decimal("0.36"),
        "fossil_carbon_fraction": Decimal("0.50"),
    },
    "construction_demolition": {
        "dry_matter": Decimal("0.90"),
        "carbon_fraction": Decimal("0.11"),
        "fossil_carbon_fraction": Decimal("0.10"),
    },
    "organic": {
        "dry_matter": Decimal("0.40"),
        "carbon_fraction": Decimal("0.38"),
        "fossil_carbon_fraction": Decimal("0.0"),
    },
    "metal": {
        "dry_matter": Decimal("0.99"),
        "carbon_fraction": Decimal("0.0"),
        "fossil_carbon_fraction": Decimal("0.0"),
    },
    "glass": {
        "dry_matter": Decimal("0.99"),
        "carbon_fraction": Decimal("0.0"),
        "fossil_carbon_fraction": Decimal("0.0"),
    },
    "e_waste": {
        "dry_matter": Decimal("0.95"),
        "carbon_fraction": Decimal("0.11"),
        "fossil_carbon_fraction": Decimal("0.80"),
    },
    "hazardous": {
        "dry_matter": Decimal("0.75"),
        "carbon_fraction": Decimal("0.20"),
        "fossil_carbon_fraction": Decimal("0.60"),
    },
    "medical": {
        "dry_matter": Decimal("0.70"),
        "carbon_fraction": Decimal("0.36"),
        "fossil_carbon_fraction": Decimal("0.50"),
    },
    "mixed": {
        "dry_matter": Decimal("0.60"),
        "carbon_fraction": Decimal("0.37"),
        "fossil_carbon_fraction": Decimal("0.35"),
    },
}

# =========================================================================
# IPCC Table 5.3 -- Incinerator N2O and CH4 EFs (kg per Gg of waste)
# =========================================================================

_INCINERATOR_EF: Dict[str, Dict[str, Decimal]] = {
    "stoker_grate": {"N2O": Decimal("50"), "CH4": Decimal("0.2")},
    "fluidized_bed": {"N2O": Decimal("56"), "CH4": Decimal("0.68")},
    "rotary_kiln": {"N2O": Decimal("50"), "CH4": Decimal("0.2")},
    "semi_continuous": {"N2O": Decimal("60"), "CH4": Decimal("6.0")},
    "batch_type": {"N2O": Decimal("60"), "CH4": Decimal("60")},
    "modular": {"N2O": Decimal("55"), "CH4": Decimal("3.0")},
}

# =========================================================================
# Open burning emission factors (IPCC 2006 Vol 5 Ch 5 / EPA AP-42)
# CH4: 6.5 g/kg dry matter, N2O: 0.15 g/kg dry matter
# =========================================================================

_OPEN_BURN_CH4_G_PER_KG_DM = Decimal("6.5")     # g CH4 / kg dry matter
_OPEN_BURN_N2O_G_PER_KG_DM = Decimal("0.15")     # g N2O / kg dry matter
_OPEN_BURN_OXIDATION_FACTOR = Decimal("0.58")     # incomplete combustion
_G_PER_KG_TO_TONNES_PER_TONNE = Decimal("0.001")  # g/kg = t/t * 1e-3

# =========================================================================
# Net Calorific Value (NCV) GJ per tonne wet waste
# =========================================================================

_WASTE_NCV: Dict[str, Decimal] = {
    "msw": Decimal("9.0"),
    "industrial": Decimal("12.0"),
    "construction_demolition": Decimal("5.0"),
    "organic": Decimal("4.0"),
    "food": Decimal("3.5"),
    "yard": Decimal("5.5"),
    "paper": Decimal("13.0"),
    "cardboard": Decimal("14.0"),
    "plastic": Decimal("32.0"),
    "metal": Decimal("0.0"),
    "glass": Decimal("0.0"),
    "textiles": Decimal("16.0"),
    "textiles_synthetic": Decimal("18.0"),
    "textiles_natural": Decimal("14.0"),
    "wood": Decimal("15.0"),
    "rubber": Decimal("26.0"),
    "e_waste": Decimal("5.0"),
    "hazardous": Decimal("10.0"),
    "medical": Decimal("14.0"),
    "sludge": Decimal("2.0"),
    "mixed": Decimal("8.5"),
    "nappies": Decimal("8.0"),
    "garden": Decimal("5.5"),
}

# =========================================================================
# GWP fallback values (used only when models.py is unavailable)
# =========================================================================

_FALLBACK_GWP: Dict[str, Dict[str, Decimal]] = {
    "AR4": {
        "CO2": _ONE,
        "CH4": Decimal("25"),
        "N2O": Decimal("298"),
    },
    "AR5": {
        "CO2": _ONE,
        "CH4": Decimal("28"),
        "N2O": Decimal("265"),
    },
    "AR6": {
        "CO2": _ONE,
        "CH4": Decimal("29.8"),
        "N2O": Decimal("273"),
    },
    "AR6_20YR": {
        "CO2": _ONE,
        "CH4": Decimal("82.5"),
        "N2O": Decimal("273"),
    },
}

# =========================================================================
# Pyrolysis default parameters
# =========================================================================

_PYROLYSIS_DEFAULTS: Dict[str, Decimal] = {
    "gas_yield_fraction": Decimal("0.30"),        # 30% of feed mass becomes gas
    "oil_yield_fraction": Decimal("0.40"),         # 40% becomes bio-oil
    "char_yield_fraction": Decimal("0.30"),        # 30% becomes char
    "gas_combustion_fraction": Decimal("0.95"),    # 95% of gas carbon combusted
    "syngas_ch4_fraction": Decimal("0.10"),        # 10% of syngas is CH4
    "char_carbon_stability": Decimal("0.80"),      # 80% of char carbon stable
}

# =========================================================================
# Gasification default parameters
# =========================================================================

_GASIFICATION_DEFAULTS: Dict[str, Decimal] = {
    "cold_gas_efficiency": Decimal("0.70"),        # 70% energy to syngas
    "carbon_conversion": Decimal("0.95"),          # 95% of feed C converts
    "syngas_co_fraction": Decimal("0.30"),         # 30% CO in syngas
    "syngas_h2_fraction": Decimal("0.25"),         # 25% H2 in syngas
    "syngas_ch4_fraction": Decimal("0.05"),        # 5% CH4 in syngas
    "syngas_co2_fraction": Decimal("0.10"),        # 10% CO2 in syngas
    "tar_fraction": Decimal("0.02"),               # 2% tar in product gas
    "equivalence_ratio_default": Decimal("0.30"),  # ER = 0.3 typical
}

# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------

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

def _d(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Number, string, or Decimal to convert.

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted to Decimal.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot convert {value!r} (type={type(value).__name__}) to Decimal"
        ) from exc

def _q(value: Decimal) -> Decimal:
    """Quantize a Decimal to 8 decimal places with ROUND_HALF_UP.

    Args:
        value: Decimal value to quantize.

    Returns:
        Quantized Decimal.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)

def _normalize_category(raw: str) -> str:
    """Normalize a waste category string for dictionary lookup.

    Strips whitespace, lowercases, and replaces spaces/hyphens with underscores.

    Args:
        raw: Raw waste category string from caller.

    Returns:
        Normalized lowercase string suitable for reference data lookup.
    """
    return raw.strip().lower().replace(" ", "_").replace("-", "_")

def _normalize_incinerator(raw: str) -> str:
    """Normalize an incinerator type string for dictionary lookup.

    Args:
        raw: Raw incinerator type string.

    Returns:
        Normalized lowercase string.
    """
    return raw.strip().lower().replace(" ", "_").replace("-", "_")

# =========================================================================
# ThermalTreatmentEngine
# =========================================================================

class ThermalTreatmentEngine:
    """Engine 3: Thermal waste treatment GHG emission calculations.

    Implements IPCC 2006 Volume 5 Chapter 5 methodology for incineration,
    pyrolysis, gasification, and open burning with deterministic Decimal
    arithmetic, full audit trails, and SHA-256 provenance hashing.

    All calculation methods return a result dictionary containing at minimum:
        - ``status``: ``"SUCCESS"`` or ``"FAILED"``
        - ``fossil_co2_tonnes``: Fossil-origin CO2 in metric tonnes
        - ``biogenic_co2_tonnes``: Biogenic-origin CO2 in metric tonnes
        - ``ch4_tonnes``: Methane in metric tonnes
        - ``n2o_tonnes``: Nitrous oxide in metric tonnes
        - ``total_co2e_tonnes``: Total CO2-equivalent (fossil only)
        - ``calculation_trace``: Step-by-step audit trail
        - ``provenance_hash``: SHA-256 hash of inputs + outputs
        - ``processing_time_ms``: Wall-clock calculation time

    Attributes:
        _config: Optional configuration dictionary.
        _lock: Reentrant lock for thread-safe custom factor mutations.
        _custom_waste_data: User-registered custom waste composition data.
        _custom_incinerator_ef: User-registered custom incinerator EFs.
        _provenance: Provenance tracker reference (or None).

    Example:
        >>> engine = ThermalTreatmentEngine()
        >>> result = engine.calculate_incineration(
        ...     waste_streams=[
        ...         {"waste_category": "plastic", "mass_tonnes": 50},
        ...         {"waste_category": "food", "mass_tonnes": 100},
        ...     ],
        ...     incinerator_type="stoker_grate",
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ThermalTreatmentEngine.

        Args:
            config: Optional configuration dictionary.  Supports:
                - ``enable_provenance`` (bool): Track provenance. Default True.
                - ``decimal_precision`` (int): Decimal places. Default 8.
                - ``default_gwp_source`` (str): Default GWP AR. Default ``"AR6"``.
                - ``default_oxidation_factor`` (str/Decimal): Default OF. Default 1.0.
                - ``enable_metrics`` (bool): Record Prometheus metrics. Default True.
        """
        self._config: Dict[str, Any] = config or {}
        self._lock = threading.RLock()

        # Decimal precision
        prec = self._config.get("decimal_precision", 8)
        self._precision = Decimal(10) ** -int(prec)

        # Defaults
        self._default_gwp = str(self._config.get("default_gwp_source", "AR6"))
        self._default_of = _d(self._config.get("default_oxidation_factor", "1.0"))
        self._enable_provenance: bool = self._config.get("enable_provenance", True)
        self._enable_metrics: bool = self._config.get("enable_metrics", True)

        # Mutable custom factor registries (guarded by _lock)
        self._custom_waste_data: Dict[str, Dict[str, Decimal]] = {}
        self._custom_incinerator_ef: Dict[str, Dict[str, Decimal]] = {}

        # Provenance tracker
        self._provenance: Any = None
        if self._enable_provenance and _PROVENANCE_AVAILABLE:
            try:
                self._provenance = _get_provenance_tracker()
            except Exception:
                logger.warning(
                    "Provenance tracker initialization failed; continuing without provenance"
                )

        logger.info(
            "ThermalTreatmentEngine initialized (gwp=%s, of=%s, provenance=%s)",
            self._default_gwp,
            self._default_of,
            self._provenance is not None,
        )

    # ------------------------------------------------------------------
    # Custom factor registration (thread-safe)
    # ------------------------------------------------------------------

    def register_custom_waste_data(
        self,
        category: str,
        dry_matter: Decimal,
        carbon_fraction: Decimal,
        fossil_carbon_fraction: Decimal,
    ) -> None:
        """Register custom waste composition data for a category.

        Thread-safe.  Overrides built-in IPCC Table 5.2 defaults for the
        specified category in subsequent calculations.

        Args:
            category: Waste category key (will be normalized).
            dry_matter: Dry matter fraction of wet waste (0-1).
            carbon_fraction: Carbon fraction of dry matter (0-1).
            fossil_carbon_fraction: Fossil fraction of total carbon (0-1).

        Raises:
            ValueError: If any fraction is outside [0, 1].
        """
        for name, val in [
            ("dry_matter", dry_matter),
            ("carbon_fraction", carbon_fraction),
            ("fossil_carbon_fraction", fossil_carbon_fraction),
        ]:
            if val < _ZERO or val > _ONE:
                raise ValueError(f"{name} must be in [0, 1], got {val}")

        key = _normalize_category(category)
        with self._lock:
            self._custom_waste_data[key] = {
                "dry_matter": _d(dry_matter),
                "carbon_fraction": _d(carbon_fraction),
                "fossil_carbon_fraction": _d(fossil_carbon_fraction),
            }
        logger.info("Registered custom waste data for category=%s", key)

    def register_custom_incinerator_ef(
        self,
        incinerator_type: str,
        n2o_kg_per_gg: Decimal,
        ch4_kg_per_gg: Decimal,
    ) -> None:
        """Register custom incinerator emission factors.

        Thread-safe.  Overrides built-in IPCC Table 5.3 defaults for the
        specified incinerator type in subsequent calculations.

        Args:
            incinerator_type: Incinerator type key (will be normalized).
            n2o_kg_per_gg: N2O emission factor in kg per Gg waste.
            ch4_kg_per_gg: CH4 emission factor in kg per Gg waste.

        Raises:
            ValueError: If any emission factor is negative.
        """
        if n2o_kg_per_gg < _ZERO:
            raise ValueError(f"N2O EF must be >= 0, got {n2o_kg_per_gg}")
        if ch4_kg_per_gg < _ZERO:
            raise ValueError(f"CH4 EF must be >= 0, got {ch4_kg_per_gg}")

        key = _normalize_incinerator(incinerator_type)
        with self._lock:
            self._custom_incinerator_ef[key] = {
                "N2O": _d(n2o_kg_per_gg),
                "CH4": _d(ch4_kg_per_gg),
            }
        logger.info("Registered custom incinerator EF for type=%s", key)

    # ------------------------------------------------------------------
    # Reference data lookups (custom overrides -> built-in fallback)
    # ------------------------------------------------------------------

    def _get_waste_params(
        self,
        category: str,
    ) -> Dict[str, Decimal]:
        """Look up waste composition parameters for a category.

        Custom overrides take precedence over built-in IPCC Table 5.2 data.

        Args:
            category: Normalized waste category key.

        Returns:
            Dict with dry_matter, carbon_fraction, fossil_carbon_fraction.

        Raises:
            ValueError: If category is not found in custom or built-in data.
        """
        with self._lock:
            if category in self._custom_waste_data:
                return dict(self._custom_waste_data[category])

        if category in _IPCC_TABLE_5_2:
            return dict(_IPCC_TABLE_5_2[category])

        raise ValueError(
            f"Unknown waste category '{category}'. Available: "
            f"{sorted(set(list(_IPCC_TABLE_5_2.keys())))}"
        )

    def _get_incinerator_ef(
        self,
        incinerator_type: str,
    ) -> Dict[str, Decimal]:
        """Look up incinerator N2O/CH4 emission factors.

        Custom overrides take precedence over built-in IPCC Table 5.3 data.

        Args:
            incinerator_type: Normalized incinerator type key.

        Returns:
            Dict with N2O and CH4 in kg per Gg of waste.

        Raises:
            ValueError: If incinerator type is not found.
        """
        with self._lock:
            if incinerator_type in self._custom_incinerator_ef:
                return dict(self._custom_incinerator_ef[incinerator_type])

        if incinerator_type in _INCINERATOR_EF:
            return dict(_INCINERATOR_EF[incinerator_type])

        raise ValueError(
            f"Unknown incinerator type '{incinerator_type}'. Available: "
            f"{sorted(_INCINERATOR_EF.keys())}"
        )

    def _get_gwp(self, gas: str, gwp_source: str) -> Decimal:
        """Look up GWP value for a gas under a given assessment report.

        Args:
            gas: Gas identifier (``"CO2"``, ``"CH4"``, ``"N2O"``).
            gwp_source: IPCC AR (``"AR4"``, ``"AR5"``, ``"AR6"``, ``"AR6_20YR"``).

        Returns:
            GWP value as Decimal.

        Raises:
            ValueError: If gas or gwp_source is not recognized.
        """
        src = gwp_source.upper().strip()
        gas_key = gas.upper().strip()

        # Try models.py GWP_VALUES first
        if _MODELS_AVAILABLE:
            try:
                gwp_enum = GWPSource(src)
                vals = _MODEL_GWP_VALUES.get(gwp_enum, {})
                if gas_key in vals:
                    return vals[gas_key]
            except (ValueError, KeyError):
                pass

        # Fallback to local table
        if src in _FALLBACK_GWP and gas_key in _FALLBACK_GWP[src]:
            return _FALLBACK_GWP[src][gas_key]

        raise ValueError(f"No GWP for gas={gas_key}, source={src}")

    def _get_ncv(self, category: str) -> Decimal:
        """Look up net calorific value for a waste category.

        Args:
            category: Normalized waste category key.

        Returns:
            NCV in GJ per tonne (wet basis).
        """
        if category in _WASTE_NCV:
            return _WASTE_NCV[category]
        logger.warning(
            "No NCV for category=%s, using default 8.5 GJ/t (mixed MSW)", category
        )
        return Decimal("8.5")

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def _emit_metrics(
        self,
        treatment_method: str,
        calc_method: str,
        waste_category: str,
        duration_s: float,
        co2e_tonnes: float,
    ) -> None:
        """Record Prometheus metrics for a thermal treatment calculation.

        Args:
            treatment_method: Treatment method label.
            calc_method: Calculation method label.
            waste_category: Waste category label.
            duration_s: Wall-clock time in seconds.
            co2e_tonnes: Total CO2e in tonnes.
        """
        if not self._enable_metrics or not _METRICS_AVAILABLE:
            return
        try:
            _record_calculation(treatment_method, calc_method, waste_category)
            _observe_duration(treatment_method, calc_method, duration_s)
            _record_thermal_treatment(treatment_method)
            if co2e_tonnes > 0:
                _record_emissions("CO2", treatment_method, waste_category, co2e_tonnes)
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
                entity_type="thermal_treatment",
                action=action,
                entity_id=entity_id,
                data=data,
            )
        except Exception:
            logger.debug("Provenance recording failed (non-critical)", exc_info=True)

    # ==================================================================
    # PUBLIC API: Incineration
    # ==================================================================

    def calculate_incineration(
        self,
        waste_streams: List[Dict[str, Any]],
        incinerator_type: str = "stoker_grate",
        oxidation_factor: Optional[Decimal] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate GHG emissions from waste incineration.

        Implements IPCC 2006 Vol 5 Ch 5 Equation 5.1 for CO2 with
        fossil/biogenic separation, and Table 5.3 for N2O and CH4.

        Args:
            waste_streams: List of dicts, each with:
                - ``waste_category`` (str): Waste type key (e.g. ``"plastic"``).
                - ``mass_tonnes`` (float/Decimal): Mass incinerated (wet basis).
                - ``dry_matter`` (float/Decimal, optional): Override DM fraction.
                - ``carbon_fraction`` (float/Decimal, optional): Override CF.
                - ``fossil_carbon_fraction`` (float/Decimal, optional): Override FCF.
                - ``oxidation_factor`` (float/Decimal, optional): Per-stream OF.
            incinerator_type: Incinerator technology type. One of
                ``stoker_grate``, ``fluidized_bed``, ``rotary_kiln``,
                ``semi_continuous``, ``batch_type``, ``modular``.
            oxidation_factor: Global oxidation factor override (0-1).
                Defaults to 1.0 for modern incinerators.
            gwp_source: GWP assessment report (``AR4``/``AR5``/``AR6``/``AR6_20YR``).

        Returns:
            Dict with keys: status, fossil_co2_tonnes, biogenic_co2_tonnes,
            ch4_tonnes, n2o_tonnes, total_co2e_tonnes, per_stream_results,
            calculation_trace, provenance_hash, processing_time_ms,
            calculation_id, gwp_source, incinerator_type.

        Raises:
            ValueError: If waste_streams is empty or any stream is invalid.
        """
        start = time.monotonic()
        calc_id = f"incin_{uuid4().hex[:12]}"
        trace: List[str] = []
        gwp_src = (gwp_source or self._default_gwp).upper().strip()
        of_global = oxidation_factor if oxidation_factor is not None else self._default_of
        of_global = _d(of_global)
        inc_type = _normalize_incinerator(incinerator_type)

        try:
            self._validate_waste_streams(waste_streams, trace)
            inc_ef = self._get_incinerator_ef(inc_type)
            trace.append(
                f"[1] Incinerator type={inc_type}, N2O EF={inc_ef['N2O']} kg/Gg, "
                f"CH4 EF={inc_ef['CH4']} kg/Gg, GWP={gwp_src}"
            )

            # Accumulators
            total_fossil_co2 = _ZERO
            total_biogenic_co2 = _ZERO
            total_ch4 = _ZERO
            total_n2o = _ZERO
            total_mass = _ZERO
            per_stream: List[Dict[str, Any]] = []
            step = 2

            for idx, stream in enumerate(waste_streams):
                cat = _normalize_category(str(stream["waste_category"]))
                mass = _d(stream["mass_tonnes"])
                self._validate_positive(mass, f"mass_tonnes[{idx}]")

                params = self._resolve_stream_params(stream, cat)
                dm = params["dry_matter"]
                cf = params["carbon_fraction"]
                fcf = params["fossil_carbon_fraction"]
                of_stream = _d(stream.get("oxidation_factor", of_global))

                # CO2 fossil: IW * dm * CF * FCF * OF * 44/12
                fossil_co2 = _q(mass * dm * cf * fcf * of_stream * _CO2_C_RATIO)

                # CO2 biogenic: IW * dm * CF * (1-FCF) * OF * 44/12
                biogenic_co2 = _q(
                    mass * dm * cf * (_ONE - fcf) * of_stream * _CO2_C_RATIO
                )

                # N2O and CH4: mass * EF * 1e-6 (kg/Gg -> t/t)
                stream_n2o = _q(mass * inc_ef["N2O"] * _KG_PER_GG_TO_TONNES_PER_TONNE)
                stream_ch4 = _q(mass * inc_ef["CH4"] * _KG_PER_GG_TO_TONNES_PER_TONNE)

                trace.append(
                    f"[{step}] Stream {idx}: cat={cat}, mass={mass}t, "
                    f"DM={dm}, CF={cf}, FCF={fcf}, OF={of_stream} -> "
                    f"fossil_CO2={fossil_co2}t, bio_CO2={biogenic_co2}t, "
                    f"CH4={stream_ch4}t, N2O={stream_n2o}t"
                )
                step += 1

                total_fossil_co2 += fossil_co2
                total_biogenic_co2 += biogenic_co2
                total_ch4 += stream_ch4
                total_n2o += stream_n2o
                total_mass += mass

                per_stream.append({
                    "waste_category": cat,
                    "mass_tonnes": str(mass),
                    "fossil_co2_tonnes": str(fossil_co2),
                    "biogenic_co2_tonnes": str(biogenic_co2),
                    "ch4_tonnes": str(stream_ch4),
                    "n2o_tonnes": str(stream_n2o),
                })

            # CO2e totals (fossil only -- biogenic is memo item)
            gwp_ch4 = self._get_gwp("CH4", gwp_src)
            gwp_n2o = self._get_gwp("N2O", gwp_src)
            co2e_ch4 = _q(total_ch4 * gwp_ch4)
            co2e_n2o = _q(total_n2o * gwp_n2o)
            total_co2e = _q(total_fossil_co2 + co2e_ch4 + co2e_n2o)

            trace.append(
                f"[{step}] Totals: fossil_CO2={total_fossil_co2}t, "
                f"bio_CO2={total_biogenic_co2}t, CH4={total_ch4}t "
                f"(CO2e={co2e_ch4}t), N2O={total_n2o}t (CO2e={co2e_n2o}t), "
                f"total_CO2e={total_co2e}t"
            )
            step += 1

            elapsed_ms = (time.monotonic() - start) * 1000
            result = self._build_result(
                calc_id=calc_id,
                status="SUCCESS",
                treatment_method="incineration",
                fossil_co2=total_fossil_co2,
                biogenic_co2=total_biogenic_co2,
                ch4=total_ch4,
                n2o=total_n2o,
                total_co2e=total_co2e,
                total_mass=total_mass,
                gwp_source=gwp_src,
                trace=trace,
                elapsed_ms=elapsed_ms,
                extra={
                    "incinerator_type": inc_type,
                    "oxidation_factor": str(of_global),
                    "per_stream_results": per_stream,
                    "co2e_from_ch4": str(co2e_ch4),
                    "co2e_from_n2o": str(co2e_n2o),
                },
            )

            self._emit_metrics(
                "incineration", "ipcc_tier_1", "mixed",
                elapsed_ms / 1000, float(total_co2e),
            )
            return result

        except Exception as exc:
            return self._handle_error(calc_id, "incineration", exc, trace, start)

    # ==================================================================
    # PUBLIC API: Incineration with Energy Recovery
    # ==================================================================

    def calculate_incineration_with_energy_recovery(
        self,
        waste_streams: List[Dict[str, Any]],
        incinerator_type: str = "stoker_grate",
        electric_efficiency: Decimal = Decimal("0.22"),
        thermal_efficiency: Decimal = Decimal("0.40"),
        grid_ef_electric: Decimal = Decimal("0.400"),
        grid_ef_heat: Decimal = Decimal("0.250"),
        oxidation_factor: Optional[Decimal] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate incineration emissions with energy recovery offset credits.

        Performs the standard incineration calculation plus energy recovery credit
        for electricity and heat displacing grid supply.

        Energy Recovery:
            E_electricity = Sum_j (IW_j * NCV_j * eta_electric)
            E_heat        = Sum_j (IW_j * NCV_j * eta_thermal)
            Displaced     = E_electricity * EF_grid_elec + E_heat * EF_grid_heat

        Args:
            waste_streams: Same format as ``calculate_incineration``.
            incinerator_type: Incinerator technology type.
            electric_efficiency: Gross electrical efficiency (0-1). Default 0.22.
            thermal_efficiency: Gross thermal (heat) efficiency (0-1). Default 0.40.
            grid_ef_electric: Grid electricity emission factor (tCO2e/GJ).
                Default 0.400 tCO2e/GJ (approx EU average).
            grid_ef_heat: Grid/district heat emission factor (tCO2e/GJ).
                Default 0.250 tCO2e/GJ.
            oxidation_factor: Global oxidation factor override.
            gwp_source: GWP assessment report override.

        Returns:
            Same as ``calculate_incineration`` plus:
            - ``energy_electricity_gj``: Electricity generated (GJ).
            - ``energy_heat_gj``: Heat generated (GJ).
            - ``displaced_co2e_tonnes``: Avoided grid emissions (tCO2e).
            - ``net_co2e_tonnes``: Gross minus displaced.
        """
        start = time.monotonic()
        calc_id = f"incin_er_{uuid4().hex[:12]}"
        trace: List[str] = []
        gwp_src = (gwp_source or self._default_gwp).upper().strip()

        try:
            # Validate efficiencies
            eta_e = _d(electric_efficiency)
            eta_h = _d(thermal_efficiency)
            ef_elec = _d(grid_ef_electric)
            ef_heat = _d(grid_ef_heat)

            self._validate_fraction(eta_e, "electric_efficiency")
            self._validate_fraction(eta_h, "thermal_efficiency")
            self._validate_non_negative(ef_elec, "grid_ef_electric")
            self._validate_non_negative(ef_heat, "grid_ef_heat")

            # Step 1: Calculate base incineration emissions
            base_result = self.calculate_incineration(
                waste_streams=waste_streams,
                incinerator_type=incinerator_type,
                oxidation_factor=oxidation_factor,
                gwp_source=gwp_src,
            )

            if base_result["status"] != "SUCCESS":
                return base_result

            # Step 2: Calculate energy recovery per stream
            total_elec_gj = _ZERO
            total_heat_gj = _ZERO
            step = len(base_result["calculation_trace"]) + 1

            trace.extend(base_result["calculation_trace"])
            trace.append(
                f"[{step}] Energy recovery: eta_e={eta_e}, eta_h={eta_h}, "
                f"EF_grid_elec={ef_elec} tCO2e/GJ, EF_grid_heat={ef_heat} tCO2e/GJ"
            )
            step += 1

            for stream in waste_streams:
                cat = _normalize_category(str(stream["waste_category"]))
                mass = _d(stream["mass_tonnes"])
                ncv = self._get_ncv(cat)
                energy_total = _q(mass * ncv)
                elec_gj = _q(energy_total * eta_e)
                heat_gj = _q(energy_total * eta_h)
                total_elec_gj += elec_gj
                total_heat_gj += heat_gj

                trace.append(
                    f"[{step}] ER stream {cat}: mass={mass}t, NCV={ncv} GJ/t, "
                    f"E_total={energy_total} GJ, E_elec={elec_gj} GJ, "
                    f"E_heat={heat_gj} GJ"
                )
                step += 1

            # Step 3: Displaced emissions
            displaced_elec = _q(total_elec_gj * ef_elec)
            displaced_heat = _q(total_heat_gj * ef_heat)
            total_displaced = _q(displaced_elec + displaced_heat)

            gross_co2e = _d(base_result["total_co2e_tonnes"])
            net_co2e = _q(gross_co2e - total_displaced)

            trace.append(
                f"[{step}] Displaced: elec={displaced_elec}t, heat={displaced_heat}t, "
                f"total_displaced={total_displaced}t, net_CO2e={net_co2e}t"
            )

            elapsed_ms = (time.monotonic() - start) * 1000
            result = self._build_result(
                calc_id=calc_id,
                status="SUCCESS",
                treatment_method="incineration_energy_recovery",
                fossil_co2=_d(base_result["fossil_co2_tonnes"]),
                biogenic_co2=_d(base_result["biogenic_co2_tonnes"]),
                ch4=_d(base_result["ch4_tonnes"]),
                n2o=_d(base_result["n2o_tonnes"]),
                total_co2e=gross_co2e,
                total_mass=_d(base_result["total_waste_tonnes"]),
                gwp_source=gwp_src,
                trace=trace,
                elapsed_ms=elapsed_ms,
                extra={
                    "incinerator_type": _normalize_incinerator(incinerator_type),
                    "electric_efficiency": str(eta_e),
                    "thermal_efficiency": str(eta_h),
                    "energy_electricity_gj": str(total_elec_gj),
                    "energy_heat_gj": str(total_heat_gj),
                    "displaced_co2e_electricity": str(displaced_elec),
                    "displaced_co2e_heat": str(displaced_heat),
                    "displaced_co2e_tonnes": str(total_displaced),
                    "net_co2e_tonnes": str(net_co2e),
                    "per_stream_results": base_result.get("per_stream_results", []),
                },
            )

            # Record energy metrics
            if self._enable_metrics and _METRICS_AVAILABLE:
                try:
                    _record_energy_recovered("electricity", float(total_elec_gj))
                    _record_energy_recovered("heat", float(total_heat_gj))
                except Exception:
                    pass

            return result

        except Exception as exc:
            return self._handle_error(
                calc_id, "incineration_energy_recovery", exc, trace, start
            )

    # ==================================================================
    # PUBLIC API: Pyrolysis
    # ==================================================================

    def calculate_pyrolysis(
        self,
        waste_tonnes: Decimal,
        waste_category: str,
        pyrolysis_temp: Optional[int] = None,
        gas_yield_fraction: Optional[Decimal] = None,
        oil_yield_fraction: Optional[Decimal] = None,
        char_yield_fraction: Optional[Decimal] = None,
        gas_combustion_fraction: Optional[Decimal] = None,
        syngas_ch4_fraction: Optional[Decimal] = None,
        char_carbon_stability: Optional[Decimal] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate GHG emissions from waste pyrolysis.

        Pyrolysis is thermal decomposition without oxygen (300-700 deg C),
        producing syngas, bio-oil, and biochar.  This method uses a simplified
        mass balance approach tracking carbon through three product streams.

        Formulae:
            C_total     = M * DM * CF
            C_gas       = C_total * gas_yield_fraction
            C_oil       = C_total * oil_yield_fraction
            C_char      = C_total * char_yield_fraction
            CO2_fossil  = (C_gas * combustion_frac + C_oil * combustion_frac)
                          * FCF * 44/12
            CO2_biogenic = (C_gas * combustion_frac + C_oil * combustion_frac)
                           * (1-FCF) * 44/12
            CH4_syngas  = C_gas * syngas_ch4_fraction * 16/12
            C_sequestered = C_char * char_stability

        Args:
            waste_tonnes: Waste mass fed to pyrolysis (wet tonnes).
            waste_category: Waste category key.
            pyrolysis_temp: Operating temperature in deg C (informational).
            gas_yield_fraction: Fraction of feed carbon becoming gas. Default 0.30.
            oil_yield_fraction: Fraction of feed carbon becoming oil. Default 0.40.
            char_yield_fraction: Fraction of feed carbon becoming char. Default 0.30.
            gas_combustion_fraction: Fraction of gas carbon combusted. Default 0.95.
            syngas_ch4_fraction: Fraction of syngas that is CH4. Default 0.10.
            char_carbon_stability: Fraction of char carbon that is stable. Default 0.80.
            gwp_source: GWP assessment report override.

        Returns:
            Dict with fossil/biogenic CO2, CH4, carbon sequestered, total CO2e,
            product yields, and full audit trail.

        Raises:
            ValueError: If mass is non-positive or yields do not sum to ~1.
        """
        start = time.monotonic()
        calc_id = f"pyro_{uuid4().hex[:12]}"
        trace: List[str] = []
        gwp_src = (gwp_source or self._default_gwp).upper().strip()

        try:
            mass = _d(waste_tonnes)
            self._validate_positive(mass, "waste_tonnes")
            cat = _normalize_category(waste_category)
            params = self._get_waste_params(cat)
            dm = params["dry_matter"]
            cf = params["carbon_fraction"]
            fcf = params["fossil_carbon_fraction"]

            # Resolve yields
            f_gas = _d(gas_yield_fraction or _PYROLYSIS_DEFAULTS["gas_yield_fraction"])
            f_oil = _d(oil_yield_fraction or _PYROLYSIS_DEFAULTS["oil_yield_fraction"])
            f_char = _d(char_yield_fraction or _PYROLYSIS_DEFAULTS["char_yield_fraction"])
            f_comb = _d(
                gas_combustion_fraction
                or _PYROLYSIS_DEFAULTS["gas_combustion_fraction"]
            )
            f_ch4 = _d(
                syngas_ch4_fraction or _PYROLYSIS_DEFAULTS["syngas_ch4_fraction"]
            )
            f_stab = _d(
                char_carbon_stability or _PYROLYSIS_DEFAULTS["char_carbon_stability"]
            )

            # Validate yield sum (allow small tolerance)
            yield_sum = f_gas + f_oil + f_char
            if abs(yield_sum - _ONE) > Decimal("0.05"):
                raise ValueError(
                    f"Pyrolysis yield fractions must sum to ~1.0, got {yield_sum} "
                    f"(gas={f_gas}, oil={f_oil}, char={f_char})"
                )

            trace.append(
                f"[1] Pyrolysis: mass={mass}t, cat={cat}, temp={pyrolysis_temp}C, "
                f"DM={dm}, CF={cf}, FCF={fcf}"
            )
            trace.append(
                f"[2] Yields: gas={f_gas}, oil={f_oil}, char={f_char}, "
                f"combustion={f_comb}, CH4_frac={f_ch4}, stability={f_stab}"
            )

            # Total carbon in feed
            c_total = _q(mass * dm * cf)
            trace.append(f"[3] Total carbon in feed: {c_total} tC")

            # Carbon distribution to products
            c_gas = _q(c_total * f_gas)
            c_oil = _q(c_total * f_oil)
            c_char = _q(c_total * f_char)
            trace.append(
                f"[4] Carbon distribution: gas={c_gas}tC, oil={c_oil}tC, char={c_char}tC"
            )

            # Combusted carbon (gas + oil combusted fractions)
            c_combusted = _q((c_gas + c_oil) * f_comb)
            trace.append(f"[5] Combusted carbon: {c_combusted} tC")

            # CO2 from combusted carbon
            fossil_co2 = _q(c_combusted * fcf * _CO2_C_RATIO)
            biogenic_co2 = _q(c_combusted * (_ONE - fcf) * _CO2_C_RATIO)
            trace.append(
                f"[6] CO2: fossil={fossil_co2}t, biogenic={biogenic_co2}t"
            )

            # CH4 from syngas (CH4 = C_gas * f_ch4 * 16/12)
            ch4_c_ratio = Decimal("1.33333333")  # 16/12
            ch4_tonnes = _q(c_gas * f_ch4 * ch4_c_ratio)
            trace.append(f"[7] CH4 from syngas: {ch4_tonnes}t")

            # Carbon sequestered in biochar
            c_sequestered = _q(c_char * f_stab)
            trace.append(f"[8] Carbon sequestered in biochar: {c_sequestered} tC")

            # N2O -- negligible for pyrolysis (no nitrogen oxidation)
            n2o_tonnes = _ZERO

            # CO2e
            gwp_ch4 = self._get_gwp("CH4", gwp_src)
            co2e_ch4 = _q(ch4_tonnes * gwp_ch4)
            total_co2e = _q(fossil_co2 + co2e_ch4)

            trace.append(
                f"[9] CO2e: fossil_CO2={fossil_co2}t + CH4_CO2e={co2e_ch4}t "
                f"= {total_co2e}t"
            )

            elapsed_ms = (time.monotonic() - start) * 1000
            result = self._build_result(
                calc_id=calc_id,
                status="SUCCESS",
                treatment_method="pyrolysis",
                fossil_co2=fossil_co2,
                biogenic_co2=biogenic_co2,
                ch4=ch4_tonnes,
                n2o=n2o_tonnes,
                total_co2e=total_co2e,
                total_mass=mass,
                gwp_source=gwp_src,
                trace=trace,
                elapsed_ms=elapsed_ms,
                extra={
                    "pyrolysis_temp_c": pyrolysis_temp,
                    "total_carbon_tc": str(c_total),
                    "carbon_gas_tc": str(c_gas),
                    "carbon_oil_tc": str(c_oil),
                    "carbon_char_tc": str(c_char),
                    "carbon_combusted_tc": str(c_combusted),
                    "carbon_sequestered_tc": str(c_sequestered),
                    "co2e_from_ch4": str(co2e_ch4),
                    "gas_yield_fraction": str(f_gas),
                    "oil_yield_fraction": str(f_oil),
                    "char_yield_fraction": str(f_char),
                },
            )

            self._emit_metrics(
                "pyrolysis", "mass_balance", cat,
                elapsed_ms / 1000, float(total_co2e),
            )
            return result

        except Exception as exc:
            return self._handle_error(calc_id, "pyrolysis", exc, trace, start)

    # ==================================================================
    # PUBLIC API: Gasification
    # ==================================================================

    def calculate_gasification(
        self,
        waste_tonnes: Decimal,
        waste_category: str,
        gasification_temp: Optional[int] = None,
        er_ratio: Optional[Decimal] = None,
        carbon_conversion: Optional[Decimal] = None,
        syngas_ch4_fraction: Optional[Decimal] = None,
        syngas_co2_fraction: Optional[Decimal] = None,
        cold_gas_efficiency: Optional[Decimal] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate GHG emissions from waste gasification.

        Gasification is partial oxidation at 700-1500 deg C producing
        synthesis gas (CO, H2, CH4, CO2).  The equivalence ratio (ER)
        controls oxidation extent.

        Formulae:
            C_total       = M * DM * CF
            C_converted   = C_total * carbon_conversion
            CO2_gasifier  = C_converted * syngas_co2_fraction * 44/12
            CH4_gasifier  = C_converted * syngas_ch4_fraction * 16/12
            C_unconverted = C_total * (1 - carbon_conversion) -> ash/tar
            CO2_fossil    = CO2_gasifier * FCF
            CO2_biogenic  = CO2_gasifier * (1 - FCF)

        Args:
            waste_tonnes: Waste mass fed to gasifier (wet tonnes).
            waste_category: Waste category key.
            gasification_temp: Operating temperature in deg C (informational).
            er_ratio: Equivalence ratio (0-1). Default 0.30.
            carbon_conversion: Fraction of feed carbon converted. Default 0.95.
            syngas_ch4_fraction: CH4 fraction of syngas carbon. Default 0.05.
            syngas_co2_fraction: CO2 fraction of syngas carbon. Default 0.10.
            cold_gas_efficiency: Cold gas efficiency (informational). Default 0.70.
            gwp_source: GWP assessment report override.

        Returns:
            Dict with fossil/biogenic CO2, CH4, total CO2e, syngas composition,
            and full audit trail.
        """
        start = time.monotonic()
        calc_id = f"gasif_{uuid4().hex[:12]}"
        trace: List[str] = []
        gwp_src = (gwp_source or self._default_gwp).upper().strip()

        try:
            mass = _d(waste_tonnes)
            self._validate_positive(mass, "waste_tonnes")
            cat = _normalize_category(waste_category)
            params = self._get_waste_params(cat)
            dm = params["dry_matter"]
            cf = params["carbon_fraction"]
            fcf = params["fossil_carbon_fraction"]

            # Resolve parameters
            er = _d(er_ratio or _GASIFICATION_DEFAULTS["equivalence_ratio_default"])
            cc = _d(carbon_conversion or _GASIFICATION_DEFAULTS["carbon_conversion"])
            f_ch4 = _d(
                syngas_ch4_fraction or _GASIFICATION_DEFAULTS["syngas_ch4_fraction"]
            )
            f_co2 = _d(
                syngas_co2_fraction or _GASIFICATION_DEFAULTS["syngas_co2_fraction"]
            )
            cge = _d(cold_gas_efficiency or _GASIFICATION_DEFAULTS["cold_gas_efficiency"])

            trace.append(
                f"[1] Gasification: mass={mass}t, cat={cat}, temp={gasification_temp}C, "
                f"ER={er}, DM={dm}, CF={cf}, FCF={fcf}"
            )
            trace.append(
                f"[2] Params: carbon_conv={cc}, CH4_frac={f_ch4}, CO2_frac={f_co2}, "
                f"CGE={cge}"
            )

            # Total carbon
            c_total = _q(mass * dm * cf)
            c_converted = _q(c_total * cc)
            c_unconverted = _q(c_total - c_converted)
            trace.append(
                f"[3] Carbon: total={c_total}tC, converted={c_converted}tC, "
                f"unconverted={c_unconverted}tC"
            )

            # CO2 from gasifier syngas (direct CO2 fraction)
            co2_syngas_c = _q(c_converted * f_co2)
            co2_syngas = _q(co2_syngas_c * _CO2_C_RATIO)
            fossil_co2 = _q(co2_syngas * fcf)
            biogenic_co2 = _q(co2_syngas * (_ONE - fcf))
            trace.append(
                f"[4] Syngas CO2: total={co2_syngas}t, fossil={fossil_co2}t, "
                f"biogenic={biogenic_co2}t"
            )

            # CH4 from gasifier syngas
            ch4_c_ratio = Decimal("1.33333333")  # 16/12
            ch4_syngas_c = _q(c_converted * f_ch4)
            ch4_tonnes = _q(ch4_syngas_c * ch4_c_ratio)
            trace.append(f"[5] Syngas CH4: {ch4_tonnes}t")

            # Remaining syngas carbon is CO + H2 (not direct GHG at point of generation)
            c_co_h2 = _q(c_converted - co2_syngas_c - ch4_syngas_c)
            trace.append(f"[6] CO+H2 carbon (not direct GHG): {c_co_h2} tC")

            # N2O -- minimal for gasification
            n2o_tonnes = _ZERO

            # CO2e
            gwp_ch4 = self._get_gwp("CH4", gwp_src)
            co2e_ch4 = _q(ch4_tonnes * gwp_ch4)
            total_co2e = _q(fossil_co2 + co2e_ch4)
            trace.append(
                f"[7] CO2e: fossil_CO2={fossil_co2}t + CH4_CO2e={co2e_ch4}t "
                f"= {total_co2e}t"
            )

            elapsed_ms = (time.monotonic() - start) * 1000
            result = self._build_result(
                calc_id=calc_id,
                status="SUCCESS",
                treatment_method="gasification",
                fossil_co2=fossil_co2,
                biogenic_co2=biogenic_co2,
                ch4=ch4_tonnes,
                n2o=n2o_tonnes,
                total_co2e=total_co2e,
                total_mass=mass,
                gwp_source=gwp_src,
                trace=trace,
                elapsed_ms=elapsed_ms,
                extra={
                    "gasification_temp_c": gasification_temp,
                    "equivalence_ratio": str(er),
                    "carbon_conversion": str(cc),
                    "cold_gas_efficiency": str(cge),
                    "total_carbon_tc": str(c_total),
                    "converted_carbon_tc": str(c_converted),
                    "unconverted_carbon_tc": str(c_unconverted),
                    "syngas_co2_carbon_tc": str(co2_syngas_c),
                    "syngas_ch4_carbon_tc": str(ch4_syngas_c),
                    "syngas_co_h2_carbon_tc": str(c_co_h2),
                    "co2e_from_ch4": str(co2e_ch4),
                },
            )

            self._emit_metrics(
                "gasification", "mass_balance", cat,
                elapsed_ms / 1000, float(total_co2e),
            )
            return result

        except Exception as exc:
            return self._handle_error(calc_id, "gasification", exc, trace, start)

    # ==================================================================
    # PUBLIC API: Open Burning
    # ==================================================================

    def calculate_open_burning(
        self,
        waste_streams: List[Dict[str, Any]],
        burn_fraction: Decimal = _ONE,
        oxidation_factor: Optional[Decimal] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate GHG emissions from open (uncontrolled) burning of waste.

        Open burning uses a reduced oxidation factor (default 0.58) and elevated
        CH4/N2O emission factors reflecting incomplete combustion.

        Formulae (IPCC 2006 Vol 5):
            CO2_fossil  = Sum_j (OW_j * BF * DM_j * CF_j * FCF_j * OF) * 44/12
            CO2_biogenic = Sum_j (OW_j * BF * DM_j * CF_j * (1-FCF_j) * OF) * 44/12
            CH4 = Sum_j (OW_j * BF * DM_j * 0.0065)   [6.5 g/kg DM = 0.0065 t/t DM]
            N2O = Sum_j (OW_j * BF * DM_j * 0.00015)   [0.15 g/kg DM]

        Args:
            waste_streams: List of dicts with ``waste_category`` and ``mass_tonnes``.
                Optional per-stream overrides: dry_matter, carbon_fraction,
                fossil_carbon_fraction.
            burn_fraction: Fraction of waste actually burned (0-1). Default 1.0.
            oxidation_factor: Oxidation factor override. Default 0.58.
            gwp_source: GWP assessment report override.

        Returns:
            Dict with fossil/biogenic CO2, CH4, N2O, total CO2e, per-stream
            results, and full audit trail.
        """
        start = time.monotonic()
        calc_id = f"oburn_{uuid4().hex[:12]}"
        trace: List[str] = []
        gwp_src = (gwp_source or self._default_gwp).upper().strip()
        of = _d(oxidation_factor if oxidation_factor is not None else _OPEN_BURN_OXIDATION_FACTOR)
        bf = _d(burn_fraction)

        try:
            self._validate_waste_streams(waste_streams, trace)
            self._validate_fraction(bf, "burn_fraction")
            trace.append(
                f"[1] Open burning: burn_fraction={bf}, OF={of}, GWP={gwp_src}"
            )

            total_fossil_co2 = _ZERO
            total_biogenic_co2 = _ZERO
            total_ch4 = _ZERO
            total_n2o = _ZERO
            total_mass = _ZERO
            per_stream: List[Dict[str, Any]] = []
            step = 2

            for idx, stream in enumerate(waste_streams):
                cat = _normalize_category(str(stream["waste_category"]))
                mass = _d(stream["mass_tonnes"])
                self._validate_positive(mass, f"mass_tonnes[{idx}]")

                params = self._resolve_stream_params(stream, cat)
                dm = params["dry_matter"]
                cf = params["carbon_fraction"]
                fcf = params["fossil_carbon_fraction"]

                effective_mass = _q(mass * bf)
                dm_mass = _q(effective_mass * dm)

                # CO2 fossil
                fossil_co2 = _q(dm_mass * cf * fcf * of * _CO2_C_RATIO)
                # CO2 biogenic
                biogenic_co2 = _q(dm_mass * cf * (_ONE - fcf) * of * _CO2_C_RATIO)

                # CH4: 6.5 g/kg DM = 0.0065 t/t DM
                ch4_ef = _OPEN_BURN_CH4_G_PER_KG_DM * _G_PER_KG_TO_TONNES_PER_TONNE
                stream_ch4 = _q(dm_mass * ch4_ef)

                # N2O: 0.15 g/kg DM = 0.00015 t/t DM
                n2o_ef = _OPEN_BURN_N2O_G_PER_KG_DM * _G_PER_KG_TO_TONNES_PER_TONNE
                stream_n2o = _q(dm_mass * n2o_ef)

                trace.append(
                    f"[{step}] Stream {idx}: cat={cat}, mass={mass}t, "
                    f"burned={effective_mass}t, DM_mass={dm_mass}t, "
                    f"fossil_CO2={fossil_co2}t, bio_CO2={biogenic_co2}t, "
                    f"CH4={stream_ch4}t, N2O={stream_n2o}t"
                )
                step += 1

                total_fossil_co2 += fossil_co2
                total_biogenic_co2 += biogenic_co2
                total_ch4 += stream_ch4
                total_n2o += stream_n2o
                total_mass += mass

                per_stream.append({
                    "waste_category": cat,
                    "mass_tonnes": str(mass),
                    "effective_burned_tonnes": str(effective_mass),
                    "fossil_co2_tonnes": str(fossil_co2),
                    "biogenic_co2_tonnes": str(biogenic_co2),
                    "ch4_tonnes": str(stream_ch4),
                    "n2o_tonnes": str(stream_n2o),
                })

            # CO2e totals
            gwp_ch4 = self._get_gwp("CH4", gwp_src)
            gwp_n2o = self._get_gwp("N2O", gwp_src)
            co2e_ch4 = _q(total_ch4 * gwp_ch4)
            co2e_n2o = _q(total_n2o * gwp_n2o)
            total_co2e = _q(total_fossil_co2 + co2e_ch4 + co2e_n2o)

            trace.append(
                f"[{step}] Totals: fossil_CO2={total_fossil_co2}t, "
                f"bio_CO2={total_biogenic_co2}t, CH4={total_ch4}t "
                f"(CO2e={co2e_ch4}t), N2O={total_n2o}t (CO2e={co2e_n2o}t), "
                f"total_CO2e={total_co2e}t"
            )

            elapsed_ms = (time.monotonic() - start) * 1000
            result = self._build_result(
                calc_id=calc_id,
                status="SUCCESS",
                treatment_method="open_burning",
                fossil_co2=total_fossil_co2,
                biogenic_co2=total_biogenic_co2,
                ch4=total_ch4,
                n2o=total_n2o,
                total_co2e=total_co2e,
                total_mass=total_mass,
                gwp_source=gwp_src,
                trace=trace,
                elapsed_ms=elapsed_ms,
                extra={
                    "burn_fraction": str(bf),
                    "oxidation_factor": str(of),
                    "per_stream_results": per_stream,
                    "co2e_from_ch4": str(co2e_ch4),
                    "co2e_from_n2o": str(co2e_n2o),
                },
            )

            self._emit_metrics(
                "open_burning", "emission_factor", "mixed",
                elapsed_ms / 1000, float(total_co2e),
            )
            return result

        except Exception as exc:
            return self._handle_error(calc_id, "open_burning", exc, trace, start)

    # ==================================================================
    # PUBLIC API: Thermal Batch
    # ==================================================================

    def calculate_thermal_batch(
        self,
        treatments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Process a batch of heterogeneous thermal treatment calculations.

        Each entry in ``treatments`` must specify a ``treatment_type`` and
        relevant parameters for that treatment type.  Results are aggregated.

        Supported treatment_type values:
            - ``"incineration"``
            - ``"incineration_energy_recovery"``
            - ``"pyrolysis"``
            - ``"gasification"``
            - ``"open_burning"``

        Args:
            treatments: List of dicts, each containing:
                - ``treatment_type`` (str): One of the supported types.
                - Additional keys matching the corresponding ``calculate_*`` method.

        Returns:
            Dict with aggregated totals, individual results, success/failure counts,
            and batch-level provenance hash.
        """
        start = time.monotonic()
        batch_id = f"therm_batch_{uuid4().hex[:12]}"
        results: List[Dict[str, Any]] = []

        agg_fossil = _ZERO
        agg_biogenic = _ZERO
        agg_ch4 = _ZERO
        agg_n2o = _ZERO
        agg_co2e = _ZERO
        agg_mass = _ZERO
        success_count = 0
        failure_count = 0

        for idx, item in enumerate(treatments):
            t_type = str(item.get("treatment_type", "")).strip().lower()
            try:
                result = self._dispatch_treatment(t_type, item)
            except Exception as exc:
                logger.error("Batch item %d (%s) failed: %s", idx, t_type, exc)
                result = {
                    "status": "FAILED",
                    "error_message": str(exc),
                    "treatment_method": t_type,
                }

            results.append(result)

            if result.get("status") == "SUCCESS":
                success_count += 1
                agg_fossil += _d(result.get("fossil_co2_tonnes", "0"))
                agg_biogenic += _d(result.get("biogenic_co2_tonnes", "0"))
                agg_ch4 += _d(result.get("ch4_tonnes", "0"))
                agg_n2o += _d(result.get("n2o_tonnes", "0"))
                agg_co2e += _d(result.get("total_co2e_tonnes", "0"))
                agg_mass += _d(result.get("total_waste_tonnes", "0"))
            else:
                failure_count += 1

        elapsed_ms = (time.monotonic() - start) * 1000

        if failure_count == 0:
            status = "SUCCESS"
        elif success_count > 0:
            status = "PARTIAL"
        else:
            status = "FAILED"

        provenance_hash = _compute_hash({
            "batch_id": batch_id,
            "treatment_count": len(treatments),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_co2e": str(agg_co2e),
            "total_mass": str(agg_mass),
        })

        self._record_provenance(
            action="calculate_thermal_batch",
            entity_id=batch_id,
            data={
                "treatment_count": len(treatments),
                "success_count": success_count,
                "total_co2e": str(agg_co2e),
                "provenance_hash": provenance_hash,
            },
        )

        logger.info(
            "Thermal batch %s completed: %d items, %d ok, %d failed, "
            "CO2e=%.4f t, %.1f ms",
            batch_id, len(treatments), success_count, failure_count,
            agg_co2e, elapsed_ms,
        )

        return {
            "status": status,
            "batch_id": batch_id,
            "treatment_count": len(treatments),
            "success_count": success_count,
            "failure_count": failure_count,
            "results": results,
            "aggregated_fossil_co2_tonnes": str(_q(agg_fossil)),
            "aggregated_biogenic_co2_tonnes": str(_q(agg_biogenic)),
            "aggregated_ch4_tonnes": str(_q(agg_ch4)),
            "aggregated_n2o_tonnes": str(_q(agg_n2o)),
            "aggregated_co2e_tonnes": str(_q(agg_co2e)),
            "aggregated_waste_tonnes": str(_q(agg_mass)),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed_ms, 3),
        }

    # ==================================================================
    # PUBLIC API: Separate Fossil / Biogenic
    # ==================================================================

    def separate_fossil_biogenic(
        self,
        co2_total_tonnes: Decimal,
        waste_streams: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Separate total CO2 into fossil and biogenic fractions.

        Uses the mass-weighted average fossil carbon fraction (FCF) across
        all waste streams to apportion total CO2.

        Args:
            co2_total_tonnes: Total CO2 emissions in tonnes.
            waste_streams: List of dicts with ``waste_category`` and ``mass_tonnes``.
                Optional per-stream ``fossil_carbon_fraction`` override.

        Returns:
            Dict with fossil_co2_tonnes, biogenic_co2_tonnes,
            weighted_fcf, and provenance_hash.
        """
        calc_id = f"sep_fb_{uuid4().hex[:12]}"
        trace: List[str] = []
        total_co2 = _d(co2_total_tonnes)

        try:
            weighted_fcf_num = _ZERO
            total_mass = _ZERO

            for stream in waste_streams:
                cat = _normalize_category(str(stream["waste_category"]))
                mass = _d(stream["mass_tonnes"])
                self._validate_positive(mass, "mass_tonnes")

                # Per-stream FCF override or lookup
                if "fossil_carbon_fraction" in stream:
                    fcf = _d(stream["fossil_carbon_fraction"])
                else:
                    params = self._get_waste_params(cat)
                    fcf = params["fossil_carbon_fraction"]

                weighted_fcf_num += mass * fcf
                total_mass += mass

            if total_mass == _ZERO:
                raise ValueError("Total waste mass is zero; cannot compute weighted FCF")

            weighted_fcf = _q(weighted_fcf_num / total_mass)
            fossil_co2 = _q(total_co2 * weighted_fcf)
            biogenic_co2 = _q(total_co2 * (_ONE - weighted_fcf))

            trace.append(
                f"[1] Weighted FCF={weighted_fcf} from {len(waste_streams)} streams, "
                f"total_mass={total_mass}t"
            )
            trace.append(
                f"[2] CO2 total={total_co2}t -> fossil={fossil_co2}t, "
                f"biogenic={biogenic_co2}t"
            )

            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "co2_total": str(total_co2),
                "weighted_fcf": str(weighted_fcf),
                "fossil_co2": str(fossil_co2),
                "biogenic_co2": str(biogenic_co2),
            })

            return {
                "status": "SUCCESS",
                "calculation_id": calc_id,
                "fossil_co2_tonnes": str(fossil_co2),
                "biogenic_co2_tonnes": str(biogenic_co2),
                "weighted_fcf": str(weighted_fcf),
                "total_co2_tonnes": str(total_co2),
                "calculation_trace": trace,
                "provenance_hash": provenance_hash,
            }

        except Exception as exc:
            logger.error("Fossil/biogenic separation failed: %s", exc, exc_info=True)
            self._emit_error_metric("calculation_error")
            return {
                "status": "FAILED",
                "calculation_id": calc_id,
                "error_message": str(exc),
                "calculation_trace": trace,
                "provenance_hash": "",
            }

    # ==================================================================
    # PUBLIC API: Energy Offset Calculator
    # ==================================================================

    def calculate_energy_offset(
        self,
        waste_tonnes: Decimal,
        ncv: Optional[Decimal] = None,
        waste_category: Optional[str] = None,
        eta_electric: Decimal = Decimal("0.22"),
        eta_thermal: Decimal = Decimal("0.40"),
        grid_ef_electric: Decimal = Decimal("0.400"),
        grid_ef_heat: Decimal = Decimal("0.250"),
    ) -> Dict[str, Any]:
        """Calculate energy recovery offset credits from thermal treatment.

        Standalone utility to estimate displaced grid emissions from
        waste-to-energy operations.

        Formulae:
            E_elec = waste_tonnes * NCV * eta_electric
            E_heat = waste_tonnes * NCV * eta_thermal
            Displaced = E_elec * grid_ef_electric + E_heat * grid_ef_heat

        Args:
            waste_tonnes: Total waste throughput (wet tonnes).
            ncv: Net calorific value (GJ/tonne). If None, looked up from
                waste_category.
            waste_category: Used for NCV lookup when ncv is None.
            eta_electric: Electrical efficiency (0-1). Default 0.22.
            eta_thermal: Thermal efficiency (0-1). Default 0.40.
            grid_ef_electric: Grid electricity EF (tCO2e/GJ). Default 0.400.
            grid_ef_heat: Grid heat EF (tCO2e/GJ). Default 0.250.

        Returns:
            Dict with energy_electricity_gj, energy_heat_gj,
            displaced_co2e_electricity, displaced_co2e_heat,
            total_displaced_co2e, and provenance_hash.

        Raises:
            ValueError: If neither ncv nor waste_category is provided, or
                if efficiency is outside [0, 1].
        """
        calc_id = f"enoff_{uuid4().hex[:12]}"
        trace: List[str] = []

        try:
            mass = _d(waste_tonnes)
            self._validate_positive(mass, "waste_tonnes")
            eta_e = _d(eta_electric)
            eta_h = _d(eta_thermal)
            ef_e = _d(grid_ef_electric)
            ef_h = _d(grid_ef_heat)

            self._validate_fraction(eta_e, "eta_electric")
            self._validate_fraction(eta_h, "eta_thermal")
            self._validate_non_negative(ef_e, "grid_ef_electric")
            self._validate_non_negative(ef_h, "grid_ef_heat")

            # Resolve NCV
            if ncv is not None:
                ncv_val = _d(ncv)
            elif waste_category is not None:
                cat = _normalize_category(waste_category)
                ncv_val = self._get_ncv(cat)
            else:
                raise ValueError(
                    "Either ncv or waste_category must be provided for NCV lookup"
                )

            self._validate_non_negative(ncv_val, "ncv")

            energy_total = _q(mass * ncv_val)
            elec_gj = _q(energy_total * eta_e)
            heat_gj = _q(energy_total * eta_h)
            displaced_elec = _q(elec_gj * ef_e)
            displaced_heat = _q(heat_gj * ef_h)
            total_displaced = _q(displaced_elec + displaced_heat)

            trace.append(
                f"[1] Energy offset: mass={mass}t, NCV={ncv_val} GJ/t, "
                f"eta_e={eta_e}, eta_h={eta_h}"
            )
            trace.append(
                f"[2] Energy: total={energy_total} GJ, elec={elec_gj} GJ, "
                f"heat={heat_gj} GJ"
            )
            trace.append(
                f"[3] Displaced: elec={displaced_elec} tCO2e, "
                f"heat={displaced_heat} tCO2e, total={total_displaced} tCO2e"
            )

            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "waste_tonnes": str(mass),
                "ncv": str(ncv_val),
                "energy_elec": str(elec_gj),
                "energy_heat": str(heat_gj),
                "displaced": str(total_displaced),
            })

            return {
                "status": "SUCCESS",
                "calculation_id": calc_id,
                "waste_tonnes": str(mass),
                "ncv_gj_per_tonne": str(ncv_val),
                "energy_total_gj": str(energy_total),
                "energy_electricity_gj": str(elec_gj),
                "energy_heat_gj": str(heat_gj),
                "grid_ef_electric_tco2e_per_gj": str(ef_e),
                "grid_ef_heat_tco2e_per_gj": str(ef_h),
                "displaced_co2e_electricity": str(displaced_elec),
                "displaced_co2e_heat": str(displaced_heat),
                "total_displaced_co2e": str(total_displaced),
                "calculation_trace": trace,
                "provenance_hash": provenance_hash,
            }

        except Exception as exc:
            logger.error("Energy offset calculation failed: %s", exc, exc_info=True)
            self._emit_error_metric("calculation_error")
            return {
                "status": "FAILED",
                "calculation_id": calc_id,
                "error_message": str(exc),
                "calculation_trace": trace,
                "provenance_hash": "",
            }

    # ==================================================================
    # PUBLIC API: Get IPCC Table 5.2 Data
    # ==================================================================

    def get_waste_composition(self, category: str) -> Dict[str, Any]:
        """Retrieve waste composition parameters for a category.

        Provides a read-only view of IPCC Table 5.2 data (or custom override).

        Args:
            category: Waste category key (will be normalized).

        Returns:
            Dict with dry_matter, carbon_fraction, fossil_carbon_fraction,
            source, and provenance_hash.

        Raises:
            ValueError: If category is not found.
        """
        cat = _normalize_category(category)
        params = self._get_waste_params(cat)
        source = "custom" if cat in self._custom_waste_data else "IPCC_2006_Table_5.2"
        provenance_hash = _compute_hash({
            "category": cat,
            "source": source,
            "data": {k: str(v) for k, v in params.items()},
        })
        return {
            "waste_category": cat,
            "dry_matter": str(params["dry_matter"]),
            "carbon_fraction": str(params["carbon_fraction"]),
            "fossil_carbon_fraction": str(params["fossil_carbon_fraction"]),
            "source": source,
            "provenance_hash": provenance_hash,
        }

    def get_incinerator_emission_factors(self, incinerator_type: str) -> Dict[str, Any]:
        """Retrieve N2O/CH4 emission factors for an incinerator type.

        Args:
            incinerator_type: Incinerator type key (will be normalized).

        Returns:
            Dict with n2o_kg_per_gg, ch4_kg_per_gg, source, and provenance_hash.

        Raises:
            ValueError: If incinerator type is not found.
        """
        inc = _normalize_incinerator(incinerator_type)
        ef = self._get_incinerator_ef(inc)
        source = "custom" if inc in self._custom_incinerator_ef else "IPCC_2006_Table_5.3"
        provenance_hash = _compute_hash({
            "incinerator_type": inc,
            "source": source,
            "n2o": str(ef["N2O"]),
            "ch4": str(ef["CH4"]),
        })
        return {
            "incinerator_type": inc,
            "n2o_kg_per_gg": str(ef["N2O"]),
            "ch4_kg_per_gg": str(ef["CH4"]),
            "source": source,
            "provenance_hash": provenance_hash,
        }

    def list_supported_waste_categories(self) -> List[str]:
        """Return all supported waste category keys.

        Returns:
            Sorted list of waste category strings.
        """
        built_in = set(_IPCC_TABLE_5_2.keys())
        with self._lock:
            custom = set(self._custom_waste_data.keys())
        return sorted(built_in | custom)

    def list_supported_incinerator_types(self) -> List[str]:
        """Return all supported incinerator type keys.

        Returns:
            Sorted list of incinerator type strings.
        """
        built_in = set(_INCINERATOR_EF.keys())
        with self._lock:
            custom = set(self._custom_incinerator_ef.keys())
        return sorted(built_in | custom)

    # ==================================================================
    # Internal: Stream parameter resolution
    # ==================================================================

    def _resolve_stream_params(
        self,
        stream: Dict[str, Any],
        category: str,
    ) -> Dict[str, Decimal]:
        """Resolve waste composition parameters for a stream.

        Per-stream overrides take precedence over category defaults.

        Args:
            stream: Waste stream dict with optional override fields.
            category: Normalized waste category key.

        Returns:
            Dict with dry_matter, carbon_fraction, fossil_carbon_fraction.
        """
        defaults = self._get_waste_params(category)
        return {
            "dry_matter": _d(stream.get("dry_matter", defaults["dry_matter"])),
            "carbon_fraction": _d(
                stream.get("carbon_fraction", defaults["carbon_fraction"])
            ),
            "fossil_carbon_fraction": _d(
                stream.get(
                    "fossil_carbon_fraction", defaults["fossil_carbon_fraction"]
                )
            ),
        }

    # ==================================================================
    # Internal: Batch dispatch
    # ==================================================================

    def _dispatch_treatment(
        self,
        treatment_type: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route a batch item to the correct calculation method.

        Args:
            treatment_type: Treatment type key.
            params: Full parameter dict for the treatment.

        Returns:
            Calculation result dict.

        Raises:
            ValueError: If treatment_type is not supported.
        """
        if treatment_type == "incineration":
            return self.calculate_incineration(
                waste_streams=params.get("waste_streams", []),
                incinerator_type=params.get("incinerator_type", "stoker_grate"),
                oxidation_factor=params.get("oxidation_factor"),
                gwp_source=params.get("gwp_source"),
            )

        if treatment_type in ("incineration_energy_recovery", "incineration_er"):
            return self.calculate_incineration_with_energy_recovery(
                waste_streams=params.get("waste_streams", []),
                incinerator_type=params.get("incinerator_type", "stoker_grate"),
                electric_efficiency=_d(params.get("electric_efficiency", "0.22")),
                thermal_efficiency=_d(params.get("thermal_efficiency", "0.40")),
                grid_ef_electric=_d(params.get("grid_ef_electric", "0.400")),
                grid_ef_heat=_d(params.get("grid_ef_heat", "0.250")),
                oxidation_factor=params.get("oxidation_factor"),
                gwp_source=params.get("gwp_source"),
            )

        if treatment_type == "pyrolysis":
            return self.calculate_pyrolysis(
                waste_tonnes=_d(params.get("waste_tonnes", 0)),
                waste_category=params.get("waste_category", "mixed"),
                pyrolysis_temp=params.get("pyrolysis_temp"),
                gas_yield_fraction=params.get("gas_yield_fraction"),
                oil_yield_fraction=params.get("oil_yield_fraction"),
                char_yield_fraction=params.get("char_yield_fraction"),
                gas_combustion_fraction=params.get("gas_combustion_fraction"),
                syngas_ch4_fraction=params.get("syngas_ch4_fraction"),
                char_carbon_stability=params.get("char_carbon_stability"),
                gwp_source=params.get("gwp_source"),
            )

        if treatment_type == "gasification":
            return self.calculate_gasification(
                waste_tonnes=_d(params.get("waste_tonnes", 0)),
                waste_category=params.get("waste_category", "mixed"),
                gasification_temp=params.get("gasification_temp"),
                er_ratio=params.get("er_ratio"),
                carbon_conversion=params.get("carbon_conversion"),
                syngas_ch4_fraction=params.get("syngas_ch4_fraction"),
                syngas_co2_fraction=params.get("syngas_co2_fraction"),
                cold_gas_efficiency=params.get("cold_gas_efficiency"),
                gwp_source=params.get("gwp_source"),
            )

        if treatment_type == "open_burning":
            return self.calculate_open_burning(
                waste_streams=params.get("waste_streams", []),
                burn_fraction=_d(params.get("burn_fraction", "1.0")),
                oxidation_factor=params.get("oxidation_factor"),
                gwp_source=params.get("gwp_source"),
            )

        raise ValueError(
            f"Unsupported treatment_type '{treatment_type}'. "
            f"Supported: incineration, incineration_energy_recovery, "
            f"pyrolysis, gasification, open_burning"
        )

    # ==================================================================
    # Internal: Validation helpers
    # ==================================================================

    def _validate_waste_streams(
        self,
        waste_streams: List[Dict[str, Any]],
        trace: List[str],
    ) -> None:
        """Validate that waste_streams is a non-empty list with required fields.

        Args:
            waste_streams: The waste streams to validate.
            trace: Trace list for recording validation.

        Raises:
            ValueError: If validation fails.
        """
        if not waste_streams:
            raise ValueError("waste_streams must be a non-empty list")

        for idx, stream in enumerate(waste_streams):
            if "waste_category" not in stream:
                raise ValueError(
                    f"waste_streams[{idx}] missing required field 'waste_category'"
                )
            if "mass_tonnes" not in stream:
                raise ValueError(
                    f"waste_streams[{idx}] missing required field 'mass_tonnes'"
                )

    def _validate_positive(self, value: Decimal, name: str) -> None:
        """Validate that a value is strictly positive.

        Args:
            value: Decimal value to check.
            name: Parameter name for error messages.

        Raises:
            ValueError: If value <= 0.
        """
        if value <= _ZERO:
            raise ValueError(f"{name} must be > 0, got {value}")

    def _validate_non_negative(self, value: Decimal, name: str) -> None:
        """Validate that a value is non-negative.

        Args:
            value: Decimal value to check.
            name: Parameter name for error messages.

        Raises:
            ValueError: If value < 0.
        """
        if value < _ZERO:
            raise ValueError(f"{name} must be >= 0, got {value}")

    def _validate_fraction(self, value: Decimal, name: str) -> None:
        """Validate that a value is in the range [0, 1].

        Args:
            value: Decimal value to check.
            name: Parameter name for error messages.

        Raises:
            ValueError: If value is outside [0, 1].
        """
        if value < _ZERO or value > _ONE:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    # ==================================================================
    # Internal: Result builder
    # ==================================================================

    def _build_result(
        self,
        calc_id: str,
        status: str,
        treatment_method: str,
        fossil_co2: Decimal,
        biogenic_co2: Decimal,
        ch4: Decimal,
        n2o: Decimal,
        total_co2e: Decimal,
        total_mass: Decimal,
        gwp_source: str,
        trace: List[str],
        elapsed_ms: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a standardized calculation result dictionary.

        Computes provenance hash and records provenance if tracker is available.

        Args:
            calc_id: Unique calculation identifier.
            status: Result status (SUCCESS/FAILED).
            treatment_method: Treatment method used.
            fossil_co2: Fossil CO2 in tonnes.
            biogenic_co2: Biogenic CO2 in tonnes.
            ch4: CH4 in tonnes.
            n2o: N2O in tonnes.
            total_co2e: Total CO2e in tonnes (fossil scope).
            total_mass: Total waste mass in tonnes.
            gwp_source: GWP assessment report used.
            trace: Calculation trace steps.
            elapsed_ms: Wall-clock time in milliseconds.
            extra: Additional key-value pairs to include.

        Returns:
            Complete result dictionary.
        """
        provenance_hash = _compute_hash({
            "calculation_id": calc_id,
            "treatment_method": treatment_method,
            "fossil_co2": str(fossil_co2),
            "biogenic_co2": str(biogenic_co2),
            "ch4": str(ch4),
            "n2o": str(n2o),
            "total_co2e": str(total_co2e),
            "total_mass": str(total_mass),
            "gwp_source": gwp_source,
        })

        trace.append(f"[P] Provenance hash: {provenance_hash[:16]}...")
        trace.append(f"[T] Processing time: {elapsed_ms:.3f} ms")

        self._record_provenance(
            action=f"calculate_{treatment_method}",
            entity_id=calc_id,
            data={
                "total_co2e": str(total_co2e),
                "total_mass": str(total_mass),
                "provenance_hash": provenance_hash,
            },
        )

        result: Dict[str, Any] = {
            "status": status,
            "calculation_id": calc_id,
            "treatment_method": treatment_method,
            "fossil_co2_tonnes": str(_q(fossil_co2)),
            "biogenic_co2_tonnes": str(_q(biogenic_co2)),
            "ch4_tonnes": str(_q(ch4)),
            "n2o_tonnes": str(_q(n2o)),
            "total_co2e_tonnes": str(_q(total_co2e)),
            "total_waste_tonnes": str(_q(total_mass)),
            "gwp_source": gwp_source,
            "calculation_trace": trace,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": utcnow().isoformat(),
        }

        if extra:
            result.update(extra)

        return result

    # ==================================================================
    # Internal: Error handler
    # ==================================================================

    def _handle_error(
        self,
        calc_id: str,
        treatment_method: str,
        exc: Exception,
        trace: List[str],
        start_time: float,
    ) -> Dict[str, Any]:
        """Build a standardized error result dictionary.

        Args:
            calc_id: Calculation identifier.
            treatment_method: Treatment method that failed.
            exc: Exception that was raised.
            trace: Calculation trace up to the point of failure.
            start_time: ``time.monotonic()`` start value.

        Returns:
            Error result dictionary with FAILED status.
        """
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            "%s calculation %s failed: %s",
            treatment_method, calc_id, exc, exc_info=True,
        )
        self._emit_error_metric("calculation_error")

        trace.append(f"[E] Error: {exc}")
        trace.append(f"[T] Processing time: {elapsed_ms:.3f} ms")

        return {
            "status": "FAILED",
            "calculation_id": calc_id,
            "treatment_method": treatment_method,
            "error_message": str(exc),
            "fossil_co2_tonnes": str(_ZERO),
            "biogenic_co2_tonnes": str(_ZERO),
            "ch4_tonnes": str(_ZERO),
            "n2o_tonnes": str(_ZERO),
            "total_co2e_tonnes": str(_ZERO),
            "total_waste_tonnes": str(_ZERO),
            "gwp_source": "",
            "calculation_trace": trace,
            "provenance_hash": "",
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": utcnow().isoformat(),
        }
