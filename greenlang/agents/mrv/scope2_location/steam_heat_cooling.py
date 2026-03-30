# -*- coding: utf-8 -*-
"""
SteamHeatCoolingEngine - Engine 3: Scope 2 Location-Based Emissions Agent
(AGENT-MRV-009)

Calculates Scope 2 emissions from purchased steam, district heating, and
cooling using deterministic Decimal arithmetic with full calculation trace
and SHA-256 provenance hashing.

Core Formulas:
    Steam:   Emissions (kgCO2e) = Consumption (GJ) x Steam EF (kgCO2e/GJ)
    Heating: Emissions (kgCO2e) = Consumption (GJ) x Heating EF (kgCO2e/GJ)
    Cooling: Emissions (kgCO2e) = Consumption (GJ) x Cooling EF (kgCO2e/GJ)

Electric heating/cooling adjusts for COP:
    Electrical Input (GJ) = Thermal Output (GJ) / COP
    Electrical Input (MWh) = Electrical Input (GJ) x 0.277778
    Emissions (kgCO2e) = Electrical Input (MWh) x Grid EF (kgCO2e/MWh)

CHP allocation distributes combined fuel emissions between heat and power
outputs using efficiency, energy, or exergy methods per GHG Protocol.

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places)
    - No LLM calls in the calculation path
    - Every step is recorded in the calculation trace
    - SHA-256 provenance hash for every result
    - Identical inputs always produce identical outputs

Thread Safety:
    Stateless per-calculation. Mutable counters protected by reentrant lock.

Example:
    >>> from greenlang.agents.mrv.scope2_location.steam_heat_cooling import SteamHeatCoolingEngine
    >>> from decimal import Decimal
    >>> engine = SteamHeatCoolingEngine()
    >>> result = engine.calculate_steam_emissions(
    ...     consumption_gj=Decimal("500"),
    ...     steam_type="natural_gas",
    ... )
    >>> assert result["status"] == "SUCCESS"
    >>> assert Decimal(result["total_co2e_kg"]) > 0

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-009 Scope 2 Location-Based Emissions (GL-MRV-SCOPE2-001)
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
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["SteamHeatCoolingEngine"]

# ---------------------------------------------------------------------------
# Conditional imports -- metrics & provenance
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.scope2_location.metrics import (
        Scope2LocationMetrics,
        get_metrics as _get_metrics,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _get_metrics = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Deterministic SHA-256 hash helper
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

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
# Decimal precision constants
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_PRECISION_3 = Decimal("0.001")    # 3 decimal places for tCO2e output
_ZERO = Decimal("0")
_ONE = Decimal("1")
_THOUSAND = Decimal("1000")

# ---------------------------------------------------------------------------
# Unit conversion constants (all Decimal, zero-hallucination)
# ---------------------------------------------------------------------------

_MMBTU_TO_GJ = Decimal("1.05506")
_THERM_TO_GJ = Decimal("0.105506")
_KWH_TO_GJ = Decimal("0.0036")
_MWH_TO_GJ = Decimal("3.6")
_GJ_TO_MWH = Decimal("0.277778")  # 1 GJ = 0.277778 MWh
_KG_TO_TONNES = Decimal("0.001")
_TJ_TO_GJ = Decimal("1000")
_BTU_TO_GJ = Decimal("0.0000010551")
_KBTU_TO_GJ = Decimal("0.0010551")
_TON_REFRIGERATION_HR_TO_GJ = Decimal("0.01267")  # 1 ton-hr = 12,670 BTU

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

def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to the standard 8-decimal-place precision.

    Args:
        value: Decimal value to quantize.

    Returns:
        Quantized Decimal with ROUND_HALF_UP.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)

def _quantize_3(value: Decimal) -> Decimal:
    """Quantize a Decimal to 3 decimal places for tCO2e output.

    Args:
        value: Decimal value to quantize.

    Returns:
        Quantized Decimal with ROUND_HALF_UP to 3 decimal places.
    """
    return value.quantize(_PRECISION_3, rounding=ROUND_HALF_UP)

# ===========================================================================
# Enumerations
# ===========================================================================

class SteamType(str, Enum):
    """Supported steam fuel types for emission factor lookup."""

    NATURAL_GAS = "natural_gas"
    COAL = "coal"
    BIOMASS = "biomass"
    OIL = "oil"
    MIXED = "mixed"

class HeatingType(str, Enum):
    """Supported heating system types for emission factor lookup."""

    DISTRICT = "district"
    GAS_BOILER = "gas_boiler"
    ELECTRIC = "electric"
    HEAT_PUMP = "heat_pump"
    BIOMASS = "biomass"

class CoolingType(str, Enum):
    """Supported cooling system types for emission factor lookup."""

    ELECTRIC_CHILLER = "electric_chiller"
    ABSORPTION = "absorption"
    DISTRICT = "district"
    FREE_COOLING = "free_cooling"

class EnergyType(str, Enum):
    """Broad energy type categories."""

    STEAM = "steam"
    HEATING = "heating"
    COOLING = "cooling"

class CHPMethod(str, Enum):
    """CHP emission allocation methods per GHG Protocol."""

    EFFICIENCY_METHOD = "efficiency_method"
    ENERGY_METHOD = "energy_method"
    EXERGY_METHOD = "exergy_method"

class CalculationStatus(str, Enum):
    """Calculation result status codes."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"

# ===========================================================================
# Default Emission Factor Tables (kgCO2e per GJ, all Decimal)
# ===========================================================================

# Steam emission factors by fuel type (kgCO2e/GJ)
_DEFAULT_STEAM_EFS: Dict[str, Decimal] = {
    "natural_gas": Decimal("56.10"),
    "coal":        Decimal("94.60"),
    "biomass":     Decimal("0.00"),   # biogenic -- reported separately
    "oil":         Decimal("73.30"),
    "mixed":       Decimal("64.20"),  # average grid steam
}

# Heating emission factors by type (kgCO2e/GJ)
_DEFAULT_HEATING_EFS: Dict[str, Decimal] = {
    "district":    Decimal("43.50"),  # district heating network
    "gas_boiler":  Decimal("56.10"),
    "electric":    Decimal("0.00"),   # placeholder -- uses grid EF / COP
    "heat_pump":   Decimal("18.50"),  # average
    "biomass":     Decimal("0.00"),   # biogenic
}

# Cooling emission factors by type (kgCO2e/GJ)
_DEFAULT_COOLING_EFS: Dict[str, Decimal] = {
    "electric_chiller": Decimal("0.00"),   # placeholder -- uses grid EF / COP
    "absorption":       Decimal("32.10"),  # gas-fired absorption
    "district":         Decimal("28.50"),  # district cooling
    "free_cooling":     Decimal("0.00"),
}

# Default grid emission factor when no country_code or grid_factor_db
# provided (kgCO2e/MWh) -- world average approximation
_DEFAULT_GRID_EF_KG_PER_MWH = Decimal("442.00")

# Default COP ranges for validation
_COP_RANGES: Dict[str, Tuple[Decimal, Decimal]] = {
    "electric_chiller": (Decimal("2.0"), Decimal("10.0")),
    "absorption":       (Decimal("0.5"), Decimal("2.0")),
    "heat_pump":        (Decimal("1.5"), Decimal("8.0")),
    "electric":         (Decimal("0.9"), Decimal("1.1")),
    "direct_electric":  (Decimal("0.9"), Decimal("1.1")),
}

# Default boiler efficiency ranges for validation
_BOILER_EFFICIENCY_RANGE = (Decimal("0.50"), Decimal("1.00"))

# Exergy quality factor for heat (Carnot-based approximation)
# Steam at ~150C, ambient ~20C: exergy factor ~ 0.30
_DEFAULT_HEAT_EXERGY_FACTOR = Decimal("0.30")
# Electricity exergy factor is 1.0 (pure exergy)
_ELECTRICITY_EXERGY_FACTOR = Decimal("1.00")

# ===========================================================================
# SteamHeatCoolingEngine
# ===========================================================================

class SteamHeatCoolingEngine:
    """Engine 3: Calculates Scope 2 emissions from purchased steam,
    district heating, and cooling.

    Implements GHG Protocol Scope 2 Guidance for non-electricity purchased
    energy using deterministic Decimal arithmetic throughout. Supports
    direct emission factor application, electric heating/cooling with COP,
    CHP allocation, and district energy calculations.

    All numeric lookups use built-in default emission factor tables or
    delegate to an optional grid_factor_db for electricity-dependent
    calculations. Every result includes a complete calculation trace and
    SHA-256 provenance hash.

    Attributes:
        _grid_factor_db: Optional external grid factor database for
            electricity EF lookup by country/region.
        _config: Optional configuration dictionary.
        _metrics: Optional Scope2LocationMetrics instance.
        _provenance: Optional provenance tracker instance.

    Example:
        >>> engine = SteamHeatCoolingEngine()
        >>> result = engine.calculate_steam_emissions(
        ...     consumption_gj=Decimal("500"),
        ...     steam_type="natural_gas",
        ... )
        >>> assert result["status"] == "SUCCESS"
        >>> assert Decimal(result["total_co2e_tonnes"]) > 0
    """

    def __init__(
        self,
        grid_factor_db: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Any] = None,
        provenance: Optional[Any] = None,
    ) -> None:
        """Initialize the SteamHeatCoolingEngine.

        Args:
            grid_factor_db: Optional database or service for looking up
                grid electricity emission factors by country/region. Must
                implement ``get_factor(country_code: str) -> Decimal``.
                If None, the engine uses _DEFAULT_GRID_EF_KG_PER_MWH.
            config: Optional configuration dictionary. Supports:
                - decimal_precision (int): Decimal places (default 8).
                - default_grid_ef (str): Default grid EF in kgCO2e/MWh.
                - enable_provenance (bool): Enable provenance tracking.
                - enable_biogenic_tracking (bool): Track biogenic CO2.
            metrics: Optional Scope2LocationMetrics instance. If None,
                attempts to load the module-level singleton.
            provenance: Optional provenance tracker instance.
        """
        self._grid_factor_db = grid_factor_db
        self._config = config or {}
        self._lock = threading.RLock()

        # Configuration
        self._precision_places: int = self._config.get("decimal_precision", 8)
        self._precision_quantizer = Decimal(10) ** -self._precision_places
        self._default_grid_ef: Decimal = _D(
            self._config.get("default_grid_ef", str(_DEFAULT_GRID_EF_KG_PER_MWH))
        )
        self._enable_provenance: bool = self._config.get("enable_provenance", True)
        self._enable_biogenic: bool = self._config.get(
            "enable_biogenic_tracking", True
        )

        # Metrics
        if metrics is not None:
            self._metrics = metrics
        elif _METRICS_AVAILABLE and _get_metrics is not None:
            try:
                self._metrics = _get_metrics()
            except Exception:
                self._metrics = None
        else:
            self._metrics = None

        # Provenance
        self._provenance = provenance

        # Custom EF overrides (populated via config if provided)
        self._custom_steam_efs: Dict[str, Decimal] = {}
        self._custom_heating_efs: Dict[str, Decimal] = {}
        self._custom_cooling_efs: Dict[str, Decimal] = {}
        if "custom_steam_efs" in self._config:
            for k, v in self._config["custom_steam_efs"].items():
                self._custom_steam_efs[k] = _D(v)
        if "custom_heating_efs" in self._config:
            for k, v in self._config["custom_heating_efs"].items():
                self._custom_heating_efs[k] = _D(v)
        if "custom_cooling_efs" in self._config:
            for k, v in self._config["custom_cooling_efs"].items():
                self._custom_cooling_efs[k] = _D(v)

        # Statistics counters
        self._total_calculations: int = 0
        self._total_steam: int = 0
        self._total_heating: int = 0
        self._total_cooling: int = 0
        self._total_chp: int = 0
        self._total_batches: int = 0
        self._total_errors: int = 0
        self._total_co2e_kg_cumulative: Decimal = _ZERO
        self._total_biogenic_kg_cumulative: Decimal = _ZERO

        logger.info(
            "SteamHeatCoolingEngine initialized "
            "(precision=%d, grid_ef=%s kgCO2e/MWh, provenance=%s, "
            "grid_db=%s, metrics=%s)",
            self._precision_places,
            self._default_grid_ef,
            self._enable_provenance,
            "connected" if self._grid_factor_db else "default",
            "active" if self._metrics else "disabled",
        )

    # ===================================================================
    # UNIT CONVERSIONS (Public API, methods 14-17)
    # ===================================================================

    def mmbtu_to_gj(self, mmbtu: Decimal) -> Decimal:
        """Convert MMBtu to GJ.

        1 MMBtu = 1.05506 GJ (exact conversion per ISO 80000-5).

        Args:
            mmbtu: Energy quantity in MMBtu. Must be >= 0.

        Returns:
            Energy quantity in GJ, quantized to 8 decimal places.

        Raises:
            ValueError: If mmbtu is negative or not convertible.
        """
        mmbtu = _D(mmbtu)
        if mmbtu < _ZERO:
            raise ValueError(f"MMBtu must be >= 0, got {mmbtu}")
        return _quantize(mmbtu * _MMBTU_TO_GJ)

    def therms_to_gj(self, therms: Decimal) -> Decimal:
        """Convert therms to GJ.

        1 therm = 0.105506 GJ (= 100,000 BTU).

        Args:
            therms: Energy quantity in therms. Must be >= 0.

        Returns:
            Energy quantity in GJ, quantized to 8 decimal places.

        Raises:
            ValueError: If therms is negative or not convertible.
        """
        therms = _D(therms)
        if therms < _ZERO:
            raise ValueError(f"Therms must be >= 0, got {therms}")
        return _quantize(therms * _THERM_TO_GJ)

    def kwh_to_gj(self, kwh: Decimal) -> Decimal:
        """Convert kWh to GJ.

        1 kWh = 0.0036 GJ (exact).

        Args:
            kwh: Energy quantity in kWh. Must be >= 0.

        Returns:
            Energy quantity in GJ, quantized to 8 decimal places.

        Raises:
            ValueError: If kwh is negative or not convertible.
        """
        kwh = _D(kwh)
        if kwh < _ZERO:
            raise ValueError(f"kWh must be >= 0, got {kwh}")
        return _quantize(kwh * _KWH_TO_GJ)

    def normalize_to_gj(self, quantity: Decimal, unit: str) -> Decimal:
        """Normalize any supported energy unit to GJ.

        Supported units: GJ, MWh, kWh, MMBtu, therm, TJ, kBtu, Btu,
        ton_refrigeration_hr.

        Args:
            quantity: Energy quantity in the specified unit. Must be >= 0.
            unit: Unit string (case-insensitive).

        Returns:
            Energy quantity in GJ, quantized to 8 decimal places.

        Raises:
            ValueError: If unit is unsupported or quantity is negative.
        """
        quantity = _D(quantity)
        if quantity < _ZERO:
            raise ValueError(f"Quantity must be >= 0, got {quantity}")

        unit_lower = unit.strip().lower()
        conversion_map: Dict[str, Decimal] = {
            "gj": _ONE,
            "mwh": _MWH_TO_GJ,
            "kwh": _KWH_TO_GJ,
            "mmbtu": _MMBTU_TO_GJ,
            "therm": _THERM_TO_GJ,
            "therms": _THERM_TO_GJ,
            "tj": _TJ_TO_GJ,
            "btu": _BTU_TO_GJ,
            "kbtu": _KBTU_TO_GJ,
            "ton_refrigeration_hr": _TON_REFRIGERATION_HR_TO_GJ,
        }

        factor = conversion_map.get(unit_lower)
        if factor is None:
            raise ValueError(
                f"Unsupported energy unit: {unit!r}. "
                f"Supported units: {sorted(conversion_map.keys())}"
            )

        return _quantize(quantity * factor)

    # ===================================================================
    # EMISSION FACTOR MANAGEMENT (Public API, methods 22-25)
    # ===================================================================

    def get_steam_ef(self, steam_type: str) -> Decimal:
        """Get emission factor for a steam fuel type.

        Checks custom overrides first, then falls back to built-in defaults.

        Args:
            steam_type: Steam fuel type (natural_gas, coal, biomass, oil,
                mixed).

        Returns:
            Emission factor in kgCO2e/GJ.

        Raises:
            ValueError: If steam_type is not recognized.
        """
        key = steam_type.strip().lower()
        if key in self._custom_steam_efs:
            return self._custom_steam_efs[key]
        if key in _DEFAULT_STEAM_EFS:
            return _DEFAULT_STEAM_EFS[key]
        raise ValueError(
            f"Unknown steam type: {steam_type!r}. "
            f"Valid types: {sorted(_DEFAULT_STEAM_EFS.keys())}"
        )

    def get_heating_ef(self, heating_type: str) -> Decimal:
        """Get emission factor for a heating system type.

        Checks custom overrides first, then falls back to built-in defaults.

        Args:
            heating_type: Heating system type (district, gas_boiler,
                electric, heat_pump, biomass).

        Returns:
            Emission factor in kgCO2e/GJ.

        Raises:
            ValueError: If heating_type is not recognized.
        """
        key = heating_type.strip().lower()
        if key in self._custom_heating_efs:
            return self._custom_heating_efs[key]
        if key in _DEFAULT_HEATING_EFS:
            return _DEFAULT_HEATING_EFS[key]
        raise ValueError(
            f"Unknown heating type: {heating_type!r}. "
            f"Valid types: {sorted(_DEFAULT_HEATING_EFS.keys())}"
        )

    def get_cooling_ef(self, cooling_type: str) -> Decimal:
        """Get emission factor for a cooling system type.

        Checks custom overrides first, then falls back to built-in defaults.

        Args:
            cooling_type: Cooling system type (electric_chiller, absorption,
                district, free_cooling).

        Returns:
            Emission factor in kgCO2e/GJ.

        Raises:
            ValueError: If cooling_type is not recognized.
        """
        key = cooling_type.strip().lower()
        if key in self._custom_cooling_efs:
            return self._custom_cooling_efs[key]
        if key in _DEFAULT_COOLING_EFS:
            return _DEFAULT_COOLING_EFS[key]
        raise ValueError(
            f"Unknown cooling type: {cooling_type!r}. "
            f"Valid types: {sorted(_DEFAULT_COOLING_EFS.keys())}"
        )

    def list_available_efs(self) -> Dict[str, Dict[str, str]]:
        """List all available emission factors across steam, heating, and
        cooling categories.

        Returns:
            Dictionary with keys 'steam', 'heating', 'cooling', each
            mapping type names to their EF values as strings.
        """
        result: Dict[str, Dict[str, str]] = {
            "steam": {},
            "heating": {},
            "cooling": {},
        }

        # Steam: merge defaults with custom overrides
        for key, val in _DEFAULT_STEAM_EFS.items():
            ef = self._custom_steam_efs.get(key, val)
            result["steam"][key] = str(ef)
        for key, val in self._custom_steam_efs.items():
            if key not in result["steam"]:
                result["steam"][key] = str(val)

        # Heating: merge defaults with custom overrides
        for key, val in _DEFAULT_HEATING_EFS.items():
            ef = self._custom_heating_efs.get(key, val)
            result["heating"][key] = str(ef)
        for key, val in self._custom_heating_efs.items():
            if key not in result["heating"]:
                result["heating"][key] = str(val)

        # Cooling: merge defaults with custom overrides
        for key, val in _DEFAULT_COOLING_EFS.items():
            ef = self._custom_cooling_efs.get(key, val)
            result["cooling"][key] = str(ef)
        for key, val in self._custom_cooling_efs.items():
            if key not in result["cooling"]:
                result["cooling"][key] = str(val)

        return result

    # ===================================================================
    # VALIDATION (Public API, methods 18-21)
    # ===================================================================

    def validate_steam_input(
        self,
        consumption_gj: Decimal,
        steam_type: str,
    ) -> List[str]:
        """Validate steam calculation inputs.

        Args:
            consumption_gj: Steam consumption in GJ.
            steam_type: Steam fuel type string.

        Returns:
            List of validation error strings. Empty list if valid.
        """
        errors: List[str] = []
        try:
            val = _D(consumption_gj)
            if val < _ZERO:
                errors.append("consumption_gj must be >= 0")
            if val > Decimal("100000000"):
                errors.append(
                    "consumption_gj exceeds maximum (100,000,000 GJ)"
                )
        except (ValueError, InvalidOperation):
            errors.append(f"consumption_gj is not a valid number: {consumption_gj!r}")

        key = steam_type.strip().lower() if steam_type else ""
        valid_types = set(_DEFAULT_STEAM_EFS.keys()) | set(
            self._custom_steam_efs.keys()
        )
        if key not in valid_types:
            errors.append(
                f"Unknown steam_type: {steam_type!r}. "
                f"Valid: {sorted(valid_types)}"
            )

        return errors

    def validate_heating_input(
        self,
        consumption_gj: Decimal,
        heating_type: str,
    ) -> List[str]:
        """Validate heating calculation inputs.

        Args:
            consumption_gj: Heating consumption in GJ.
            heating_type: Heating system type string.

        Returns:
            List of validation error strings. Empty list if valid.
        """
        errors: List[str] = []
        try:
            val = _D(consumption_gj)
            if val < _ZERO:
                errors.append("consumption_gj must be >= 0")
            if val > Decimal("100000000"):
                errors.append(
                    "consumption_gj exceeds maximum (100,000,000 GJ)"
                )
        except (ValueError, InvalidOperation):
            errors.append(
                f"consumption_gj is not a valid number: {consumption_gj!r}"
            )

        key = heating_type.strip().lower() if heating_type else ""
        valid_types = set(_DEFAULT_HEATING_EFS.keys()) | set(
            self._custom_heating_efs.keys()
        )
        if key not in valid_types:
            errors.append(
                f"Unknown heating_type: {heating_type!r}. "
                f"Valid: {sorted(valid_types)}"
            )

        return errors

    def validate_cooling_input(
        self,
        consumption_gj: Decimal,
        cooling_type: str,
    ) -> List[str]:
        """Validate cooling calculation inputs.

        Args:
            consumption_gj: Cooling consumption in GJ.
            cooling_type: Cooling system type string.

        Returns:
            List of validation error strings. Empty list if valid.
        """
        errors: List[str] = []
        try:
            val = _D(consumption_gj)
            if val < _ZERO:
                errors.append("consumption_gj must be >= 0")
            if val > Decimal("100000000"):
                errors.append(
                    "consumption_gj exceeds maximum (100,000,000 GJ)"
                )
        except (ValueError, InvalidOperation):
            errors.append(
                f"consumption_gj is not a valid number: {consumption_gj!r}"
            )

        key = cooling_type.strip().lower() if cooling_type else ""
        valid_types = set(_DEFAULT_COOLING_EFS.keys()) | set(
            self._custom_cooling_efs.keys()
        )
        if key not in valid_types:
            errors.append(
                f"Unknown cooling_type: {cooling_type!r}. "
                f"Valid: {sorted(valid_types)}"
            )

        return errors

    def validate_cop(
        self,
        cop: Decimal,
        system_type: str,
    ) -> List[str]:
        """Validate COP (Coefficient of Performance) for a system type.

        COP ranges:
            - electric_chiller: 2.0 - 10.0
            - absorption: 0.5 - 2.0
            - heat_pump: 1.5 - 8.0
            - electric / direct_electric: 0.9 - 1.1

        Args:
            cop: Coefficient of Performance value.
            system_type: System type string for range lookup.

        Returns:
            List of validation error strings. Empty list if valid.
        """
        errors: List[str] = []
        try:
            cop_val = _D(cop)
        except (ValueError, InvalidOperation):
            errors.append(f"COP is not a valid number: {cop!r}")
            return errors

        if cop_val <= _ZERO:
            errors.append("COP must be > 0")
            return errors

        key = system_type.strip().lower() if system_type else ""
        cop_range = _COP_RANGES.get(key)
        if cop_range is not None:
            min_cop, max_cop = cop_range
            if cop_val < min_cop or cop_val > max_cop:
                errors.append(
                    f"COP {cop_val} outside expected range "
                    f"[{min_cop}, {max_cop}] for {system_type}"
                )

        return errors

    # ===================================================================
    # STEAM CALCULATIONS (Public API, methods 1-2)
    # ===================================================================

    def calculate_steam_emissions(
        self,
        consumption_gj: Decimal,
        steam_type: str = "natural_gas",
        custom_ef: Optional[Decimal] = None,
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate Scope 2 emissions from purchased steam.

        Formula:
            Emissions (kgCO2e) = Consumption (GJ) x EF (kgCO2e/GJ)

        If steam_type is 'biomass', emissions are recorded as biogenic
        and excluded from Scope 2 totals per GHG Protocol guidance.

        Args:
            consumption_gj: Steam consumption in GJ. Must be >= 0.
            steam_type: Steam fuel type (natural_gas, coal, biomass,
                oil, mixed). Defaults to 'natural_gas'.
            custom_ef: Optional custom emission factor (kgCO2e/GJ).
                Overrides the default for the given steam_type.
            country_code: Optional ISO 3166-1 country code. Currently
                reserved for future country-specific steam EFs.

        Returns:
            Dictionary with:
                - calculation_id: Unique calculation identifier.
                - status: SUCCESS or ERROR.
                - energy_type: 'steam'.
                - consumption_gj: Input consumption.
                - ef_applied: Emission factor used (kgCO2e/GJ).
                - total_co2e_kg: Total emissions in kgCO2e.
                - total_co2e_tonnes: Total emissions in tCO2e.
                - steam_type: Steam fuel type used.
                - is_biogenic: Whether emissions are biogenic.
                - biogenic_co2_kg: Biogenic CO2 (if applicable).
                - calculation_trace: Step-by-step trace.
                - provenance_hash: SHA-256 hash.
                - processing_time_ms: Calculation time.
        """
        t0 = time.monotonic()
        calc_id = f"shc_steam_{uuid4().hex[:12]}"
        trace: List[str] = []

        try:
            # Step 1: Validate inputs
            consumption_gj = _D(consumption_gj)
            validation_errors = self.validate_steam_input(
                consumption_gj, steam_type
            )
            if validation_errors:
                raise ValueError(
                    f"Steam input validation failed: {'; '.join(validation_errors)}"
                )
            trace.append(
                f"[1] Validated: consumption={consumption_gj} GJ, "
                f"type={steam_type}"
            )

            # Step 2: Resolve emission factor
            steam_key = steam_type.strip().lower()
            is_biogenic = steam_key == "biomass"

            if custom_ef is not None:
                ef = _D(custom_ef)
                ef_source = "custom"
                trace.append(f"[2] Custom EF: {ef} kgCO2e/GJ")
            else:
                ef = self.get_steam_ef(steam_type)
                ef_source = "default"
                trace.append(
                    f"[2] Default EF for {steam_type}: {ef} kgCO2e/GJ"
                )

            # Step 3: Calculate emissions
            total_co2e_kg = _quantize(consumption_gj * ef)
            total_co2e_tonnes = _quantize_3(total_co2e_kg * _KG_TO_TONNES)
            trace.append(
                f"[3] {consumption_gj} GJ x {ef} kgCO2e/GJ = "
                f"{total_co2e_kg} kgCO2e = {total_co2e_tonnes} tCO2e"
            )

            # Step 4: Handle biogenic
            biogenic_co2_kg = _ZERO
            biogenic_co2_tonnes = _ZERO
            if is_biogenic and self._enable_biogenic:
                biogenic_co2_kg = total_co2e_kg
                biogenic_co2_tonnes = total_co2e_tonnes
                total_co2e_kg = _ZERO
                total_co2e_tonnes = _ZERO
                trace.append(
                    f"[4] Biogenic steam: {biogenic_co2_kg} kgCO2 "
                    "excluded from Scope 2"
                )
            else:
                trace.append("[4] Non-biogenic; included in Scope 2 total")

            # Step 5: Provenance hash
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            result = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "energy_type": "steam",
                "consumption_gj": str(consumption_gj),
                "ef_applied": str(ef),
                "ef_source": ef_source,
                "ef_unit": "kgCO2e/GJ",
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "steam_type": steam_key,
                "is_biogenic": is_biogenic,
                "biogenic_co2_kg": str(biogenic_co2_kg),
                "biogenic_co2_tonnes": str(biogenic_co2_tonnes),
                "country_code": country_code,
                "calculation_trace": trace,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)
            trace.append(
                f"[5] Provenance: {result['provenance_hash'][:16]}..."
            )

            # Record statistics and metrics
            self._record_success("steam", total_co2e_kg, elapsed_ms)

            logger.info(
                "Steam calc %s: %s GJ %s -> %s tCO2e in %.1fms",
                calc_id, consumption_gj, steam_type,
                total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            return self._build_error_result(
                calc_id, "steam", str(exc), trace, t0
            )

    def calculate_steam_with_efficiency(
        self,
        consumption_gj: Decimal,
        boiler_efficiency: Decimal,
        steam_type: str = "natural_gas",
    ) -> Dict[str, Any]:
        """Calculate steam emissions adjusted for boiler efficiency.

        When the steam producer's boiler efficiency is known, the effective
        emission factor is adjusted upward:
            EF_adjusted = EF_default / boiler_efficiency

        This reflects the additional fuel consumed per GJ of steam delivered.

        Args:
            consumption_gj: Steam consumption in GJ. Must be >= 0.
            boiler_efficiency: Boiler thermal efficiency as a fraction
                (0.50 to 1.00). For example, 0.85 for an 85% efficient
                boiler.
            steam_type: Steam fuel type. Defaults to 'natural_gas'.

        Returns:
            Dictionary with standard steam result fields plus:
                - boiler_efficiency: Efficiency value used.
                - ef_unadjusted: Original EF before adjustment.
                - ef_adjusted: EF after efficiency adjustment.
        """
        t0 = time.monotonic()
        calc_id = f"shc_steam_eff_{uuid4().hex[:12]}"
        trace: List[str] = []

        try:
            # Step 1: Validate
            consumption_gj = _D(consumption_gj)
            boiler_efficiency = _D(boiler_efficiency)

            validation_errors = self.validate_steam_input(
                consumption_gj, steam_type
            )
            if boiler_efficiency < _BOILER_EFFICIENCY_RANGE[0]:
                validation_errors.append(
                    f"Boiler efficiency {boiler_efficiency} below minimum "
                    f"{_BOILER_EFFICIENCY_RANGE[0]}"
                )
            if boiler_efficiency > _BOILER_EFFICIENCY_RANGE[1]:
                validation_errors.append(
                    f"Boiler efficiency {boiler_efficiency} above maximum "
                    f"{_BOILER_EFFICIENCY_RANGE[1]}"
                )
            if validation_errors:
                raise ValueError(
                    f"Validation failed: {'; '.join(validation_errors)}"
                )
            trace.append(
                f"[1] Validated: consumption={consumption_gj} GJ, "
                f"efficiency={boiler_efficiency}, type={steam_type}"
            )

            # Step 2: Get base EF
            steam_key = steam_type.strip().lower()
            is_biogenic = steam_key == "biomass"
            ef_base = self.get_steam_ef(steam_type)
            trace.append(f"[2] Base EF for {steam_type}: {ef_base} kgCO2e/GJ")

            # Step 3: Adjust for efficiency
            ef_adjusted = _quantize(ef_base / boiler_efficiency)
            trace.append(
                f"[3] Adjusted EF: {ef_base} / {boiler_efficiency} = "
                f"{ef_adjusted} kgCO2e/GJ"
            )

            # Step 4: Calculate emissions
            total_co2e_kg = _quantize(consumption_gj * ef_adjusted)
            total_co2e_tonnes = _quantize_3(total_co2e_kg * _KG_TO_TONNES)
            trace.append(
                f"[4] {consumption_gj} GJ x {ef_adjusted} = "
                f"{total_co2e_kg} kgCO2e = {total_co2e_tonnes} tCO2e"
            )

            # Step 5: Biogenic handling
            biogenic_co2_kg = _ZERO
            biogenic_co2_tonnes = _ZERO
            if is_biogenic and self._enable_biogenic:
                biogenic_co2_kg = total_co2e_kg
                biogenic_co2_tonnes = total_co2e_tonnes
                total_co2e_kg = _ZERO
                total_co2e_tonnes = _ZERO
                trace.append("[5] Biogenic; excluded from Scope 2")
            else:
                trace.append("[5] Non-biogenic; included in Scope 2")

            # Step 6: Provenance
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            result = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "energy_type": "steam",
                "consumption_gj": str(consumption_gj),
                "boiler_efficiency": str(boiler_efficiency),
                "ef_unadjusted": str(ef_base),
                "ef_adjusted": str(ef_adjusted),
                "ef_applied": str(ef_adjusted),
                "ef_source": "efficiency_adjusted",
                "ef_unit": "kgCO2e/GJ",
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "steam_type": steam_key,
                "is_biogenic": is_biogenic,
                "biogenic_co2_kg": str(biogenic_co2_kg),
                "biogenic_co2_tonnes": str(biogenic_co2_tonnes),
                "calculation_trace": trace,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)
            trace.append(
                f"[6] Provenance: {result['provenance_hash'][:16]}..."
            )

            self._record_success("steam", total_co2e_kg, elapsed_ms)

            logger.info(
                "Steam+eff calc %s: %s GJ @ eff=%s -> %s tCO2e in %.1fms",
                calc_id, consumption_gj, boiler_efficiency,
                total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            return self._build_error_result(
                calc_id, "steam", str(exc), trace, t0
            )

    # ===================================================================
    # HEATING CALCULATIONS (Public API, methods 3-5)
    # ===================================================================

    def calculate_heating_emissions(
        self,
        consumption_gj: Decimal,
        heating_type: str = "district",
        custom_ef: Optional[Decimal] = None,
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate Scope 2 emissions from purchased heating.

        Formula:
            Emissions (kgCO2e) = Consumption (GJ) x Heating EF (kgCO2e/GJ)

        For electric heating, use calculate_electric_heating() instead.

        Args:
            consumption_gj: Heating consumption in GJ. Must be >= 0.
            heating_type: Heating system type (district, gas_boiler,
                electric, heat_pump, biomass). Defaults to 'district'.
            custom_ef: Optional custom EF (kgCO2e/GJ).
            country_code: Optional ISO country code for future use.

        Returns:
            Dictionary with standard result fields.
        """
        t0 = time.monotonic()
        calc_id = f"shc_heat_{uuid4().hex[:12]}"
        trace: List[str] = []

        try:
            consumption_gj = _D(consumption_gj)
            validation_errors = self.validate_heating_input(
                consumption_gj, heating_type
            )
            if validation_errors:
                raise ValueError(
                    f"Heating validation failed: {'; '.join(validation_errors)}"
                )
            trace.append(
                f"[1] Validated: consumption={consumption_gj} GJ, "
                f"type={heating_type}"
            )

            heating_key = heating_type.strip().lower()
            is_biogenic = heating_key == "biomass"

            if custom_ef is not None:
                ef = _D(custom_ef)
                ef_source = "custom"
                trace.append(f"[2] Custom EF: {ef} kgCO2e/GJ")
            else:
                ef = self.get_heating_ef(heating_type)
                ef_source = "default"
                trace.append(
                    f"[2] Default EF for {heating_type}: {ef} kgCO2e/GJ"
                )

            # For electric types with 0.00 EF, warn that dedicated method
            # should be used
            if heating_key == "electric" and ef == _ZERO and custom_ef is None:
                trace.append(
                    "[2a] WARNING: Electric heating EF is 0.00; "
                    "use calculate_electric_heating() for grid-based calc"
                )

            total_co2e_kg = _quantize(consumption_gj * ef)
            total_co2e_tonnes = _quantize_3(total_co2e_kg * _KG_TO_TONNES)
            trace.append(
                f"[3] {consumption_gj} GJ x {ef} = "
                f"{total_co2e_kg} kgCO2e = {total_co2e_tonnes} tCO2e"
            )

            biogenic_co2_kg = _ZERO
            biogenic_co2_tonnes = _ZERO
            if is_biogenic and self._enable_biogenic:
                biogenic_co2_kg = total_co2e_kg
                biogenic_co2_tonnes = total_co2e_tonnes
                total_co2e_kg = _ZERO
                total_co2e_tonnes = _ZERO
                trace.append("[4] Biogenic; excluded from Scope 2")
            else:
                trace.append("[4] Non-biogenic; included in Scope 2")

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            result = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "energy_type": "heating",
                "consumption_gj": str(consumption_gj),
                "ef_applied": str(ef),
                "ef_source": ef_source,
                "ef_unit": "kgCO2e/GJ",
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "heating_type": heating_key,
                "is_biogenic": is_biogenic,
                "biogenic_co2_kg": str(biogenic_co2_kg),
                "biogenic_co2_tonnes": str(biogenic_co2_tonnes),
                "country_code": country_code,
                "calculation_trace": trace,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)
            trace.append(
                f"[5] Provenance: {result['provenance_hash'][:16]}..."
            )

            self._record_success("heating", total_co2e_kg, elapsed_ms)

            logger.info(
                "Heating calc %s: %s GJ %s -> %s tCO2e in %.1fms",
                calc_id, consumption_gj, heating_type,
                total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            return self._build_error_result(
                calc_id, "heating", str(exc), trace, t0
            )

    def calculate_electric_heating(
        self,
        consumption_gj: Decimal,
        cop: Decimal = Decimal("1.0"),
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate Scope 2 emissions from electric heating.

        Electric heating converts heat output to electrical input via COP,
        then applies the grid electricity emission factor.

        Formula:
            Electrical Input (GJ) = Heat Output (GJ) / COP
            Electrical Input (MWh) = Electrical Input (GJ) x 0.277778
            Emissions (kgCO2e) = Electrical Input (MWh) x Grid EF (kgCO2e/MWh)

        Args:
            consumption_gj: Heat output consumed in GJ. Must be >= 0.
            cop: Coefficient of Performance. Direct electric heating
                has COP = 1.0 (default). Must be > 0.
            country_code: Optional ISO country code for grid EF lookup.

        Returns:
            Dictionary with standard result fields plus:
                - cop: COP value used.
                - electrical_input_gj: Electrical energy input in GJ.
                - electrical_input_mwh: Electrical energy input in MWh.
                - grid_ef_kg_per_mwh: Grid EF used.
        """
        t0 = time.monotonic()
        calc_id = f"shc_eheat_{uuid4().hex[:12]}"
        trace: List[str] = []

        try:
            consumption_gj = _D(consumption_gj)
            cop = _D(cop)

            if consumption_gj < _ZERO:
                raise ValueError(f"consumption_gj must be >= 0, got {consumption_gj}")
            if cop <= _ZERO:
                raise ValueError(f"COP must be > 0, got {cop}")

            cop_errors = self.validate_cop(cop, "electric")
            if cop_errors:
                trace.append(f"[0] COP warning: {'; '.join(cop_errors)}")

            trace.append(
                f"[1] Electric heating: {consumption_gj} GJ output, COP={cop}"
            )

            # Step 2: Convert to electrical input
            electrical_gj = _quantize(consumption_gj / cop)
            electrical_mwh = _quantize(electrical_gj * _GJ_TO_MWH)
            trace.append(
                f"[2] Electrical input: {consumption_gj}/{cop} = "
                f"{electrical_gj} GJ = {electrical_mwh} MWh"
            )

            # Step 3: Get grid EF
            grid_ef = self._get_grid_ef(country_code)
            trace.append(
                f"[3] Grid EF ({country_code or 'default'}): "
                f"{grid_ef} kgCO2e/MWh"
            )

            # Step 4: Calculate emissions
            total_co2e_kg = _quantize(electrical_mwh * grid_ef)
            total_co2e_tonnes = _quantize_3(total_co2e_kg * _KG_TO_TONNES)
            trace.append(
                f"[4] {electrical_mwh} MWh x {grid_ef} = "
                f"{total_co2e_kg} kgCO2e = {total_co2e_tonnes} tCO2e"
            )

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            result = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "energy_type": "heating",
                "heating_type": "electric",
                "consumption_gj": str(consumption_gj),
                "cop": str(cop),
                "electrical_input_gj": str(electrical_gj),
                "electrical_input_mwh": str(electrical_mwh),
                "grid_ef_kg_per_mwh": str(grid_ef),
                "ef_applied": str(grid_ef),
                "ef_source": "grid",
                "ef_unit": "kgCO2e/MWh",
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "is_biogenic": False,
                "biogenic_co2_kg": str(_ZERO),
                "biogenic_co2_tonnes": str(_ZERO),
                "country_code": country_code,
                "calculation_trace": trace,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)
            trace.append(
                f"[5] Provenance: {result['provenance_hash'][:16]}..."
            )

            self._record_success("heating", total_co2e_kg, elapsed_ms)

            logger.info(
                "Electric heating %s: %s GJ COP=%s -> %s tCO2e in %.1fms",
                calc_id, consumption_gj, cop, total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            return self._build_error_result(
                calc_id, "heating", str(exc), trace, t0
            )

    def calculate_heat_pump(
        self,
        consumption_gj: Decimal,
        cop: Decimal = Decimal("3.5"),
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate Scope 2 emissions from a heat pump.

        A heat pump delivers thermal energy greater than its electrical
        input by a factor of its COP. The electrical input is the
        relevant quantity for Scope 2 emissions.

        Formula:
            Electrical Input (GJ) = Heat Output (GJ) / COP
            Electrical Input (MWh) = Electrical Input (GJ) x 0.277778
            Emissions (kgCO2e) = Electrical Input (MWh) x Grid EF

        Args:
            consumption_gj: Heat output delivered in GJ. Must be >= 0.
            cop: Coefficient of Performance. Typical range 2.5-5.0.
                Defaults to 3.5.
            country_code: Optional ISO country code for grid EF lookup.

        Returns:
            Dictionary with standard result fields plus COP and
            electrical input details.
        """
        t0 = time.monotonic()
        calc_id = f"shc_hp_{uuid4().hex[:12]}"
        trace: List[str] = []

        try:
            consumption_gj = _D(consumption_gj)
            cop = _D(cop)

            if consumption_gj < _ZERO:
                raise ValueError(
                    f"consumption_gj must be >= 0, got {consumption_gj}"
                )
            if cop <= _ZERO:
                raise ValueError(f"COP must be > 0, got {cop}")

            cop_errors = self.validate_cop(cop, "heat_pump")
            if cop_errors:
                trace.append(f"[0] COP warning: {'; '.join(cop_errors)}")

            trace.append(
                f"[1] Heat pump: {consumption_gj} GJ output, COP={cop}"
            )

            # Electrical input = heat output / COP
            electrical_gj = _quantize(consumption_gj / cop)
            electrical_mwh = _quantize(electrical_gj * _GJ_TO_MWH)
            trace.append(
                f"[2] Electrical input: {consumption_gj}/{cop} = "
                f"{electrical_gj} GJ = {electrical_mwh} MWh"
            )

            grid_ef = self._get_grid_ef(country_code)
            trace.append(
                f"[3] Grid EF ({country_code or 'default'}): "
                f"{grid_ef} kgCO2e/MWh"
            )

            total_co2e_kg = _quantize(electrical_mwh * grid_ef)
            total_co2e_tonnes = _quantize_3(total_co2e_kg * _KG_TO_TONNES)
            trace.append(
                f"[4] {electrical_mwh} MWh x {grid_ef} = "
                f"{total_co2e_kg} kgCO2e = {total_co2e_tonnes} tCO2e"
            )

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            result = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "energy_type": "heating",
                "heating_type": "heat_pump",
                "consumption_gj": str(consumption_gj),
                "cop": str(cop),
                "electrical_input_gj": str(electrical_gj),
                "electrical_input_mwh": str(electrical_mwh),
                "grid_ef_kg_per_mwh": str(grid_ef),
                "ef_applied": str(grid_ef),
                "ef_source": "grid",
                "ef_unit": "kgCO2e/MWh",
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "is_biogenic": False,
                "biogenic_co2_kg": str(_ZERO),
                "biogenic_co2_tonnes": str(_ZERO),
                "country_code": country_code,
                "calculation_trace": trace,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)
            trace.append(
                f"[5] Provenance: {result['provenance_hash'][:16]}..."
            )

            self._record_success("heating", total_co2e_kg, elapsed_ms)

            logger.info(
                "Heat pump %s: %s GJ COP=%s -> %s tCO2e in %.1fms",
                calc_id, consumption_gj, cop, total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            return self._build_error_result(
                calc_id, "heating", str(exc), trace, t0
            )

    # ===================================================================
    # COOLING CALCULATIONS (Public API, methods 6-8)
    # ===================================================================

    def calculate_cooling_emissions(
        self,
        consumption_gj: Decimal,
        cooling_type: str = "electric_chiller",
        custom_ef: Optional[Decimal] = None,
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate Scope 2 emissions from purchased cooling.

        Formula:
            Emissions (kgCO2e) = Consumption (GJ) x Cooling EF (kgCO2e/GJ)

        For electric chillers, use calculate_electric_chiller() instead
        for COP-adjusted grid-based calculation.

        Args:
            consumption_gj: Cooling consumption in GJ. Must be >= 0.
            cooling_type: Cooling system type (electric_chiller, absorption,
                district, free_cooling). Defaults to 'electric_chiller'.
            custom_ef: Optional custom EF (kgCO2e/GJ).
            country_code: Optional ISO country code.

        Returns:
            Dictionary with standard result fields.
        """
        t0 = time.monotonic()
        calc_id = f"shc_cool_{uuid4().hex[:12]}"
        trace: List[str] = []

        try:
            consumption_gj = _D(consumption_gj)
            validation_errors = self.validate_cooling_input(
                consumption_gj, cooling_type
            )
            if validation_errors:
                raise ValueError(
                    f"Cooling validation failed: {'; '.join(validation_errors)}"
                )
            trace.append(
                f"[1] Validated: consumption={consumption_gj} GJ, "
                f"type={cooling_type}"
            )

            cooling_key = cooling_type.strip().lower()

            if custom_ef is not None:
                ef = _D(custom_ef)
                ef_source = "custom"
                trace.append(f"[2] Custom EF: {ef} kgCO2e/GJ")
            else:
                ef = self.get_cooling_ef(cooling_type)
                ef_source = "default"
                trace.append(
                    f"[2] Default EF for {cooling_type}: {ef} kgCO2e/GJ"
                )

            if cooling_key == "electric_chiller" and ef == _ZERO and custom_ef is None:
                trace.append(
                    "[2a] WARNING: Electric chiller EF is 0.00; "
                    "use calculate_electric_chiller() for grid-based calc"
                )

            total_co2e_kg = _quantize(consumption_gj * ef)
            total_co2e_tonnes = _quantize_3(total_co2e_kg * _KG_TO_TONNES)
            trace.append(
                f"[3] {consumption_gj} GJ x {ef} = "
                f"{total_co2e_kg} kgCO2e = {total_co2e_tonnes} tCO2e"
            )

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            result = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "energy_type": "cooling",
                "consumption_gj": str(consumption_gj),
                "ef_applied": str(ef),
                "ef_source": ef_source,
                "ef_unit": "kgCO2e/GJ",
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "cooling_type": cooling_key,
                "is_biogenic": False,
                "biogenic_co2_kg": str(_ZERO),
                "biogenic_co2_tonnes": str(_ZERO),
                "country_code": country_code,
                "calculation_trace": trace,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)
            trace.append(
                f"[4] Provenance: {result['provenance_hash'][:16]}..."
            )

            self._record_success("cooling", total_co2e_kg, elapsed_ms)

            logger.info(
                "Cooling calc %s: %s GJ %s -> %s tCO2e in %.1fms",
                calc_id, consumption_gj, cooling_type,
                total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            return self._build_error_result(
                calc_id, "cooling", str(exc), trace, t0
            )

    def calculate_electric_chiller(
        self,
        consumption_gj: Decimal,
        cop: Decimal = Decimal("5.0"),
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate Scope 2 emissions from an electric chiller.

        Electric chiller converts cooling output to electrical input via
        COP, then applies the grid electricity emission factor.

        Formula:
            Electrical Input (GJ) = Cooling Output (GJ) / COP
            Electrical Input (MWh) = Electrical Input (GJ) x 0.277778
            Emissions (kgCO2e) = Electrical Input (MWh) x Grid EF

        Args:
            consumption_gj: Cooling output delivered in GJ. Must be >= 0.
            cop: Coefficient of Performance. Typical range 4.0-6.0.
                Defaults to 5.0.
            country_code: Optional ISO country code for grid EF lookup.

        Returns:
            Dictionary with standard result fields plus COP and
            electrical input details.
        """
        t0 = time.monotonic()
        calc_id = f"shc_echiller_{uuid4().hex[:12]}"
        trace: List[str] = []

        try:
            consumption_gj = _D(consumption_gj)
            cop = _D(cop)

            if consumption_gj < _ZERO:
                raise ValueError(
                    f"consumption_gj must be >= 0, got {consumption_gj}"
                )
            if cop <= _ZERO:
                raise ValueError(f"COP must be > 0, got {cop}")

            cop_errors = self.validate_cop(cop, "electric_chiller")
            if cop_errors:
                trace.append(f"[0] COP warning: {'; '.join(cop_errors)}")

            trace.append(
                f"[1] Electric chiller: {consumption_gj} GJ output, COP={cop}"
            )

            electrical_gj = _quantize(consumption_gj / cop)
            electrical_mwh = _quantize(electrical_gj * _GJ_TO_MWH)
            trace.append(
                f"[2] Electrical input: {consumption_gj}/{cop} = "
                f"{electrical_gj} GJ = {electrical_mwh} MWh"
            )

            grid_ef = self._get_grid_ef(country_code)
            trace.append(
                f"[3] Grid EF ({country_code or 'default'}): "
                f"{grid_ef} kgCO2e/MWh"
            )

            total_co2e_kg = _quantize(electrical_mwh * grid_ef)
            total_co2e_tonnes = _quantize_3(total_co2e_kg * _KG_TO_TONNES)
            trace.append(
                f"[4] {electrical_mwh} MWh x {grid_ef} = "
                f"{total_co2e_kg} kgCO2e = {total_co2e_tonnes} tCO2e"
            )

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            result = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "energy_type": "cooling",
                "cooling_type": "electric_chiller",
                "consumption_gj": str(consumption_gj),
                "cop": str(cop),
                "electrical_input_gj": str(electrical_gj),
                "electrical_input_mwh": str(electrical_mwh),
                "grid_ef_kg_per_mwh": str(grid_ef),
                "ef_applied": str(grid_ef),
                "ef_source": "grid",
                "ef_unit": "kgCO2e/MWh",
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "is_biogenic": False,
                "biogenic_co2_kg": str(_ZERO),
                "biogenic_co2_tonnes": str(_ZERO),
                "country_code": country_code,
                "calculation_trace": trace,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)
            trace.append(
                f"[5] Provenance: {result['provenance_hash'][:16]}..."
            )

            self._record_success("cooling", total_co2e_kg, elapsed_ms)

            logger.info(
                "Electric chiller %s: %s GJ COP=%s -> %s tCO2e in %.1fms",
                calc_id, consumption_gj, cop, total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            return self._build_error_result(
                calc_id, "cooling", str(exc), trace, t0
            )

    def calculate_absorption_cooling(
        self,
        consumption_gj: Decimal,
        cop: Decimal = Decimal("1.2"),
    ) -> Dict[str, Any]:
        """Calculate Scope 2 emissions from gas-fired absorption cooling.

        Absorption chillers use heat (typically natural gas) to drive the
        cooling cycle. The COP adjusts the default absorption EF.

        Formula:
            Fuel Input (GJ) = Cooling Output (GJ) / COP
            Emissions (kgCO2e) = Fuel Input (GJ) x Absorption EF (kgCO2e/GJ)

        Args:
            consumption_gj: Cooling output delivered in GJ. Must be >= 0.
            cop: COP for absorption chiller. Typical range 0.7-1.4.
                Defaults to 1.2.

        Returns:
            Dictionary with standard result fields plus COP and fuel
            input details.
        """
        t0 = time.monotonic()
        calc_id = f"shc_absorb_{uuid4().hex[:12]}"
        trace: List[str] = []

        try:
            consumption_gj = _D(consumption_gj)
            cop = _D(cop)

            if consumption_gj < _ZERO:
                raise ValueError(
                    f"consumption_gj must be >= 0, got {consumption_gj}"
                )
            if cop <= _ZERO:
                raise ValueError(f"COP must be > 0, got {cop}")

            cop_errors = self.validate_cop(cop, "absorption")
            if cop_errors:
                trace.append(f"[0] COP warning: {'; '.join(cop_errors)}")

            trace.append(
                f"[1] Absorption cooling: {consumption_gj} GJ output, COP={cop}"
            )

            # Fuel input = cooling output / COP
            fuel_input_gj = _quantize(consumption_gj / cop)
            trace.append(
                f"[2] Fuel input: {consumption_gj}/{cop} = {fuel_input_gj} GJ"
            )

            # Apply absorption EF to fuel input
            absorption_ef = self.get_cooling_ef("absorption")
            total_co2e_kg = _quantize(fuel_input_gj * absorption_ef)
            total_co2e_tonnes = _quantize_3(total_co2e_kg * _KG_TO_TONNES)
            trace.append(
                f"[3] {fuel_input_gj} GJ x {absorption_ef} kgCO2e/GJ = "
                f"{total_co2e_kg} kgCO2e = {total_co2e_tonnes} tCO2e"
            )

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            result = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "energy_type": "cooling",
                "cooling_type": "absorption",
                "consumption_gj": str(consumption_gj),
                "cop": str(cop),
                "fuel_input_gj": str(fuel_input_gj),
                "ef_applied": str(absorption_ef),
                "ef_source": "default",
                "ef_unit": "kgCO2e/GJ",
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "is_biogenic": False,
                "biogenic_co2_kg": str(_ZERO),
                "biogenic_co2_tonnes": str(_ZERO),
                "calculation_trace": trace,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)
            trace.append(
                f"[4] Provenance: {result['provenance_hash'][:16]}..."
            )

            self._record_success("cooling", total_co2e_kg, elapsed_ms)

            logger.info(
                "Absorption cooling %s: %s GJ COP=%s -> %s tCO2e in %.1fms",
                calc_id, consumption_gj, cop, total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            return self._build_error_result(
                calc_id, "cooling", str(exc), trace, t0
            )

    # ===================================================================
    # CHP / DISTRICT ENERGY (Public API, methods 9-11)
    # ===================================================================

    def allocate_chp_emissions(
        self,
        total_fuel_gj: Decimal,
        heat_output_gj: Decimal,
        power_output_mwh: Decimal,
        method: str = "efficiency_method",
    ) -> Dict[str, Any]:
        """Allocate CHP (Combined Heat and Power) emissions between
        Scope 1 (power) and Scope 2 (heat) per GHG Protocol.

        Three allocation methods are supported:

        1. Efficiency Method:
            heat_share = (heat_output / heat_eff) /
                         (heat_output / heat_eff + power_output_gj / power_eff)
            Using default efficiencies: heat=0.80, power=0.35.

        2. Energy Method:
            heat_share = heat_output_gj /
                         (heat_output_gj + power_output_gj)

        3. Exergy Method:
            heat_exergy = heat_output_gj x heat_exergy_factor
            power_exergy = power_output_gj x 1.0
            heat_share = heat_exergy / (heat_exergy + power_exergy)

        Args:
            total_fuel_gj: Total fuel input to CHP in GJ. Must be > 0.
            heat_output_gj: Useful heat output in GJ. Must be >= 0.
            power_output_mwh: Electrical power output in MWh. Must be >= 0.
            method: Allocation method (efficiency_method, energy_method,
                exergy_method). Defaults to 'efficiency_method'.

        Returns:
            Dictionary with:
                - heat_allocation_fraction: Fraction allocated to heat.
                - power_allocation_fraction: Fraction allocated to power.
                - heat_fuel_gj: Fuel attributed to heat production.
                - power_fuel_gj: Fuel attributed to power production.
                - heat_emissions_kg: Scope 2 emissions from heat (kgCO2e).
                - power_emissions_kg: Scope 1 emissions from power (kgCO2e).
                - method: Allocation method used.
                - provenance_hash: SHA-256 hash.
        """
        t0 = time.monotonic()
        calc_id = f"shc_chp_{uuid4().hex[:12]}"
        trace: List[str] = []

        try:
            total_fuel_gj = _D(total_fuel_gj)
            heat_output_gj = _D(heat_output_gj)
            power_output_mwh = _D(power_output_mwh)

            if total_fuel_gj <= _ZERO:
                raise ValueError(
                    f"total_fuel_gj must be > 0, got {total_fuel_gj}"
                )
            if heat_output_gj < _ZERO:
                raise ValueError(
                    f"heat_output_gj must be >= 0, got {heat_output_gj}"
                )
            if power_output_mwh < _ZERO:
                raise ValueError(
                    f"power_output_mwh must be >= 0, got {power_output_mwh}"
                )

            # Convert power to GJ for comparison
            power_output_gj = _quantize(power_output_mwh * _MWH_TO_GJ)
            trace.append(
                f"[1] CHP: fuel={total_fuel_gj} GJ, "
                f"heat={heat_output_gj} GJ, "
                f"power={power_output_mwh} MWh ({power_output_gj} GJ)"
            )

            method_key = method.strip().lower()

            try:
                chp_method = CHPMethod(method_key)
            except ValueError:
                raise ValueError(
                    f"Unknown CHP method: {method!r}. "
                    f"Valid: {[m.value for m in CHPMethod]}"
                )

            # Calculate allocation fraction
            if chp_method == CHPMethod.EFFICIENCY_METHOD:
                heat_allocation = self._chp_efficiency_method(
                    heat_output_gj, power_output_gj, trace
                )
            elif chp_method == CHPMethod.ENERGY_METHOD:
                heat_allocation = self._chp_energy_method(
                    heat_output_gj, power_output_gj, trace
                )
            elif chp_method == CHPMethod.EXERGY_METHOD:
                heat_allocation = self._chp_exergy_method(
                    heat_output_gj, power_output_gj, trace
                )
            else:
                raise ValueError(f"Unimplemented CHP method: {chp_method}")

            power_allocation = _quantize(_ONE - heat_allocation)
            trace.append(
                f"[3] Allocation: heat={heat_allocation}, power={power_allocation}"
            )

            # Allocate fuel
            heat_fuel_gj = _quantize(total_fuel_gj * heat_allocation)
            power_fuel_gj = _quantize(total_fuel_gj * power_allocation)
            trace.append(
                f"[4] Fuel: heat={heat_fuel_gj} GJ, power={power_fuel_gj} GJ"
            )

            # Calculate emissions using mixed steam EF as proxy for CHP fuel
            chp_ef = self.get_steam_ef("mixed")
            heat_emissions_kg = _quantize(heat_fuel_gj * chp_ef)
            heat_emissions_tonnes = _quantize_3(
                heat_emissions_kg * _KG_TO_TONNES
            )
            power_emissions_kg = _quantize(power_fuel_gj * chp_ef)
            power_emissions_tonnes = _quantize_3(
                power_emissions_kg * _KG_TO_TONNES
            )
            trace.append(
                f"[5] Emissions (EF={chp_ef}): "
                f"heat={heat_emissions_kg} kg, power={power_emissions_kg} kg"
            )

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            result = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "energy_type": "chp",
                "method": chp_method.value,
                "total_fuel_gj": str(total_fuel_gj),
                "heat_output_gj": str(heat_output_gj),
                "power_output_mwh": str(power_output_mwh),
                "power_output_gj": str(power_output_gj),
                "heat_allocation_fraction": str(heat_allocation),
                "power_allocation_fraction": str(power_allocation),
                "heat_fuel_gj": str(heat_fuel_gj),
                "power_fuel_gj": str(power_fuel_gj),
                "chp_ef_applied": str(chp_ef),
                "heat_emissions_kg": str(heat_emissions_kg),
                "heat_emissions_tonnes": str(heat_emissions_tonnes),
                "power_emissions_kg": str(power_emissions_kg),
                "power_emissions_tonnes": str(power_emissions_tonnes),
                "scope2_heat_co2e_kg": str(heat_emissions_kg),
                "scope2_heat_co2e_tonnes": str(heat_emissions_tonnes),
                "scope1_power_co2e_kg": str(power_emissions_kg),
                "scope1_power_co2e_tonnes": str(power_emissions_tonnes),
                "calculation_trace": trace,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)
            trace.append(
                f"[6] Provenance: {result['provenance_hash'][:16]}..."
            )

            with self._lock:
                self._total_chp += 1
                self._total_calculations += 1

            logger.info(
                "CHP alloc %s [%s]: heat=%.1f%% power=%.1f%% in %.1fms",
                calc_id, chp_method.value,
                float(heat_allocation) * 100,
                float(power_allocation) * 100,
                elapsed_ms,
            )
            return result

        except Exception as exc:
            return self._build_error_result(
                calc_id, "chp", str(exc), trace, t0
            )

    def calculate_district_heating(
        self,
        consumption_gj: Decimal,
        district_ef: Optional[Decimal] = None,
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate Scope 2 emissions from district heating.

        Uses the district heating default EF (43.50 kgCO2e/GJ) or a
        custom district-specific EF.

        Args:
            consumption_gj: District heating consumed in GJ.
            district_ef: Optional district-specific EF (kgCO2e/GJ).
            country_code: Optional ISO country code.

        Returns:
            Dictionary with standard result fields.
        """
        return self.calculate_heating_emissions(
            consumption_gj=consumption_gj,
            heating_type="district",
            custom_ef=district_ef,
            country_code=country_code,
        )

    def calculate_district_cooling(
        self,
        consumption_gj: Decimal,
        district_ef: Optional[Decimal] = None,
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate Scope 2 emissions from district cooling.

        Uses the district cooling default EF (28.50 kgCO2e/GJ) or a
        custom district-specific EF.

        Args:
            consumption_gj: District cooling consumed in GJ.
            district_ef: Optional district-specific EF (kgCO2e/GJ).
            country_code: Optional ISO country code.

        Returns:
            Dictionary with standard result fields.
        """
        return self.calculate_cooling_emissions(
            consumption_gj=consumption_gj,
            cooling_type="district",
            custom_ef=district_ef,
            country_code=country_code,
        )

    # ===================================================================
    # COMBINED / BATCH CALCULATIONS (Public API, methods 12-13)
    # ===================================================================

    def calculate_all_non_electric(
        self,
        requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate steam + heating + cooling emissions together.

        Processes a list of non-electric energy consumption requests and
        aggregates totals by energy type.

        Each request dictionary must contain:
            - energy_type (str): 'steam', 'heating', or 'cooling'.
            - consumption_gj (str or Decimal): Consumption in GJ.
            - sub_type (str): Steam type, heating type, or cooling type.
            - custom_ef (str or Decimal, optional): Custom EF.
            - country_code (str, optional): ISO country code.

        Args:
            requests: List of request dictionaries.

        Returns:
            Dictionary with:
                - steam_total_co2e_kg: Aggregated steam emissions.
                - heating_total_co2e_kg: Aggregated heating emissions.
                - cooling_total_co2e_kg: Aggregated cooling emissions.
                - combined_co2e_kg: Sum of all categories.
                - combined_co2e_tonnes: Sum in tonnes.
                - results: List of individual calculation results.
                - success_count / error_count: Tallies.
                - provenance_hash: SHA-256 hash of combined result.
        """
        t0 = time.monotonic()
        batch_id = f"shc_combined_{uuid4().hex[:12]}"

        steam_total = _ZERO
        heating_total = _ZERO
        cooling_total = _ZERO
        biogenic_total = _ZERO
        results: List[Dict[str, Any]] = []
        success_count = 0
        error_count = 0

        for idx, req in enumerate(requests):
            energy_type = req.get("energy_type", "").strip().lower()
            consumption_gj = _D(req.get("consumption_gj", "0"))
            sub_type = req.get("sub_type", "")
            custom_ef_raw = req.get("custom_ef")
            custom_ef = _D(custom_ef_raw) if custom_ef_raw is not None else None
            country_code = req.get("country_code")

            if energy_type == "steam":
                result = self.calculate_steam_emissions(
                    consumption_gj=consumption_gj,
                    steam_type=sub_type or "natural_gas",
                    custom_ef=custom_ef,
                    country_code=country_code,
                )
            elif energy_type == "heating":
                result = self.calculate_heating_emissions(
                    consumption_gj=consumption_gj,
                    heating_type=sub_type or "district",
                    custom_ef=custom_ef,
                    country_code=country_code,
                )
            elif energy_type == "cooling":
                result = self.calculate_cooling_emissions(
                    consumption_gj=consumption_gj,
                    cooling_type=sub_type or "district",
                    custom_ef=custom_ef,
                    country_code=country_code,
                )
            else:
                result = {
                    "calculation_id": f"shc_err_{uuid4().hex[:12]}",
                    "status": CalculationStatus.ERROR.value,
                    "error": f"Unknown energy_type: {energy_type!r}",
                    "request_index": idx,
                }

            results.append(result)

            if result.get("status") == CalculationStatus.SUCCESS.value:
                success_count += 1
                co2e = _D(result.get("total_co2e_kg", "0"))
                bio = _D(result.get("biogenic_co2_kg", "0"))
                if energy_type == "steam":
                    steam_total += co2e
                elif energy_type == "heating":
                    heating_total += co2e
                elif energy_type == "cooling":
                    cooling_total += co2e
                biogenic_total += bio
            else:
                error_count += 1

        combined_co2e_kg = _quantize(steam_total + heating_total + cooling_total)
        combined_co2e_tonnes = _quantize_3(combined_co2e_kg * _KG_TO_TONNES)
        biogenic_total_tonnes = _quantize_3(biogenic_total * _KG_TO_TONNES)

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        combined_result = {
            "batch_id": batch_id,
            "status": (
                CalculationStatus.SUCCESS.value if error_count == 0
                else CalculationStatus.PARTIAL.value if success_count > 0
                else CalculationStatus.ERROR.value
            ),
            "request_count": len(requests),
            "success_count": success_count,
            "error_count": error_count,
            "steam_total_co2e_kg": str(_quantize(steam_total)),
            "steam_total_co2e_tonnes": str(
                _quantize_3(steam_total * _KG_TO_TONNES)
            ),
            "heating_total_co2e_kg": str(_quantize(heating_total)),
            "heating_total_co2e_tonnes": str(
                _quantize_3(heating_total * _KG_TO_TONNES)
            ),
            "cooling_total_co2e_kg": str(_quantize(cooling_total)),
            "cooling_total_co2e_tonnes": str(
                _quantize_3(cooling_total * _KG_TO_TONNES)
            ),
            "combined_co2e_kg": str(combined_co2e_kg),
            "combined_co2e_tonnes": str(combined_co2e_tonnes),
            "biogenic_total_co2_kg": str(_quantize(biogenic_total)),
            "biogenic_total_co2_tonnes": str(biogenic_total_tonnes),
            "results": results,
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": utcnow().isoformat(),
        }
        combined_result["provenance_hash"] = _compute_hash({
            "batch_id": batch_id,
            "combined_co2e_kg": str(combined_co2e_kg),
            "success_count": success_count,
            "error_count": error_count,
        })

        logger.info(
            "Combined calc %s: %d requests (%d ok, %d err), "
            "total=%s tCO2e in %.1fms",
            batch_id, len(requests), success_count, error_count,
            combined_co2e_tonnes, elapsed_ms,
        )

        return combined_result

    def calculate_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Batch-process multiple steam/heating/cooling calculations.

        Each request is a dictionary that must specify a ``calculation_type``
        field to route to the correct method:
            - 'steam': Routes to calculate_steam_emissions.
            - 'steam_efficiency': Routes to calculate_steam_with_efficiency.
            - 'heating': Routes to calculate_heating_emissions.
            - 'electric_heating': Routes to calculate_electric_heating.
            - 'heat_pump': Routes to calculate_heat_pump.
            - 'cooling': Routes to calculate_cooling_emissions.
            - 'electric_chiller': Routes to calculate_electric_chiller.
            - 'absorption': Routes to calculate_absorption_cooling.
            - 'district_heating': Routes to calculate_district_heating.
            - 'district_cooling': Routes to calculate_district_cooling.
            - 'chp': Routes to allocate_chp_emissions.

        Additional fields depend on the calculation_type. See individual
        method docstrings for details.

        Args:
            requests: List of request dictionaries.

        Returns:
            Dictionary with:
                - batch_id: Unique batch identifier.
                - status: SUCCESS, PARTIAL, or ERROR.
                - total_co2e_kg / total_co2e_tonnes: Aggregated emissions.
                - results: List of individual results.
                - success_count / error_count: Tallies.
                - provenance_hash: SHA-256 hash.
        """
        t0 = time.monotonic()
        batch_id = f"shc_batch_{uuid4().hex[:12]}"

        results: List[Dict[str, Any]] = []
        total_co2e_kg = _ZERO
        total_biogenic_kg = _ZERO
        success_count = 0
        error_count = 0

        dispatch: Dict[str, Any] = {
            "steam": self._dispatch_steam,
            "steam_efficiency": self._dispatch_steam_efficiency,
            "heating": self._dispatch_heating,
            "electric_heating": self._dispatch_electric_heating,
            "heat_pump": self._dispatch_heat_pump,
            "cooling": self._dispatch_cooling,
            "electric_chiller": self._dispatch_electric_chiller,
            "absorption": self._dispatch_absorption,
            "district_heating": self._dispatch_district_heating,
            "district_cooling": self._dispatch_district_cooling,
            "chp": self._dispatch_chp,
        }

        for idx, req in enumerate(requests):
            calc_type = req.get("calculation_type", "").strip().lower()
            handler = dispatch.get(calc_type)

            if handler is None:
                result: Dict[str, Any] = {
                    "calculation_id": f"shc_err_{uuid4().hex[:12]}",
                    "status": CalculationStatus.ERROR.value,
                    "error": (
                        f"Unknown calculation_type: {calc_type!r}. "
                        f"Valid: {sorted(dispatch.keys())}"
                    ),
                    "request_index": idx,
                }
            else:
                try:
                    result = handler(req)
                except Exception as exc:
                    result = {
                        "calculation_id": f"shc_err_{uuid4().hex[:12]}",
                        "status": CalculationStatus.ERROR.value,
                        "error": str(exc),
                        "request_index": idx,
                    }

            result["request_index"] = idx
            results.append(result)

            if result.get("status") == CalculationStatus.SUCCESS.value:
                success_count += 1
                co2e = _D(result.get("total_co2e_kg", "0"))
                bio = _D(result.get("biogenic_co2_kg", "0"))
                total_co2e_kg += co2e
                total_biogenic_kg += bio
                # For CHP, add heat-side emissions
                if calc_type == "chp":
                    total_co2e_kg += _D(
                        result.get("scope2_heat_co2e_kg", "0")
                    ) - co2e
            else:
                error_count += 1

        total_co2e_kg = _quantize(total_co2e_kg)
        total_co2e_tonnes = _quantize_3(total_co2e_kg * _KG_TO_TONNES)
        total_biogenic_tonnes = _quantize_3(total_biogenic_kg * _KG_TO_TONNES)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        batch_result = {
            "batch_id": batch_id,
            "status": (
                CalculationStatus.SUCCESS.value if error_count == 0
                else CalculationStatus.PARTIAL.value if success_count > 0
                else CalculationStatus.ERROR.value
            ),
            "request_count": len(requests),
            "success_count": success_count,
            "error_count": error_count,
            "total_co2e_kg": str(total_co2e_kg),
            "total_co2e_tonnes": str(total_co2e_tonnes),
            "total_biogenic_co2_kg": str(_quantize(total_biogenic_kg)),
            "total_biogenic_co2_tonnes": str(total_biogenic_tonnes),
            "results": results,
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": utcnow().isoformat(),
        }
        batch_result["provenance_hash"] = _compute_hash({
            "batch_id": batch_id,
            "total_co2e_kg": str(total_co2e_kg),
            "success_count": success_count,
            "error_count": error_count,
        })

        with self._lock:
            self._total_batches += 1

        logger.info(
            "Batch %s: %d requests (%d ok, %d err), "
            "total=%s tCO2e in %.1fms",
            batch_id, len(requests), success_count, error_count,
            total_co2e_tonnes, elapsed_ms,
        )

        return batch_result

    # ===================================================================
    # STATISTICS (Public API, method 26)
    # ===================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine statistics.

        Returns:
            Dictionary with cumulative counts and totals:
                - total_calculations: Total individual calculations.
                - total_steam: Steam calculations.
                - total_heating: Heating calculations.
                - total_cooling: Cooling calculations.
                - total_chp: CHP allocation calculations.
                - total_batches: Batch operations.
                - total_errors: Error count.
                - total_co2e_kg_cumulative: Cumulative emissions tracked.
                - total_biogenic_kg_cumulative: Cumulative biogenic CO2.
        """
        with self._lock:
            return {
                "total_calculations": self._total_calculations,
                "total_steam": self._total_steam,
                "total_heating": self._total_heating,
                "total_cooling": self._total_cooling,
                "total_chp": self._total_chp,
                "total_batches": self._total_batches,
                "total_errors": self._total_errors,
                "total_co2e_kg_cumulative": str(
                    self._total_co2e_kg_cumulative
                ),
                "total_biogenic_kg_cumulative": str(
                    self._total_biogenic_kg_cumulative
                ),
            }

    # ===================================================================
    # PRIVATE: Grid EF Lookup
    # ===================================================================

    def _get_grid_ef(self, country_code: Optional[str] = None) -> Decimal:
        """Look up the grid electricity emission factor.

        Delegates to grid_factor_db if available; otherwise returns the
        configured default.

        Args:
            country_code: Optional ISO 3166-1 country code.

        Returns:
            Grid emission factor in kgCO2e/MWh.
        """
        if country_code and self._grid_factor_db is not None:
            try:
                factor = self._grid_factor_db.get_factor(country_code)
                if factor is not None:
                    ef = _D(factor)
                    if self._metrics is not None:
                        try:
                            self._metrics.record_grid_factor_lookup("CUSTOM")
                        except Exception:
                            pass
                    return ef
            except Exception as exc:
                logger.warning(
                    "Grid factor lookup failed for %s: %s; using default",
                    country_code, exc,
                )

        return self._default_grid_ef

    # ===================================================================
    # PRIVATE: CHP Allocation Methods
    # ===================================================================

    def _chp_efficiency_method(
        self,
        heat_gj: Decimal,
        power_gj: Decimal,
        trace: List[str],
    ) -> Decimal:
        """CHP efficiency method allocation.

        Uses reference efficiencies:
            heat_efficiency = 0.80 (typical boiler)
            power_efficiency = 0.35 (typical grid power plant)

        heat_fuel_equivalent = heat_gj / heat_eff
        power_fuel_equivalent = power_gj / power_eff
        heat_share = heat_fuel_equivalent / (heat_fuel_equivalent + power_fuel_equivalent)

        Args:
            heat_gj: Useful heat output in GJ.
            power_gj: Electrical output in GJ.
            trace: Calculation trace list.

        Returns:
            Heat allocation fraction as Decimal [0, 1].
        """
        heat_eff = Decimal("0.80")
        power_eff = Decimal("0.35")

        heat_fuel_eq = _quantize(heat_gj / heat_eff)
        power_fuel_eq = _quantize(power_gj / power_eff)
        total_fuel_eq = heat_fuel_eq + power_fuel_eq

        if total_fuel_eq == _ZERO:
            trace.append("[2] Efficiency method: zero total output; 50/50 split")
            return Decimal("0.50000000")

        heat_share = _quantize(heat_fuel_eq / total_fuel_eq)
        trace.append(
            f"[2] Efficiency method: heat_fuel_eq={heat_fuel_eq}, "
            f"power_fuel_eq={power_fuel_eq}, heat_share={heat_share}"
        )
        return heat_share

    def _chp_energy_method(
        self,
        heat_gj: Decimal,
        power_gj: Decimal,
        trace: List[str],
    ) -> Decimal:
        """CHP energy method allocation.

        Proportional to energy content:
            heat_share = heat_gj / (heat_gj + power_gj)

        Args:
            heat_gj: Useful heat output in GJ.
            power_gj: Electrical output in GJ.
            trace: Calculation trace list.

        Returns:
            Heat allocation fraction as Decimal [0, 1].
        """
        total = heat_gj + power_gj
        if total == _ZERO:
            trace.append("[2] Energy method: zero total output; 50/50 split")
            return Decimal("0.50000000")

        heat_share = _quantize(heat_gj / total)
        trace.append(
            f"[2] Energy method: heat={heat_gj}, power={power_gj}, "
            f"heat_share={heat_share}"
        )
        return heat_share

    def _chp_exergy_method(
        self,
        heat_gj: Decimal,
        power_gj: Decimal,
        trace: List[str],
    ) -> Decimal:
        """CHP exergy method allocation.

        Based on thermodynamic quality:
            heat_exergy = heat_gj x heat_exergy_factor (default 0.30)
            power_exergy = power_gj x 1.0
            heat_share = heat_exergy / (heat_exergy + power_exergy)

        Args:
            heat_gj: Useful heat output in GJ.
            power_gj: Electrical output in GJ.
            trace: Calculation trace list.

        Returns:
            Heat allocation fraction as Decimal [0, 1].
        """
        heat_exergy = _quantize(heat_gj * _DEFAULT_HEAT_EXERGY_FACTOR)
        power_exergy = _quantize(power_gj * _ELECTRICITY_EXERGY_FACTOR)
        total_exergy = heat_exergy + power_exergy

        if total_exergy == _ZERO:
            trace.append("[2] Exergy method: zero total exergy; 50/50 split")
            return Decimal("0.50000000")

        heat_share = _quantize(heat_exergy / total_exergy)
        trace.append(
            f"[2] Exergy method: heat_exergy={heat_exergy} "
            f"(factor={_DEFAULT_HEAT_EXERGY_FACTOR}), "
            f"power_exergy={power_exergy}, heat_share={heat_share}"
        )
        return heat_share

    # ===================================================================
    # PRIVATE: Batch Dispatch Handlers
    # ===================================================================

    def _dispatch_steam(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return self.calculate_steam_emissions(
            consumption_gj=_D(req.get("consumption_gj", "0")),
            steam_type=req.get("steam_type", "natural_gas"),
            custom_ef=_D(req["custom_ef"]) if req.get("custom_ef") else None,
            country_code=req.get("country_code"),
        )

    def _dispatch_steam_efficiency(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return self.calculate_steam_with_efficiency(
            consumption_gj=_D(req.get("consumption_gj", "0")),
            boiler_efficiency=_D(req.get("boiler_efficiency", "0.85")),
            steam_type=req.get("steam_type", "natural_gas"),
        )

    def _dispatch_heating(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return self.calculate_heating_emissions(
            consumption_gj=_D(req.get("consumption_gj", "0")),
            heating_type=req.get("heating_type", "district"),
            custom_ef=_D(req["custom_ef"]) if req.get("custom_ef") else None,
            country_code=req.get("country_code"),
        )

    def _dispatch_electric_heating(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return self.calculate_electric_heating(
            consumption_gj=_D(req.get("consumption_gj", "0")),
            cop=_D(req.get("cop", "1.0")),
            country_code=req.get("country_code"),
        )

    def _dispatch_heat_pump(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return self.calculate_heat_pump(
            consumption_gj=_D(req.get("consumption_gj", "0")),
            cop=_D(req.get("cop", "3.5")),
            country_code=req.get("country_code"),
        )

    def _dispatch_cooling(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return self.calculate_cooling_emissions(
            consumption_gj=_D(req.get("consumption_gj", "0")),
            cooling_type=req.get("cooling_type", "electric_chiller"),
            custom_ef=_D(req["custom_ef"]) if req.get("custom_ef") else None,
            country_code=req.get("country_code"),
        )

    def _dispatch_electric_chiller(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return self.calculate_electric_chiller(
            consumption_gj=_D(req.get("consumption_gj", "0")),
            cop=_D(req.get("cop", "5.0")),
            country_code=req.get("country_code"),
        )

    def _dispatch_absorption(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return self.calculate_absorption_cooling(
            consumption_gj=_D(req.get("consumption_gj", "0")),
            cop=_D(req.get("cop", "1.2")),
        )

    def _dispatch_district_heating(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return self.calculate_district_heating(
            consumption_gj=_D(req.get("consumption_gj", "0")),
            district_ef=_D(req["district_ef"]) if req.get("district_ef") else None,
            country_code=req.get("country_code"),
        )

    def _dispatch_district_cooling(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return self.calculate_district_cooling(
            consumption_gj=_D(req.get("consumption_gj", "0")),
            district_ef=_D(req["district_ef"]) if req.get("district_ef") else None,
            country_code=req.get("country_code"),
        )

    def _dispatch_chp(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return self.allocate_chp_emissions(
            total_fuel_gj=_D(req.get("total_fuel_gj", "0")),
            heat_output_gj=_D(req.get("heat_output_gj", "0")),
            power_output_mwh=_D(req.get("power_output_mwh", "0")),
            method=req.get("method", "efficiency_method"),
        )

    # ===================================================================
    # PRIVATE: Statistics & Metrics Recording
    # ===================================================================

    def _record_success(
        self,
        energy_type: str,
        co2e_kg: Decimal,
        elapsed_ms: float,
    ) -> None:
        """Record a successful calculation in statistics and metrics.

        Args:
            energy_type: Energy type (steam, heating, cooling).
            co2e_kg: Emissions in kgCO2e.
            elapsed_ms: Processing time in milliseconds.
        """
        with self._lock:
            self._total_calculations += 1
            self._total_co2e_kg_cumulative += co2e_kg

            if energy_type == "steam":
                self._total_steam += 1
            elif energy_type == "heating":
                self._total_heating += 1
            elif energy_type == "cooling":
                self._total_cooling += 1

        if self._metrics is not None:
            try:
                co2e_tonnes = float(co2e_kg * _KG_TO_TONNES)
                self._metrics.record_calculation(
                    energy_type=energy_type,
                    method="location_based",
                    duration=elapsed_ms / 1000.0,
                    co2e_tonnes=co2e_tonnes,
                )
                self._metrics.record_steam_heat_cooling(
                    energy_type=energy_type,
                    duration=elapsed_ms / 1000.0,
                )
            except Exception as exc:
                logger.debug("Metrics recording error: %s", exc)

    def _record_error(self, energy_type: str) -> None:
        """Record a failed calculation.

        Args:
            energy_type: Energy type that failed.
        """
        with self._lock:
            self._total_errors += 1

        if self._metrics is not None:
            try:
                self._metrics.record_error("calculation_error")
            except Exception:
                pass

    # ===================================================================
    # PRIVATE: Error Result Builder
    # ===================================================================

    def _build_error_result(
        self,
        calc_id: str,
        energy_type: str,
        error_msg: str,
        trace: List[str],
        t0: float,
    ) -> Dict[str, Any]:
        """Build a standardized error result dictionary.

        Args:
            calc_id: Calculation identifier.
            energy_type: Energy type (steam, heating, cooling, chp).
            error_msg: Error description.
            trace: Calculation trace accumulated so far.
            t0: Start time from time.monotonic().

        Returns:
            Error result dictionary with provenance hash.
        """
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._record_error(energy_type)

        trace.append(f"[ERROR] {error_msg}")

        error_result = {
            "calculation_id": calc_id,
            "status": CalculationStatus.ERROR.value,
            "energy_type": energy_type,
            "error": error_msg,
            "error_type": "CalculationError",
            "total_co2e_kg": str(_ZERO),
            "total_co2e_tonnes": str(_ZERO),
            "calculation_trace": trace,
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": utcnow().isoformat(),
        }
        error_result["provenance_hash"] = _compute_hash(error_result)

        logger.error(
            "SHC calc %s [%s] failed: %s (%.1fms)",
            calc_id, energy_type, error_msg, elapsed_ms,
        )

        return error_result
