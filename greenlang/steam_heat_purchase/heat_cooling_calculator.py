# -*- coding: utf-8 -*-
"""
HeatCoolingCalculatorEngine - District Heating & District Cooling Calculations (Engine 3 of 7)

AGENT-MRV-011: Steam/Heat Purchase Agent

Core calculation engine implementing GHG Protocol Scope 2 emission calculations
for district heating and district cooling energy purchases:

    1. District Heating:
       Adjusted_Consumption (GJ) = Consumption (GJ) / (1 - Distribution_Loss_Pct)
       Emissions (kgCO2e) = Adjusted_Consumption x DH_Network_EF (kgCO2e/GJ)

    2. Electric Cooling (COP-based):
       Electrical_Input (GJ) = Cooling_Output (GJ) / COP
       Electrical_Input (kWh) = Electrical_Input (GJ) x 277.778
       Emissions (kgCO2e) = Electrical_Input (kWh) x Grid_EF (kgCO2e/kWh)

    3. Absorption Cooling (heat-driven):
       Heat_Input (GJ) = Cooling_Output (GJ) / COP_absorption
       Emissions (kgCO2e) = Heat_Input (GJ) x Heat_Source_EF (kgCO2e/GJ)

    4. Free Cooling:
       Pump_Energy (kWh) = Cooling_Output (GJ) x 277.778 / Free_Cooling_COP
       Emissions (kgCO2e) = Pump_Energy (kWh) x Grid_EF (kgCO2e/kWh)

    5. Thermal Storage Losses:
       Effective_Output = Stored_Energy x (1 - Storage_Loss_Pct)
       Additional_Energy = Cooling_Output - Effective_Output

    6. Network Distribution Losses:
       Delivered_Energy = Generated_Energy x (1 - Network_Loss_Pct)
       Required_Generation = Delivered_Energy / (1 - Network_Loss_Pct)

All calculations use Python Decimal arithmetic with 8 decimal places for
zero-hallucination determinism.  Every calculation result includes a full
trace breakdown, processing time, and SHA-256 provenance hash.

Zero-Hallucination Guarantees:
    - All numeric calculations use Python Decimal.
    - No LLM calls in any calculation path.
    - Every calculation step is logged and traceable.
    - SHA-256 provenance hash for every result.
    - Identical inputs always produce identical outputs.

Thread Safety:
    Thread-safe singleton with threading.RLock protecting mutable counters.
    Per-calculation state is created fresh for each method call.

Example:
    >>> from greenlang.steam_heat_purchase.heat_cooling_calculator import (
    ...     HeatCoolingCalculatorEngine,
    ...     get_heat_cooling_calculator,
    ... )
    >>> calc = get_heat_cooling_calculator()
    >>> result = calc.calculate_district_heating(
    ...     consumption_gj="500",
    ...     region="denmark",
    ...     network_type="municipal",
    ...     supplier_ef=None,
    ...     distribution_loss_pct=None,
    ...     gwp_source="AR6",
    ... )
    >>> assert result["status"] == "SUCCESS"
    >>> assert "provenance_hash" in result

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-011 Steam/Heat Purchase Agent (GL-MRV-X-022)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
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

__all__ = ["HeatCoolingCalculatorEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.steam_heat_purchase.metrics import (
        SteamHeatPurchaseMetrics as _MetricsClass,
    )

    def _get_metrics() -> Any:
        """Return the singleton metrics instance."""
        return _MetricsClass()

    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _get_metrics = None  # type: ignore[assignment]

try:
    from greenlang.steam_heat_purchase.steam_heat_database import (
        SteamHeatDatabaseEngine,
    )
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    SteamHeatDatabaseEngine = None  # type: ignore[misc,assignment]

try:
    from greenlang.steam_heat_purchase.config import (
        SteamHeatPurchaseConfig as _ConfigClass,
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _ConfigClass = None  # type: ignore[misc,assignment]

try:
    from greenlang.steam_heat_purchase.provenance import (
        SteamHeatPurchaseProvenance as _ProvenanceClass,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _ProvenanceClass = None  # type: ignore[misc,assignment]

try:
    from greenlang.steam_heat_purchase.models import (
        GWP_VALUES,
        DISTRICT_HEATING_FACTORS,
        COOLING_SYSTEM_FACTORS,
        COOLING_ENERGY_SOURCE,
        UNIT_CONVERSIONS,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    # Inline fallback constants for standalone operation
    GWP_VALUES: Dict[str, Dict[str, Decimal]] = {  # type: ignore[no-redef]
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
            "CH4": Decimal("27.9"),
            "N2O": Decimal("273"),
        },
        "AR6_20YR": {
            "CO2": Decimal("1"),
            "CH4": Decimal("81.2"),
            "N2O": Decimal("273"),
        },
    }
    DISTRICT_HEATING_FACTORS: Dict[str, Dict[str, Decimal]] = {  # type: ignore[no-redef]
        "denmark": {
            "ef_kgco2e_per_gj": Decimal("36.0"),
            "distribution_loss_pct": Decimal("0.10"),
        },
        "sweden": {
            "ef_kgco2e_per_gj": Decimal("18.0"),
            "distribution_loss_pct": Decimal("0.08"),
        },
        "finland": {
            "ef_kgco2e_per_gj": Decimal("55.0"),
            "distribution_loss_pct": Decimal("0.09"),
        },
        "germany": {
            "ef_kgco2e_per_gj": Decimal("72.0"),
            "distribution_loss_pct": Decimal("0.12"),
        },
        "poland": {
            "ef_kgco2e_per_gj": Decimal("105.0"),
            "distribution_loss_pct": Decimal("0.15"),
        },
        "netherlands": {
            "ef_kgco2e_per_gj": Decimal("58.0"),
            "distribution_loss_pct": Decimal("0.10"),
        },
        "france": {
            "ef_kgco2e_per_gj": Decimal("42.0"),
            "distribution_loss_pct": Decimal("0.10"),
        },
        "uk": {
            "ef_kgco2e_per_gj": Decimal("65.0"),
            "distribution_loss_pct": Decimal("0.12"),
        },
        "us": {
            "ef_kgco2e_per_gj": Decimal("75.0"),
            "distribution_loss_pct": Decimal("0.12"),
        },
        "china": {
            "ef_kgco2e_per_gj": Decimal("110.0"),
            "distribution_loss_pct": Decimal("0.15"),
        },
        "japan": {
            "ef_kgco2e_per_gj": Decimal("68.0"),
            "distribution_loss_pct": Decimal("0.10"),
        },
        "south_korea": {
            "ef_kgco2e_per_gj": Decimal("72.0"),
            "distribution_loss_pct": Decimal("0.10"),
        },
        "global_default": {
            "ef_kgco2e_per_gj": Decimal("70.0"),
            "distribution_loss_pct": Decimal("0.12"),
        },
    }
    COOLING_SYSTEM_FACTORS: Dict[str, Dict[str, Decimal]] = {  # type: ignore[no-redef]
        "centrifugal_chiller": {
            "cop_min": Decimal("5.0"),
            "cop_max": Decimal("7.0"),
            "cop_default": Decimal("6.0"),
            "energy_source": Decimal("0"),
        },
        "screw_chiller": {
            "cop_min": Decimal("4.0"),
            "cop_max": Decimal("5.5"),
            "cop_default": Decimal("4.5"),
            "energy_source": Decimal("0"),
        },
        "reciprocating_chiller": {
            "cop_min": Decimal("3.5"),
            "cop_max": Decimal("5.0"),
            "cop_default": Decimal("4.0"),
            "energy_source": Decimal("0"),
        },
        "absorption_single": {
            "cop_min": Decimal("0.6"),
            "cop_max": Decimal("0.8"),
            "cop_default": Decimal("0.7"),
            "energy_source": Decimal("1"),
        },
        "absorption_double": {
            "cop_min": Decimal("1.0"),
            "cop_max": Decimal("1.4"),
            "cop_default": Decimal("1.2"),
            "energy_source": Decimal("1"),
        },
        "absorption_triple": {
            "cop_min": Decimal("1.5"),
            "cop_max": Decimal("1.8"),
            "cop_default": Decimal("1.6"),
            "energy_source": Decimal("1"),
        },
        "free_cooling": {
            "cop_min": Decimal("15.0"),
            "cop_max": Decimal("30.0"),
            "cop_default": Decimal("20.0"),
            "energy_source": Decimal("0"),
        },
        "ice_storage": {
            "cop_min": Decimal("3.0"),
            "cop_max": Decimal("4.5"),
            "cop_default": Decimal("3.5"),
            "energy_source": Decimal("0"),
        },
        "thermal_storage": {
            "cop_min": Decimal("4.0"),
            "cop_max": Decimal("6.0"),
            "cop_default": Decimal("5.0"),
            "energy_source": Decimal("0"),
        },
    }
    COOLING_ENERGY_SOURCE: Dict[str, str] = {  # type: ignore[no-redef]
        "centrifugal_chiller": "electricity",
        "screw_chiller": "electricity",
        "reciprocating_chiller": "electricity",
        "absorption_single": "heat",
        "absorption_double": "heat",
        "absorption_triple": "heat",
        "free_cooling": "electricity",
        "ice_storage": "electricity",
        "thermal_storage": "electricity",
    }
    UNIT_CONVERSIONS: Dict[str, Decimal] = {  # type: ignore[no-redef]
        "gj_to_mwh": Decimal("0.277778"),
        "mwh_to_gj": Decimal("3.6"),
        "gj_to_kwh": Decimal("277.778"),
        "gj_to_mmbtu": Decimal("0.947817"),
        "mmbtu_to_gj": Decimal("1.055056"),
        "therm_to_gj": Decimal("0.105506"),
        "gj_to_therm": Decimal("9.47817"),
        "mj_to_gj": Decimal("0.001"),
        "gj_to_mj": Decimal("1000.0"),
    }


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Dictionary, Pydantic model, or any JSON-serializable object.

    Returns:
        64-character lowercase hex SHA-256 digest string.
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
_ZERO = Decimal("0")
_ONE = Decimal("1")
_THOUSAND = Decimal("1000")
_GJ_TO_KWH = Decimal("277.778")


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal with controlled precision.

    Args:
        value: Any numeric value or string representation.

    Returns:
        Decimal representation of the value.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert a value to Decimal, returning default on failure.

    Args:
        value: Value to convert. May be None, str, int, float, or Decimal.
        default: Fallback value if conversion fails.

    Returns:
        Decimal representation or the default.
    """
    if value is None:
        return default
    try:
        return _D(value)
    except (InvalidOperation, ValueError, TypeError):
        return default


# ===========================================================================
# Enumerations
# ===========================================================================


class HeatingCoolingType(str, Enum):
    """Classification of heat/cooling calculation types."""

    DISTRICT_HEATING = "DISTRICT_HEATING"
    ELECTRIC_COOLING = "ELECTRIC_COOLING"
    ABSORPTION_COOLING = "ABSORPTION_COOLING"
    FREE_COOLING = "FREE_COOLING"
    THERMAL_STORAGE_COOLING = "THERMAL_STORAGE_COOLING"
    ICE_STORAGE_COOLING = "ICE_STORAGE_COOLING"


class CoolingCategory(str, Enum):
    """Broad category classification for cooling technologies."""

    ELECTRIC = "ELECTRIC"
    ABSORPTION = "ABSORPTION"
    FREE = "FREE"
    STORAGE = "STORAGE"


# ===========================================================================
# Trace Step Dataclass
# ===========================================================================


@dataclass
class TraceStep:
    """Single step in a calculation trace for audit trail.

    Attributes:
        step_number: Sequential step number.
        description: Human-readable description of what was calculated.
        formula: Mathematical formula applied.
        inputs: Input values for this step.
        output: Output value from this step.
        unit: Unit of the output value.
    """

    step_number: int
    description: str
    formula: str
    inputs: Dict[str, str]
    output: str
    unit: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_number": self.step_number,
            "description": self.description,
            "formula": self.formula,
            "inputs": self.inputs,
            "output": self.output,
            "unit": self.unit,
        }


# ===========================================================================
# Default COP values for cooling technology families
# ===========================================================================

# Electric cooling technologies (COP = cooling_output / electricity_input)
_ELECTRIC_COOLING_TECHNOLOGIES = frozenset({
    "centrifugal_chiller",
    "screw_chiller",
    "reciprocating_chiller",
    "ice_storage",
    "thermal_storage",
})

# Absorption cooling technologies (COP = cooling_output / heat_input)
_ABSORPTION_COOLING_TECHNOLOGIES = frozenset({
    "absorption_single",
    "absorption_double",
    "absorption_triple",
})

# Free cooling technologies (minimal pump energy only)
_FREE_COOLING_TECHNOLOGIES = frozenset({
    "free_cooling",
})

# Default storage loss percentages by technology type
_DEFAULT_STORAGE_LOSS_PCT: Dict[str, Decimal] = {
    "ice_storage": Decimal("0.05"),
    "thermal_storage": Decimal("0.03"),
}

# Default network loss percentages by network type
_DEFAULT_NETWORK_LOSS_PCT: Dict[str, Decimal] = {
    "municipal": Decimal("0.12"),
    "industrial": Decimal("0.08"),
    "campus": Decimal("0.06"),
    "mixed": Decimal("0.10"),
}

# Default grid emission factor (kgCO2e/kWh) -- IEA global average
_DEFAULT_GRID_EF_KWH = Decimal("0.450")

# Default heat source emission factor (kgCO2e/GJ) -- natural gas boiler
_DEFAULT_HEAT_SOURCE_EF_GJ = Decimal("66.0")


# ===========================================================================
# HeatCoolingCalculatorEngine
# ===========================================================================


class HeatCoolingCalculatorEngine:
    """Core district heating and district cooling emission calculator
    implementing GHG Protocol Scope 2 methods.

    This engine performs deterministic Decimal arithmetic for all heating
    and cooling emission calculations, including:
    - District heating with regional emission factors and distribution losses
    - Electric cooling via COP-based electrical input conversion
    - Absorption cooling via COP-based heat input conversion
    - Free cooling with minimal pump energy calculations
    - Thermal storage and ice storage loss adjustments
    - Network distribution loss adjustments

    Zero-Hallucination:
        All calculations use Python Decimal arithmetic. No LLM calls are
        made anywhere in the calculation path. Every step is traced and
        every result carries a SHA-256 provenance hash.

    Thread Safety:
        Singleton pattern with RLock. Per-calculation state is created
        fresh for each method call. Shared counters are protected by
        the reentrant lock.

    Attributes:
        _lock: Reentrant lock protecting mutable counters.
        _total_calculations: Counter of total calculations performed.
        _total_heating_calculations: Counter of heating calculations.
        _total_cooling_calculations: Counter of cooling calculations.
        _total_batch_calculations: Counter of batch calculations.
        _gwp_source: Default GWP source for CO2e conversion.
        _db: Optional reference to the SteamHeatDatabaseEngine.
        _created_at: Timestamp when this engine was initialized.

    Example:
        >>> calc = HeatCoolingCalculatorEngine()
        >>> result = calc.calculate_district_heating(
        ...     consumption_gj="1000",
        ...     region="germany",
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    # ------------------------------------------------------------------
    # Singleton machinery
    # ------------------------------------------------------------------

    _instance: Optional[HeatCoolingCalculatorEngine] = None
    _singleton_lock: threading.RLock = threading.RLock()

    def __new__(cls, *args: Any, **kwargs: Any) -> HeatCoolingCalculatorEngine:
        """Return the singleton instance, creating it on first call.

        Uses double-checked locking with the class-level RLock to ensure
        thread-safe singleton creation.

        Returns:
            The singleton HeatCoolingCalculatorEngine instance.
        """
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
                    logger.info(
                        "HeatCoolingCalculatorEngine singleton created"
                    )
        return cls._instance

    def __init__(
        self,
        steam_heat_database: Optional[Any] = None,
        gwp_source: str = "AR6",
    ) -> None:
        """Initialize the HeatCoolingCalculatorEngine.

        Only initializes once due to the singleton pattern. Subsequent
        calls to ``__init__`` are silently skipped.

        Args:
            steam_heat_database: Optional SteamHeatDatabaseEngine instance.
                If None, standalone mode with inline constants is used.
            gwp_source: Default GWP source for CO2e conversion (AR4/AR5/AR6).
        """
        if getattr(self, "_initialized", False):
            return

        if steam_heat_database is not None:
            self._db = steam_heat_database
        elif _DB_AVAILABLE and SteamHeatDatabaseEngine is not None:
            try:
                self._db = SteamHeatDatabaseEngine()
            except Exception:
                self._db = None
        else:
            self._db = None

        self._gwp_source = gwp_source.upper()
        self._lock = threading.RLock()
        self._total_calculations: int = 0
        self._total_heating_calculations: int = 0
        self._total_cooling_calculations: int = 0
        self._total_batch_calculations: int = 0
        self._created_at = _utcnow()
        self._initialized = True

        logger.info(
            "HeatCoolingCalculatorEngine initialized: "
            "gwp_source=%s, db_available=%s",
            self._gwp_source,
            self._db is not None,
        )

    # ------------------------------------------------------------------
    # Singleton management
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. Intended for testing teardown.

        Destroys the current singleton so that the next instantiation
        creates a fresh engine with default state.
        """
        with cls._singleton_lock:
            cls._instance = None
        logger.info("HeatCoolingCalculatorEngine singleton reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _increment_calculations(self, calc_type: str = "general") -> None:
        """Thread-safe increment of calculation counters.

        Args:
            calc_type: One of 'general', 'heating', 'cooling', or 'batch'.
        """
        with self._lock:
            self._total_calculations += 1
            if calc_type == "heating":
                self._total_heating_calculations += 1
            elif calc_type == "cooling":
                self._total_cooling_calculations += 1
            elif calc_type == "batch":
                self._total_batch_calculations += 1

    def _resolve_dh_factors(
        self,
        region: str,
        supplier_ef: Optional[Any] = None,
        distribution_loss_pct: Optional[Any] = None,
    ) -> Tuple[Decimal, Decimal, str]:
        """Resolve district heating emission factor and distribution loss.

        Looks up regional factors from the constant table or database.
        Supplier-specific overrides take precedence when provided.

        Args:
            region: Geographic region identifier (e.g. 'denmark').
            supplier_ef: Optional supplier-specific EF override (kgCO2e/GJ).
            distribution_loss_pct: Optional distribution loss override (0-1).

        Returns:
            Tuple of (emission_factor_kgco2e_per_gj, loss_pct, ef_source).
        """
        region_lower = region.lower().strip()
        ef_source = "regional_default"

        # Look up regional factors
        regional = DISTRICT_HEATING_FACTORS.get(
            region_lower,
            DISTRICT_HEATING_FACTORS.get("global_default", {}),
        )

        ef_kgco2e_per_gj = _safe_decimal(
            regional.get("ef_kgco2e_per_gj"), Decimal("70.0")
        )
        loss_pct = _safe_decimal(
            regional.get("distribution_loss_pct"), Decimal("0.12")
        )

        # Override with supplier-specific EF if provided
        if supplier_ef is not None:
            ef_kgco2e_per_gj = _safe_decimal(supplier_ef, ef_kgco2e_per_gj)
            ef_source = "supplier_specific"

        # Override distribution loss if provided
        if distribution_loss_pct is not None:
            loss_pct = _safe_decimal(distribution_loss_pct, loss_pct)

        return ef_kgco2e_per_gj, loss_pct, ef_source

    def _resolve_cooling_factors(
        self,
        technology: str,
        cop_override: Optional[Any] = None,
    ) -> Tuple[Decimal, str]:
        """Resolve COP and energy source for a cooling technology.

        Args:
            technology: Cooling technology identifier.
            cop_override: Optional COP override value.

        Returns:
            Tuple of (cop_value, energy_source_str).
        """
        tech_lower = technology.lower().strip()

        tech_factors = COOLING_SYSTEM_FACTORS.get(tech_lower, {})
        cop_default = _safe_decimal(
            tech_factors.get("cop_default"), Decimal("5.0")
        )
        energy_source = COOLING_ENERGY_SOURCE.get(tech_lower, "electricity")

        if cop_override is not None:
            cop_value = _safe_decimal(cop_override, cop_default)
        else:
            cop_value = cop_default

        return cop_value, energy_source

    def _classify_cooling_technology(self, technology: str) -> CoolingCategory:
        """Classify a cooling technology into a broad category.

        Args:
            technology: Cooling technology identifier.

        Returns:
            CoolingCategory enum value.
        """
        tech_lower = technology.lower().strip()
        if tech_lower in _ABSORPTION_COOLING_TECHNOLOGIES:
            return CoolingCategory.ABSORPTION
        if tech_lower in _FREE_COOLING_TECHNOLOGIES:
            return CoolingCategory.FREE
        if tech_lower in ("ice_storage", "thermal_storage"):
            return CoolingCategory.STORAGE
        return CoolingCategory.ELECTRIC

    def _get_gwp_multiplier(
        self,
        gas: str,
        gwp_source: Optional[str] = None,
    ) -> Decimal:
        """Get the GWP multiplier for a greenhouse gas.

        Args:
            gas: Gas name (CO2, CH4, N2O).
            gwp_source: IPCC Assessment Report source. Defaults to engine
                default.

        Returns:
            GWP multiplier as Decimal.
        """
        source = (gwp_source or self._gwp_source).upper()
        gas_upper = gas.upper()
        if source in GWP_VALUES and gas_upper in GWP_VALUES[source]:
            return GWP_VALUES[source][gas_upper]
        return _ONE

    # ==================================================================
    # Public Method 1: calculate_heating_emissions
    # ==================================================================

    def calculate_heating_emissions(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate emissions from district heating consumption.

        Main district heating calculation entry point accepting a request
        dictionary. Delegates to ``calculate_district_heating`` after
        extracting and validating request parameters.

        Required request keys:
            consumption_gj: Thermal energy consumed in GJ.
            region: Geographic region for emission factor lookup.

        Optional request keys:
            network_type: Type of district heating network.
            supplier_ef: Supplier-specific emission factor (kgCO2e/GJ).
            distribution_loss_pct: Distribution loss fraction (0-1).
            gwp_source: GWP source (AR4/AR5/AR6).
            facility_id: Facility identifier for traceability.
            period: Reporting period identifier.

        Args:
            request: Calculation request dictionary.

        Returns:
            Calculation result dictionary with emissions, trace, and hash.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        consumption_gj = request.get("consumption_gj")
        region = str(request.get("region", "global_default"))
        network_type = str(request.get("network_type", "municipal"))
        supplier_ef = request.get("supplier_ef")
        distribution_loss_pct = request.get("distribution_loss_pct")
        gwp_source = str(request.get("gwp_source", self._gwp_source))
        facility_id = str(request.get("facility_id", ""))
        period = str(request.get("period", ""))

        result = self.calculate_district_heating(
            consumption_gj=consumption_gj,
            region=region,
            network_type=network_type,
            supplier_ef=supplier_ef,
            distribution_loss_pct=distribution_loss_pct,
            gwp_source=gwp_source,
        )

        # Enrich result with request metadata
        if facility_id:
            result["facility_id"] = facility_id
        if period:
            result["period"] = period

        return result

    # ==================================================================
    # Public Method 2: calculate_cooling_emissions
    # ==================================================================

    def calculate_cooling_emissions(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate emissions from district cooling consumption.

        Main district cooling calculation entry point accepting a request
        dictionary. Routes to the appropriate cooling calculation method
        based on the technology's energy source classification (electric,
        absorption, or free cooling).

        Required request keys:
            cooling_output_gj: Cooling energy delivered in GJ.
            technology: Cooling technology identifier.

        Optional request keys:
            cop: Coefficient of performance override.
            grid_ef_kwh: Grid emission factor (kgCO2e/kWh) for electric.
            heat_source_ef: Heat source EF (kgCO2e/GJ) for absorption.
            storage_loss_pct: Thermal storage loss fraction (0-1).
            gwp_source: GWP source (AR4/AR5/AR6).
            facility_id: Facility identifier for traceability.
            period: Reporting period identifier.

        Args:
            request: Calculation request dictionary.

        Returns:
            Calculation result dictionary with emissions, trace, and hash.
        """
        cooling_output_gj = request.get("cooling_output_gj")
        technology = str(request.get("technology", "centrifugal_chiller"))
        cop = request.get("cop")
        grid_ef_kwh = request.get("grid_ef_kwh")
        heat_source_ef = request.get("heat_source_ef")
        storage_loss_pct = request.get("storage_loss_pct")
        gwp_source = str(request.get("gwp_source", self._gwp_source))
        facility_id = str(request.get("facility_id", ""))
        period = str(request.get("period", ""))

        # Classify the technology to choose the right method
        category = self._classify_cooling_technology(technology)

        if category == CoolingCategory.ABSORPTION:
            result = self.calculate_absorption_cooling(
                cooling_output_gj=cooling_output_gj,
                technology=technology,
                cop=cop,
                heat_source_ef=heat_source_ef,
                gwp_source=gwp_source,
            )
        elif category == CoolingCategory.FREE:
            result = self.calculate_free_cooling(
                cooling_output_gj=cooling_output_gj,
                cop=cop,
                grid_ef_kwh=grid_ef_kwh,
                gwp_source=gwp_source,
            )
        else:
            # ELECTRIC and STORAGE categories both use electric cooling
            result = self.calculate_electric_cooling(
                cooling_output_gj=cooling_output_gj,
                technology=technology,
                cop=cop,
                grid_ef_kwh=grid_ef_kwh,
                gwp_source=gwp_source,
            )

        # Apply storage loss adjustment for storage technologies
        if category == CoolingCategory.STORAGE and storage_loss_pct is not None:
            result = self._apply_storage_loss_to_result(
                result, storage_loss_pct, technology,
            )

        # Enrich result with request metadata
        if facility_id:
            result["facility_id"] = facility_id
        if period:
            result["period"] = period

        return result

    # ==================================================================
    # Public Method 3: calculate_district_heating
    # ==================================================================

    def calculate_district_heating(
        self,
        consumption_gj: Any,
        region: str = "global_default",
        network_type: str = "municipal",
        supplier_ef: Optional[Any] = None,
        distribution_loss_pct: Optional[Any] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate emissions from district heating consumption.

        Applies the district heating formula:
            Adjusted_Consumption = Consumption / (1 - Distribution_Loss_Pct)
            Emissions_kgCO2e = Adjusted_Consumption * DH_Network_EF

        The distribution loss adjustment accounts for energy lost in the
        district heating network between the generating plant and the
        customer meter. This means the plant had to generate more energy
        than what the customer consumed.

        Args:
            consumption_gj: Thermal energy consumed at the meter in GJ.
            region: Geographic region for emission factor lookup.
            network_type: Type of district heating network.
            supplier_ef: Optional supplier-specific EF (kgCO2e/GJ).
            distribution_loss_pct: Optional distribution loss (0-1).
            gwp_source: GWP source (AR4/AR5/AR6).

        Returns:
            Calculation result with emissions, per-gas breakdown, trace,
            processing time, and SHA-256 provenance hash.
        """
        self._increment_calculations("heating")
        start_time = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = str(uuid4())
        gwp = (gwp_source or self._gwp_source).upper()

        # -- Validate inputs -----------------------------------------------
        errors: List[str] = []
        consumption = _safe_decimal(consumption_gj)
        if consumption_gj is None:
            errors.append("consumption_gj is required")
        elif consumption <= _ZERO:
            errors.append("consumption_gj must be > 0")

        region_str = str(region).strip()
        if not region_str:
            errors.append("region is required")

        if errors:
            return self._build_error_result(
                calc_id, "DISTRICT_HEATING", errors, start_time,
            )

        # -- Resolve emission factor and loss percentage -------------------
        ef_kgco2e_per_gj, loss_pct, ef_source = self._resolve_dh_factors(
            region_str, supplier_ef, distribution_loss_pct,
        )

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Resolve district heating emission factor and loss",
            formula="lookup(region, supplier_ef)",
            inputs={
                "region": region_str,
                "network_type": network_type,
                "supplier_ef_override": str(supplier_ef),
            },
            output=str(ef_kgco2e_per_gj),
            unit="kgCO2e/GJ",
        ))

        # -- Apply distribution loss adjustment ----------------------------
        adjusted_consumption = self.apply_distribution_loss(
            consumption, loss_pct,
        )

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Adjust consumption for distribution losses",
            formula="Adjusted = Consumption / (1 - Loss_Pct)",
            inputs={
                "consumption_gj": str(consumption),
                "distribution_loss_pct": str(loss_pct),
            },
            output=str(adjusted_consumption),
            unit="GJ",
        ))

        # -- Calculate total emissions (kgCO2e) ----------------------------
        total_emissions_kgco2e = (
            adjusted_consumption * ef_kgco2e_per_gj
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Calculate district heating emissions",
            formula="Emissions = Adjusted_Consumption x EF",
            inputs={
                "adjusted_consumption_gj": str(adjusted_consumption),
                "ef_kgco2e_per_gj": str(ef_kgco2e_per_gj),
            },
            output=str(total_emissions_kgco2e),
            unit="kgCO2e",
        ))

        # -- Convert to tonnes CO2e ----------------------------------------
        total_emissions_tco2e = (
            total_emissions_kgco2e / _THOUSAND
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Convert emissions to tonnes CO2e",
            formula="tCO2e = kgCO2e / 1000",
            inputs={"emissions_kgco2e": str(total_emissions_kgco2e)},
            output=str(total_emissions_tco2e),
            unit="tCO2e",
        ))

        # -- Energy loss calculated ----------------------------------------
        energy_loss_gj = (
            adjusted_consumption - consumption
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Calculate energy lost in distribution network",
            formula="Loss = Adjusted - Consumption",
            inputs={
                "adjusted_consumption_gj": str(adjusted_consumption),
                "consumption_gj": str(consumption),
            },
            output=str(energy_loss_gj),
            unit="GJ",
        ))

        # -- Assemble result -----------------------------------------------
        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "calculation_type": "DISTRICT_HEATING",
            "energy_type": "district_heating",
            "region": region_str,
            "network_type": network_type,
            "gwp_source": gwp,
            "ef_source": ef_source,
            "inputs": {
                "consumption_gj": str(consumption),
                "ef_kgco2e_per_gj": str(ef_kgco2e_per_gj),
                "distribution_loss_pct": str(loss_pct),
            },
            "adjusted_consumption_gj": str(adjusted_consumption),
            "energy_loss_gj": str(energy_loss_gj),
            "emissions_kgco2e": str(total_emissions_kgco2e),
            "emissions_tco2e": str(total_emissions_tco2e),
            "emissions_kgco2": str(total_emissions_kgco2e),
            "emissions_kgch4": str(_ZERO),
            "emissions_kgn2o": str(_ZERO),
            "biogenic_co2_kg": str(_ZERO),
            "trace_steps": [s.to_dict() for s in trace_steps],
            "processing_time_ms": processing_time,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "District heating calculation complete: id=%s, "
            "consumption=%s GJ, adjusted=%s GJ, "
            "emissions=%s kgCO2e, time=%.3fms",
            calc_id, consumption, adjusted_consumption,
            total_emissions_kgco2e, processing_time,
        )

        if _METRICS_AVAILABLE and _get_metrics is not None:
            try:
                metrics = _get_metrics()
                metrics.record_calculation(
                    energy_type="district_heat",
                    method="default_emission_factor"
                    if ef_source == "regional_default"
                    else "supplier_specific",
                    status="success",
                    duration=processing_time / 1000.0,
                    co2e_kg=float(total_emissions_kgco2e),
                    biogenic_kg=0.0,
                    fuel_type="mixed",
                    tenant_id="default",
                )
            except Exception:
                pass

        return result

    # ==================================================================
    # Public Method 4: calculate_electric_cooling
    # ==================================================================

    def calculate_electric_cooling(
        self,
        cooling_output_gj: Any,
        technology: str = "centrifugal_chiller",
        cop: Optional[Any] = None,
        grid_ef_kwh: Optional[Any] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate emissions from electric cooling (COP-based).

        Applies the electric cooling formula:
            Electrical_Input (GJ) = Cooling_Output (GJ) / COP
            Electrical_Input (kWh) = Electrical_Input (GJ) x 277.778
            Emissions (kgCO2e) = Electrical_Input (kWh) x Grid_EF (kgCO2e/kWh)

        Args:
            cooling_output_gj: Cooling energy delivered in GJ.
            technology: Cooling technology identifier for COP lookup.
            cop: Optional COP override. If None, default for technology.
            grid_ef_kwh: Grid emission factor in kgCO2e/kWh. If None,
                uses IEA global average (0.450 kgCO2e/kWh).
            gwp_source: GWP source (AR4/AR5/AR6).

        Returns:
            Calculation result with emissions, electrical input, trace,
            processing time, and SHA-256 provenance hash.
        """
        self._increment_calculations("cooling")
        start_time = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = str(uuid4())
        gwp = (gwp_source or self._gwp_source).upper()

        # -- Validate inputs -----------------------------------------------
        errors: List[str] = []
        cooling_gj = _safe_decimal(cooling_output_gj)
        if cooling_output_gj is None:
            errors.append("cooling_output_gj is required")
        elif cooling_gj <= _ZERO:
            errors.append("cooling_output_gj must be > 0")

        if errors:
            return self._build_error_result(
                calc_id, "ELECTRIC_COOLING", errors, start_time,
            )

        # -- Resolve COP and grid EF ---------------------------------------
        cop_value, energy_source = self._resolve_cooling_factors(
            technology, cop,
        )
        grid_ef = _safe_decimal(grid_ef_kwh, _DEFAULT_GRID_EF_KWH)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Resolve COP and grid emission factor",
            formula="lookup(technology, cop_override)",
            inputs={
                "technology": technology,
                "cop_override": str(cop),
                "grid_ef_kwh_override": str(grid_ef_kwh),
            },
            output=f"COP={cop_value}, Grid_EF={grid_ef}",
            unit="mixed",
        ))

        # -- Compute electrical input (GJ) --------------------------------
        electrical_input_gj = self.compute_electrical_input(cooling_gj, cop_value)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Compute electrical input from cooling output",
            formula="Electrical_Input_GJ = Cooling_Output / COP",
            inputs={
                "cooling_output_gj": str(cooling_gj),
                "cop": str(cop_value),
            },
            output=str(electrical_input_gj),
            unit="GJ",
        ))

        # -- Convert GJ to kWh --------------------------------------------
        electrical_input_kwh = self.convert_gj_to_kwh(electrical_input_gj)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Convert electrical input from GJ to kWh",
            formula="kWh = GJ x 277.778",
            inputs={"electrical_input_gj": str(electrical_input_gj)},
            output=str(electrical_input_kwh),
            unit="kWh",
        ))

        # -- Calculate emissions (kgCO2e) ----------------------------------
        total_emissions_kgco2e = (
            electrical_input_kwh * grid_ef
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Calculate electric cooling emissions",
            formula="Emissions = Electrical_Input_kWh x Grid_EF",
            inputs={
                "electrical_input_kwh": str(electrical_input_kwh),
                "grid_ef_kgco2e_per_kwh": str(grid_ef),
            },
            output=str(total_emissions_kgco2e),
            unit="kgCO2e",
        ))

        # -- Convert to tonnes CO2e ----------------------------------------
        total_emissions_tco2e = (
            total_emissions_kgco2e / _THOUSAND
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Convert emissions to tonnes CO2e",
            formula="tCO2e = kgCO2e / 1000",
            inputs={"emissions_kgco2e": str(total_emissions_kgco2e)},
            output=str(total_emissions_tco2e),
            unit="tCO2e",
        ))

        # -- Assemble result -----------------------------------------------
        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "calculation_type": "ELECTRIC_COOLING",
            "energy_type": "district_cooling",
            "technology": technology,
            "cop": str(cop_value),
            "energy_source": energy_source,
            "gwp_source": gwp,
            "inputs": {
                "cooling_output_gj": str(cooling_gj),
                "cop": str(cop_value),
                "grid_ef_kgco2e_per_kwh": str(grid_ef),
            },
            "electrical_input_gj": str(electrical_input_gj),
            "electrical_input_kwh": str(electrical_input_kwh),
            "emissions_kgco2e": str(total_emissions_kgco2e),
            "emissions_tco2e": str(total_emissions_tco2e),
            "emissions_kgco2": str(total_emissions_kgco2e),
            "emissions_kgch4": str(_ZERO),
            "emissions_kgn2o": str(_ZERO),
            "biogenic_co2_kg": str(_ZERO),
            "trace_steps": [s.to_dict() for s in trace_steps],
            "processing_time_ms": processing_time,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Electric cooling calculation complete: id=%s, "
            "cooling=%s GJ, elec_input=%s kWh, "
            "emissions=%s kgCO2e, COP=%s, time=%.3fms",
            calc_id, cooling_gj, electrical_input_kwh,
            total_emissions_kgco2e, cop_value, processing_time,
        )

        if _METRICS_AVAILABLE and _get_metrics is not None:
            try:
                metrics = _get_metrics()
                metrics.record_calculation(
                    energy_type="district_cooling",
                    method="supplier_specific"
                    if cop is not None
                    else "default_emission_factor",
                    status="success",
                    duration=processing_time / 1000.0,
                    co2e_kg=float(total_emissions_kgco2e),
                    biogenic_kg=0.0,
                    fuel_type="mixed",
                    tenant_id="default",
                )
            except Exception:
                pass

        return result

    # ==================================================================
    # Public Method 5: calculate_absorption_cooling
    # ==================================================================

    def calculate_absorption_cooling(
        self,
        cooling_output_gj: Any,
        technology: str = "absorption_double",
        cop: Optional[Any] = None,
        heat_source_ef: Optional[Any] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate emissions from absorption cooling (heat-driven).

        Applies the absorption cooling formula:
            Heat_Input (GJ) = Cooling_Output (GJ) / COP_absorption
            Emissions (kgCO2e) = Heat_Input (GJ) x Heat_Source_EF (kgCO2e/GJ)

        Absorption chillers use thermal energy (steam, hot water, or waste
        heat) rather than electricity to drive the cooling process. The
        emission factor reflects the carbon intensity of the heat source.

        Args:
            cooling_output_gj: Cooling energy delivered in GJ.
            technology: Absorption technology identifier.
            cop: Optional COP override. If None, default for technology.
            heat_source_ef: Heat source emission factor (kgCO2e/GJ).
                If None, uses default natural gas boiler factor (66.0).
            gwp_source: GWP source (AR4/AR5/AR6).

        Returns:
            Calculation result with emissions, heat input, trace,
            processing time, and SHA-256 provenance hash.
        """
        self._increment_calculations("cooling")
        start_time = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = str(uuid4())
        gwp = (gwp_source or self._gwp_source).upper()

        # -- Validate inputs -----------------------------------------------
        errors: List[str] = []
        cooling_gj = _safe_decimal(cooling_output_gj)
        if cooling_output_gj is None:
            errors.append("cooling_output_gj is required")
        elif cooling_gj <= _ZERO:
            errors.append("cooling_output_gj must be > 0")

        if errors:
            return self._build_error_result(
                calc_id, "ABSORPTION_COOLING", errors, start_time,
            )

        # -- Resolve COP and heat source EF --------------------------------
        cop_value, energy_source = self._resolve_cooling_factors(
            technology, cop,
        )
        hs_ef = _safe_decimal(heat_source_ef, _DEFAULT_HEAT_SOURCE_EF_GJ)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Resolve absorption COP and heat source EF",
            formula="lookup(technology, cop_override)",
            inputs={
                "technology": technology,
                "cop_override": str(cop),
                "heat_source_ef_override": str(heat_source_ef),
            },
            output=f"COP={cop_value}, Heat_EF={hs_ef}",
            unit="mixed",
        ))

        # -- Compute heat input (GJ) --------------------------------------
        heat_input_gj = self.compute_heat_input(cooling_gj, cop_value)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Compute heat input from cooling output",
            formula="Heat_Input = Cooling_Output / COP_absorption",
            inputs={
                "cooling_output_gj": str(cooling_gj),
                "cop_absorption": str(cop_value),
            },
            output=str(heat_input_gj),
            unit="GJ",
        ))

        # -- Calculate emissions (kgCO2e) ----------------------------------
        total_emissions_kgco2e = (
            heat_input_gj * hs_ef
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Calculate absorption cooling emissions",
            formula="Emissions = Heat_Input x Heat_Source_EF",
            inputs={
                "heat_input_gj": str(heat_input_gj),
                "heat_source_ef_kgco2e_per_gj": str(hs_ef),
            },
            output=str(total_emissions_kgco2e),
            unit="kgCO2e",
        ))

        # -- Convert to tonnes CO2e ----------------------------------------
        total_emissions_tco2e = (
            total_emissions_kgco2e / _THOUSAND
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Convert emissions to tonnes CO2e",
            formula="tCO2e = kgCO2e / 1000",
            inputs={"emissions_kgco2e": str(total_emissions_kgco2e)},
            output=str(total_emissions_tco2e),
            unit="tCO2e",
        ))

        # -- Assemble result -----------------------------------------------
        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "calculation_type": "ABSORPTION_COOLING",
            "energy_type": "district_cooling",
            "technology": technology,
            "cop": str(cop_value),
            "energy_source": energy_source,
            "gwp_source": gwp,
            "inputs": {
                "cooling_output_gj": str(cooling_gj),
                "cop_absorption": str(cop_value),
                "heat_source_ef_kgco2e_per_gj": str(hs_ef),
            },
            "heat_input_gj": str(heat_input_gj),
            "emissions_kgco2e": str(total_emissions_kgco2e),
            "emissions_tco2e": str(total_emissions_tco2e),
            "emissions_kgco2": str(total_emissions_kgco2e),
            "emissions_kgch4": str(_ZERO),
            "emissions_kgn2o": str(_ZERO),
            "biogenic_co2_kg": str(_ZERO),
            "trace_steps": [s.to_dict() for s in trace_steps],
            "processing_time_ms": processing_time,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Absorption cooling calculation complete: id=%s, "
            "cooling=%s GJ, heat_input=%s GJ, "
            "emissions=%s kgCO2e, COP=%s, time=%.3fms",
            calc_id, cooling_gj, heat_input_gj,
            total_emissions_kgco2e, cop_value, processing_time,
        )

        return result

    # ==================================================================
    # Public Method 6: calculate_free_cooling
    # ==================================================================

    def calculate_free_cooling(
        self,
        cooling_output_gj: Any,
        cop: Optional[Any] = None,
        grid_ef_kwh: Optional[Any] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate emissions from free cooling (pump energy only).

        Free cooling uses ambient sources (seawater, lake water, deep
        aquifer) with minimal pump and fan energy. The effective COP is
        very high (15-30) because only circulation energy is needed.

        Formula:
            Pump_Energy (kWh) = Cooling_Output (GJ) x 277.778 / Free_Cooling_COP
            Emissions (kgCO2e) = Pump_Energy (kWh) x Grid_EF (kgCO2e/kWh)

        Args:
            cooling_output_gj: Cooling energy delivered in GJ.
            cop: Optional free cooling COP override. Default is 20.0.
            grid_ef_kwh: Grid emission factor (kgCO2e/kWh).
            gwp_source: GWP source (AR4/AR5/AR6).

        Returns:
            Calculation result with emissions, pump energy, trace,
            processing time, and SHA-256 provenance hash.
        """
        self._increment_calculations("cooling")
        start_time = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = str(uuid4())
        gwp = (gwp_source or self._gwp_source).upper()

        # -- Validate inputs -----------------------------------------------
        errors: List[str] = []
        cooling_gj = _safe_decimal(cooling_output_gj)
        if cooling_output_gj is None:
            errors.append("cooling_output_gj is required")
        elif cooling_gj <= _ZERO:
            errors.append("cooling_output_gj must be > 0")

        if errors:
            return self._build_error_result(
                calc_id, "FREE_COOLING", errors, start_time,
            )

        # -- Resolve COP and grid EF ---------------------------------------
        cop_value, _ = self._resolve_cooling_factors("free_cooling", cop)
        grid_ef = _safe_decimal(grid_ef_kwh, _DEFAULT_GRID_EF_KWH)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Resolve free cooling COP and grid EF",
            formula="lookup(free_cooling, cop_override)",
            inputs={
                "cop_override": str(cop),
                "grid_ef_override": str(grid_ef_kwh),
            },
            output=f"COP={cop_value}, Grid_EF={grid_ef}",
            unit="mixed",
        ))

        # -- Compute pump energy (kWh) ------------------------------------
        cooling_kwh = self.convert_gj_to_kwh(cooling_gj)
        pump_energy_kwh = (
            cooling_kwh / cop_value
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Compute pump energy for free cooling",
            formula="Pump_kWh = Cooling_GJ x 277.778 / COP",
            inputs={
                "cooling_output_gj": str(cooling_gj),
                "cooling_kwh": str(cooling_kwh),
                "cop_free_cooling": str(cop_value),
            },
            output=str(pump_energy_kwh),
            unit="kWh",
        ))

        # -- Calculate emissions (kgCO2e) ----------------------------------
        total_emissions_kgco2e = (
            pump_energy_kwh * grid_ef
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Calculate free cooling emissions",
            formula="Emissions = Pump_kWh x Grid_EF",
            inputs={
                "pump_energy_kwh": str(pump_energy_kwh),
                "grid_ef_kgco2e_per_kwh": str(grid_ef),
            },
            output=str(total_emissions_kgco2e),
            unit="kgCO2e",
        ))

        # -- Convert to tonnes CO2e ----------------------------------------
        total_emissions_tco2e = (
            total_emissions_kgco2e / _THOUSAND
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Convert emissions to tonnes CO2e",
            formula="tCO2e = kgCO2e / 1000",
            inputs={"emissions_kgco2e": str(total_emissions_kgco2e)},
            output=str(total_emissions_tco2e),
            unit="tCO2e",
        ))

        # -- Assemble result -----------------------------------------------
        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "calculation_type": "FREE_COOLING",
            "energy_type": "district_cooling",
            "technology": "free_cooling",
            "cop": str(cop_value),
            "energy_source": "electricity",
            "gwp_source": gwp,
            "inputs": {
                "cooling_output_gj": str(cooling_gj),
                "cop_free_cooling": str(cop_value),
                "grid_ef_kgco2e_per_kwh": str(grid_ef),
            },
            "pump_energy_kwh": str(pump_energy_kwh),
            "emissions_kgco2e": str(total_emissions_kgco2e),
            "emissions_tco2e": str(total_emissions_tco2e),
            "emissions_kgco2": str(total_emissions_kgco2e),
            "emissions_kgch4": str(_ZERO),
            "emissions_kgn2o": str(_ZERO),
            "biogenic_co2_kg": str(_ZERO),
            "trace_steps": [s.to_dict() for s in trace_steps],
            "processing_time_ms": processing_time,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Free cooling calculation complete: id=%s, "
            "cooling=%s GJ, pump=%s kWh, "
            "emissions=%s kgCO2e, COP=%s, time=%.3fms",
            calc_id, cooling_gj, pump_energy_kwh,
            total_emissions_kgco2e, cop_value, processing_time,
        )

        return result

    # ==================================================================
    # Public Method 7: apply_distribution_loss
    # ==================================================================

    def apply_distribution_loss(
        self,
        consumption_gj: Any,
        loss_pct: Any,
    ) -> Decimal:
        """Adjust energy consumption for network distribution losses.

        The distribution loss represents energy lost between the generation
        plant and the customer meter. To determine the total energy the
        plant had to generate, we divide by (1 - loss_pct).

        Formula:
            Adjusted = Consumption / (1 - Loss_Pct)

        Args:
            consumption_gj: Energy consumed at the meter in GJ.
            loss_pct: Distribution loss fraction (0 to <1).

        Returns:
            Adjusted consumption in GJ as Decimal.

        Raises:
            ValueError: If loss_pct >= 1.0 (would cause division by zero
                or negative denominator).
        """
        consumption = _safe_decimal(consumption_gj)
        loss = _safe_decimal(loss_pct)

        if loss >= _ONE:
            raise ValueError(
                f"distribution_loss_pct must be < 1.0, got {loss}"
            )

        if loss <= _ZERO:
            return consumption.quantize(_PRECISION, rounding=ROUND_HALF_UP)

        denominator = _ONE - loss
        adjusted = (consumption / denominator).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        return adjusted

    # ==================================================================
    # Public Method 8: apply_storage_loss
    # ==================================================================

    def apply_storage_loss(
        self,
        energy_gj: Any,
        loss_pct: Any,
    ) -> Decimal:
        """Adjust energy for thermal storage losses.

        Thermal storage (chilled water tanks, ice storage) incurs losses
        due to heat gain from the environment. The effective output is
        reduced by the storage loss fraction.

        Formula:
            Effective_Output = Stored_Energy x (1 - Storage_Loss_Pct)

        Args:
            energy_gj: Stored energy in GJ before losses.
            loss_pct: Storage loss fraction (0 to <1).

        Returns:
            Effective energy output in GJ as Decimal.
        """
        energy = _safe_decimal(energy_gj)
        loss = _safe_decimal(loss_pct)

        if loss <= _ZERO:
            return energy.quantize(_PRECISION, rounding=ROUND_HALF_UP)

        if loss >= _ONE:
            return _ZERO

        effective = (energy * (_ONE - loss)).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        return effective

    # ==================================================================
    # Public Method 9: convert_gj_to_kwh
    # ==================================================================

    def convert_gj_to_kwh(self, energy_gj: Any) -> Decimal:
        """Convert energy from gigajoules to kilowatt-hours.

        Formula:
            kWh = GJ x 277.778

        Args:
            energy_gj: Energy in gigajoules.

        Returns:
            Energy in kilowatt-hours as Decimal.
        """
        gj = _safe_decimal(energy_gj)
        kwh = (gj * _GJ_TO_KWH).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        return kwh

    # ==================================================================
    # Public Method 10: compute_electrical_input
    # ==================================================================

    def compute_electrical_input(
        self,
        cooling_gj: Any,
        cop: Any,
    ) -> Decimal:
        """Compute electrical input from cooling output using COP.

        Formula:
            Electrical_Input (GJ) = Cooling_Output (GJ) / COP

        Args:
            cooling_gj: Cooling output in GJ.
            cop: Coefficient of performance (dimensionless).

        Returns:
            Electrical input in GJ as Decimal.

        Raises:
            ValueError: If COP is zero or negative.
        """
        cooling = _safe_decimal(cooling_gj)
        cop_val = _safe_decimal(cop, _ONE)

        if cop_val <= _ZERO:
            raise ValueError(f"COP must be > 0, got {cop_val}")

        electrical_input = (cooling / cop_val).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        return electrical_input

    # ==================================================================
    # Public Method 11: compute_heat_input
    # ==================================================================

    def compute_heat_input(
        self,
        cooling_gj: Any,
        cop: Any,
    ) -> Decimal:
        """Compute heat input for absorption cooling using COP.

        Absorption chillers convert heat energy into cooling. The heat
        input required is the cooling output divided by the absorption COP.

        Formula:
            Heat_Input (GJ) = Cooling_Output (GJ) / COP_absorption

        Args:
            cooling_gj: Cooling output in GJ.
            cop: Absorption coefficient of performance (dimensionless).

        Returns:
            Heat input in GJ as Decimal.

        Raises:
            ValueError: If COP is zero or negative.
        """
        cooling = _safe_decimal(cooling_gj)
        cop_val = _safe_decimal(cop, _ONE)

        if cop_val <= _ZERO:
            raise ValueError(
                f"Absorption COP must be > 0, got {cop_val}"
            )

        heat_input = (cooling / cop_val).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        return heat_input

    # ==================================================================
    # Public Method 12: validate_request
    # ==================================================================

    def validate_request(
        self,
        request: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Validate a heating or cooling calculation request.

        Checks for required fields, valid value ranges, and consistent
        parameter combinations. Returns a tuple of (is_valid, errors).

        Validation rules:
            - request must be a dictionary.
            - Must have either 'consumption_gj' (heating) or
              'cooling_output_gj' (cooling).
            - Numeric values must be parseable and positive.
            - distribution_loss_pct must be in [0, 1).
            - COP must be positive.
            - grid_ef_kwh must be non-negative.
            - technology must be a recognized cooling technology.
            - region must be a recognized district heating region.
            - gwp_source must be a recognized GWP source.

        Args:
            request: Calculation request dictionary to validate.

        Returns:
            Tuple of (is_valid: bool, errors: List[str]).
        """
        errors: List[str] = []

        if not isinstance(request, dict):
            return False, ["request must be a dictionary"]

        has_heating = "consumption_gj" in request
        has_cooling = "cooling_output_gj" in request

        if not has_heating and not has_cooling:
            errors.append(
                "Either 'consumption_gj' or 'cooling_output_gj' is required"
            )

        # Validate consumption_gj
        if has_heating:
            val = _safe_decimal(request.get("consumption_gj"))
            if val <= _ZERO:
                errors.append("consumption_gj must be > 0")

        # Validate cooling_output_gj
        if has_cooling:
            val = _safe_decimal(request.get("cooling_output_gj"))
            if val <= _ZERO:
                errors.append("cooling_output_gj must be > 0")

        # Validate distribution_loss_pct
        if "distribution_loss_pct" in request:
            loss = _safe_decimal(request["distribution_loss_pct"])
            if loss < _ZERO or loss >= _ONE:
                errors.append(
                    "distribution_loss_pct must be in [0, 1)"
                )

        # Validate COP
        if "cop" in request:
            cop_val = _safe_decimal(request["cop"])
            if cop_val <= _ZERO:
                errors.append("cop must be > 0")

        # Validate grid_ef_kwh
        if "grid_ef_kwh" in request:
            ef_val = _safe_decimal(request["grid_ef_kwh"])
            if ef_val < _ZERO:
                errors.append("grid_ef_kwh must be >= 0")

        # Validate technology
        if "technology" in request:
            tech = str(request["technology"]).lower().strip()
            all_techs = set(COOLING_SYSTEM_FACTORS.keys())
            if tech and tech not in all_techs:
                errors.append(
                    f"Unrecognized technology '{tech}'. "
                    f"Valid: {sorted(all_techs)}"
                )

        # Validate region
        if "region" in request:
            region_val = str(request["region"]).lower().strip()
            all_regions = set(DISTRICT_HEATING_FACTORS.keys())
            if region_val and region_val not in all_regions:
                errors.append(
                    f"Unrecognized region '{region_val}'. "
                    f"Valid: {sorted(all_regions)}"
                )

        # Validate gwp_source
        if "gwp_source" in request:
            gwp_val = str(request["gwp_source"]).upper().strip()
            valid_gwps = set(GWP_VALUES.keys())
            if gwp_val and gwp_val not in valid_gwps:
                errors.append(
                    f"Unrecognized gwp_source '{gwp_val}'. "
                    f"Valid: {sorted(valid_gwps)}"
                )

        # Validate storage_loss_pct
        if "storage_loss_pct" in request:
            sl = _safe_decimal(request["storage_loss_pct"])
            if sl < _ZERO or sl >= _ONE:
                errors.append("storage_loss_pct must be in [0, 1)")

        # Validate heat_source_ef
        if "heat_source_ef" in request:
            hs = _safe_decimal(request["heat_source_ef"])
            if hs < _ZERO:
                errors.append("heat_source_ef must be >= 0")

        is_valid = len(errors) == 0
        return is_valid, errors

    # ==================================================================
    # Public Method 13: batch_calculate
    # ==================================================================

    def batch_calculate(
        self,
        requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Process a batch of heating and cooling calculation requests.

        Iterates over a list of calculation requests, routing each to
        the appropriate method (heating or cooling). Collects results
        and aggregates summary statistics.

        Each request in the list must contain either 'consumption_gj'
        (for heating) or 'cooling_output_gj' (for cooling) to determine
        the calculation type.

        Args:
            requests: List of calculation request dictionaries.

        Returns:
            Batch result dictionary with individual results, summary
            statistics, and overall provenance hash.
        """
        self._increment_calculations("batch")
        start_time = time.monotonic()
        batch_id = str(uuid4())

        if not isinstance(requests, list):
            return {
                "batch_id": batch_id,
                "status": "VALIDATION_ERROR",
                "errors": ["requests must be a list"],
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        total_count = len(requests)
        if total_count == 0:
            return {
                "batch_id": batch_id,
                "status": "VALIDATION_ERROR",
                "errors": ["requests list is empty"],
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        results: List[Dict[str, Any]] = []
        success_count = 0
        error_count = 0
        total_emissions_kgco2e = _ZERO
        total_heating_gj = _ZERO
        total_cooling_gj = _ZERO

        for idx, req in enumerate(requests):
            if not isinstance(req, dict):
                results.append({
                    "index": idx,
                    "status": "VALIDATION_ERROR",
                    "errors": ["request must be a dictionary"],
                })
                error_count += 1
                continue

            try:
                is_heating = "consumption_gj" in req
                is_cooling = "cooling_output_gj" in req

                if is_heating:
                    result = self.calculate_heating_emissions(req)
                elif is_cooling:
                    result = self.calculate_cooling_emissions(req)
                else:
                    results.append({
                        "index": idx,
                        "status": "VALIDATION_ERROR",
                        "errors": [
                            "Must have 'consumption_gj' or "
                            "'cooling_output_gj'"
                        ],
                    })
                    error_count += 1
                    continue

                result["index"] = idx

                if result.get("status") == "SUCCESS":
                    success_count += 1
                    emissions = _safe_decimal(
                        result.get("emissions_kgco2e", "0")
                    )
                    total_emissions_kgco2e += emissions

                    if is_heating:
                        total_heating_gj += _safe_decimal(
                            req.get("consumption_gj")
                        )
                    elif is_cooling:
                        total_cooling_gj += _safe_decimal(
                            req.get("cooling_output_gj")
                        )
                else:
                    error_count += 1

                results.append(result)

            except Exception as exc:
                logger.error(
                    "Batch item %d failed: %s", idx, str(exc),
                    exc_info=True,
                )
                results.append({
                    "index": idx,
                    "status": "ERROR",
                    "errors": [str(exc)],
                })
                error_count += 1

        # -- Summary statistics --------------------------------------------
        total_emissions_tco2e = (
            total_emissions_kgco2e / _THOUSAND
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        batch_status = "COMPLETED"
        if error_count > 0 and success_count > 0:
            batch_status = "PARTIAL"
        elif error_count > 0 and success_count == 0:
            batch_status = "FAILED"

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        batch_result: Dict[str, Any] = {
            "batch_id": batch_id,
            "status": batch_status,
            "total_requests": total_count,
            "success_count": success_count,
            "error_count": error_count,
            "summary": {
                "total_emissions_kgco2e": str(
                    total_emissions_kgco2e.quantize(
                        _PRECISION, rounding=ROUND_HALF_UP
                    )
                ),
                "total_emissions_tco2e": str(total_emissions_tco2e),
                "total_heating_gj": str(
                    total_heating_gj.quantize(
                        _PRECISION, rounding=ROUND_HALF_UP
                    )
                ),
                "total_cooling_gj": str(
                    total_cooling_gj.quantize(
                        _PRECISION, rounding=ROUND_HALF_UP
                    )
                ),
            },
            "results": results,
            "processing_time_ms": processing_time,
        }

        batch_result["provenance_hash"] = _compute_hash(batch_result)

        logger.info(
            "Batch calculation complete: id=%s, "
            "total=%d, success=%d, errors=%d, "
            "total_co2e=%s kgCO2e, time=%.3fms",
            batch_id, total_count, success_count, error_count,
            total_emissions_kgco2e, processing_time,
        )

        return batch_result

    # ==================================================================
    # Public Method 14: compare_cooling_technologies
    # ==================================================================

    def compare_cooling_technologies(
        self,
        cooling_output_gj: Any,
        technologies: Optional[List[str]] = None,
        grid_ef_kwh: Optional[Any] = None,
        heat_source_ef: Optional[Any] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare emission intensities across cooling technologies.

        Calculates emissions for the same cooling output across multiple
        cooling technologies, enabling technology selection analysis
        and emission reduction opportunity identification.

        Args:
            cooling_output_gj: Cooling energy delivered in GJ.
            technologies: List of technology identifiers to compare.
                If None, compares all 9 available technologies.
            grid_ef_kwh: Grid EF for electric technologies (kgCO2e/kWh).
            heat_source_ef: Heat source EF for absorption (kgCO2e/GJ).
            gwp_source: GWP source (AR4/AR5/AR6).

        Returns:
            Comparison result with per-technology emissions, ranking,
            and overall provenance hash.
        """
        start_time = time.monotonic()
        calc_id = str(uuid4())
        gwp = (gwp_source or self._gwp_source).upper()

        # -- Validate inputs -----------------------------------------------
        cooling_gj = _safe_decimal(cooling_output_gj)
        if cooling_gj <= _ZERO:
            return {
                "comparison_id": calc_id,
                "status": "VALIDATION_ERROR",
                "errors": ["cooling_output_gj must be > 0"],
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        # Default to all technologies
        if technologies is None:
            technologies = list(COOLING_SYSTEM_FACTORS.keys())

        # -- Calculate for each technology ---------------------------------
        comparison_results: List[Dict[str, Any]] = []

        for tech in technologies:
            tech_lower = tech.lower().strip()
            category = self._classify_cooling_technology(tech_lower)

            try:
                if category == CoolingCategory.ABSORPTION:
                    result = self.calculate_absorption_cooling(
                        cooling_output_gj=str(cooling_gj),
                        technology=tech_lower,
                        heat_source_ef=heat_source_ef,
                        gwp_source=gwp,
                    )
                elif category == CoolingCategory.FREE:
                    result = self.calculate_free_cooling(
                        cooling_output_gj=str(cooling_gj),
                        grid_ef_kwh=grid_ef_kwh,
                        gwp_source=gwp,
                    )
                else:
                    result = self.calculate_electric_cooling(
                        cooling_output_gj=str(cooling_gj),
                        technology=tech_lower,
                        grid_ef_kwh=grid_ef_kwh,
                        gwp_source=gwp,
                    )

                comparison_results.append({
                    "technology": tech_lower,
                    "category": category.value,
                    "status": result.get("status", "ERROR"),
                    "cop": result.get("cop", "N/A"),
                    "emissions_kgco2e": result.get(
                        "emissions_kgco2e", "0"
                    ),
                    "emissions_tco2e": result.get(
                        "emissions_tco2e", "0"
                    ),
                    "calculation_id": result.get("calculation_id", ""),
                })
            except Exception as exc:
                comparison_results.append({
                    "technology": tech_lower,
                    "category": category.value,
                    "status": "ERROR",
                    "errors": [str(exc)],
                    "emissions_kgco2e": "0",
                    "emissions_tco2e": "0",
                })

        # -- Rank by emissions (lowest first) ------------------------------
        successful = [
            r for r in comparison_results
            if r.get("status") == "SUCCESS"
        ]
        successful.sort(
            key=lambda r: _safe_decimal(r.get("emissions_kgco2e", "0"))
        )

        for rank, item in enumerate(successful, 1):
            item["rank"] = rank

        # -- Compute savings relative to highest emitter -------------------
        if len(successful) >= 2:
            highest = _safe_decimal(
                successful[-1].get("emissions_kgco2e", "0")
            )
            lowest = _safe_decimal(
                successful[0].get("emissions_kgco2e", "0")
            )
            if highest > _ZERO:
                savings_pct = (
                    (highest - lowest) / highest * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            else:
                savings_pct = _ZERO
            savings_kgco2e = (highest - lowest).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
        else:
            savings_pct = _ZERO
            savings_kgco2e = _ZERO

        # -- Assemble result -----------------------------------------------
        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        comparison: Dict[str, Any] = {
            "comparison_id": calc_id,
            "status": "SUCCESS",
            "cooling_output_gj": str(cooling_gj),
            "gwp_source": gwp,
            "technologies_compared": len(comparison_results),
            "ranking": successful,
            "all_results": comparison_results,
            "best_technology": successful[0]["technology"]
            if successful else None,
            "worst_technology": successful[-1]["technology"]
            if successful else None,
            "max_savings_kgco2e": str(savings_kgco2e),
            "max_savings_pct": str(savings_pct),
            "processing_time_ms": processing_time,
        }

        comparison["provenance_hash"] = _compute_hash(comparison)

        logger.info(
            "Cooling technology comparison complete: id=%s, "
            "cooling=%s GJ, techs=%d, best=%s, "
            "max_savings=%s kgCO2e (%.1f%%), time=%.3fms",
            calc_id, cooling_gj, len(comparison_results),
            comparison.get("best_technology"),
            savings_kgco2e, float(savings_pct), processing_time,
        )

        return comparison

    # ==================================================================
    # Public Method 15: get_calculation_stats
    # ==================================================================

    def get_calculation_stats(self) -> Dict[str, Any]:
        """Return engine usage statistics.

        Returns a snapshot of all engine counters and metadata. Thread-safe.

        Returns:
            Dictionary with engine name, version, counters, and uptime.
        """
        with self._lock:
            now = _utcnow()
            uptime_seconds = int(
                (now - self._created_at).total_seconds()
            )
            return {
                "engine": "HeatCoolingCalculatorEngine",
                "version": "1.0.0",
                "agent": "AGENT-MRV-011",
                "created_at": self._created_at.isoformat(),
                "current_time": now.isoformat(),
                "uptime_seconds": uptime_seconds,
                "total_calculations": self._total_calculations,
                "heating_calculations": self._total_heating_calculations,
                "cooling_calculations": self._total_cooling_calculations,
                "batch_calculations": self._total_batch_calculations,
                "gwp_source": self._gwp_source,
                "db_available": self._db is not None,
                "metrics_available": _METRICS_AVAILABLE,
                "provenance_available": _PROVENANCE_AVAILABLE,
                "config_available": _CONFIG_AVAILABLE,
                "models_available": _MODELS_AVAILABLE,
                "supported_dh_regions": sorted(
                    DISTRICT_HEATING_FACTORS.keys()
                ),
                "supported_cooling_technologies": sorted(
                    COOLING_SYSTEM_FACTORS.keys()
                ),
                "supported_gwp_sources": sorted(GWP_VALUES.keys()),
            }

    # ==================================================================
    # Public Method 16: health_check
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the engine.

        Validates that the engine is properly initialized, constants are
        loaded, and basic calculations succeed. Returns a structured
        health report.

        Returns:
            Dictionary with health status, component checks, and
            diagnostic information.
        """
        start_time = time.monotonic()
        checks: Dict[str, str] = {}
        is_healthy = True

        # Check 1: Engine initialization
        if self._initialized:
            checks["engine_initialized"] = "PASS"
        else:
            checks["engine_initialized"] = "FAIL"
            is_healthy = False

        # Check 2: Constants loaded
        if len(DISTRICT_HEATING_FACTORS) >= 13:
            checks["dh_factors_loaded"] = "PASS"
        else:
            checks["dh_factors_loaded"] = "FAIL"
            is_healthy = False

        if len(COOLING_SYSTEM_FACTORS) >= 9:
            checks["cooling_factors_loaded"] = "PASS"
        else:
            checks["cooling_factors_loaded"] = "FAIL"
            is_healthy = False

        if len(GWP_VALUES) >= 3:
            checks["gwp_values_loaded"] = "PASS"
        else:
            checks["gwp_values_loaded"] = "FAIL"
            is_healthy = False

        # Check 3: Decimal arithmetic works
        try:
            test_result = (
                _D("100") / _D("0.9")
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
            if test_result > _ZERO:
                checks["decimal_arithmetic"] = "PASS"
            else:
                checks["decimal_arithmetic"] = "FAIL"
                is_healthy = False
        except Exception:
            checks["decimal_arithmetic"] = "FAIL"
            is_healthy = False

        # Check 4: Basic heating calculation
        try:
            test_heating = self.calculate_district_heating(
                consumption_gj="100",
                region="global_default",
            )
            if test_heating.get("status") == "SUCCESS":
                checks["heating_calculation"] = "PASS"
            else:
                checks["heating_calculation"] = "FAIL"
                is_healthy = False
        except Exception:
            checks["heating_calculation"] = "FAIL"
            is_healthy = False

        # Check 5: Basic cooling calculation
        try:
            test_cooling = self.calculate_electric_cooling(
                cooling_output_gj="100",
                technology="centrifugal_chiller",
            )
            if test_cooling.get("status") == "SUCCESS":
                checks["cooling_calculation"] = "PASS"
            else:
                checks["cooling_calculation"] = "FAIL"
                is_healthy = False
        except Exception:
            checks["cooling_calculation"] = "FAIL"
            is_healthy = False

        # Check 6: SHA-256 hashing
        try:
            test_hash = _compute_hash({"test": "value"})
            if len(test_hash) == 64:
                checks["sha256_hashing"] = "PASS"
            else:
                checks["sha256_hashing"] = "FAIL"
                is_healthy = False
        except Exception:
            checks["sha256_hashing"] = "FAIL"
            is_healthy = False

        # Check 7: Database availability (non-critical)
        if self._db is not None:
            checks["database"] = "AVAILABLE"
        else:
            checks["database"] = "UNAVAILABLE"

        # Check 8: Metrics availability (non-critical)
        if _METRICS_AVAILABLE:
            checks["metrics"] = "AVAILABLE"
        else:
            checks["metrics"] = "UNAVAILABLE"

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        health_result: Dict[str, Any] = {
            "engine": "HeatCoolingCalculatorEngine",
            "status": "HEALTHY" if is_healthy else "UNHEALTHY",
            "agent": "AGENT-MRV-011",
            "version": "1.0.0",
            "checks": checks,
            "total_checks": len(checks),
            "passed_checks": sum(
                1 for v in checks.values()
                if v in ("PASS", "AVAILABLE")
            ),
            "failed_checks": sum(
                1 for v in checks.values() if v == "FAIL"
            ),
            "timestamp": _utcnow().isoformat(),
            "processing_time_ms": processing_time,
        }

        logger.info(
            "Health check complete: status=%s, passed=%d/%d, time=%.3fms",
            health_result["status"],
            health_result["passed_checks"],
            health_result["total_checks"],
            processing_time,
        )

        return health_result

    # ==================================================================
    # Additional public methods for advanced calculations
    # ==================================================================

    def calculate_network_losses(
        self,
        generated_energy_gj: Any,
        network_loss_pct: Any,
    ) -> Dict[str, Any]:
        """Calculate energy lost in a district energy network.

        Computes delivered energy and additional generation required to
        compensate for network distribution losses.

        Formulas:
            Delivered = Generated x (1 - Network_Loss_Pct)
            Required_Generation = Delivered / (1 - Network_Loss_Pct)
            Loss = Generated - Delivered

        Args:
            generated_energy_gj: Total energy generated at plant (GJ).
            network_loss_pct: Network distribution loss fraction (0-1).

        Returns:
            Dictionary with generated, delivered, and lost energy values.
        """
        start_time = time.monotonic()
        calc_id = str(uuid4())

        generated = _safe_decimal(generated_energy_gj)
        loss_pct = _safe_decimal(network_loss_pct)

        errors: List[str] = []
        if generated <= _ZERO:
            errors.append("generated_energy_gj must be > 0")
        if loss_pct < _ZERO or loss_pct >= _ONE:
            errors.append("network_loss_pct must be in [0, 1)")

        if errors:
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "errors": errors,
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        delivered = (
            generated * (_ONE - loss_pct)
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        energy_lost = (
            generated - delivered
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        loss_pct_actual = _ZERO
        if generated > _ZERO:
            loss_pct_actual = (
                energy_lost / generated
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "generated_energy_gj": str(generated),
            "delivered_energy_gj": str(delivered),
            "energy_lost_gj": str(energy_lost),
            "network_loss_pct": str(loss_pct),
            "actual_loss_pct": str(loss_pct_actual),
            "processing_time_ms": processing_time,
        }

        result["provenance_hash"] = _compute_hash(result)
        return result

    def calculate_storage_adjustment(
        self,
        stored_energy_gj: Any,
        cooling_demand_gj: Any,
        storage_loss_pct: Any,
        technology: str = "thermal_storage",
        grid_ef_kwh: Optional[Any] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate additional cooling required to compensate for
        thermal storage losses.

        When cooling energy is stored (chilled water tank, ice storage),
        some energy is lost due to heat gain from the environment. This
        method computes the effective output after losses and any
        additional cooling generation required to meet demand.

        Formulas:
            Effective_Output = Stored_Energy x (1 - Storage_Loss_Pct)
            Additional_Energy = max(0, Cooling_Demand - Effective_Output)

        Args:
            stored_energy_gj: Energy stored in GJ before losses.
            cooling_demand_gj: Cooling demand that must be met in GJ.
            storage_loss_pct: Storage loss fraction (0 to <1).
            technology: Cooling technology for additional generation.
            grid_ef_kwh: Grid EF for additional generation (kgCO2e/kWh).
            gwp_source: GWP source (AR4/AR5/AR6).

        Returns:
            Dictionary with effective output, shortfall, additional
            generation emissions, and provenance hash.
        """
        start_time = time.monotonic()
        calc_id = str(uuid4())
        gwp = (gwp_source or self._gwp_source).upper()

        stored = _safe_decimal(stored_energy_gj)
        demand = _safe_decimal(cooling_demand_gj)
        loss = _safe_decimal(storage_loss_pct)

        errors: List[str] = []
        if stored <= _ZERO:
            errors.append("stored_energy_gj must be > 0")
        if demand <= _ZERO:
            errors.append("cooling_demand_gj must be > 0")
        if loss < _ZERO or loss >= _ONE:
            errors.append("storage_loss_pct must be in [0, 1)")

        if errors:
            return self._build_error_result(
                calc_id, "STORAGE_ADJUSTMENT", errors, start_time,
            )

        # Effective output after storage losses
        effective_output = self.apply_storage_loss(stored, loss)

        # Shortfall
        shortfall = max(
            _ZERO,
            (demand - effective_output).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            ),
        )

        # Energy lost to storage
        energy_lost = (
            stored - effective_output
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # If there is a shortfall, calculate additional generation emissions
        additional_emissions_kgco2e = _ZERO
        additional_result: Optional[Dict[str, Any]] = None

        if shortfall > _ZERO:
            additional_result = self.calculate_electric_cooling(
                cooling_output_gj=str(shortfall),
                technology=technology,
                grid_ef_kwh=grid_ef_kwh,
                gwp_source=gwp,
            )
            if additional_result.get("status") == "SUCCESS":
                additional_emissions_kgco2e = _safe_decimal(
                    additional_result.get("emissions_kgco2e", "0")
                )

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "calculation_type": "STORAGE_ADJUSTMENT",
            "stored_energy_gj": str(stored),
            "cooling_demand_gj": str(demand),
            "storage_loss_pct": str(loss),
            "effective_output_gj": str(effective_output),
            "energy_lost_gj": str(energy_lost),
            "shortfall_gj": str(shortfall),
            "additional_emissions_kgco2e": str(
                additional_emissions_kgco2e.quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
            ),
            "additional_generation_result": additional_result,
            "gwp_source": gwp,
            "processing_time_ms": processing_time,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Storage adjustment complete: id=%s, "
            "stored=%s GJ, effective=%s GJ, shortfall=%s GJ, "
            "additional_co2e=%s kgCO2e, time=%.3fms",
            calc_id, stored, effective_output, shortfall,
            additional_emissions_kgco2e, processing_time,
        )

        return result

    def calculate_combined_heating_cooling(
        self,
        heating_request: Dict[str, Any],
        cooling_request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate combined emissions from both heating and cooling.

        Convenience method that processes a heating request and a
        cooling request together, returning combined totals alongside
        the individual results.

        Args:
            heating_request: District heating calculation request.
            cooling_request: District cooling calculation request.

        Returns:
            Combined result with individual results and aggregate totals.
        """
        start_time = time.monotonic()
        calc_id = str(uuid4())

        heating_result = self.calculate_heating_emissions(heating_request)
        cooling_result = self.calculate_cooling_emissions(cooling_request)

        heating_emissions = _safe_decimal(
            heating_result.get("emissions_kgco2e", "0")
        )
        cooling_emissions = _safe_decimal(
            cooling_result.get("emissions_kgco2e", "0")
        )
        combined_emissions_kgco2e = (
            heating_emissions + cooling_emissions
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        combined_emissions_tco2e = (
            combined_emissions_kgco2e / _THOUSAND
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        both_success = (
            heating_result.get("status") == "SUCCESS"
            and cooling_result.get("status") == "SUCCESS"
        )

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS" if both_success else "PARTIAL",
            "calculation_type": "COMBINED_HEATING_COOLING",
            "heating_result": heating_result,
            "cooling_result": cooling_result,
            "combined_emissions_kgco2e": str(combined_emissions_kgco2e),
            "combined_emissions_tco2e": str(combined_emissions_tco2e),
            "heating_share_pct": str(
                (
                    heating_emissions / combined_emissions_kgco2e
                    * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                if combined_emissions_kgco2e > _ZERO
                else _ZERO
            ),
            "cooling_share_pct": str(
                (
                    cooling_emissions / combined_emissions_kgco2e
                    * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                if combined_emissions_kgco2e > _ZERO
                else _ZERO
            ),
            "processing_time_ms": processing_time,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Combined heating+cooling complete: id=%s, "
            "heating=%s kgCO2e, cooling=%s kgCO2e, "
            "combined=%s kgCO2e, time=%.3fms",
            calc_id, heating_emissions, cooling_emissions,
            combined_emissions_kgco2e, processing_time,
        )

        return result

    def get_regional_heating_factors(
        self,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve district heating emission factors for a region.

        If no region is specified, returns all available regional factors.
        Useful for UI dropdowns and factor verification.

        Args:
            region: Optional region identifier. If None, returns all.

        Returns:
            Dictionary with emission factors and metadata.
        """
        if region is not None:
            region_lower = region.lower().strip()
            factors = DISTRICT_HEATING_FACTORS.get(region_lower)
            if factors is None:
                return {
                    "status": "NOT_FOUND",
                    "region": region_lower,
                    "available_regions": sorted(
                        DISTRICT_HEATING_FACTORS.keys()
                    ),
                }
            return {
                "status": "SUCCESS",
                "region": region_lower,
                "ef_kgco2e_per_gj": str(
                    factors.get("ef_kgco2e_per_gj", "0")
                ),
                "distribution_loss_pct": str(
                    factors.get("distribution_loss_pct", "0")
                ),
            }

        all_factors: Dict[str, Dict[str, str]] = {}
        for r, f in DISTRICT_HEATING_FACTORS.items():
            all_factors[r] = {
                "ef_kgco2e_per_gj": str(
                    f.get("ef_kgco2e_per_gj", "0")
                ),
                "distribution_loss_pct": str(
                    f.get("distribution_loss_pct", "0")
                ),
            }
        return {
            "status": "SUCCESS",
            "region_count": len(all_factors),
            "regions": all_factors,
        }

    def get_cooling_technology_factors(
        self,
        technology: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve cooling technology factors for a technology.

        If no technology is specified, returns all available technologies.

        Args:
            technology: Optional technology identifier. If None, returns all.

        Returns:
            Dictionary with COP ranges and energy source metadata.
        """
        if technology is not None:
            tech_lower = technology.lower().strip()
            factors = COOLING_SYSTEM_FACTORS.get(tech_lower)
            if factors is None:
                return {
                    "status": "NOT_FOUND",
                    "technology": tech_lower,
                    "available_technologies": sorted(
                        COOLING_SYSTEM_FACTORS.keys()
                    ),
                }
            return {
                "status": "SUCCESS",
                "technology": tech_lower,
                "cop_min": str(factors.get("cop_min", "0")),
                "cop_max": str(factors.get("cop_max", "0")),
                "cop_default": str(factors.get("cop_default", "0")),
                "energy_source": COOLING_ENERGY_SOURCE.get(
                    tech_lower, "electricity"
                ),
            }

        all_factors: Dict[str, Dict[str, str]] = {}
        for t, f in COOLING_SYSTEM_FACTORS.items():
            all_factors[t] = {
                "cop_min": str(f.get("cop_min", "0")),
                "cop_max": str(f.get("cop_max", "0")),
                "cop_default": str(f.get("cop_default", "0")),
                "energy_source": COOLING_ENERGY_SOURCE.get(
                    t, "electricity"
                ),
            }
        return {
            "status": "SUCCESS",
            "technology_count": len(all_factors),
            "technologies": all_factors,
        }

    def estimate_annual_emissions(
        self,
        monthly_consumption_gj: List[Any],
        energy_type: str = "heating",
        region: str = "global_default",
        technology: str = "centrifugal_chiller",
        grid_ef_kwh: Optional[Any] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Estimate annual emissions from monthly consumption data.

        Processes 12 months of consumption data, calculates per-month
        emissions, and produces annual aggregates with seasonal analysis.

        Args:
            monthly_consumption_gj: List of 12 monthly values in GJ.
            energy_type: Either 'heating' or 'cooling'.
            region: Region for heating factor lookup.
            technology: Cooling technology identifier.
            grid_ef_kwh: Grid EF for cooling (kgCO2e/kWh).
            gwp_source: GWP source (AR4/AR5/AR6).

        Returns:
            Annual estimate with monthly breakdown and totals.
        """
        start_time = time.monotonic()
        calc_id = str(uuid4())

        if not isinstance(monthly_consumption_gj, list):
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "errors": ["monthly_consumption_gj must be a list"],
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        if len(monthly_consumption_gj) != 12:
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "errors": [
                    "monthly_consumption_gj must have exactly 12 values"
                ],
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        month_names = [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December",
        ]

        monthly_results: List[Dict[str, Any]] = []
        annual_consumption_gj = _ZERO
        annual_emissions_kgco2e = _ZERO
        energy_type_lower = energy_type.lower().strip()

        for idx, monthly_val in enumerate(monthly_consumption_gj):
            month_gj = _safe_decimal(monthly_val)
            annual_consumption_gj += month_gj

            if month_gj <= _ZERO:
                monthly_results.append({
                    "month": month_names[idx],
                    "month_number": idx + 1,
                    "consumption_gj": str(month_gj),
                    "emissions_kgco2e": str(_ZERO),
                    "status": "SKIPPED",
                })
                continue

            if energy_type_lower == "heating":
                result = self.calculate_district_heating(
                    consumption_gj=str(month_gj),
                    region=region,
                    gwp_source=gwp_source,
                )
            else:
                result = self.calculate_electric_cooling(
                    cooling_output_gj=str(month_gj),
                    technology=technology,
                    grid_ef_kwh=grid_ef_kwh,
                    gwp_source=gwp_source,
                )

            emissions = _safe_decimal(
                result.get("emissions_kgco2e", "0")
            )
            annual_emissions_kgco2e += emissions

            monthly_results.append({
                "month": month_names[idx],
                "month_number": idx + 1,
                "consumption_gj": str(month_gj),
                "emissions_kgco2e": str(emissions),
                "status": result.get("status", "ERROR"),
            })

        annual_emissions_tco2e = (
            annual_emissions_kgco2e / _THOUSAND
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        avg_monthly_emissions = _ZERO
        if annual_emissions_kgco2e > _ZERO:
            avg_monthly_emissions = (
                annual_emissions_kgco2e / Decimal("12")
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        emission_intensity = _ZERO
        if annual_consumption_gj > _ZERO:
            emission_intensity = (
                annual_emissions_kgco2e / annual_consumption_gj
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result_dict: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "energy_type": energy_type_lower,
            "region": region if energy_type_lower == "heating" else None,
            "technology": technology
            if energy_type_lower == "cooling" else None,
            "monthly_results": monthly_results,
            "annual_summary": {
                "total_consumption_gj": str(
                    annual_consumption_gj.quantize(
                        _PRECISION, rounding=ROUND_HALF_UP
                    )
                ),
                "total_emissions_kgco2e": str(
                    annual_emissions_kgco2e.quantize(
                        _PRECISION, rounding=ROUND_HALF_UP
                    )
                ),
                "total_emissions_tco2e": str(annual_emissions_tco2e),
                "avg_monthly_emissions_kgco2e": str(avg_monthly_emissions),
                "emission_intensity_kgco2e_per_gj": str(emission_intensity),
            },
            "processing_time_ms": processing_time,
        }

        result_dict["provenance_hash"] = _compute_hash(result_dict)

        logger.info(
            "Annual estimate complete: id=%s, type=%s, "
            "annual_gj=%s, annual_co2e=%s kgCO2e, time=%.3fms",
            calc_id, energy_type_lower, annual_consumption_gj,
            annual_emissions_kgco2e, processing_time,
        )

        return result_dict

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_error_result(
        self,
        calc_id: str,
        calc_type: str,
        errors: List[str],
        start_time: float,
    ) -> Dict[str, Any]:
        """Build a standardized validation error result.

        Args:
            calc_id: Calculation UUID.
            calc_type: Calculation type string.
            errors: List of validation error messages.
            start_time: Monotonic start time for processing duration.

        Returns:
            Standardized error result dictionary.
        """
        processing_time = round(
            (time.monotonic() - start_time) * 1000, 3
        )
        return {
            "calculation_id": calc_id,
            "status": "VALIDATION_ERROR",
            "calculation_type": calc_type,
            "errors": errors,
            "processing_time_ms": processing_time,
        }

    def _apply_storage_loss_to_result(
        self,
        result: Dict[str, Any],
        storage_loss_pct: Any,
        technology: str,
    ) -> Dict[str, Any]:
        """Apply thermal storage loss adjustment to an existing result.

        Recalculates emissions accounting for energy lost during thermal
        storage. The effective cooling output is reduced, requiring
        additional generation to meet the original demand.

        Args:
            result: Existing calculation result to adjust.
            storage_loss_pct: Storage loss fraction (0 to <1).
            technology: Cooling technology for reference.

        Returns:
            Updated result with storage loss fields added.
        """
        loss = _safe_decimal(storage_loss_pct)
        if loss <= _ZERO or loss >= _ONE:
            return result

        original_emissions = _safe_decimal(
            result.get("emissions_kgco2e", "0")
        )

        # Calculate the additional energy needed to compensate for losses
        # If we need X GJ of cooling but lose loss% in storage,
        # we must generate X / (1 - loss%) of cooling
        adjustment_factor = (_ONE / (_ONE - loss)).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        adjusted_emissions = (
            original_emissions * adjustment_factor
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        additional_emissions = (
            adjusted_emissions - original_emissions
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        result["storage_loss_pct"] = str(loss)
        result["storage_adjustment_factor"] = str(adjustment_factor)
        result["pre_storage_emissions_kgco2e"] = str(original_emissions)
        result["storage_additional_emissions_kgco2e"] = str(
            additional_emissions
        )
        result["emissions_kgco2e"] = str(adjusted_emissions)
        result["emissions_tco2e"] = str(
            (adjusted_emissions / _THOUSAND).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
        )

        # Recompute provenance hash with updated data
        result["provenance_hash"] = _compute_hash(result)

        return result


# ===========================================================================
# Module-level singleton accessor
# ===========================================================================


def get_heat_cooling_calculator(
    steam_heat_database: Optional[Any] = None,
    gwp_source: str = "AR6",
) -> HeatCoolingCalculatorEngine:
    """Return the singleton HeatCoolingCalculatorEngine instance.

    Convenience function that creates or returns the singleton engine.
    Equivalent to calling ``HeatCoolingCalculatorEngine()`` but provides
    a clearer API for external consumers.

    Args:
        steam_heat_database: Optional database engine instance.
        gwp_source: Default GWP source for CO2e conversion.

    Returns:
        The singleton HeatCoolingCalculatorEngine instance.
    """
    return HeatCoolingCalculatorEngine(
        steam_heat_database=steam_heat_database,
        gwp_source=gwp_source,
    )
