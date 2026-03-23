# -*- coding: utf-8 -*-
"""
CHPAllocationEngine - Engine 4: Steam/Heat Purchase Agent (AGENT-MRV-011)

Allocates combined heat and power (CHP) / cogeneration plant emissions between
thermal (heat/steam) and electrical (power) outputs using GHG Protocol-compliant
allocation methods. CHP plants produce both electricity and useful heat from a
single fuel input; the total fuel combustion emissions must be apportioned to
each output for accurate Scope 2 reporting by the energy purchaser.

Core Capabilities:
    - Efficiency-based allocation (GHG Protocol recommended method)
    - Energy-based allocation (simple energy content ratio)
    - Exergy-based allocation (Carnot thermodynamic quality weighting)
    - Multi-product allocation (heat + power + cooling three-way split)
    - Total CHP fuel emission computation (CO2, CH4, N2O with GWP)
    - Carnot factor calculation for exergy analysis
    - Primary Energy Savings (PES) per EU Energy Efficiency Directive
    - High-efficiency CHP determination (EU EED 2012/27/EU)
    - Cross-method comparison analysis
    - Default CHP efficiency lookup by fuel type
    - Overall CHP efficiency computation
    - Batch allocation across multiple CHP plants
    - Request validation with detailed error reporting
    - SHA-256 provenance hashing on every allocation result
    - Thread-safe singleton pattern with RLock

Zero-Hallucination Guarantees:
    - All arithmetic uses Python ``Decimal`` with ``ROUND_HALF_UP``
    - No LLM calls in any calculation path
    - Every result carries a SHA-256 provenance hash
    - Deterministic: identical inputs produce identical outputs (bit-perfect)
    - Thread-safe via ``threading.RLock``

Formula References:
    Efficiency Method (GHG Protocol recommended):
        Heat_Fuel_Equiv  = Heat_Output_GJ / eta_thermal
        Power_Fuel_Equiv = Power_Output_GJ / eta_electrical
        Total_Fuel_Equiv = Heat_Fuel_Equiv + Power_Fuel_Equiv
        Heat_Share       = Heat_Fuel_Equiv / Total_Fuel_Equiv
        Power_Share      = Power_Fuel_Equiv / Total_Fuel_Equiv

    Energy Method (simple allocation):
        Total_Output = Heat_Output_GJ + Power_Output_GJ
        Heat_Share   = Heat_Output_GJ / Total_Output
        Power_Share  = Power_Output_GJ / Total_Output

    Exergy Method (Carnot-based):
        Carnot_Factor = 1 - (T_ambient + 273.15) / (T_steam + 273.15)
        Exergy_Heat   = Heat_Output_GJ * Carnot_Factor
        Exergy_Power  = Power_Output_GJ  (power is pure exergy)
        Total_Exergy  = Exergy_Heat + Exergy_Power
        Heat_Share    = Exergy_Heat / Total_Exergy
        Power_Share   = Exergy_Power / Total_Exergy

    Primary Energy Savings (EU EED 2012/27/EU Annex II):
        PES = 1 - 1 / ((eta_elec / eta_ref_elec) + (eta_therm / eta_ref_therm))
        High-efficiency CHP: PES > 10% (large >= 1MW) or PES > 0% (small < 1MW)

    Multi-product CHP (heat + power + cooling):
        Cooling_Fuel_Equiv = Cooling_Output_GJ / eta_cooling
        Three-way allocation: heat_share + power_share + cooling_share = 1.0

    Total Fuel Emissions:
        CO2_kg  = Fuel_Input_GJ * CO2_EF_per_GJ
        CH4_kg  = Fuel_Input_GJ * CH4_EF_per_GJ
        N2O_kg  = Fuel_Input_GJ * N2O_EF_per_GJ
        CO2e_kg = CO2_kg + (CH4_kg * GWP_CH4) + (N2O_kg * GWP_N2O)

Reference Efficiencies (EU EED 2012/27/EU Annex II):
    eta_ref_electrical: 0.525 (gas turbine), 0.442 (steam turbine),
                        0.532 (combined cycle)
    eta_ref_thermal:    0.90

Supported Disclosure Frameworks:
    - GHG Protocol Scope 2 Guidance (2015) - CHP allocation
    - GHG Protocol Corporate Standard (Ch. 7)
    - ISO 14064-1:2018 (Category 2)
    - EU Energy Efficiency Directive 2012/27/EU
    - EPA CHP Partnership methodology
    - CSRD / ESRS E1 (Energy-related)
    - CDP Climate Change Questionnaire

Example:
    >>> from greenlang.agents.mrv.steam_heat_purchase.chp_allocation import CHPAllocationEngine
    >>> from decimal import Decimal
    >>> engine = CHPAllocationEngine()
    >>> result = engine.allocate_efficiency_method(
    ...     total_fuel_gj=Decimal("1000"),
    ...     fuel_type="natural_gas",
    ...     heat_output_gj=Decimal("400"),
    ...     power_output_gj=Decimal("350"),
    ...     eta_thermal=Decimal("0.45"),
    ...     eta_electrical=Decimal("0.35"),
    ... )
    >>> assert result["heat_share"] > Decimal("0")
    >>> assert result["power_share"] > Decimal("0")
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
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.steam_heat_purchase.metrics import get_metrics as _get_metrics
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _get_metrics = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.steam_heat_purchase.steam_heat_database import SteamHeatDatabaseEngine
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    SteamHeatDatabaseEngine = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Decimal precision constants
# ---------------------------------------------------------------------------

#: 8 decimal places for deterministic CHP allocation arithmetic.
_PRECISION = Decimal("0.00000001")

#: 6 decimal places for percentage outputs.
_PCT_PRECISION = Decimal("0.000001")

#: Kelvin offset for Celsius to Kelvin conversion.
_KELVIN_OFFSET = Decimal("273.15")

#: Decimal zero constant.
_ZERO = Decimal("0")

#: Decimal one constant.
_ONE = Decimal("1")

#: Decimal one hundred constant.
_HUNDRED = Decimal("100")


# ---------------------------------------------------------------------------
# Built-in Constants: GWP Values
# ---------------------------------------------------------------------------

#: IPCC Global Warming Potential values by assessment report.
#: Source: IPCC AR4 (2007), AR5 (2014), AR6 (2021).
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
        "CH4": Decimal("27.9"),
        "N2O": Decimal("273"),
    },
    "AR6_20YR": {
        "CO2": Decimal("1"),
        "CH4": Decimal("81.2"),
        "N2O": Decimal("273"),
    },
}


# ---------------------------------------------------------------------------
# Built-in Constants: Fuel Emission Factors
# ---------------------------------------------------------------------------

#: Emission factors by fuel type for CHP plant fuel input.
#: Units: kgCO2, kgCH4, kgN2O per GJ of fuel input (HHV basis).
#: Source: IPCC 2006 Vol 2 Ch 2 Table 2.2/2.4, US EPA AP-42.
FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "natural_gas": {
        "co2_ef": Decimal("56.100"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0001"),
        "default_efficiency": Decimal("0.85"),
        "is_biogenic": Decimal("0"),
    },
    "fuel_oil_2": {
        "co2_ef": Decimal("74.100"),
        "ch4_ef": Decimal("0.003"),
        "n2o_ef": Decimal("0.0006"),
        "default_efficiency": Decimal("0.82"),
        "is_biogenic": Decimal("0"),
    },
    "fuel_oil_6": {
        "co2_ef": Decimal("77.400"),
        "ch4_ef": Decimal("0.003"),
        "n2o_ef": Decimal("0.0006"),
        "default_efficiency": Decimal("0.80"),
        "is_biogenic": Decimal("0"),
    },
    "coal_bituminous": {
        "co2_ef": Decimal("94.600"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0015"),
        "default_efficiency": Decimal("0.78"),
        "is_biogenic": Decimal("0"),
    },
    "coal_subbituminous": {
        "co2_ef": Decimal("96.100"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0015"),
        "default_efficiency": Decimal("0.75"),
        "is_biogenic": Decimal("0"),
    },
    "coal_lignite": {
        "co2_ef": Decimal("101.000"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0015"),
        "default_efficiency": Decimal("0.72"),
        "is_biogenic": Decimal("0"),
    },
    "lpg": {
        "co2_ef": Decimal("63.100"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0001"),
        "default_efficiency": Decimal("0.85"),
        "is_biogenic": Decimal("0"),
    },
    "biomass_wood": {
        "co2_ef": Decimal("112.000"),
        "ch4_ef": Decimal("0.030"),
        "n2o_ef": Decimal("0.004"),
        "default_efficiency": Decimal("0.70"),
        "is_biogenic": Decimal("1"),
    },
    "biomass_biogas": {
        "co2_ef": Decimal("54.600"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0001"),
        "default_efficiency": Decimal("0.80"),
        "is_biogenic": Decimal("1"),
    },
    "municipal_waste": {
        "co2_ef": Decimal("91.700"),
        "ch4_ef": Decimal("0.030"),
        "n2o_ef": Decimal("0.004"),
        "default_efficiency": Decimal("0.65"),
        "is_biogenic": Decimal("0"),
    },
}


# ---------------------------------------------------------------------------
# Built-in Constants: CHP Default Efficiencies
# ---------------------------------------------------------------------------

#: Default electrical and thermal efficiencies for CHP plants by fuel type.
#: Source: IEA CHP Country Scorecard data, EPA CHP Partnership.
CHP_DEFAULT_EFFICIENCIES: Dict[str, Dict[str, Decimal]] = {
    "natural_gas": {
        "electrical_efficiency": Decimal("0.35"),
        "thermal_efficiency": Decimal("0.45"),
        "overall_efficiency": Decimal("0.80"),
    },
    "coal": {
        "electrical_efficiency": Decimal("0.30"),
        "thermal_efficiency": Decimal("0.40"),
        "overall_efficiency": Decimal("0.70"),
    },
    "biomass": {
        "electrical_efficiency": Decimal("0.25"),
        "thermal_efficiency": Decimal("0.50"),
        "overall_efficiency": Decimal("0.75"),
    },
    "fuel_oil": {
        "electrical_efficiency": Decimal("0.32"),
        "thermal_efficiency": Decimal("0.43"),
        "overall_efficiency": Decimal("0.75"),
    },
    "municipal_waste": {
        "electrical_efficiency": Decimal("0.20"),
        "thermal_efficiency": Decimal("0.45"),
        "overall_efficiency": Decimal("0.65"),
    },
}

#: Mapping from specific fuel types to CHP efficiency lookup keys.
#: Multiple specific fuel types map to a single CHP category.
_FUEL_TYPE_TO_CHP_KEY: Dict[str, str] = {
    "natural_gas": "natural_gas",
    "lpg": "natural_gas",
    "biomass_biogas": "natural_gas",
    "fuel_oil_2": "fuel_oil",
    "fuel_oil_6": "fuel_oil",
    "coal_bituminous": "coal",
    "coal_subbituminous": "coal",
    "coal_lignite": "coal",
    "biomass_wood": "biomass",
    "municipal_waste": "municipal_waste",
}


# ---------------------------------------------------------------------------
# Built-in Constants: Reference Efficiencies (EU EED 2012/27/EU Annex II)
# ---------------------------------------------------------------------------

#: Reference electrical efficiencies by CHP technology type.
#: Used in Primary Energy Savings (PES) calculation.
#: Source: EU EED 2012/27/EU Annex II, Delegated Regulation (EU) 2015/2402.
REFERENCE_ELECTRICAL_EFFICIENCIES: Dict[str, Decimal] = {
    "gas_turbine": Decimal("0.525"),
    "steam_turbine": Decimal("0.442"),
    "combined_cycle": Decimal("0.532"),
    "internal_combustion": Decimal("0.442"),
    "micro_turbine": Decimal("0.280"),
    "fuel_cell": Decimal("0.530"),
    "stirling": Decimal("0.350"),
    "orc": Decimal("0.330"),
}

#: Reference thermal efficiency for separate heat production.
#: Source: EU EED 2012/27/EU Annex II.
REFERENCE_THERMAL_EFFICIENCY: Decimal = Decimal("0.90")

#: Default reference electrical efficiency when technology is unknown.
DEFAULT_REFERENCE_ELECTRICAL_EFFICIENCY: Decimal = Decimal("0.525")

#: Default cooling efficiency (COP equivalent as thermal fraction).
DEFAULT_COOLING_EFFICIENCY: Decimal = Decimal("0.70")

#: Default ambient temperature in Celsius for exergy calculations.
DEFAULT_AMBIENT_TEMP_C: Decimal = Decimal("25")

#: Default steam temperature in Celsius for exergy calculations.
DEFAULT_STEAM_TEMP_C: Decimal = Decimal("180")

#: PES threshold for high-efficiency CHP (large plants >= 1 MW).
PES_THRESHOLD_LARGE: Decimal = Decimal("10")

#: PES threshold for high-efficiency CHP (small plants < 1 MW).
PES_THRESHOLD_SMALL: Decimal = _ZERO

#: Capacity boundary between small and large CHP (MW).
SMALL_CHP_CAPACITY_MW: Decimal = _ONE

#: Version string for this engine.
VERSION: str = "1.0.0"

#: Database table prefix.
TABLE_PREFIX: str = "gl_shp_"


# ---------------------------------------------------------------------------
# Utility: UTC now helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Utility: Safe Decimal conversion
# ---------------------------------------------------------------------------

def _to_decimal(value: Any) -> Decimal:
    """Convert a value to Decimal safely.

    Args:
        value: int, float, str, or Decimal.

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal: {exc}") from exc


def _safe_get_decimal(
    data: Dict[str, Any], key: str, default: Decimal = _ZERO,
) -> Decimal:
    """Safely extract a Decimal from a dictionary.

    Args:
        data: Source dictionary.
        key: Key to look up.
        default: Default value if key is missing or None.

    Returns:
        Decimal value.
    """
    raw = data.get(key)
    if raw is None:
        return default
    return _to_decimal(raw)


# ---------------------------------------------------------------------------
# Utility: SHA-256 hash for CHP allocation results
# ---------------------------------------------------------------------------

def _hash_allocation(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of a CHP allocation payload.

    Args:
        data: Dictionary payload to hash.

    Returns:
        64-character lowercase hexadecimal SHA-256 digest.
    """
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Provenance helper (lightweight inline tracker)
# ---------------------------------------------------------------------------

class _CHPAllocationProvenance:
    """Chain-hashing provenance tracker for CHP allocation operations.

    Each recorded entry chains its SHA-256 hash to the previous entry,
    producing a tamper-evident audit log for every allocation, comparison,
    and batch operation.
    """

    _GENESIS = "GL-MRV-011-STEAM-HEAT-CHP-ALLOCATION-GENESIS"

    def __init__(self) -> None:
        """Initialize with genesis hash."""
        self._genesis: str = hashlib.sha256(
            self._GENESIS.encode("utf-8")
        ).hexdigest()
        self._last_hash: str = self._genesis
        self._entries: List[Dict[str, Any]] = []
        self._lock: threading.RLock = threading.RLock()

    def record(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a provenance entry and return it.

        Args:
            entity_type: Category of the entity (e.g. 'chp_allocation').
            action: Action performed (e.g. 'allocate', 'validate').
            entity_id: Unique identifier for the entity.
            data: Optional data payload to hash.

        Returns:
            Dictionary with entity_type, entity_id, action, hash_value,
            parent_hash, timestamp, and data_hash.
        """
        ts = _utcnow().isoformat()
        data_hash = hashlib.sha256(
            json.dumps(data or {}, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        with self._lock:
            parent = self._last_hash
            payload = f"{parent}|{data_hash}|{action}|{ts}"
            chain_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            entry = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": action,
                "hash_value": chain_hash,
                "parent_hash": parent,
                "timestamp": ts,
                "data_hash": data_hash,
            }
            self._entries.append(entry)
            self._last_hash = chain_hash
        return entry

    def verify_chain(self) -> bool:
        """Verify integrity of the provenance chain.

        Returns:
            True if chain is intact, False if tampered.
        """
        with self._lock:
            if not self._entries:
                return True
            expected_parent = self._genesis
            for entry in self._entries:
                if entry["parent_hash"] != expected_parent:
                    return False
                expected_parent = entry["hash_value"]
            return True

    def get_entries(self) -> List[Dict[str, Any]]:
        """Return a copy of all provenance entries."""
        with self._lock:
            return list(self._entries)

    @property
    def entry_count(self) -> int:
        """Return number of provenance entries."""
        with self._lock:
            return len(self._entries)

    def reset(self) -> None:
        """Reset to genesis state."""
        with self._lock:
            self._entries.clear()
            self._last_hash = self._genesis


# ===========================================================================
# CHPAllocationEngine
# ===========================================================================


class CHPAllocationEngine:
    """CHP emission allocation engine for GHG Protocol Scope 2 compliance.

    Allocates total fuel combustion emissions from combined heat and power
    (CHP) / cogeneration plants between their thermal (heat/steam) and
    electrical (power) outputs using one of three GHG Protocol-compliant
    allocation methods: efficiency, energy, or exergy.

    This engine also supports multi-product allocation (heat + power +
    cooling), computes primary energy savings (PES) per the EU Energy
    Efficiency Directive 2012/27/EU, and determines whether a CHP plant
    qualifies as high-efficiency.

    Thread Safety:
        All mutable state is protected by ``threading.RLock``. Multiple
        threads may safely call any public method concurrently.

    Zero-Hallucination:
        No LLM calls. All calculations are pure Decimal arithmetic.
        Every result carries a SHA-256 provenance hash.

    Singleton Pattern:
        Uses ``__new__`` with double-checked locking to ensure exactly
        one instance exists. Use ``reset_singleton()`` for test isolation.

    Attributes:
        ENGINE_ID: Constant identifier for this engine.
        ENGINE_VERSION: Semantic version string.

    Example:
        >>> engine = CHPAllocationEngine()
        >>> result = engine.allocate_efficiency_method(
        ...     total_fuel_gj=Decimal("1000"),
        ...     fuel_type="natural_gas",
        ...     heat_output_gj=Decimal("400"),
        ...     power_output_gj=Decimal("350"),
        ...     eta_thermal=Decimal("0.45"),
        ...     eta_electrical=Decimal("0.35"),
        ... )
        >>> assert result["heat_share"] + result["power_share"] == Decimal("1.00000000")
    """

    ENGINE_ID: str = "chp_allocation"
    ENGINE_VERSION: str = "1.0.0"

    _instance: Optional[CHPAllocationEngine] = None
    _cls_lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton construction
    # ------------------------------------------------------------------

    def __new__(
        cls,
        config: Optional[Dict[str, Any]] = None,
    ) -> CHPAllocationEngine:
        """Return the singleton CHPAllocationEngine instance.

        Uses double-checked locking with an RLock to ensure exactly one
        instance is created even under concurrent first-access.

        Args:
            config: Optional configuration dictionary (ignored after first init).

        Returns:
            The singleton instance.
        """
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize CHPAllocationEngine.

        Idempotent: after the first call, subsequent invocations are
        silently skipped to prevent duplicate initialisation.

        Args:
            config: Optional configuration dictionary. Supports:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                    Default True.
                - ``decimal_precision`` (int): Decimal places. Default 8.
                - ``default_gwp_source`` (str): Default GWP source. Default 'AR6'.
                - ``default_ambient_temp_c`` (Decimal): Ambient temp. Default 25.
                - ``default_steam_temp_c`` (Decimal): Steam temp. Default 180.
                - ``default_cooling_efficiency`` (Decimal): Cooling COP-equivalent.
                    Default 0.70.
                - ``tenant_id`` (str): Default tenant. Default 'default'.
        """
        if self._initialized:
            return

        config = config or {}
        self._enable_provenance: bool = config.get("enable_provenance", True)
        self._precision_places: int = config.get("decimal_precision", 8)
        self._precision: Decimal = Decimal(10) ** (-self._precision_places)
        self._default_gwp_source: str = config.get("default_gwp_source", "AR6")
        self._default_ambient_temp_c: Decimal = _to_decimal(
            config.get("default_ambient_temp_c", DEFAULT_AMBIENT_TEMP_C)
        )
        self._default_steam_temp_c: Decimal = _to_decimal(
            config.get("default_steam_temp_c", DEFAULT_STEAM_TEMP_C)
        )
        self._default_cooling_efficiency: Decimal = _to_decimal(
            config.get("default_cooling_efficiency", DEFAULT_COOLING_EFFICIENCY)
        )
        self._tenant_id: str = config.get("tenant_id", "default")

        # Provenance tracker
        self._provenance: Optional[_CHPAllocationProvenance] = (
            _CHPAllocationProvenance() if self._enable_provenance else None
        )

        # Thread safety for mutable state
        self._lock: threading.RLock = threading.RLock()

        # Statistics
        self._created_at: datetime = _utcnow()
        self._total_allocations: int = 0
        self._total_efficiency_allocations: int = 0
        self._total_energy_allocations: int = 0
        self._total_exergy_allocations: int = 0
        self._total_multiproduct_allocations: int = 0
        self._total_fuel_emissions_computed: int = 0
        self._total_pes_computed: int = 0
        self._total_comparisons: int = 0
        self._total_batch_allocations: int = 0
        self._total_validations: int = 0
        self._total_errors: int = 0

        self._initialized = True

        logger.info(
            "CHPAllocationEngine initialized: provenance=%s, precision=%d, "
            "gwp=%s, ambient=%.1f C, tenant=%s",
            self._enable_provenance,
            self._precision_places,
            self._default_gwp_source,
            float(self._default_ambient_temp_c),
            self._tenant_id,
        )

    # ------------------------------------------------------------------
    # Internal: Decimal quantize helpers
    # ------------------------------------------------------------------

    def _q(self, value: Decimal) -> Decimal:
        """Quantize a Decimal to the configured precision.

        Args:
            value: Decimal to quantize.

        Returns:
            Quantized Decimal with ROUND_HALF_UP.
        """
        return value.quantize(self._precision, rounding=ROUND_HALF_UP)

    def _q_pct(self, value: Decimal) -> Decimal:
        """Quantize a percentage Decimal to 6 decimal places.

        Args:
            value: Decimal percentage to quantize.

        Returns:
            Quantized Decimal with ROUND_HALF_UP.
        """
        return value.quantize(_PCT_PRECISION, rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Internal: Provenance recording helper
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Record a provenance entry if tracking is enabled.

        Args:
            entity_type: Category of the entity.
            action: Action performed.
            entity_id: Unique identifier.
            data: Optional data payload.

        Returns:
            Hash value string if provenance is enabled, else None.
        """
        if self._provenance is not None:
            entry = self._provenance.record(
                entity_type=entity_type,
                action=action,
                entity_id=entity_id,
                data=data,
            )
            return entry["hash_value"]
        return None

    # ------------------------------------------------------------------
    # Internal: Error counter increment
    # ------------------------------------------------------------------

    def _increment_error(self) -> None:
        """Increment the error counter under lock."""
        with self._lock:
            self._total_errors += 1

    # ------------------------------------------------------------------
    # Internal: Metrics recording helper
    # ------------------------------------------------------------------

    def _record_metrics(self, method: str, fuel_type: str) -> None:
        """Record CHP allocation metrics if available.

        Args:
            method: Allocation method used.
            fuel_type: Fuel type of the CHP plant.
        """
        if _METRICS_AVAILABLE and _get_metrics is not None:
            try:
                metrics = _get_metrics()
                metrics.record_chp_allocation(
                    method=method,
                    fuel_type=fuel_type,
                    tenant_id=self._tenant_id,
                )
            except Exception:
                pass  # Metrics are best-effort

    # ------------------------------------------------------------------
    # Internal: Resolve GWP values
    # ------------------------------------------------------------------

    def _resolve_gwp(self, gwp_source: Optional[str] = None) -> Dict[str, Decimal]:
        """Resolve GWP values for the given source.

        Args:
            gwp_source: IPCC Assessment Report key. If None, uses the
                configured default.

        Returns:
            Dictionary with CO2, CH4, N2O GWP multipliers.

        Raises:
            ValueError: If gwp_source is not recognized.
        """
        source = gwp_source or self._default_gwp_source
        source_upper = source.upper()
        if source_upper not in GWP_VALUES:
            raise ValueError(
                f"Unknown GWP source '{source}'. "
                f"Valid sources: {list(GWP_VALUES.keys())}"
            )
        return GWP_VALUES[source_upper]

    # ------------------------------------------------------------------
    # Internal: Resolve fuel emission factors
    # ------------------------------------------------------------------

    def _resolve_fuel_ef(self, fuel_type: str) -> Dict[str, Decimal]:
        """Resolve emission factors for the given fuel type.

        Args:
            fuel_type: Fuel type key (e.g. 'natural_gas').

        Returns:
            Dictionary with co2_ef, ch4_ef, n2o_ef, default_efficiency,
            is_biogenic fields.

        Raises:
            ValueError: If fuel_type is not recognized.
        """
        fuel_key = fuel_type.lower().strip()
        if fuel_key not in FUEL_EMISSION_FACTORS:
            raise ValueError(
                f"Unknown fuel type '{fuel_type}'. "
                f"Valid fuel types: {list(FUEL_EMISSION_FACTORS.keys())}"
            )
        return FUEL_EMISSION_FACTORS[fuel_key]

    # ------------------------------------------------------------------
    # Internal: Resolve CHP default efficiencies
    # ------------------------------------------------------------------

    def _resolve_chp_defaults(self, fuel_type: str) -> Dict[str, Decimal]:
        """Resolve default CHP efficiencies for the given fuel type.

        Maps specific fuel type keys to CHP efficiency categories and
        returns the default electrical and thermal efficiencies.

        Args:
            fuel_type: Fuel type key (e.g. 'natural_gas', 'coal_bituminous').

        Returns:
            Dictionary with electrical_efficiency, thermal_efficiency,
            overall_efficiency fields.
        """
        fuel_key = fuel_type.lower().strip()
        chp_key = _FUEL_TYPE_TO_CHP_KEY.get(fuel_key, "natural_gas")
        return CHP_DEFAULT_EFFICIENCIES.get(
            chp_key, CHP_DEFAULT_EFFICIENCIES["natural_gas"]
        )

    # ==================================================================
    # PUBLIC METHOD 1: allocate_chp_emissions (main entry point)
    # ==================================================================

    def allocate_chp_emissions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main CHP emission allocation entry point.

        Validates the request, auto-selects the appropriate allocation
        method based on the ``method`` field, and dispatches to the
        corresponding implementation method.

        Args:
            request: Allocation request dictionary with keys:
                - total_fuel_gj (required): Total fuel input in GJ.
                - fuel_type (required): Fuel type string.
                - heat_output_gj (required): Heat output in GJ.
                - power_output_gj (required): Power output in GJ.
                - method (optional): 'efficiency', 'energy', or 'exergy'.
                    Default 'efficiency'.
                - electrical_efficiency (optional): Explicit eta_electrical.
                - thermal_efficiency (optional): Explicit eta_thermal.
                - cooling_output_gj (optional): Cooling output in GJ.
                - cooling_efficiency (optional): Cooling system COP-equiv.
                - steam_temperature_c (optional): Steam temp for exergy.
                - ambient_temperature_c (optional): Ambient temp for exergy.
                - gwp_source (optional): GWP source string.

        Returns:
            Complete allocation result dictionary with shares, emissions,
            trace, and provenance_hash.

        Raises:
            ValueError: If request validation fails.
        """
        start = time.monotonic()
        alloc_id = str(uuid.uuid4())

        try:
            # Step 1: Validate
            is_valid, errors = self.validate_request(request)
            if not is_valid:
                raise ValueError(
                    f"CHP allocation request validation failed: "
                    f"{'; '.join(errors)}"
                )

            # Step 2: Extract parameters
            total_fuel_gj = _to_decimal(request["total_fuel_gj"])
            fuel_type = str(request["fuel_type"]).lower().strip()
            heat_gj = _to_decimal(request["heat_output_gj"])
            power_gj = _to_decimal(request["power_output_gj"])
            method = str(request.get("method", "efficiency")).lower().strip()
            gwp_source = request.get("gwp_source")
            cooling_gj = _to_decimal(request.get("cooling_output_gj", 0))

            # Step 3: Determine if multi-product
            if cooling_gj > _ZERO:
                efficiencies = self._extract_efficiencies(request, fuel_type)
                result = self.allocate_multiproduct(
                    total_fuel_gj=total_fuel_gj,
                    fuel_type=fuel_type,
                    heat_gj=heat_gj,
                    power_gj=power_gj,
                    cooling_gj=cooling_gj,
                    method=method,
                    efficiencies=efficiencies,
                    gwp_source=gwp_source,
                )
            elif method == "efficiency":
                eta_t, eta_e = self._extract_eta_pair(request, fuel_type)
                result = self.allocate_efficiency_method(
                    total_fuel_gj=total_fuel_gj,
                    fuel_type=fuel_type,
                    heat_output_gj=heat_gj,
                    power_output_gj=power_gj,
                    eta_thermal=eta_t,
                    eta_electrical=eta_e,
                    gwp_source=gwp_source,
                )
            elif method == "energy":
                result = self.allocate_energy_method(
                    total_fuel_gj=total_fuel_gj,
                    fuel_type=fuel_type,
                    heat_output_gj=heat_gj,
                    power_output_gj=power_gj,
                    gwp_source=gwp_source,
                )
            elif method == "exergy":
                steam_temp = _to_decimal(
                    request.get("steam_temperature_c", self._default_steam_temp_c)
                )
                ambient_temp = _to_decimal(
                    request.get("ambient_temperature_c", self._default_ambient_temp_c)
                )
                result = self.allocate_exergy_method(
                    total_fuel_gj=total_fuel_gj,
                    fuel_type=fuel_type,
                    heat_output_gj=heat_gj,
                    power_output_gj=power_gj,
                    steam_temp_c=steam_temp,
                    ambient_temp_c=ambient_temp,
                    gwp_source=gwp_source,
                )
            else:
                raise ValueError(
                    f"Unknown allocation method '{method}'. "
                    f"Valid methods: efficiency, energy, exergy"
                )

            # Step 4: Enrich result
            elapsed = time.monotonic() - start
            result["allocation_id"] = alloc_id
            result["processing_time_ms"] = self._q(
                Decimal(str(elapsed * 1000))
            )

            with self._lock:
                self._total_allocations += 1

            self._record_provenance(
                "chp_allocation", "allocate", alloc_id,
                {"method": method, "fuel_type": fuel_type},
            )

            logger.info(
                "CHP allocation %s completed: method=%s, fuel=%s, "
                "heat_share=%s, power_share=%s, elapsed=%.3fms",
                alloc_id, method, fuel_type,
                result.get("heat_share"), result.get("power_share"),
                elapsed * 1000,
            )

            return result

        except Exception as exc:
            self._increment_error()
            logger.error(
                "CHP allocation failed: %s", str(exc), exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Internal: Extract efficiency pair from request
    # ------------------------------------------------------------------

    def _extract_eta_pair(
        self, request: Dict[str, Any], fuel_type: str,
    ) -> Tuple[Decimal, Decimal]:
        """Extract thermal and electrical efficiencies from request or defaults.

        Args:
            request: Allocation request dictionary.
            fuel_type: Fuel type for default lookup.

        Returns:
            Tuple of (eta_thermal, eta_electrical).
        """
        defaults = self._resolve_chp_defaults(fuel_type)
        eta_t = _to_decimal(
            request.get("thermal_efficiency")
            or defaults["thermal_efficiency"]
        )
        eta_e = _to_decimal(
            request.get("electrical_efficiency")
            or defaults["electrical_efficiency"]
        )
        return eta_t, eta_e

    def _extract_efficiencies(
        self, request: Dict[str, Any], fuel_type: str,
    ) -> Dict[str, Decimal]:
        """Extract all efficiency parameters from request or defaults.

        Args:
            request: Allocation request dictionary.
            fuel_type: Fuel type for default lookup.

        Returns:
            Dictionary with eta_thermal, eta_electrical, eta_cooling.
        """
        defaults = self._resolve_chp_defaults(fuel_type)
        return {
            "eta_thermal": _to_decimal(
                request.get("thermal_efficiency")
                or defaults["thermal_efficiency"]
            ),
            "eta_electrical": _to_decimal(
                request.get("electrical_efficiency")
                or defaults["electrical_efficiency"]
            ),
            "eta_cooling": _to_decimal(
                request.get("cooling_efficiency")
                or self._default_cooling_efficiency
            ),
        }

    # ==================================================================
    # PUBLIC METHOD 2: allocate_efficiency_method
    # ==================================================================

    def allocate_efficiency_method(
        self,
        total_fuel_gj: Decimal,
        fuel_type: str,
        heat_output_gj: Decimal,
        power_output_gj: Decimal,
        eta_thermal: Decimal,
        eta_electrical: Decimal,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Allocate CHP emissions using the efficiency method.

        The GHG Protocol recommended method. Allocates emissions
        proportionally to the fuel equivalent of each output, where fuel
        equivalent = output / conversion efficiency. This reflects the
        thermodynamic quality difference between electricity and heat.

        Formula:
            Heat_Fuel_Equiv  = heat_output_gj / eta_thermal
            Power_Fuel_Equiv = power_output_gj / eta_electrical
            Total_Fuel_Equiv = Heat_Fuel_Equiv + Power_Fuel_Equiv
            Heat_Share       = Heat_Fuel_Equiv / Total_Fuel_Equiv
            Power_Share      = Power_Fuel_Equiv / Total_Fuel_Equiv
            Heat_Emissions   = Total_Fuel_Emissions * Heat_Share
            Power_Emissions  = Total_Fuel_Emissions * Power_Share

        Args:
            total_fuel_gj: Total fuel input to the CHP plant in GJ.
            fuel_type: Fuel type string for emission factor lookup.
            heat_output_gj: Useful thermal output in GJ.
            power_output_gj: Electrical power output in GJ.
            eta_thermal: Thermal conversion efficiency (0-1).
            eta_electrical: Electrical conversion efficiency (0-1).
            gwp_source: IPCC GWP source. Default from config.

        Returns:
            Dictionary with method, heat_share, power_share,
            heat_emissions_kgco2e, power_emissions_kgco2e,
            total_fuel_emissions_kgco2e, fuel_emissions_detail,
            trace, provenance_hash.

        Raises:
            ValueError: If any input is invalid.
        """
        start = time.monotonic()
        trace: List[str] = []

        # Validate inputs
        total_fuel_gj = _to_decimal(total_fuel_gj)
        heat_output_gj = _to_decimal(heat_output_gj)
        power_output_gj = _to_decimal(power_output_gj)
        eta_thermal = _to_decimal(eta_thermal)
        eta_electrical = _to_decimal(eta_electrical)

        self._validate_positive("total_fuel_gj", total_fuel_gj)
        self._validate_positive("heat_output_gj", heat_output_gj)
        self._validate_positive("power_output_gj", power_output_gj)
        self._validate_efficiency("eta_thermal", eta_thermal)
        self._validate_efficiency("eta_electrical", eta_electrical)

        trace.append(
            f"Inputs: total_fuel={total_fuel_gj} GJ, fuel={fuel_type}, "
            f"heat={heat_output_gj} GJ, power={power_output_gj} GJ, "
            f"eta_t={eta_thermal}, eta_e={eta_electrical}"
        )

        # Step 1: Compute fuel emissions
        fuel_emissions = self.compute_fuel_emissions(
            total_fuel_gj, fuel_type, gwp_source,
        )
        total_co2e = _to_decimal(fuel_emissions["total_co2e_kg"])
        trace.append(
            f"Total fuel emissions: {total_co2e} kgCO2e "
            f"(CO2={fuel_emissions['co2_kg']}, "
            f"CH4={fuel_emissions['ch4_kg']}, "
            f"N2O={fuel_emissions['n2o_kg']})"
        )

        # Step 2: Compute fuel equivalents
        heat_fuel_equiv = self._q(heat_output_gj / eta_thermal)
        power_fuel_equiv = self._q(power_output_gj / eta_electrical)
        total_fuel_equiv = self._q(heat_fuel_equiv + power_fuel_equiv)

        trace.append(
            f"Fuel equivalents: heat={heat_fuel_equiv} GJ, "
            f"power={power_fuel_equiv} GJ, total={total_fuel_equiv} GJ"
        )

        # Step 3: Compute shares
        if total_fuel_equiv == _ZERO:
            raise ValueError(
                "Total fuel equivalent is zero; cannot compute allocation shares"
            )

        heat_share = self._q(heat_fuel_equiv / total_fuel_equiv)
        power_share = self._q(_ONE - heat_share)

        trace.append(
            f"Allocation shares: heat={heat_share}, power={power_share}"
        )

        # Step 4: Allocate emissions
        heat_emissions = self._q(total_co2e * heat_share)
        power_emissions = self._q(total_co2e * power_share)

        trace.append(
            f"Allocated emissions: heat={heat_emissions} kgCO2e, "
            f"power={power_emissions} kgCO2e"
        )

        # Build result
        elapsed = time.monotonic() - start
        result_data = {
            "method": "efficiency",
            "heat_share": heat_share,
            "power_share": power_share,
            "cooling_share": self._q(_ZERO),
            "heat_emissions_kgco2e": heat_emissions,
            "power_emissions_kgco2e": power_emissions,
            "cooling_emissions_kgco2e": self._q(_ZERO),
            "total_fuel_emissions_kgco2e": self._q(total_co2e),
            "fuel_emissions_detail": fuel_emissions,
            "heat_fuel_equivalent_gj": heat_fuel_equiv,
            "power_fuel_equivalent_gj": power_fuel_equiv,
            "total_fuel_equivalent_gj": total_fuel_equiv,
            "eta_thermal": eta_thermal,
            "eta_electrical": eta_electrical,
            "fuel_type": fuel_type,
            "gwp_source": gwp_source or self._default_gwp_source,
            "trace": trace,
            "processing_time_ms": self._q(Decimal(str(elapsed * 1000))),
        }

        result_data["provenance_hash"] = _hash_allocation(result_data)

        with self._lock:
            self._total_efficiency_allocations += 1

        self._record_metrics("efficiency", fuel_type)

        logger.debug(
            "Efficiency allocation: heat_share=%s, power_share=%s",
            heat_share, power_share,
        )

        return result_data

    # ==================================================================
    # PUBLIC METHOD 3: allocate_energy_method
    # ==================================================================

    def allocate_energy_method(
        self,
        total_fuel_gj: Decimal,
        fuel_type: str,
        heat_output_gj: Decimal,
        power_output_gj: Decimal,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Allocate CHP emissions using the energy method.

        Simple allocation proportional to the energy content of each
        output (GJ) without adjusting for conversion efficiency. This
        method does not reflect the higher thermodynamic value of
        electricity versus heat, but is simpler to apply.

        Formula:
            Total_Output = heat_output_gj + power_output_gj
            Heat_Share   = heat_output_gj / Total_Output
            Power_Share  = power_output_gj / Total_Output

        Args:
            total_fuel_gj: Total fuel input to the CHP plant in GJ.
            fuel_type: Fuel type string for emission factor lookup.
            heat_output_gj: Useful thermal output in GJ.
            power_output_gj: Electrical power output in GJ.
            gwp_source: IPCC GWP source. Default from config.

        Returns:
            Dictionary with method, heat_share, power_share,
            heat_emissions_kgco2e, power_emissions_kgco2e,
            total_fuel_emissions_kgco2e, fuel_emissions_detail,
            trace, provenance_hash.

        Raises:
            ValueError: If any input is invalid.
        """
        start = time.monotonic()
        trace: List[str] = []

        total_fuel_gj = _to_decimal(total_fuel_gj)
        heat_output_gj = _to_decimal(heat_output_gj)
        power_output_gj = _to_decimal(power_output_gj)

        self._validate_positive("total_fuel_gj", total_fuel_gj)
        self._validate_positive("heat_output_gj", heat_output_gj)
        self._validate_positive("power_output_gj", power_output_gj)

        trace.append(
            f"Inputs: total_fuel={total_fuel_gj} GJ, fuel={fuel_type}, "
            f"heat={heat_output_gj} GJ, power={power_output_gj} GJ"
        )

        # Step 1: Compute fuel emissions
        fuel_emissions = self.compute_fuel_emissions(
            total_fuel_gj, fuel_type, gwp_source,
        )
        total_co2e = _to_decimal(fuel_emissions["total_co2e_kg"])
        trace.append(
            f"Total fuel emissions: {total_co2e} kgCO2e"
        )

        # Step 2: Compute total output and shares
        total_output = self._q(heat_output_gj + power_output_gj)

        if total_output == _ZERO:
            raise ValueError(
                "Total energy output is zero; cannot compute allocation shares"
            )

        heat_share = self._q(heat_output_gj / total_output)
        power_share = self._q(_ONE - heat_share)

        trace.append(
            f"Total output: {total_output} GJ; "
            f"heat_share={heat_share}, power_share={power_share}"
        )

        # Step 3: Allocate emissions
        heat_emissions = self._q(total_co2e * heat_share)
        power_emissions = self._q(total_co2e * power_share)

        trace.append(
            f"Allocated emissions: heat={heat_emissions} kgCO2e, "
            f"power={power_emissions} kgCO2e"
        )

        elapsed = time.monotonic() - start
        result_data = {
            "method": "energy",
            "heat_share": heat_share,
            "power_share": power_share,
            "cooling_share": self._q(_ZERO),
            "heat_emissions_kgco2e": heat_emissions,
            "power_emissions_kgco2e": power_emissions,
            "cooling_emissions_kgco2e": self._q(_ZERO),
            "total_fuel_emissions_kgco2e": self._q(total_co2e),
            "fuel_emissions_detail": fuel_emissions,
            "total_energy_output_gj": total_output,
            "fuel_type": fuel_type,
            "gwp_source": gwp_source or self._default_gwp_source,
            "trace": trace,
            "processing_time_ms": self._q(Decimal(str(elapsed * 1000))),
        }

        result_data["provenance_hash"] = _hash_allocation(result_data)

        with self._lock:
            self._total_energy_allocations += 1

        self._record_metrics("energy", fuel_type)

        logger.debug(
            "Energy allocation: heat_share=%s, power_share=%s",
            heat_share, power_share,
        )

        return result_data

    # ==================================================================
    # PUBLIC METHOD 4: allocate_exergy_method
    # ==================================================================

    def allocate_exergy_method(
        self,
        total_fuel_gj: Decimal,
        fuel_type: str,
        heat_output_gj: Decimal,
        power_output_gj: Decimal,
        steam_temp_c: Optional[Decimal] = None,
        ambient_temp_c: Optional[Decimal] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Allocate CHP emissions using the exergy method.

        Allocates emissions proportionally to the exergy (available work)
        content of each output. Electricity is pure exergy. Heat exergy
        is discounted by the Carnot factor based on the steam delivery
        temperature relative to ambient. This is the most thermodynamically
        rigorous method.

        Formula:
            Carnot_Factor = 1 - (T_ambient + 273.15) / (T_steam + 273.15)
            Exergy_Heat   = heat_output_gj * Carnot_Factor
            Exergy_Power  = power_output_gj  (power is pure exergy)
            Total_Exergy  = Exergy_Heat + Exergy_Power
            Heat_Share    = Exergy_Heat / Total_Exergy
            Power_Share   = Exergy_Power / Total_Exergy

        Args:
            total_fuel_gj: Total fuel input to the CHP plant in GJ.
            fuel_type: Fuel type string for emission factor lookup.
            heat_output_gj: Useful thermal output in GJ.
            power_output_gj: Electrical power output in GJ.
            steam_temp_c: Steam/hot water supply temperature in Celsius.
                Default from config.
            ambient_temp_c: Ambient reference temperature in Celsius.
                Default from config.
            gwp_source: IPCC GWP source. Default from config.

        Returns:
            Dictionary with method, heat_share, power_share,
            heat_emissions_kgco2e, power_emissions_kgco2e,
            total_fuel_emissions_kgco2e, carnot_factor,
            exergy_heat_gj, exergy_power_gj,
            fuel_emissions_detail, trace, provenance_hash.

        Raises:
            ValueError: If any input is invalid.
        """
        start = time.monotonic()
        trace: List[str] = []

        total_fuel_gj = _to_decimal(total_fuel_gj)
        heat_output_gj = _to_decimal(heat_output_gj)
        power_output_gj = _to_decimal(power_output_gj)
        steam_temp_c = _to_decimal(
            steam_temp_c if steam_temp_c is not None
            else self._default_steam_temp_c
        )
        ambient_temp_c = _to_decimal(
            ambient_temp_c if ambient_temp_c is not None
            else self._default_ambient_temp_c
        )

        self._validate_positive("total_fuel_gj", total_fuel_gj)
        self._validate_positive("heat_output_gj", heat_output_gj)
        self._validate_positive("power_output_gj", power_output_gj)

        if steam_temp_c <= ambient_temp_c:
            raise ValueError(
                f"steam_temp_c ({steam_temp_c}) must be greater than "
                f"ambient_temp_c ({ambient_temp_c}) for exergy allocation"
            )

        trace.append(
            f"Inputs: total_fuel={total_fuel_gj} GJ, fuel={fuel_type}, "
            f"heat={heat_output_gj} GJ, power={power_output_gj} GJ, "
            f"T_steam={steam_temp_c} C, T_ambient={ambient_temp_c} C"
        )

        # Step 1: Compute fuel emissions
        fuel_emissions = self.compute_fuel_emissions(
            total_fuel_gj, fuel_type, gwp_source,
        )
        total_co2e = _to_decimal(fuel_emissions["total_co2e_kg"])
        trace.append(f"Total fuel emissions: {total_co2e} kgCO2e")

        # Step 2: Compute Carnot factor
        carnot_factor = self.compute_carnot_factor(steam_temp_c, ambient_temp_c)
        trace.append(f"Carnot factor: {carnot_factor}")

        # Step 3: Compute exergy of each output
        exergy_heat = self._q(heat_output_gj * carnot_factor)
        exergy_power = self._q(power_output_gj)  # Power is pure exergy
        total_exergy = self._q(exergy_heat + exergy_power)

        trace.append(
            f"Exergy: heat={exergy_heat} GJ, power={exergy_power} GJ, "
            f"total={total_exergy} GJ"
        )

        if total_exergy == _ZERO:
            raise ValueError(
                "Total exergy is zero; cannot compute allocation shares"
            )

        # Step 4: Compute shares
        heat_share = self._q(exergy_heat / total_exergy)
        power_share = self._q(_ONE - heat_share)

        trace.append(
            f"Allocation shares: heat={heat_share}, power={power_share}"
        )

        # Step 5: Allocate emissions
        heat_emissions = self._q(total_co2e * heat_share)
        power_emissions = self._q(total_co2e * power_share)

        trace.append(
            f"Allocated emissions: heat={heat_emissions} kgCO2e, "
            f"power={power_emissions} kgCO2e"
        )

        elapsed = time.monotonic() - start
        result_data = {
            "method": "exergy",
            "heat_share": heat_share,
            "power_share": power_share,
            "cooling_share": self._q(_ZERO),
            "heat_emissions_kgco2e": heat_emissions,
            "power_emissions_kgco2e": power_emissions,
            "cooling_emissions_kgco2e": self._q(_ZERO),
            "total_fuel_emissions_kgco2e": self._q(total_co2e),
            "fuel_emissions_detail": fuel_emissions,
            "carnot_factor": carnot_factor,
            "exergy_heat_gj": exergy_heat,
            "exergy_power_gj": exergy_power,
            "total_exergy_gj": total_exergy,
            "steam_temperature_c": steam_temp_c,
            "ambient_temperature_c": ambient_temp_c,
            "fuel_type": fuel_type,
            "gwp_source": gwp_source or self._default_gwp_source,
            "trace": trace,
            "processing_time_ms": self._q(Decimal(str(elapsed * 1000))),
        }

        result_data["provenance_hash"] = _hash_allocation(result_data)

        with self._lock:
            self._total_exergy_allocations += 1

        self._record_metrics("exergy", fuel_type)

        logger.debug(
            "Exergy allocation: carnot=%s, heat_share=%s, power_share=%s",
            carnot_factor, heat_share, power_share,
        )

        return result_data

    # ==================================================================
    # PUBLIC METHOD 5: allocate_multiproduct
    # ==================================================================

    def allocate_multiproduct(
        self,
        total_fuel_gj: Decimal,
        fuel_type: str,
        heat_gj: Decimal,
        power_gj: Decimal,
        cooling_gj: Decimal,
        method: str = "efficiency",
        efficiencies: Optional[Dict[str, Decimal]] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Allocate CHP emissions across heat, power, and cooling outputs.

        Extends the two-product allocation to three products when the CHP
        plant also drives an absorption chiller for cooling. The cooling
        output is treated as a third energy product with its own fuel
        equivalent (cooling_gj / eta_cooling).

        For the efficiency method, fuel equivalents are:
            Heat_Fuel_Equiv    = heat_gj / eta_thermal
            Power_Fuel_Equiv   = power_gj / eta_electrical
            Cooling_Fuel_Equiv = cooling_gj / eta_cooling
            Total_Fuel_Equiv   = sum of all three

        For the energy method, shares are proportional to GJ output:
            Total_Output = heat_gj + power_gj + cooling_gj

        For the exergy method, cooling exergy uses the reverse Carnot
        factor (COP-based approximation).

        Args:
            total_fuel_gj: Total fuel input in GJ.
            fuel_type: Fuel type string for emission factor lookup.
            heat_gj: Thermal heat output in GJ.
            power_gj: Electrical power output in GJ.
            cooling_gj: Cooling output in GJ.
            method: Allocation method ('efficiency', 'energy', 'exergy').
            efficiencies: Dictionary with eta_thermal, eta_electrical,
                eta_cooling. If None, defaults are used.
            gwp_source: IPCC GWP source.

        Returns:
            Dictionary with three-way allocation shares, emissions for
            each product, trace, and provenance_hash.

        Raises:
            ValueError: If any input is invalid.
        """
        start = time.monotonic()
        trace: List[str] = []

        total_fuel_gj = _to_decimal(total_fuel_gj)
        heat_gj = _to_decimal(heat_gj)
        power_gj = _to_decimal(power_gj)
        cooling_gj = _to_decimal(cooling_gj)

        self._validate_positive("total_fuel_gj", total_fuel_gj)
        self._validate_positive("heat_gj", heat_gj)
        self._validate_positive("power_gj", power_gj)
        self._validate_positive("cooling_gj", cooling_gj)

        method = method.lower().strip()

        # Resolve efficiencies
        defaults = self._resolve_chp_defaults(fuel_type)
        eff = efficiencies or {}
        eta_t = _to_decimal(eff.get("eta_thermal", defaults["thermal_efficiency"]))
        eta_e = _to_decimal(eff.get("eta_electrical", defaults["electrical_efficiency"]))
        eta_c = _to_decimal(eff.get("eta_cooling", self._default_cooling_efficiency))

        self._validate_efficiency("eta_thermal", eta_t)
        self._validate_efficiency("eta_electrical", eta_e)
        self._validate_efficiency("eta_cooling", eta_c)

        trace.append(
            f"Multiproduct inputs: fuel={total_fuel_gj} GJ, "
            f"heat={heat_gj} GJ, power={power_gj} GJ, "
            f"cooling={cooling_gj} GJ, method={method}"
        )
        trace.append(
            f"Efficiencies: eta_t={eta_t}, eta_e={eta_e}, eta_c={eta_c}"
        )

        # Step 1: Compute fuel emissions
        fuel_emissions = self.compute_fuel_emissions(
            total_fuel_gj, fuel_type, gwp_source,
        )
        total_co2e = _to_decimal(fuel_emissions["total_co2e_kg"])
        trace.append(f"Total fuel emissions: {total_co2e} kgCO2e")

        # Step 2: Compute shares based on method
        if method == "efficiency":
            heat_share, power_share, cooling_share = (
                self._multiproduct_efficiency_shares(
                    heat_gj, power_gj, cooling_gj,
                    eta_t, eta_e, eta_c, trace,
                )
            )
        elif method == "energy":
            heat_share, power_share, cooling_share = (
                self._multiproduct_energy_shares(
                    heat_gj, power_gj, cooling_gj, trace,
                )
            )
        elif method == "exergy":
            heat_share, power_share, cooling_share = (
                self._multiproduct_exergy_shares(
                    heat_gj, power_gj, cooling_gj,
                    eta_c, trace,
                )
            )
        else:
            raise ValueError(
                f"Unknown multiproduct allocation method '{method}'. "
                f"Valid methods: efficiency, energy, exergy"
            )

        # Step 3: Allocate emissions
        heat_emissions = self._q(total_co2e * heat_share)
        power_emissions = self._q(total_co2e * power_share)
        cooling_emissions = self._q(total_co2e * cooling_share)

        trace.append(
            f"Allocated: heat={heat_emissions} kgCO2e, "
            f"power={power_emissions} kgCO2e, "
            f"cooling={cooling_emissions} kgCO2e"
        )

        elapsed = time.monotonic() - start
        result_data = {
            "method": method,
            "is_multiproduct": True,
            "heat_share": heat_share,
            "power_share": power_share,
            "cooling_share": cooling_share,
            "heat_emissions_kgco2e": heat_emissions,
            "power_emissions_kgco2e": power_emissions,
            "cooling_emissions_kgco2e": cooling_emissions,
            "total_fuel_emissions_kgco2e": self._q(total_co2e),
            "fuel_emissions_detail": fuel_emissions,
            "eta_thermal": eta_t,
            "eta_electrical": eta_e,
            "eta_cooling": eta_c,
            "fuel_type": fuel_type,
            "gwp_source": gwp_source or self._default_gwp_source,
            "trace": trace,
            "processing_time_ms": self._q(Decimal(str(elapsed * 1000))),
        }

        result_data["provenance_hash"] = _hash_allocation(result_data)

        with self._lock:
            self._total_multiproduct_allocations += 1

        self._record_metrics(f"multiproduct_{method}", fuel_type)

        logger.debug(
            "Multiproduct allocation: heat=%s, power=%s, cooling=%s",
            heat_share, power_share, cooling_share,
        )

        return result_data

    # ------------------------------------------------------------------
    # Internal: Multiproduct share computations
    # ------------------------------------------------------------------

    def _multiproduct_efficiency_shares(
        self,
        heat_gj: Decimal,
        power_gj: Decimal,
        cooling_gj: Decimal,
        eta_t: Decimal,
        eta_e: Decimal,
        eta_c: Decimal,
        trace: List[str],
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Compute three-way efficiency-based shares.

        Args:
            heat_gj: Heat output in GJ.
            power_gj: Power output in GJ.
            cooling_gj: Cooling output in GJ.
            eta_t: Thermal efficiency.
            eta_e: Electrical efficiency.
            eta_c: Cooling efficiency.
            trace: Trace list for recording steps.

        Returns:
            Tuple of (heat_share, power_share, cooling_share).
        """
        heat_fuel_equiv = self._q(heat_gj / eta_t)
        power_fuel_equiv = self._q(power_gj / eta_e)
        cooling_fuel_equiv = self._q(cooling_gj / eta_c)
        total_fuel_equiv = self._q(
            heat_fuel_equiv + power_fuel_equiv + cooling_fuel_equiv
        )

        trace.append(
            f"Fuel equivalents: heat={heat_fuel_equiv}, "
            f"power={power_fuel_equiv}, cooling={cooling_fuel_equiv}, "
            f"total={total_fuel_equiv}"
        )

        if total_fuel_equiv == _ZERO:
            raise ValueError(
                "Total fuel equivalent is zero in multiproduct allocation"
            )

        heat_share = self._q(heat_fuel_equiv / total_fuel_equiv)
        cooling_share = self._q(cooling_fuel_equiv / total_fuel_equiv)
        power_share = self._q(_ONE - heat_share - cooling_share)

        trace.append(
            f"Shares: heat={heat_share}, power={power_share}, "
            f"cooling={cooling_share}"
        )

        return heat_share, power_share, cooling_share

    def _multiproduct_energy_shares(
        self,
        heat_gj: Decimal,
        power_gj: Decimal,
        cooling_gj: Decimal,
        trace: List[str],
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Compute three-way energy-based shares.

        Args:
            heat_gj: Heat output in GJ.
            power_gj: Power output in GJ.
            cooling_gj: Cooling output in GJ.
            trace: Trace list for recording steps.

        Returns:
            Tuple of (heat_share, power_share, cooling_share).
        """
        total_output = self._q(heat_gj + power_gj + cooling_gj)

        if total_output == _ZERO:
            raise ValueError(
                "Total energy output is zero in multiproduct allocation"
            )

        heat_share = self._q(heat_gj / total_output)
        cooling_share = self._q(cooling_gj / total_output)
        power_share = self._q(_ONE - heat_share - cooling_share)

        trace.append(
            f"Energy shares: heat={heat_share}, power={power_share}, "
            f"cooling={cooling_share}, total_output={total_output} GJ"
        )

        return heat_share, power_share, cooling_share

    def _multiproduct_exergy_shares(
        self,
        heat_gj: Decimal,
        power_gj: Decimal,
        cooling_gj: Decimal,
        eta_c: Decimal,
        trace: List[str],
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Compute three-way exergy-based shares.

        Uses the default steam/ambient temperatures for the heat Carnot
        factor. Cooling exergy is approximated as cooling_gj / COP
        (inverse of the cooling efficiency), representing the work input
        equivalent of the cooling output.

        Args:
            heat_gj: Heat output in GJ.
            power_gj: Power output in GJ.
            cooling_gj: Cooling output in GJ.
            eta_c: Cooling efficiency (COP equivalent).
            trace: Trace list for recording steps.

        Returns:
            Tuple of (heat_share, power_share, cooling_share).
        """
        carnot = self.compute_carnot_factor(
            self._default_steam_temp_c, self._default_ambient_temp_c,
        )
        exergy_heat = self._q(heat_gj * carnot)
        exergy_power = self._q(power_gj)
        # Cooling exergy: approximate as work equivalent
        exergy_cooling = self._q(cooling_gj / eta_c) if eta_c > _ZERO else _ZERO
        total_exergy = self._q(exergy_heat + exergy_power + exergy_cooling)

        trace.append(
            f"Exergy: heat={exergy_heat} (carnot={carnot}), "
            f"power={exergy_power}, cooling={exergy_cooling}, "
            f"total={total_exergy}"
        )

        if total_exergy == _ZERO:
            raise ValueError(
                "Total exergy is zero in multiproduct allocation"
            )

        heat_share = self._q(exergy_heat / total_exergy)
        cooling_share = self._q(exergy_cooling / total_exergy)
        power_share = self._q(_ONE - heat_share - cooling_share)

        trace.append(
            f"Exergy shares: heat={heat_share}, power={power_share}, "
            f"cooling={cooling_share}"
        )

        return heat_share, power_share, cooling_share

    # ==================================================================
    # PUBLIC METHOD 6: compute_fuel_emissions
    # ==================================================================

    def compute_fuel_emissions(
        self,
        total_fuel_gj: Decimal,
        fuel_type: str,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute total CHP fuel combustion emissions.

        Calculates CO2, CH4, N2O emissions from fuel input using the
        IPCC emission factors, then converts to CO2e using GWP values.

        Formula:
            CO2_kg  = total_fuel_gj * co2_ef_per_gj
            CH4_kg  = total_fuel_gj * ch4_ef_per_gj
            N2O_kg  = total_fuel_gj * n2o_ef_per_gj
            CO2e_kg = CO2_kg + (CH4_kg * GWP_CH4) + (N2O_kg * GWP_N2O)

        For biogenic fuels, CO2 is reported separately in biogenic_co2_kg
        and excluded from the fossil total_co2e_kg.

        Args:
            total_fuel_gj: Total fuel input in GJ.
            fuel_type: Fuel type string for emission factor lookup.
            gwp_source: IPCC GWP source. Default from config.

        Returns:
            Dictionary with co2_kg, ch4_kg, n2o_kg, co2e_ch4_kg,
            co2e_n2o_kg, total_co2e_kg, biogenic_co2_kg,
            is_biogenic, gwp_source, fuel_type.

        Raises:
            ValueError: If fuel_type or gwp_source is invalid.
        """
        total_fuel_gj = _to_decimal(total_fuel_gj)
        self._validate_non_negative("total_fuel_gj", total_fuel_gj)

        fuel_ef = self._resolve_fuel_ef(fuel_type)
        gwp = self._resolve_gwp(gwp_source)

        co2_ef = fuel_ef["co2_ef"]
        ch4_ef = fuel_ef["ch4_ef"]
        n2o_ef = fuel_ef["n2o_ef"]
        is_biogenic = fuel_ef["is_biogenic"] == _ONE

        # Raw gas emissions (kg)
        co2_kg = self._q(total_fuel_gj * co2_ef)
        ch4_kg = self._q(total_fuel_gj * ch4_ef)
        n2o_kg = self._q(total_fuel_gj * n2o_ef)

        # GWP-weighted CO2e (kg)
        co2e_ch4 = self._q(ch4_kg * gwp["CH4"])
        co2e_n2o = self._q(n2o_kg * gwp["N2O"])

        # For biogenic fuels, CO2 is reported separately
        if is_biogenic:
            biogenic_co2_kg = co2_kg
            fossil_co2_kg = self._q(_ZERO)
            total_co2e = self._q(co2e_ch4 + co2e_n2o)
        else:
            biogenic_co2_kg = self._q(_ZERO)
            fossil_co2_kg = co2_kg
            total_co2e = self._q(fossil_co2_kg + co2e_ch4 + co2e_n2o)

        with self._lock:
            self._total_fuel_emissions_computed += 1

        return {
            "co2_kg": fossil_co2_kg if not is_biogenic else self._q(_ZERO),
            "ch4_kg": ch4_kg,
            "n2o_kg": n2o_kg,
            "co2e_ch4_kg": co2e_ch4,
            "co2e_n2o_kg": co2e_n2o,
            "total_co2e_kg": total_co2e,
            "biogenic_co2_kg": biogenic_co2_kg,
            "is_biogenic": is_biogenic,
            "gwp_source": gwp_source or self._default_gwp_source,
            "fuel_type": fuel_type,
            "total_fuel_gj": total_fuel_gj,
            "co2_ef_per_gj": co2_ef,
            "ch4_ef_per_gj": ch4_ef,
            "n2o_ef_per_gj": n2o_ef,
        }

    # ==================================================================
    # PUBLIC METHOD 7: compute_carnot_factor
    # ==================================================================

    def compute_carnot_factor(
        self,
        steam_temp_c: Decimal,
        ambient_temp_c: Decimal,
    ) -> Decimal:
        """Compute the Carnot efficiency factor for exergy analysis.

        The Carnot factor represents the maximum theoretical fraction of
        thermal energy that can be converted to work, based on the
        temperature difference between the heat source and the ambient
        environment.

        Formula:
            Carnot_Factor = 1 - (T_ambient + 273.15) / (T_steam + 273.15)

        Args:
            steam_temp_c: Steam/hot water supply temperature in Celsius.
            ambient_temp_c: Ambient reference temperature in Celsius.

        Returns:
            Carnot factor as Decimal (0-1).

        Raises:
            ValueError: If steam temp <= ambient temp.
        """
        steam_temp_c = _to_decimal(steam_temp_c)
        ambient_temp_c = _to_decimal(ambient_temp_c)

        if steam_temp_c <= ambient_temp_c:
            raise ValueError(
                f"steam_temp_c ({steam_temp_c}) must be greater than "
                f"ambient_temp_c ({ambient_temp_c})"
            )

        t_steam_k = steam_temp_c + _KELVIN_OFFSET
        t_ambient_k = ambient_temp_c + _KELVIN_OFFSET

        if t_steam_k <= _ZERO:
            raise ValueError(
                f"Steam temperature in Kelvin ({t_steam_k}) must be positive"
            )

        carnot = self._q(_ONE - (t_ambient_k / t_steam_k))

        logger.debug(
            "Carnot factor: T_steam=%s K, T_ambient=%s K, carnot=%s",
            t_steam_k, t_ambient_k, carnot,
        )

        return carnot

    # ==================================================================
    # PUBLIC METHOD 8: compute_primary_energy_savings
    # ==================================================================

    def compute_primary_energy_savings(
        self,
        eta_electrical: Decimal,
        eta_thermal: Decimal,
        eta_ref_electrical: Optional[Decimal] = None,
        eta_ref_thermal: Optional[Decimal] = None,
    ) -> Decimal:
        """Compute Primary Energy Savings (PES) per EU EED.

        PES measures the fuel savings achieved by CHP compared to
        separate production of heat and electricity at reference
        efficiencies. A positive PES means the CHP is more efficient.

        Formula (EU EED 2012/27/EU Annex II):
            PES = 1 - 1 / ((eta_elec / eta_ref_elec) + (eta_therm / eta_ref_therm))
            Expressed as percentage (PES * 100).

        Args:
            eta_electrical: CHP electrical efficiency (0-1).
            eta_thermal: CHP thermal efficiency (0-1).
            eta_ref_electrical: Reference electrical efficiency for
                separate generation. Default 0.525 (gas turbine).
            eta_ref_thermal: Reference thermal efficiency for separate
                generation. Default 0.90.

        Returns:
            PES as a percentage Decimal. Positive values indicate
            primary energy savings.

        Raises:
            ValueError: If efficiencies are invalid.
        """
        eta_electrical = _to_decimal(eta_electrical)
        eta_thermal = _to_decimal(eta_thermal)
        eta_ref_e = _to_decimal(
            eta_ref_electrical if eta_ref_electrical is not None
            else DEFAULT_REFERENCE_ELECTRICAL_EFFICIENCY
        )
        eta_ref_t = _to_decimal(
            eta_ref_thermal if eta_ref_thermal is not None
            else REFERENCE_THERMAL_EFFICIENCY
        )

        self._validate_efficiency("eta_electrical", eta_electrical)
        self._validate_efficiency("eta_thermal", eta_thermal)
        self._validate_efficiency("eta_ref_electrical", eta_ref_e)
        self._validate_efficiency("eta_ref_thermal", eta_ref_t)

        # PES = 1 - 1 / ((eta_e / eta_ref_e) + (eta_t / eta_ref_t))
        elec_ratio = self._q(eta_electrical / eta_ref_e)
        therm_ratio = self._q(eta_thermal / eta_ref_t)
        denominator = self._q(elec_ratio + therm_ratio)

        if denominator == _ZERO:
            raise ValueError(
                "PES denominator is zero; check efficiency values"
            )

        pes_fraction = self._q(_ONE - (_ONE / denominator))
        pes_pct = self._q_pct(pes_fraction * _HUNDRED)

        with self._lock:
            self._total_pes_computed += 1

        logger.debug(
            "PES: eta_e=%s, eta_t=%s, ref_e=%s, ref_t=%s -> PES=%s%%",
            eta_electrical, eta_thermal, eta_ref_e, eta_ref_t, pes_pct,
        )

        return pes_pct

    # ==================================================================
    # PUBLIC METHOD 9: is_high_efficiency_chp
    # ==================================================================

    def is_high_efficiency_chp(
        self,
        pes: Decimal,
        capacity_mw: Optional[Decimal] = None,
    ) -> bool:
        """Determine whether a CHP plant qualifies as high-efficiency.

        Per EU EED 2012/27/EU Article 2(34):
        - Large CHP (>= 1 MW): PES > 10%
        - Small CHP (< 1 MW):  PES > 0%

        Args:
            pes: Primary Energy Savings percentage.
            capacity_mw: CHP plant electrical capacity in MW.
                If None, applies the large plant threshold.

        Returns:
            True if the CHP plant qualifies as high-efficiency.
        """
        pes = _to_decimal(pes)
        if capacity_mw is not None:
            capacity_mw = _to_decimal(capacity_mw)
            if capacity_mw < SMALL_CHP_CAPACITY_MW:
                return pes > PES_THRESHOLD_SMALL
        return pes > PES_THRESHOLD_LARGE

    # ==================================================================
    # PUBLIC METHOD 10: compare_allocation_methods
    # ==================================================================

    def compare_allocation_methods(
        self,
        total_fuel_gj: Decimal,
        fuel_type: str,
        heat_gj: Decimal,
        power_gj: Decimal,
        efficiencies: Optional[Dict[str, Decimal]] = None,
        steam_temp_c: Optional[Decimal] = None,
        ambient_temp_c: Optional[Decimal] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare all three allocation methods side by side.

        Runs efficiency, energy, and exergy allocations with the same
        inputs and returns a comparison matrix showing how the choice
        of method affects the allocation of emissions to heat vs power.

        Args:
            total_fuel_gj: Total fuel input in GJ.
            fuel_type: Fuel type string.
            heat_gj: Heat output in GJ.
            power_gj: Power output in GJ.
            efficiencies: Dictionary with eta_thermal, eta_electrical.
            steam_temp_c: Steam temperature for exergy method.
            ambient_temp_c: Ambient temperature for exergy method.
            gwp_source: IPCC GWP source.

        Returns:
            Dictionary with:
                - efficiency: result from efficiency method
                - energy: result from energy method
                - exergy: result from exergy method
                - comparison_summary: comparative statistics
                - provenance_hash: SHA-256 of the comparison
        """
        start = time.monotonic()
        comp_id = str(uuid.uuid4())

        total_fuel_gj = _to_decimal(total_fuel_gj)
        heat_gj = _to_decimal(heat_gj)
        power_gj = _to_decimal(power_gj)

        defaults = self._resolve_chp_defaults(fuel_type)
        eff = efficiencies or {}
        eta_t = _to_decimal(eff.get("eta_thermal", defaults["thermal_efficiency"]))
        eta_e = _to_decimal(eff.get("eta_electrical", defaults["electrical_efficiency"]))

        steam_t = _to_decimal(
            steam_temp_c if steam_temp_c is not None
            else self._default_steam_temp_c
        )
        ambient_t = _to_decimal(
            ambient_temp_c if ambient_temp_c is not None
            else self._default_ambient_temp_c
        )

        # Run all three methods
        efficiency_result = self.allocate_efficiency_method(
            total_fuel_gj, fuel_type, heat_gj, power_gj,
            eta_t, eta_e, gwp_source,
        )
        energy_result = self.allocate_energy_method(
            total_fuel_gj, fuel_type, heat_gj, power_gj, gwp_source,
        )
        exergy_result = self.allocate_exergy_method(
            total_fuel_gj, fuel_type, heat_gj, power_gj,
            steam_t, ambient_t, gwp_source,
        )

        # Compute comparison summary
        heat_shares = {
            "efficiency": efficiency_result["heat_share"],
            "energy": energy_result["heat_share"],
            "exergy": exergy_result["heat_share"],
        }
        power_shares = {
            "efficiency": efficiency_result["power_share"],
            "energy": energy_result["power_share"],
            "exergy": exergy_result["power_share"],
        }

        heat_share_values = list(heat_shares.values())
        heat_share_min = min(heat_share_values)
        heat_share_max = max(heat_share_values)
        heat_share_range = self._q(heat_share_max - heat_share_min)

        heat_emissions_map = {
            "efficiency": efficiency_result["heat_emissions_kgco2e"],
            "energy": energy_result["heat_emissions_kgco2e"],
            "exergy": exergy_result["heat_emissions_kgco2e"],
        }

        power_emissions_map = {
            "efficiency": efficiency_result["power_emissions_kgco2e"],
            "energy": energy_result["power_emissions_kgco2e"],
            "exergy": exergy_result["power_emissions_kgco2e"],
        }

        # Method with highest heat allocation
        max_heat_method = max(heat_shares, key=heat_shares.get)  # type: ignore[arg-type]
        min_heat_method = min(heat_shares, key=heat_shares.get)  # type: ignore[arg-type]

        summary = {
            "heat_shares": {k: str(v) for k, v in heat_shares.items()},
            "power_shares": {k: str(v) for k, v in power_shares.items()},
            "heat_share_range": heat_share_range,
            "heat_emissions_range_kgco2e": self._q(
                max(heat_emissions_map.values())
                - min(heat_emissions_map.values())
            ),
            "max_heat_allocation_method": max_heat_method,
            "min_heat_allocation_method": min_heat_method,
            "recommended_method": "efficiency",
            "recommendation_reason": (
                "GHG Protocol Scope 2 Guidance recommends the efficiency "
                "method as it reflects the thermodynamic quality difference "
                "between electricity and heat outputs."
            ),
        }

        elapsed = time.monotonic() - start
        result = {
            "comparison_id": comp_id,
            "efficiency": efficiency_result,
            "energy": energy_result,
            "exergy": exergy_result,
            "comparison_summary": summary,
            "fuel_type": fuel_type,
            "total_fuel_gj": str(total_fuel_gj),
            "heat_output_gj": str(heat_gj),
            "power_output_gj": str(power_gj),
            "processing_time_ms": self._q(Decimal(str(elapsed * 1000))),
        }

        result["provenance_hash"] = _hash_allocation(result)

        with self._lock:
            self._total_comparisons += 1

        self._record_provenance(
            "chp_comparison", "compare", comp_id,
            {"fuel_type": fuel_type, "methods": ["efficiency", "energy", "exergy"]},
        )

        logger.info(
            "CHP method comparison %s: heat_share_range=%s, "
            "recommended=efficiency, elapsed=%.3fms",
            comp_id, heat_share_range, elapsed * 1000,
        )

        return result

    # ==================================================================
    # PUBLIC METHOD 11: validate_request
    # ==================================================================

    def validate_request(
        self, request: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Validate a CHP allocation request dictionary.

        Checks that all required fields are present and have valid types
        and ranges. Returns a tuple of (is_valid, error_messages).

        Args:
            request: Allocation request dictionary to validate.

        Returns:
            Tuple of (bool, List[str]) where the boolean indicates
            validity and the list contains error message strings.
        """
        errors: List[str] = []

        # Required fields
        required_fields = [
            "total_fuel_gj",
            "fuel_type",
            "heat_output_gj",
            "power_output_gj",
        ]
        for field in required_fields:
            if field not in request:
                errors.append(f"Missing required field: '{field}'")

        if errors:
            with self._lock:
                self._total_validations += 1
            return False, errors

        # Validate numeric fields are positive
        numeric_positive = ["total_fuel_gj", "heat_output_gj", "power_output_gj"]
        for field in numeric_positive:
            try:
                val = _to_decimal(request[field])
                if val <= _ZERO:
                    errors.append(f"'{field}' must be positive, got {val}")
            except (ValueError, TypeError) as exc:
                errors.append(f"'{field}' is not a valid number: {exc}")

        # Validate fuel type
        fuel_type = str(request.get("fuel_type", "")).lower().strip()
        if fuel_type and fuel_type not in FUEL_EMISSION_FACTORS:
            errors.append(
                f"Unknown fuel_type '{fuel_type}'. "
                f"Valid: {list(FUEL_EMISSION_FACTORS.keys())}"
            )

        # Validate method if provided
        method = str(request.get("method", "efficiency")).lower().strip()
        valid_methods = {"efficiency", "energy", "exergy"}
        if method not in valid_methods:
            errors.append(
                f"Unknown method '{method}'. Valid: {valid_methods}"
            )

        # Validate optional efficiency fields
        for eff_field in ["electrical_efficiency", "thermal_efficiency", "cooling_efficiency"]:
            if eff_field in request and request[eff_field] is not None:
                try:
                    val = _to_decimal(request[eff_field])
                    if val <= _ZERO or val >= _ONE:
                        errors.append(
                            f"'{eff_field}' must be between 0 and 1 "
                            f"(exclusive), got {val}"
                        )
                except (ValueError, TypeError) as exc:
                    errors.append(
                        f"'{eff_field}' is not a valid number: {exc}"
                    )

        # Validate optional cooling_output_gj
        if "cooling_output_gj" in request and request["cooling_output_gj"] is not None:
            try:
                val = _to_decimal(request["cooling_output_gj"])
                if val < _ZERO:
                    errors.append(
                        f"'cooling_output_gj' must be non-negative, got {val}"
                    )
            except (ValueError, TypeError) as exc:
                errors.append(
                    f"'cooling_output_gj' is not a valid number: {exc}"
                )

        # Validate GWP source if provided
        gwp_source = request.get("gwp_source")
        if gwp_source is not None:
            source_upper = str(gwp_source).upper()
            if source_upper not in GWP_VALUES:
                errors.append(
                    f"Unknown gwp_source '{gwp_source}'. "
                    f"Valid: {list(GWP_VALUES.keys())}"
                )

        # Validate temperature fields for exergy
        if method == "exergy":
            steam_t = request.get("steam_temperature_c")
            ambient_t = request.get("ambient_temperature_c")
            if steam_t is not None and ambient_t is not None:
                try:
                    st = _to_decimal(steam_t)
                    at = _to_decimal(ambient_t)
                    if st <= at:
                        errors.append(
                            f"steam_temperature_c ({st}) must be greater "
                            f"than ambient_temperature_c ({at})"
                        )
                except (ValueError, TypeError) as exc:
                    errors.append(f"Temperature validation error: {exc}")

        with self._lock:
            self._total_validations += 1

        is_valid = len(errors) == 0
        return is_valid, errors

    # ==================================================================
    # PUBLIC METHOD 12: batch_allocate
    # ==================================================================

    def batch_allocate(
        self, requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform batch CHP emission allocation across multiple plants.

        Processes a list of allocation requests, collecting results and
        errors. Returns a summary with individual results, aggregate
        statistics, and provenance hash.

        Args:
            requests: List of allocation request dictionaries.

        Returns:
            Dictionary with:
                - batch_id: UUID for this batch
                - total_requests: Number of requests
                - successful: Number of successful allocations
                - failed: Number of failed allocations
                - results: List of individual result dictionaries
                - errors: List of error dictionaries
                - aggregate: Aggregate statistics across all results
                - provenance_hash: SHA-256 of the batch
        """
        start = time.monotonic()
        batch_id = str(uuid.uuid4())

        if not requests:
            return {
                "batch_id": batch_id,
                "total_requests": 0,
                "successful": 0,
                "failed": 0,
                "results": [],
                "errors": [],
                "aggregate": {},
                "provenance_hash": _hash_allocation({"batch_id": batch_id}),
            }

        results: List[Dict[str, Any]] = []
        batch_errors: List[Dict[str, Any]] = []
        total_heat_emissions = _ZERO
        total_power_emissions = _ZERO
        total_cooling_emissions = _ZERO
        total_fuel_emissions = _ZERO

        for idx, req in enumerate(requests):
            try:
                result = self.allocate_chp_emissions(req)
                results.append(result)

                total_heat_emissions += _to_decimal(
                    result.get("heat_emissions_kgco2e", 0)
                )
                total_power_emissions += _to_decimal(
                    result.get("power_emissions_kgco2e", 0)
                )
                total_cooling_emissions += _to_decimal(
                    result.get("cooling_emissions_kgco2e", 0)
                )
                total_fuel_emissions += _to_decimal(
                    result.get("total_fuel_emissions_kgco2e", 0)
                )

            except Exception as exc:
                batch_errors.append({
                    "index": idx,
                    "error": str(exc),
                    "request_keys": list(req.keys()) if isinstance(req, dict) else [],
                })

        # Aggregate statistics
        aggregate = {
            "total_heat_emissions_kgco2e": self._q(total_heat_emissions),
            "total_power_emissions_kgco2e": self._q(total_power_emissions),
            "total_cooling_emissions_kgco2e": self._q(total_cooling_emissions),
            "total_fuel_emissions_kgco2e": self._q(total_fuel_emissions),
            "average_heat_share": self._q(
                sum(
                    _to_decimal(r.get("heat_share", 0)) for r in results
                ) / Decimal(str(len(results)))
            ) if results else _ZERO,
            "average_power_share": self._q(
                sum(
                    _to_decimal(r.get("power_share", 0)) for r in results
                ) / Decimal(str(len(results)))
            ) if results else _ZERO,
        }

        elapsed = time.monotonic() - start
        batch_result = {
            "batch_id": batch_id,
            "total_requests": len(requests),
            "successful": len(results),
            "failed": len(batch_errors),
            "results": results,
            "errors": batch_errors,
            "aggregate": aggregate,
            "processing_time_ms": self._q(Decimal(str(elapsed * 1000))),
        }

        batch_result["provenance_hash"] = _hash_allocation(batch_result)

        with self._lock:
            self._total_batch_allocations += 1

        self._record_provenance(
            "chp_batch", "batch_allocate", batch_id,
            {
                "total_requests": len(requests),
                "successful": len(results),
                "failed": len(batch_errors),
            },
        )

        logger.info(
            "CHP batch allocation %s: %d/%d successful, "
            "%d failed, elapsed=%.3fms",
            batch_id, len(results), len(requests),
            len(batch_errors), elapsed * 1000,
        )

        return batch_result

    # ==================================================================
    # PUBLIC METHOD 13: get_allocation_stats
    # ==================================================================

    def get_allocation_stats(self) -> Dict[str, Any]:
        """Return operational statistics for this engine.

        Returns:
            Dictionary with allocation counts, error count, uptime,
            engine version, and provenance chain status.
        """
        with self._lock:
            uptime_seconds = (
                _utcnow() - self._created_at
            ).total_seconds()

            stats = {
                "engine_id": self.ENGINE_ID,
                "engine_version": self.ENGINE_VERSION,
                "created_at": self._created_at.isoformat(),
                "uptime_seconds": uptime_seconds,
                "total_allocations": self._total_allocations,
                "efficiency_allocations": self._total_efficiency_allocations,
                "energy_allocations": self._total_energy_allocations,
                "exergy_allocations": self._total_exergy_allocations,
                "multiproduct_allocations": self._total_multiproduct_allocations,
                "fuel_emissions_computed": self._total_fuel_emissions_computed,
                "pes_computed": self._total_pes_computed,
                "comparisons": self._total_comparisons,
                "batch_allocations": self._total_batch_allocations,
                "validations": self._total_validations,
                "total_errors": self._total_errors,
                "provenance_enabled": self._enable_provenance,
                "provenance_entries": (
                    self._provenance.entry_count
                    if self._provenance is not None else 0
                ),
                "provenance_chain_valid": (
                    self._provenance.verify_chain()
                    if self._provenance is not None else None
                ),
                "default_gwp_source": self._default_gwp_source,
                "default_ambient_temp_c": str(self._default_ambient_temp_c),
                "default_steam_temp_c": str(self._default_steam_temp_c),
                "tenant_id": self._tenant_id,
            }

        stats["provenance_hash"] = _hash_allocation(stats)
        return stats

    # ==================================================================
    # PUBLIC METHOD 14: health_check
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the CHP allocation engine.

        Validates that the engine is operational by running a minimal
        test allocation and verifying the provenance chain.

        Returns:
            Dictionary with status ('healthy' or 'unhealthy'), checks
            performed, and diagnostic details.
        """
        checks: List[Dict[str, Any]] = []
        overall_healthy = True

        # Check 1: Engine initialized
        checks.append({
            "check": "engine_initialized",
            "status": "pass" if self._initialized else "fail",
        })
        if not self._initialized:
            overall_healthy = False

        # Check 2: GWP values loaded
        gwp_ok = len(GWP_VALUES) >= 3
        checks.append({
            "check": "gwp_values_loaded",
            "status": "pass" if gwp_ok else "fail",
            "detail": f"{len(GWP_VALUES)} GWP sources available",
        })
        if not gwp_ok:
            overall_healthy = False

        # Check 3: Fuel emission factors loaded
        ef_ok = len(FUEL_EMISSION_FACTORS) >= 10
        checks.append({
            "check": "fuel_emission_factors_loaded",
            "status": "pass" if ef_ok else "fail",
            "detail": f"{len(FUEL_EMISSION_FACTORS)} fuel types available",
        })
        if not ef_ok:
            overall_healthy = False

        # Check 4: CHP default efficiencies loaded
        chp_ok = len(CHP_DEFAULT_EFFICIENCIES) >= 5
        checks.append({
            "check": "chp_default_efficiencies_loaded",
            "status": "pass" if chp_ok else "fail",
            "detail": f"{len(CHP_DEFAULT_EFFICIENCIES)} CHP fuel categories",
        })
        if not chp_ok:
            overall_healthy = False

        # Check 5: Test allocation
        try:
            test_result = self.allocate_energy_method(
                total_fuel_gj=Decimal("100"),
                fuel_type="natural_gas",
                heat_output_gj=Decimal("40"),
                power_output_gj=Decimal("35"),
            )
            alloc_ok = (
                test_result.get("heat_share", _ZERO) > _ZERO
                and test_result.get("power_share", _ZERO) > _ZERO
                and "provenance_hash" in test_result
            )
            checks.append({
                "check": "test_allocation",
                "status": "pass" if alloc_ok else "fail",
                "detail": "Energy method test allocation succeeded",
            })
            if not alloc_ok:
                overall_healthy = False
        except Exception as exc:
            checks.append({
                "check": "test_allocation",
                "status": "fail",
                "detail": f"Test allocation failed: {exc}",
            })
            overall_healthy = False

        # Check 6: Provenance chain integrity
        if self._provenance is not None:
            chain_ok = self._provenance.verify_chain()
            checks.append({
                "check": "provenance_chain_integrity",
                "status": "pass" if chain_ok else "fail",
                "detail": f"{self._provenance.entry_count} entries in chain",
            })
            if not chain_ok:
                overall_healthy = False
        else:
            checks.append({
                "check": "provenance_chain_integrity",
                "status": "skip",
                "detail": "Provenance tracking disabled",
            })

        result = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "engine_id": self.ENGINE_ID,
            "engine_version": self.ENGINE_VERSION,
            "timestamp": _utcnow().isoformat(),
            "checks": checks,
            "checks_passed": sum(1 for c in checks if c["status"] == "pass"),
            "checks_total": len(checks),
        }

        result["provenance_hash"] = _hash_allocation(result)
        return result

    # ==================================================================
    # PUBLIC METHOD 15: get_default_efficiencies
    # ==================================================================

    def get_default_efficiencies(self, fuel_type: str) -> Dict[str, Any]:
        """Get default CHP efficiencies for a fuel type.

        Returns the default electrical, thermal, and overall efficiencies
        used when explicit values are not provided in the allocation
        request.

        Args:
            fuel_type: Fuel type string (e.g. 'natural_gas', 'coal_bituminous').

        Returns:
            Dictionary with:
                - fuel_type: The input fuel type
                - chp_category: The CHP efficiency category used
                - electrical_efficiency: Default eta_electrical
                - thermal_efficiency: Default eta_thermal
                - overall_efficiency: Default combined efficiency
                - reference_electrical: EU EED reference eta_electrical
                - reference_thermal: EU EED reference eta_thermal
        """
        fuel_key = fuel_type.lower().strip()
        chp_key = _FUEL_TYPE_TO_CHP_KEY.get(fuel_key, "natural_gas")
        defaults = CHP_DEFAULT_EFFICIENCIES.get(
            chp_key, CHP_DEFAULT_EFFICIENCIES["natural_gas"]
        )

        return {
            "fuel_type": fuel_type,
            "chp_category": chp_key,
            "electrical_efficiency": defaults["electrical_efficiency"],
            "thermal_efficiency": defaults["thermal_efficiency"],
            "overall_efficiency": defaults["overall_efficiency"],
            "reference_electrical_efficiency": DEFAULT_REFERENCE_ELECTRICAL_EFFICIENCY,
            "reference_thermal_efficiency": REFERENCE_THERMAL_EFFICIENCY,
            "reference_efficiencies_by_technology": dict(REFERENCE_ELECTRICAL_EFFICIENCIES),
        }

    # ==================================================================
    # PUBLIC METHOD 16: compute_overall_efficiency
    # ==================================================================

    def compute_overall_efficiency(
        self,
        eta_electrical: Decimal,
        eta_thermal: Decimal,
    ) -> Decimal:
        """Compute the overall CHP efficiency.

        The overall efficiency is the sum of electrical and thermal
        efficiencies, representing the total fraction of fuel energy
        converted to useful output. A well-designed CHP plant typically
        achieves 65-85% overall efficiency.

        Formula:
            Overall_Efficiency = eta_electrical + eta_thermal

        Args:
            eta_electrical: Electrical conversion efficiency (0-1).
            eta_thermal: Thermal conversion efficiency (0-1).

        Returns:
            Overall efficiency as Decimal.

        Raises:
            ValueError: If efficiencies are invalid or sum exceeds 1.
        """
        eta_electrical = _to_decimal(eta_electrical)
        eta_thermal = _to_decimal(eta_thermal)

        self._validate_efficiency("eta_electrical", eta_electrical)
        self._validate_efficiency("eta_thermal", eta_thermal)

        overall = self._q(eta_electrical + eta_thermal)

        if overall > _ONE:
            raise ValueError(
                f"Overall efficiency ({overall}) exceeds 1.0. "
                f"eta_electrical={eta_electrical}, eta_thermal={eta_thermal}"
            )

        return overall

    # ==================================================================
    # Validation helpers
    # ==================================================================

    def _validate_positive(self, name: str, value: Decimal) -> None:
        """Validate that a Decimal value is strictly positive.

        Args:
            name: Field name for error messages.
            value: Decimal value to validate.

        Raises:
            ValueError: If value is not positive.
        """
        if value <= _ZERO:
            raise ValueError(f"'{name}' must be positive, got {value}")

    def _validate_non_negative(self, name: str, value: Decimal) -> None:
        """Validate that a Decimal value is non-negative.

        Args:
            name: Field name for error messages.
            value: Decimal value to validate.

        Raises:
            ValueError: If value is negative.
        """
        if value < _ZERO:
            raise ValueError(f"'{name}' must be non-negative, got {value}")

    def _validate_efficiency(self, name: str, value: Decimal) -> None:
        """Validate that a Decimal value is a valid efficiency (0,1).

        Args:
            name: Field name for error messages.
            value: Decimal value to validate.

        Raises:
            ValueError: If value is not in (0, 1).
        """
        if value <= _ZERO or value >= _ONE:
            raise ValueError(
                f"'{name}' must be between 0 and 1 (exclusive), got {value}"
            )

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(self) -> None:
        """Reset all mutable state to initial values.

        Clears operational counters and provenance entries. Built-in
        constants (GWP, emission factors, efficiencies) are not affected.

        Primarily used for testing and development.
        """
        with self._lock:
            self._total_allocations = 0
            self._total_efficiency_allocations = 0
            self._total_energy_allocations = 0
            self._total_exergy_allocations = 0
            self._total_multiproduct_allocations = 0
            self._total_fuel_emissions_computed = 0
            self._total_pes_computed = 0
            self._total_comparisons = 0
            self._total_batch_allocations = 0
            self._total_validations = 0
            self._total_errors = 0

        if self._provenance is not None:
            self._provenance.reset()

        logger.info("CHPAllocationEngine reset: all mutable state cleared")

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance for testing.

        After calling this method, the next instantiation will create
        a fresh instance. This is intended for test isolation only.
        """
        with cls._cls_lock:
            cls._instance = None
        logger.info("CHPAllocationEngine singleton reset")

    # ==================================================================
    # Dunder methods
    # ==================================================================

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"CHPAllocationEngine("
            f"allocations={self._total_allocations}, "
            f"efficiency={self._total_efficiency_allocations}, "
            f"energy={self._total_energy_allocations}, "
            f"exergy={self._total_exergy_allocations}, "
            f"multiproduct={self._total_multiproduct_allocations}, "
            f"errors={self._total_errors})"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        return (
            f"CHPAllocationEngine v{VERSION} - "
            f"{self._total_allocations} allocations, "
            f"{self._total_comparisons} comparisons, "
            f"{self._total_batch_allocations} batches"
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_module_engine: Optional[CHPAllocationEngine] = None
_module_lock = threading.Lock()


def get_chp_allocator(
    config: Optional[Dict[str, Any]] = None,
) -> CHPAllocationEngine:
    """Get or create the module-level CHPAllocationEngine singleton.

    This is the recommended entry point for obtaining a CHPAllocationEngine
    instance. The singleton is created on first call and reused thereafter.

    Args:
        config: Optional configuration dictionary (used only on first call).

    Returns:
        The CHPAllocationEngine singleton instance.

    Example:
        >>> engine = get_chp_allocator()
        >>> result = engine.allocate_energy_method(
        ...     total_fuel_gj=Decimal("500"),
        ...     fuel_type="natural_gas",
        ...     heat_output_gj=Decimal("200"),
        ...     power_output_gj=Decimal("175"),
        ... )
    """
    global _module_engine
    if _module_engine is None:
        with _module_lock:
            if _module_engine is None:
                _module_engine = CHPAllocationEngine(config=config)
    return _module_engine


def reset_chp_allocator() -> None:
    """Reset the module-level CHPAllocationEngine singleton.

    After calling this function, the next call to ``get_chp_allocator()``
    will create a fresh instance. Intended for test isolation.
    """
    global _module_engine
    with _module_lock:
        if _module_engine is not None:
            _module_engine.reset()
        _module_engine = None
    CHPAllocationEngine.reset_singleton()
    logger.info("Module-level CHPAllocationEngine reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = ["CHPAllocationEngine"]
