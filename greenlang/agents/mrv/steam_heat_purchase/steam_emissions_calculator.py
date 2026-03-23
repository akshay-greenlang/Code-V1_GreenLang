# -*- coding: utf-8 -*-
"""
SteamEmissionsCalculatorEngine - Engine 2: Steam/Heat Purchase Agent (AGENT-MRV-011)

Core purchased-steam emission calculations using fuel-based, supplier-specific,
and blended multi-fuel methods with condensate return adjustment, biogenic CO2
separation, per-gas GHG breakdown (CO2, CH4, N2O), and GWP conversion per the
GHG Protocol Scope 2 Guidance for purchased steam and heat.

Formulas:
    Direct supplier EF:
        Emissions (kgCO2e) = Consumption (GJ) x Supplier_EF (kgCO2e/GJ)

    Fuel-based calculation:
        Fuel_Input (GJ) = Steam_Consumption (GJ) / Boiler_Efficiency
        Emissions_CO2 (kg) = Fuel_Input (GJ) x CO2_EF (kgCO2/GJ)
        Emissions_CH4 (kg) = Fuel_Input (GJ) x CH4_EF (kgCH4/GJ)
        Emissions_N2O (kg) = Fuel_Input (GJ) x N2O_EF (kgN2O/GJ)
        Total_CO2e (kg) = CO2 + (CH4 x GWP_CH4) + (N2O x GWP_N2O)

    Condensate return adjustment:
        Effective_Consumption (GJ) = Consumption (GJ) x (1 - Condensate_Return_Pct / 100)

    Multi-fuel blended steam:
        Blended_EF = SUM(Fuel_Fraction_i x EF_i)
        Emissions = Fuel_Input x Blended_EF

    Biogenic CO2 separation:
        For biogenic fuels: CO2 reported as biogenic_co2 (separate line item)
        Non-biogenic CH4 and N2O still count as fossil GHG

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places internal)
    - No LLM calls in the calculation path
    - Every step is recorded in the calculation trace
    - SHA-256 provenance hash for every result
    - Deterministic: same input -> same output (bit-perfect)

Example:
    >>> from greenlang.agents.mrv.steam_heat_purchase.steam_emissions_calculator import (
    ...     SteamEmissionsCalculatorEngine,
    ... )
    >>> from decimal import Decimal
    >>> engine = SteamEmissionsCalculatorEngine()
    >>> result = engine.calculate_with_supplier_ef(
    ...     consumption_gj=Decimal("1000"),
    ...     supplier_ef=Decimal("66.5"),
    ... )
    >>> assert result["total_co2e_kg"] == Decimal("66500.000")

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
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional imports: metrics
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.steam_heat_purchase.metrics import (
        SteamHeatPurchaseMetrics,
        get_metrics as _get_metrics,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _get_metrics = None  # type: ignore[assignment]
    SteamHeatPurchaseMetrics = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Conditional imports: database engine
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.steam_heat_purchase.steam_heat_database import (
        SteamHeatDatabaseEngine,
    )
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    SteamHeatDatabaseEngine = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Decimal precision constants
# ---------------------------------------------------------------------------
_PRECISION_INTERNAL = Decimal("0.00000001")  # 8 decimal places internal
_PRECISION_OUTPUT = Decimal("0.001")          # 3 decimal places output

_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")
_THOUSAND = Decimal("1000")


# ---------------------------------------------------------------------------
# Unit Conversion Constants (all Decimal for zero-hallucination)
# ---------------------------------------------------------------------------

#: 1 MWh = 3.6 GJ
_MWH_TO_GJ = Decimal("3.6")

#: 1 GJ = 0.277778 MWh (1 / 3.6)
_GJ_TO_MWH = Decimal("0.277778")

#: 1 MMBTU = 1.055056 GJ
_MMBTU_TO_GJ = Decimal("1.055056")

#: 1 therm = 0.105506 GJ
_THERM_TO_GJ = Decimal("0.105506")

#: 1 MJ = 0.001 GJ
_MJ_TO_GJ = Decimal("0.001")

#: 1 kWh = 0.0036 GJ
_KWH_TO_GJ = Decimal("0.0036")

#: kg to tonnes
_KG_TO_TONNES = Decimal("0.001")

#: tonnes to kg
_TONNES_TO_KG = Decimal("1000")


# ---------------------------------------------------------------------------
# GWP Values by IPCC Assessment Report
# ---------------------------------------------------------------------------

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
# Fuel Emission Factors (kgCO2, kgCH4, kgN2O per GJ of fuel input, HHV)
# ---------------------------------------------------------------------------
# Sources:
#   IPCC 2006 Guidelines Vol 2, Ch 2, Tables 2.2, 2.4
#   US EPA AP-42 Compilation of Air Emission Factors
#   UK BEIS GHG Reporting Conversion Factors

FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "natural_gas": {
        "co2_ef": Decimal("56.100"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0001"),
        "default_efficiency": Decimal("0.85"),
        "is_biogenic": False,
        "description": "Pipeline-quality natural gas",
    },
    "fuel_oil_2": {
        "co2_ef": Decimal("74.100"),
        "ch4_ef": Decimal("0.003"),
        "n2o_ef": Decimal("0.0006"),
        "default_efficiency": Decimal("0.82"),
        "is_biogenic": False,
        "description": "Light distillate fuel oil (No. 2 / diesel oil)",
    },
    "fuel_oil_6": {
        "co2_ef": Decimal("77.400"),
        "ch4_ef": Decimal("0.003"),
        "n2o_ef": Decimal("0.0006"),
        "default_efficiency": Decimal("0.80"),
        "is_biogenic": False,
        "description": "Heavy residual fuel oil (No. 6 / bunker C)",
    },
    "coal_bituminous": {
        "co2_ef": Decimal("94.600"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0015"),
        "default_efficiency": Decimal("0.78"),
        "is_biogenic": False,
        "description": "Bituminous coal (hard coal)",
    },
    "coal_subbituminous": {
        "co2_ef": Decimal("96.100"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0015"),
        "default_efficiency": Decimal("0.75"),
        "is_biogenic": False,
        "description": "Subbituminous coal",
    },
    "coal_lignite": {
        "co2_ef": Decimal("101.000"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0015"),
        "default_efficiency": Decimal("0.72"),
        "is_biogenic": False,
        "description": "Lignite (brown coal)",
    },
    "lpg": {
        "co2_ef": Decimal("63.100"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0001"),
        "default_efficiency": Decimal("0.83"),
        "is_biogenic": False,
        "description": "Liquefied petroleum gas (propane/butane mix)",
    },
    "biomass_wood": {
        "co2_ef": Decimal("112.000"),
        "ch4_ef": Decimal("0.030"),
        "n2o_ef": Decimal("0.004"),
        "default_efficiency": Decimal("0.70"),
        "is_biogenic": True,
        "description": "Wood chips, wood pellets, or other woody biomass",
    },
    "biomass_biogas": {
        "co2_ef": Decimal("54.600"),
        "ch4_ef": Decimal("0.005"),
        "n2o_ef": Decimal("0.001"),
        "default_efficiency": Decimal("0.75"),
        "is_biogenic": True,
        "description": "Biogas from anaerobic digestion",
    },
    "municipal_waste": {
        "co2_ef": Decimal("91.700"),
        "ch4_ef": Decimal("0.030"),
        "n2o_ef": Decimal("0.004"),
        "default_efficiency": Decimal("0.65"),
        "is_biogenic": False,
        "description": "Municipal solid waste incineration with heat recovery",
    },
    "waste_heat": {
        "co2_ef": Decimal("0"),
        "ch4_ef": Decimal("0"),
        "n2o_ef": Decimal("0"),
        "default_efficiency": Decimal("1.00"),
        "is_biogenic": False,
        "description": "Recovered waste heat - zero direct emissions",
    },
    "geothermal": {
        "co2_ef": Decimal("0"),
        "ch4_ef": Decimal("0"),
        "n2o_ef": Decimal("0"),
        "default_efficiency": Decimal("1.00"),
        "is_biogenic": False,
        "description": "Geothermal heat",
    },
    "solar_thermal": {
        "co2_ef": Decimal("0"),
        "ch4_ef": Decimal("0"),
        "n2o_ef": Decimal("0"),
        "default_efficiency": Decimal("1.00"),
        "is_biogenic": False,
        "description": "Solar thermal collectors",
    },
    "electric": {
        "co2_ef": Decimal("0"),
        "ch4_ef": Decimal("0"),
        "n2o_ef": Decimal("0"),
        "default_efficiency": Decimal("0.98"),
        "is_biogenic": False,
        "description": "Electric boiler or heat pump (grid emissions separate)",
    },
}

#: Set of biogenic fuel types for quick lookup.
BIOGENIC_FUEL_TYPES: Set[str] = {
    ft for ft, props in FUEL_EMISSION_FACTORS.items()
    if props.get("is_biogenic") is True
}

#: Set of zero-emission fuel types (waste heat, geothermal, solar, electric).
ZERO_EMISSION_FUEL_TYPES: Set[str] = {
    ft for ft, props in FUEL_EMISSION_FACTORS.items()
    if props["co2_ef"] == _ZERO and props["ch4_ef"] == _ZERO and props["n2o_ef"] == _ZERO
}

#: Valid fuel type identifiers.
VALID_FUEL_TYPES: Set[str] = set(FUEL_EMISSION_FACTORS.keys())


# ---------------------------------------------------------------------------
# Default boiler efficiency bounds
# ---------------------------------------------------------------------------
_MIN_BOILER_EFFICIENCY = Decimal("0.50")
_MAX_BOILER_EFFICIENCY = Decimal("1.00")
_DEFAULT_BOILER_EFFICIENCY = Decimal("0.80")

# ---------------------------------------------------------------------------
# Condensate return bounds
# ---------------------------------------------------------------------------
_MIN_CONDENSATE_RETURN_PCT = Decimal("0")
_MAX_CONDENSATE_RETURN_PCT = Decimal("95")

# ---------------------------------------------------------------------------
# Maximum batch size
# ---------------------------------------------------------------------------
_MAX_BATCH_SIZE = 10_000


# ---------------------------------------------------------------------------
# Prometheus metrics helpers (graceful fallback)
# ---------------------------------------------------------------------------

def _record_calculation_metric(
    energy_type: str,
    method: str,
    status: str,
    duration: float,
    co2e_kg: float,
    biogenic_kg: float,
    fuel_type: str,
    tenant_id: str = "default",
) -> None:
    """Record a steam emission calculation metric.

    Args:
        energy_type: Type of thermal energy (steam, hot_water, etc.).
        method: Calculation method used.
        status: Outcome (success, failure).
        duration: Calculation duration in seconds.
        co2e_kg: Total CO2e emissions in kg.
        biogenic_kg: Biogenic CO2 emissions in kg.
        fuel_type: Fuel type used.
        tenant_id: Tenant identifier.
    """
    if _METRICS_AVAILABLE and _get_metrics is not None:
        try:
            _get_metrics().record_calculation(
                energy_type=energy_type,
                method=method,
                status=status,
                duration=duration,
                co2e_kg=co2e_kg,
                biogenic_kg=biogenic_kg,
                fuel_type=fuel_type,
                tenant_id=tenant_id,
            )
        except Exception:
            pass


def _record_batch_metric(
    status: str,
    size: int,
    tenant_id: str = "default",
) -> None:
    """Record a batch calculation metric.

    Args:
        status: Batch outcome (success, failure, partial).
        size: Number of records in the batch.
        tenant_id: Tenant identifier.
    """
    if _METRICS_AVAILABLE and _get_metrics is not None:
        try:
            _get_metrics().record_batch(
                status=status, size=size, tenant_id=tenant_id,
            )
        except Exception:
            pass


def _record_error_metric(
    error_type: str,
    tenant_id: str = "default",
) -> None:
    """Record a calculator error metric.

    Args:
        error_type: Classification of the error.
        tenant_id: Tenant identifier.
    """
    if _METRICS_AVAILABLE and _get_metrics is not None:
        try:
            _get_metrics().record_error(
                engine="calculator",
                error_type=error_type,
                tenant_id=tenant_id,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Provenance helper (lightweight inline tracker)
# ---------------------------------------------------------------------------

class _ProvenanceTracker:
    """Lightweight chain-hashing provenance tracker for steam emissions.

    Each recorded entry chains its SHA-256 hash to the previous entry,
    producing a tamper-evident audit log.
    """

    def __init__(
        self,
        genesis: str = "GL-MRV-011-STEAM-EMISSIONS-CALCULATOR-GENESIS",
    ) -> None:
        """Initialize the provenance tracker.

        Args:
            genesis: Seed string for the genesis hash.
        """
        self._genesis: str = hashlib.sha256(genesis.encode("utf-8")).hexdigest()
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
            entity_type: Category of the recorded entity.
            action: Action name (e.g. method name).
            entity_id: Unique identifier for this entry.
            data: Optional data payload for hashing.

        Returns:
            Dictionary with entry metadata and chain hash.
        """
        ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
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
        """Verify integrity of the entire provenance chain.

        Returns:
            True if chain is valid, False if tampered.
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
        """Return a copy of all provenance entries.

        Returns:
            List of provenance entry dictionaries.
        """
        with self._lock:
            return list(self._entries)

    @property
    def entry_count(self) -> int:
        """Return the number of provenance entries recorded."""
        with self._lock:
            return len(self._entries)


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


def _q(value: Decimal) -> Decimal:
    """Quantize a Decimal to 8 internal decimal places.

    Args:
        value: Decimal to quantize.

    Returns:
        Quantized Decimal.
    """
    return value.quantize(_PRECISION_INTERNAL, rounding=ROUND_HALF_UP)


def _q_out(value: Decimal) -> Decimal:
    """Quantize a Decimal to 3 output decimal places.

    Args:
        value: Decimal to quantize.

    Returns:
        Quantized Decimal.
    """
    return value.quantize(_PRECISION_OUTPUT, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Energy unit normalization to GJ
# ---------------------------------------------------------------------------

_ENERGY_UNIT_TO_GJ: Dict[str, Decimal] = {
    "GJ": _ONE,
    "MWH": _MWH_TO_GJ,
    "KWH": _KWH_TO_GJ,
    "MMBTU": _MMBTU_TO_GJ,
    "THERM": _THERM_TO_GJ,
    "MJ": _MJ_TO_GJ,
}


def _normalize_to_gj(quantity: Decimal, unit: str) -> Decimal:
    """Convert an energy quantity from any supported unit to GJ.

    Args:
        quantity: Energy quantity. Must be >= 0.
        unit: Unit string (case-insensitive). Supported: GJ, MWH, KWH,
            MMBTU, THERM, MJ.

    Returns:
        Energy in GJ (Decimal).

    Raises:
        ValueError: If unit is unsupported or quantity is negative.
    """
    if quantity < _ZERO:
        raise ValueError(f"Energy quantity must be >= 0, got {quantity}")
    unit_upper = unit.strip().upper()
    factor = _ENERGY_UNIT_TO_GJ.get(unit_upper)
    if factor is None:
        raise ValueError(
            f"Unsupported energy unit '{unit}'. "
            f"Supported: {sorted(_ENERGY_UNIT_TO_GJ.keys())}"
        )
    return _q(quantity * factor)


# ===========================================================================
# SteamEmissionsCalculatorEngine
# ===========================================================================


class SteamEmissionsCalculatorEngine:
    """Core purchased-steam emission calculation engine (Engine 2).

    Implements GHG Protocol Scope 2 steam emission calculations using
    supplier-specific emission factors, fuel-based calculations with
    per-gas breakdown, multi-fuel blended steam, condensate return
    adjustment, and biogenic CO2 separation.

    Zero-Hallucination Guarantees:
        - All arithmetic uses ``Decimal`` with ``ROUND_HALF_UP``
        - 8 decimal places for internal precision, 3 for output
        - No LLM calls in the calculation path
        - SHA-256 provenance hash on every result
        - Complete calculation trace for audit
        - Thread-safe via ``threading.RLock``

    Attributes:
        _config: Configuration dictionary.
        _provenance: Chain-hashing provenance tracker.
        _lock: Reentrant thread lock for shared mutable state.

    Example:
        >>> engine = SteamEmissionsCalculatorEngine()
        >>> r = engine.calculate_with_supplier_ef(Decimal("500"), Decimal("66.5"))
        >>> assert r["total_co2e_kg"] == Decimal("33250.000")
    """

    # ------------------------------------------------------------------
    # Singleton support
    # ------------------------------------------------------------------
    _instance: Optional[SteamEmissionsCalculatorEngine] = None
    _singleton_lock: threading.RLock = threading.RLock()

    def __new__(
        cls,
        config: Optional[Dict[str, Any]] = None,
        database: Optional[Any] = None,
        provenance: Optional[Any] = None,
    ) -> SteamEmissionsCalculatorEngine:
        """Return the singleton SteamEmissionsCalculatorEngine instance.

        Uses double-checked locking to ensure exactly one instance is
        created even under concurrent first-access.

        Args:
            config: Optional configuration dictionary. Ignored after
                first construction.
            database: Optional external database engine. Ignored after
                first construction.
            provenance: Optional external provenance tracker. Ignored
                after first construction.

        Returns:
            The singleton :class:`SteamEmissionsCalculatorEngine` instance.
        """
        if cls._instance is None:
            with cls._singleton_lock:
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
        database: Optional[Any] = None,
        provenance: Optional[Any] = None,
    ) -> None:
        """Initialize SteamEmissionsCalculatorEngine.

        Args:
            config: Optional configuration dictionary. Supports:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                    Default True.
                - ``default_gwp_source`` (str): Default GWP source.
                    Default ``AR5``.
                - ``default_boiler_efficiency`` (Decimal): Default boiler
                    efficiency. Default ``0.80``.
                - ``default_condensate_return_pct`` (Decimal): Default
                    condensate return %. Default ``0``.
                - ``enable_biogenic_separation`` (bool): Enable biogenic
                    CO2 separation. Default True.
                - ``tenant_id`` (str): Default tenant ID. Default
                    ``default``.
            database: Optional external database engine for emission
                factor lookups.
            provenance: Optional external provenance tracker. If None,
                an internal ``_ProvenanceTracker`` is created.
        """
        if self._initialized:
            return

        self._config: Dict[str, Any] = config or {}
        self._database = database
        self._lock = threading.RLock()

        # Provenance
        self._enable_provenance: bool = self._config.get(
            "enable_provenance", True
        )
        if provenance is not None:
            self._provenance = provenance
        elif self._enable_provenance:
            self._provenance = _ProvenanceTracker()
        else:
            self._provenance = None

        # Defaults
        self._default_gwp: str = self._config.get(
            "default_gwp_source", "AR5"
        )
        self._default_boiler_efficiency: Decimal = _to_decimal(
            self._config.get("default_boiler_efficiency", "0.80")
        )
        self._default_condensate_pct: Decimal = _to_decimal(
            self._config.get("default_condensate_return_pct", "0")
        )
        self._enable_biogenic: bool = self._config.get(
            "enable_biogenic_separation", True
        )
        self._tenant_id: str = self._config.get("tenant_id", "default")

        # Statistics counters
        self._stats_lock = threading.RLock()
        self._total_calculations: int = 0
        self._total_batches: int = 0
        self._total_consumption_gj = _ZERO
        self._total_co2e_kg_processed = _ZERO
        self._total_biogenic_co2_kg = _ZERO
        self._total_errors: int = 0
        self._fuel_type_counts: Dict[str, int] = {}

        self._initialized = True

        logger.info(
            "SteamEmissionsCalculatorEngine initialized "
            "(provenance=%s, default_gwp=%s, default_efficiency=%s, "
            "biogenic_separation=%s)",
            self._enable_provenance,
            self._default_gwp,
            self._default_boiler_efficiency,
            self._enable_biogenic,
        )

    # ==================================================================
    # 1. calculate_steam_emissions (main entry point)
    # ==================================================================

    def calculate_steam_emissions(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate steam emissions from a structured request dictionary.

        This is the primary entry point that dispatches to the appropriate
        calculation method based on the request contents.

        The request dict must contain:
            - consumption_gj (Decimal or numeric): Steam consumption in GJ.
        And one of:
            - supplier_ef (Decimal or numeric): Direct supplier EF in
              kgCO2e/GJ. Triggers supplier-specific calculation.
            - fuel_type (str): Fuel type for fuel-based calculation.
            - fuel_mix (list[dict]): Multi-fuel blend specification.

        Optional fields:
            - boiler_efficiency (Decimal): Boiler efficiency (0.50-1.00).
            - condensate_return_pct (Decimal): Condensate return % (0-95).
            - gwp_source (str): IPCC GWP source (AR4/AR5/AR6/AR6_20YR).
            - energy_unit (str): Unit of consumption if not GJ.
            - facility_id (str): Facility identifier.
            - supplier_id (str): Supplier identifier.
            - period (str): Reporting period.

        Args:
            request: Calculation request dictionary.

        Returns:
            Dictionary with calculation results including total_co2e_kg,
            per-gas breakdown, biogenic_co2_kg, provenance_hash, and
            calculation_trace.

        Raises:
            ValueError: If the request is invalid or missing required fields.
        """
        start = time.monotonic()
        calc_id = f"steam_{uuid.uuid4().hex[:12]}"
        trace: List[str] = []

        try:
            # Validate request
            is_valid, errors = self.validate_request(request)
            if not is_valid:
                raise ValueError(
                    f"Request validation failed: {'; '.join(errors)}"
                )

            # Extract common fields
            consumption_raw = _to_decimal(request["consumption_gj"])
            energy_unit = request.get("energy_unit", "GJ")
            consumption_gj = _normalize_to_gj(consumption_raw, energy_unit)
            trace.append(
                f"[1] Input: consumption={consumption_raw} {energy_unit} "
                f"= {consumption_gj} GJ"
            )

            # Apply condensate return adjustment
            condensate_pct = _to_decimal(
                request.get(
                    "condensate_return_pct", self._default_condensate_pct
                )
            )
            if condensate_pct > _ZERO:
                consumption_gj = self.apply_condensate_return(
                    consumption_gj, condensate_pct
                )
                trace.append(
                    f"[2] Condensate return: {condensate_pct}% -> "
                    f"effective consumption = {consumption_gj} GJ"
                )
            else:
                trace.append("[2] No condensate return adjustment")

            gwp_source = request.get("gwp_source", self._default_gwp)

            # Dispatch to appropriate method
            if "supplier_ef" in request:
                supplier_ef = _to_decimal(request["supplier_ef"])
                result = self.calculate_with_supplier_ef(
                    consumption_gj=consumption_gj,
                    supplier_ef=supplier_ef,
                    gwp_source=gwp_source,
                )
                trace.append(
                    f"[3] Method: supplier_ef={supplier_ef} kgCO2e/GJ"
                )

            elif "fuel_mix" in request:
                fuel_mix = request["fuel_mix"]
                boiler_eff = _to_decimal(
                    request.get(
                        "boiler_efficiency",
                        self._default_boiler_efficiency,
                    )
                )
                result = self.calculate_blended_steam(
                    consumption_gj=consumption_gj,
                    fuel_mix=fuel_mix,
                    boiler_efficiency=boiler_eff,
                    gwp_source=gwp_source,
                )
                trace.append(
                    f"[3] Method: blended fuel mix "
                    f"({len(fuel_mix)} fuels), efficiency={boiler_eff}"
                )

            elif "fuel_type" in request:
                fuel_type = request["fuel_type"]
                boiler_eff = _to_decimal(
                    request.get(
                        "boiler_efficiency",
                        self._get_default_efficiency(fuel_type),
                    )
                )
                result = self.calculate_with_fuel(
                    consumption_gj=consumption_gj,
                    fuel_type=fuel_type,
                    boiler_efficiency=boiler_eff,
                    gwp_source=gwp_source,
                )
                trace.append(
                    f"[3] Method: fuel_type={fuel_type}, "
                    f"efficiency={boiler_eff}"
                )

            else:
                raise ValueError(
                    "Request must contain one of: supplier_ef, fuel_type, "
                    "or fuel_mix"
                )

            # Augment result with request metadata
            result["calculation_id"] = calc_id
            result["facility_id"] = request.get("facility_id", "")
            result["supplier_id"] = request.get("supplier_id", "")
            result["period"] = request.get("period", "")
            result["condensate_return_pct"] = condensate_pct
            result["energy_unit"] = energy_unit
            result["raw_consumption"] = consumption_raw
            result["effective_consumption_gj"] = consumption_gj
            result["calculation_trace"] = (
                trace + result.get("calculation_trace", [])
            )

            elapsed = time.monotonic() - start
            result["processing_time_ms"] = round(elapsed * 1000, 3)
            result["timestamp"] = _utcnow().isoformat()
            result["status"] = "SUCCESS"

            # Update stats
            self._update_stats(
                co2e_kg=_to_decimal(result.get("total_co2e_kg", 0)),
                consumption_gj=consumption_gj,
                fuel_type=request.get("fuel_type", "unknown"),
                biogenic_co2_kg=_to_decimal(
                    result.get("biogenic_co2_kg", 0)
                ),
            )

            # Record metrics
            _record_calculation_metric(
                energy_type="steam",
                method=result.get("method", "unknown"),
                status="success",
                duration=elapsed,
                co2e_kg=float(result.get("total_co2e_kg", 0)),
                biogenic_kg=float(result.get("biogenic_co2_kg", 0)),
                fuel_type=request.get("fuel_type", "unknown"),
                tenant_id=self._tenant_id,
            )

            if self._provenance is not None:
                self._provenance.record(
                    "steam_calculation",
                    "calculate_steam_emissions",
                    calc_id,
                    {
                        "total_co2e_kg": str(result.get("total_co2e_kg", 0)),
                        "consumption_gj": str(consumption_gj),
                    },
                )

            return result

        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(
                "Steam calculation failed (%s): %s",
                calc_id, exc, exc_info=True,
            )
            self._update_stats_error()
            _record_error_metric("calculation_error", self._tenant_id)

            return {
                "calculation_id": calc_id,
                "status": "FAILED",
                "total_co2e_kg": _ZERO,
                "total_co2e_tonnes": _ZERO,
                "co2_kg": _ZERO,
                "ch4_kg": _ZERO,
                "n2o_kg": _ZERO,
                "biogenic_co2_kg": _ZERO,
                "provenance_hash": "",
                "calculation_trace": trace,
                "processing_time_ms": round(elapsed * 1000, 3),
                "timestamp": _utcnow().isoformat(),
                "error_message": str(exc),
            }

    # ==================================================================
    # 2. calculate_with_supplier_ef
    # ==================================================================

    def calculate_with_supplier_ef(
        self,
        consumption_gj: Decimal,
        supplier_ef: Decimal,
        gwp_source: str = "AR5",
    ) -> Dict[str, Any]:
        """Calculate emissions using a direct supplier emission factor.

        Formula:
            Emissions (kgCO2e) = Consumption (GJ) x Supplier_EF (kgCO2e/GJ)

        This is the simplest and preferred method when the supplier
        provides a verified composite emission factor.

        Args:
            consumption_gj: Steam consumption in GJ. Must be >= 0.
            supplier_ef: Supplier emission factor in kgCO2e/GJ.
                Must be >= 0.
            gwp_source: IPCC GWP source for informational output.
                Default ``AR5``.

        Returns:
            Dictionary with total_co2e_kg, total_co2e_tonnes,
            effective_ef, provenance_hash, and calculation_trace.

        Raises:
            ValueError: If consumption or EF is negative.
        """
        start = time.monotonic()
        consumption_gj = _to_decimal(consumption_gj)
        supplier_ef = _to_decimal(supplier_ef)

        if consumption_gj < _ZERO:
            raise ValueError("consumption_gj must be >= 0")
        if supplier_ef < _ZERO:
            raise ValueError("supplier_ef must be >= 0")

        trace: List[str] = []
        trace.append(
            f"[1] Supplier EF input: consumption={consumption_gj} GJ, "
            f"ef={supplier_ef} kgCO2e/GJ"
        )

        # Calculate total emissions
        total_co2e_kg = _q(consumption_gj * supplier_ef)
        total_co2e_tonnes = _q(total_co2e_kg * _KG_TO_TONNES)
        trace.append(
            f"[2] Emissions: {consumption_gj} GJ x {supplier_ef} "
            f"kgCO2e/GJ = {total_co2e_kg} kgCO2e "
            f"= {total_co2e_tonnes} tCO2e"
        )

        # Effective EF is the supplier EF itself
        effective_ef = supplier_ef
        trace.append(f"[3] Effective EF: {effective_ef} kgCO2e/GJ")

        # Provenance
        prov_data = {
            "method": "calculate_with_supplier_ef",
            "consumption_gj": str(consumption_gj),
            "supplier_ef": str(supplier_ef),
            "total_co2e_kg": str(total_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        trace.append(f"[4] Provenance: {provenance_hash[:16]}...")

        elapsed = time.monotonic() - start

        if self._provenance is not None:
            self._provenance.record(
                "supplier_ef_calculation",
                "calculate_with_supplier_ef",
                f"sup_{uuid.uuid4().hex[:12]}",
                prov_data,
            )

        return {
            "method": "supplier_ef",
            "consumption_gj": consumption_gj,
            "supplier_ef_kgco2e_gj": supplier_ef,
            "total_co2e_kg": _q_out(total_co2e_kg),
            "total_co2e_tonnes": _q_out(total_co2e_tonnes),
            "co2_kg": _q_out(total_co2e_kg),
            "ch4_kg": _ZERO,
            "n2o_kg": _ZERO,
            "ch4_co2e_kg": _ZERO,
            "n2o_co2e_kg": _ZERO,
            "biogenic_co2_kg": _ZERO,
            "effective_ef_kgco2e_gj": _q_out(effective_ef),
            "gwp_source": gwp_source,
            "fuel_type": "unknown",
            "is_biogenic": False,
            "data_quality": "supplier_specific",
            "calculation_trace": trace,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 3. calculate_with_fuel
    # ==================================================================

    def calculate_with_fuel(
        self,
        consumption_gj: Decimal,
        fuel_type: str,
        boiler_efficiency: Optional[Decimal] = None,
        gwp_source: str = "AR5",
    ) -> Dict[str, Any]:
        """Calculate emissions using fuel-based method with per-gas breakdown.

        Formulas:
            Fuel_Input (GJ) = Steam_Consumption (GJ) / Boiler_Efficiency
            CO2 (kg) = Fuel_Input (GJ) x CO2_EF (kgCO2/GJ)
            CH4 (kg) = Fuel_Input (GJ) x CH4_EF (kgCH4/GJ)
            N2O (kg) = Fuel_Input (GJ) x N2O_EF (kgN2O/GJ)
            Total_CO2e (kg) = CO2 + (CH4 x GWP_CH4) + (N2O x GWP_N2O)

        For biogenic fuels, CO2 is reported as biogenic_co2_kg and
        excluded from the fossil total_co2e_kg.

        Args:
            consumption_gj: Steam consumption in GJ. Must be >= 0.
            fuel_type: Fuel type identifier. Must be a recognized type.
            boiler_efficiency: Boiler efficiency as a fraction (0.50-1.00).
                If None, uses the default for the fuel type.
            gwp_source: IPCC GWP source (AR4, AR5, AR6, AR6_20YR).
                Default ``AR5``.

        Returns:
            Dictionary with total_co2e_kg, per-gas breakdown (co2_kg,
            ch4_kg, n2o_kg), biogenic_co2_kg, effective_ef,
            provenance_hash, and calculation_trace.

        Raises:
            ValueError: If inputs are invalid.
        """
        start = time.monotonic()
        consumption_gj = _to_decimal(consumption_gj)
        fuel_type = fuel_type.strip().lower()

        # Validate
        errors = self._validate_fuel_inputs(
            consumption_gj, fuel_type, boiler_efficiency, gwp_source,
        )
        if errors:
            raise ValueError(f"Validation failed: {'; '.join(errors)}")

        # Resolve boiler efficiency
        if boiler_efficiency is not None:
            eff = _to_decimal(boiler_efficiency)
        else:
            eff = self._get_default_efficiency(fuel_type)

        trace: List[str] = []
        trace.append(
            f"[1] Fuel-based input: consumption={consumption_gj} GJ, "
            f"fuel={fuel_type}, efficiency={eff}, gwp={gwp_source}"
        )

        # Step 1: Compute fuel input
        fuel_input_gj = self.compute_fuel_input(consumption_gj, eff)
        trace.append(
            f"[2] Fuel input: {consumption_gj} / {eff} = "
            f"{fuel_input_gj} GJ"
        )

        # Step 2: Compute per-gas emissions
        gas_result = self.compute_gas_emissions(
            fuel_input_gj, fuel_type, gwp_source,
        )
        trace.append(
            f"[3] Gas emissions: CO2={gas_result['co2_kg']} kg, "
            f"CH4={gas_result['ch4_kg']} kg, "
            f"N2O={gas_result['n2o_kg']} kg"
        )

        # Step 3: Separate biogenic CO2
        fuel_props = FUEL_EMISSION_FACTORS.get(fuel_type, {})
        is_biogenic = fuel_props.get("is_biogenic", False)
        fossil_co2_kg, biogenic_co2_kg = self.separate_biogenic(
            fuel_type, gas_result["co2_kg"],
        )
        trace.append(
            f"[4] Biogenic separation: fossil_co2={fossil_co2_kg} kg, "
            f"biogenic_co2={biogenic_co2_kg} kg"
        )

        # Step 4: Compute total CO2e (fossil only)
        ch4_co2e = gas_result["ch4_co2e_kg"]
        n2o_co2e = gas_result["n2o_co2e_kg"]
        total_co2e_kg = _q(fossil_co2_kg + ch4_co2e + n2o_co2e)
        total_co2e_tonnes = _q(total_co2e_kg * _KG_TO_TONNES)
        trace.append(
            f"[5] Total fossil CO2e: {fossil_co2_kg} + {ch4_co2e} + "
            f"{n2o_co2e} = {total_co2e_kg} kgCO2e "
            f"= {total_co2e_tonnes} tCO2e"
        )

        # Step 5: Effective EF
        effective_ef = self.compute_effective_ef(
            total_co2e_kg, consumption_gj,
        )
        trace.append(
            f"[6] Effective EF: {total_co2e_kg} / {consumption_gj} = "
            f"{effective_ef} kgCO2e/GJ"
        )

        # Provenance
        prov_data = {
            "method": "calculate_with_fuel",
            "consumption_gj": str(consumption_gj),
            "fuel_type": fuel_type,
            "boiler_efficiency": str(eff),
            "fuel_input_gj": str(fuel_input_gj),
            "total_co2e_kg": str(total_co2e_kg),
            "biogenic_co2_kg": str(biogenic_co2_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        trace.append(f"[7] Provenance: {provenance_hash[:16]}...")

        elapsed = time.monotonic() - start

        if self._provenance is not None:
            self._provenance.record(
                "fuel_based_calculation",
                "calculate_with_fuel",
                f"fuel_{uuid.uuid4().hex[:12]}",
                prov_data,
            )

        return {
            "method": "fuel_based",
            "consumption_gj": consumption_gj,
            "fuel_type": fuel_type,
            "boiler_efficiency": eff,
            "fuel_input_gj": _q_out(fuel_input_gj),
            "co2_kg": _q_out(gas_result["co2_kg"]),
            "ch4_kg": _q_out(gas_result["ch4_kg"]),
            "n2o_kg": _q_out(gas_result["n2o_kg"]),
            "ch4_co2e_kg": _q_out(ch4_co2e),
            "n2o_co2e_kg": _q_out(n2o_co2e),
            "fossil_co2_kg": _q_out(fossil_co2_kg),
            "biogenic_co2_kg": _q_out(biogenic_co2_kg),
            "total_co2e_kg": _q_out(total_co2e_kg),
            "total_co2e_tonnes": _q_out(total_co2e_tonnes),
            "effective_ef_kgco2e_gj": _q_out(effective_ef),
            "gwp_source": gwp_source,
            "gwp_ch4": GWP_VALUES[gwp_source]["CH4"],
            "gwp_n2o": GWP_VALUES[gwp_source]["N2O"],
            "is_biogenic": is_biogenic,
            "data_quality": "fuel_specific",
            "gas_breakdown": [
                {
                    "gas": "CO2",
                    "mass_kg": _q_out(gas_result["co2_kg"]),
                    "co2e_kg": _q_out(fossil_co2_kg),
                    "gwp_factor": GWP_VALUES[gwp_source]["CO2"],
                    "is_biogenic": is_biogenic,
                },
                {
                    "gas": "CH4",
                    "mass_kg": _q_out(gas_result["ch4_kg"]),
                    "co2e_kg": _q_out(ch4_co2e),
                    "gwp_factor": GWP_VALUES[gwp_source]["CH4"],
                    "is_biogenic": False,
                },
                {
                    "gas": "N2O",
                    "mass_kg": _q_out(gas_result["n2o_kg"]),
                    "co2e_kg": _q_out(n2o_co2e),
                    "gwp_factor": GWP_VALUES[gwp_source]["N2O"],
                    "is_biogenic": False,
                },
            ],
            "calculation_trace": trace,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 4. calculate_blended_steam
    # ==================================================================

    def calculate_blended_steam(
        self,
        consumption_gj: Decimal,
        fuel_mix: List[Dict[str, Any]],
        boiler_efficiency: Optional[Decimal] = None,
        gwp_source: str = "AR5",
    ) -> Dict[str, Any]:
        """Calculate emissions for multi-fuel blended steam generation.

        Formulas:
            Blended_EF_gas = SUM(Fuel_Fraction_i x EF_gas_i) for each gas
            Fuel_Input (GJ) = Consumption (GJ) / Boiler_Efficiency
            Gas_Emissions (kg) = Fuel_Input (GJ) x Blended_EF_gas

        Each element in fuel_mix must contain:
            - fuel_type (str): Fuel type identifier.
            - fraction (Decimal or numeric): Fraction of fuel mix (0-1).
              Fractions must sum to 1.0 (+/- 0.001 tolerance).

        Optional per-fuel fields:
            - efficiency (Decimal): Override boiler efficiency for this fuel.

        Args:
            consumption_gj: Steam consumption in GJ. Must be >= 0.
            fuel_mix: List of fuel mix dictionaries.
            boiler_efficiency: Overall boiler efficiency (0.50-1.00).
                If None, uses weighted average of per-fuel defaults.
            gwp_source: IPCC GWP source. Default ``AR5``.

        Returns:
            Dictionary with blended emissions, per-fuel breakdown,
            provenance_hash, and calculation_trace.

        Raises:
            ValueError: If inputs are invalid or fractions do not sum
                to 1.0.
        """
        start = time.monotonic()
        consumption_gj = _to_decimal(consumption_gj)

        if consumption_gj < _ZERO:
            raise ValueError("consumption_gj must be >= 0")
        if not fuel_mix:
            raise ValueError("fuel_mix must not be empty")
        if gwp_source not in GWP_VALUES:
            raise ValueError(
                f"Unknown gwp_source: {gwp_source}. "
                f"Must be one of {list(GWP_VALUES.keys())}"
            )

        gwp = GWP_VALUES[gwp_source]
        trace: List[str] = []
        trace.append(
            f"[1] Blended input: consumption={consumption_gj} GJ, "
            f"fuels={len(fuel_mix)}, gwp={gwp_source}"
        )

        # Validate and normalize fuel mix
        total_fraction = _ZERO
        normalized_mix: List[Dict[str, Any]] = []

        for i, fm in enumerate(fuel_mix):
            ft = fm.get("fuel_type", "").strip().lower()
            frac = _to_decimal(fm.get("fraction", 0))

            if ft not in FUEL_EMISSION_FACTORS:
                raise ValueError(
                    f"fuel_mix[{i}]: Unknown fuel_type '{ft}'. "
                    f"Valid: {sorted(VALID_FUEL_TYPES)}"
                )
            if frac < _ZERO or frac > _ONE:
                raise ValueError(
                    f"fuel_mix[{i}]: fraction must be 0-1, got {frac}"
                )

            total_fraction += frac
            normalized_mix.append({
                "fuel_type": ft,
                "fraction": frac,
                "efficiency": _to_decimal(
                    fm.get(
                        "efficiency",
                        FUEL_EMISSION_FACTORS[ft]["default_efficiency"],
                    )
                ),
            })

        # Validate fractions sum to 1.0
        tolerance = Decimal("0.001")
        if abs(total_fraction - _ONE) > tolerance:
            raise ValueError(
                f"Fuel mix fractions must sum to 1.0 "
                f"(tolerance {tolerance}), got {total_fraction}"
            )

        # Compute weighted average efficiency if not provided
        if boiler_efficiency is not None:
            eff = _to_decimal(boiler_efficiency)
        else:
            eff = _ZERO
            for fm in normalized_mix:
                eff += fm["fraction"] * fm["efficiency"]
            eff = _q(eff)

        trace.append(f"[2] Effective boiler efficiency: {eff}")

        # Compute fuel input
        fuel_input_gj = self.compute_fuel_input(consumption_gj, eff)
        trace.append(
            f"[3] Fuel input: {consumption_gj} / {eff} = "
            f"{fuel_input_gj} GJ"
        )

        # Compute blended emission factors
        blended_co2_ef = _ZERO
        blended_ch4_ef = _ZERO
        blended_n2o_ef = _ZERO
        blended_biogenic_fraction = _ZERO

        per_fuel_results: List[Dict[str, Any]] = []

        for fm in normalized_mix:
            ft = fm["fuel_type"]
            frac = fm["fraction"]
            props = FUEL_EMISSION_FACTORS[ft]

            co2_ef = _to_decimal(props["co2_ef"])
            ch4_ef = _to_decimal(props["ch4_ef"])
            n2o_ef = _to_decimal(props["n2o_ef"])
            is_bio = props.get("is_biogenic", False)

            blended_co2_ef += frac * co2_ef
            blended_ch4_ef += frac * ch4_ef
            blended_n2o_ef += frac * n2o_ef
            if is_bio:
                blended_biogenic_fraction += frac

            per_fuel_results.append({
                "fuel_type": ft,
                "fraction": frac,
                "co2_ef_kgco2_gj": co2_ef,
                "ch4_ef_kgch4_gj": ch4_ef,
                "n2o_ef_kgn2o_gj": n2o_ef,
                "is_biogenic": is_bio,
            })

            trace.append(
                f"[4.{len(per_fuel_results)}] Fuel {ft}: frac={frac}, "
                f"CO2_EF={co2_ef}, CH4_EF={ch4_ef}, N2O_EF={n2o_ef}"
            )

        blended_co2_ef = _q(blended_co2_ef)
        blended_ch4_ef = _q(blended_ch4_ef)
        blended_n2o_ef = _q(blended_n2o_ef)
        blended_biogenic_fraction = _q(blended_biogenic_fraction)

        trace.append(
            f"[5] Blended EFs: CO2={blended_co2_ef}, "
            f"CH4={blended_ch4_ef}, N2O={blended_n2o_ef}, "
            f"biogenic_fraction={blended_biogenic_fraction}"
        )

        # Compute per-gas emissions
        co2_kg = _q(fuel_input_gj * blended_co2_ef)
        ch4_kg = _q(fuel_input_gj * blended_ch4_ef)
        n2o_kg = _q(fuel_input_gj * blended_n2o_ef)

        # Biogenic CO2 separation based on biogenic fraction
        biogenic_co2_kg = _q(co2_kg * blended_biogenic_fraction)
        fossil_co2_kg = _q(co2_kg - biogenic_co2_kg)

        # CO2e conversions
        ch4_co2e = _q(ch4_kg * gwp["CH4"])
        n2o_co2e = _q(n2o_kg * gwp["N2O"])
        total_co2e_kg = _q(fossil_co2_kg + ch4_co2e + n2o_co2e)
        total_co2e_tonnes = _q(total_co2e_kg * _KG_TO_TONNES)

        trace.append(
            f"[6] Emissions: CO2={co2_kg} kg "
            f"(fossil={fossil_co2_kg}, biogenic={biogenic_co2_kg}), "
            f"CH4={ch4_kg} kg (CO2e={ch4_co2e}), "
            f"N2O={n2o_kg} kg (CO2e={n2o_co2e})"
        )
        trace.append(
            f"[7] Total fossil CO2e: {total_co2e_kg} kgCO2e "
            f"= {total_co2e_tonnes} tCO2e"
        )

        # Effective EF
        effective_ef = self.compute_effective_ef(
            total_co2e_kg, consumption_gj,
        )
        trace.append(
            f"[8] Effective EF: {effective_ef} kgCO2e/GJ"
        )

        # Provenance
        prov_data = {
            "method": "calculate_blended_steam",
            "consumption_gj": str(consumption_gj),
            "fuel_count": len(fuel_mix),
            "boiler_efficiency": str(eff),
            "total_co2e_kg": str(total_co2e_kg),
            "biogenic_co2_kg": str(biogenic_co2_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        trace.append(f"[9] Provenance: {provenance_hash[:16]}...")

        elapsed = time.monotonic() - start

        if self._provenance is not None:
            self._provenance.record(
                "blended_calculation",
                "calculate_blended_steam",
                f"blend_{uuid.uuid4().hex[:12]}",
                prov_data,
            )

        return {
            "method": "blended_fuel",
            "consumption_gj": consumption_gj,
            "boiler_efficiency": eff,
            "fuel_input_gj": _q_out(fuel_input_gj),
            "fuel_mix": per_fuel_results,
            "fuel_count": len(fuel_mix),
            "blended_co2_ef": _q_out(blended_co2_ef),
            "blended_ch4_ef": _q_out(blended_ch4_ef),
            "blended_n2o_ef": _q_out(blended_n2o_ef),
            "biogenic_fraction": _q_out(blended_biogenic_fraction),
            "co2_kg": _q_out(co2_kg),
            "ch4_kg": _q_out(ch4_kg),
            "n2o_kg": _q_out(n2o_kg),
            "ch4_co2e_kg": _q_out(ch4_co2e),
            "n2o_co2e_kg": _q_out(n2o_co2e),
            "fossil_co2_kg": _q_out(fossil_co2_kg),
            "biogenic_co2_kg": _q_out(biogenic_co2_kg),
            "total_co2e_kg": _q_out(total_co2e_kg),
            "total_co2e_tonnes": _q_out(total_co2e_tonnes),
            "effective_ef_kgco2e_gj": _q_out(effective_ef),
            "gwp_source": gwp_source,
            "gwp_ch4": gwp["CH4"],
            "gwp_n2o": gwp["N2O"],
            "is_biogenic": blended_biogenic_fraction > _ZERO,
            "data_quality": "fuel_specific",
            "gas_breakdown": [
                {
                    "gas": "CO2",
                    "mass_kg": _q_out(co2_kg),
                    "fossil_co2e_kg": _q_out(fossil_co2_kg),
                    "biogenic_co2_kg": _q_out(biogenic_co2_kg),
                    "gwp_factor": gwp["CO2"],
                },
                {
                    "gas": "CH4",
                    "mass_kg": _q_out(ch4_kg),
                    "co2e_kg": _q_out(ch4_co2e),
                    "gwp_factor": gwp["CH4"],
                },
                {
                    "gas": "N2O",
                    "mass_kg": _q_out(n2o_kg),
                    "co2e_kg": _q_out(n2o_co2e),
                    "gwp_factor": gwp["N2O"],
                },
            ],
            "calculation_trace": trace,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 5. apply_condensate_return
    # ==================================================================

    def apply_condensate_return(
        self,
        consumption_gj: Decimal,
        condensate_return_pct: Decimal,
    ) -> Decimal:
        """Apply condensate return adjustment to steam consumption.

        Formula:
            Effective_Consumption = Consumption x (1 - Condensate_Return_Pct / 100)

        Condensate return represents hot water returned to the boiler,
        reducing the net energy required from the steam supplier.

        Args:
            consumption_gj: Steam consumption in GJ. Must be >= 0.
            condensate_return_pct: Condensate return percentage (0-95).
                Must be >= 0 and <= 95.

        Returns:
            Adjusted consumption in GJ (Decimal).

        Raises:
            ValueError: If consumption is negative or condensate_return_pct
                is out of bounds.
        """
        consumption_gj = _to_decimal(consumption_gj)
        condensate_return_pct = _to_decimal(condensate_return_pct)

        if consumption_gj < _ZERO:
            raise ValueError("consumption_gj must be >= 0")
        if condensate_return_pct < _MIN_CONDENSATE_RETURN_PCT:
            raise ValueError(
                f"condensate_return_pct must be >= "
                f"{_MIN_CONDENSATE_RETURN_PCT}, "
                f"got {condensate_return_pct}"
            )
        if condensate_return_pct > _MAX_CONDENSATE_RETURN_PCT:
            raise ValueError(
                f"condensate_return_pct must be <= "
                f"{_MAX_CONDENSATE_RETURN_PCT}, "
                f"got {condensate_return_pct}"
            )

        adjustment_factor = _q(_ONE - condensate_return_pct / _HUNDRED)
        return _q(consumption_gj * adjustment_factor)

    # ==================================================================
    # 6. compute_fuel_input
    # ==================================================================

    def compute_fuel_input(
        self,
        consumption_gj: Decimal,
        boiler_efficiency: Decimal,
    ) -> Decimal:
        """Compute fuel input GJ from steam output and boiler efficiency.

        Formula:
            Fuel_Input (GJ) = Steam_Consumption (GJ) / Boiler_Efficiency

        Args:
            consumption_gj: Steam consumption in GJ. Must be >= 0.
            boiler_efficiency: Boiler efficiency as a fraction (0.50-1.00).

        Returns:
            Fuel input in GJ (Decimal).

        Raises:
            ValueError: If consumption is negative or efficiency is
                out of bounds.
        """
        consumption_gj = _to_decimal(consumption_gj)
        boiler_efficiency = _to_decimal(boiler_efficiency)

        if consumption_gj < _ZERO:
            raise ValueError("consumption_gj must be >= 0")
        if boiler_efficiency < _MIN_BOILER_EFFICIENCY:
            raise ValueError(
                f"boiler_efficiency must be >= {_MIN_BOILER_EFFICIENCY}, "
                f"got {boiler_efficiency}"
            )
        if boiler_efficiency > _MAX_BOILER_EFFICIENCY:
            raise ValueError(
                f"boiler_efficiency must be <= {_MAX_BOILER_EFFICIENCY}, "
                f"got {boiler_efficiency}"
            )

        return _q(consumption_gj / boiler_efficiency)

    # ==================================================================
    # 7. compute_gas_emissions
    # ==================================================================

    def compute_gas_emissions(
        self,
        fuel_input_gj: Decimal,
        fuel_type: str,
        gwp_source: str = "AR5",
    ) -> Dict[str, Decimal]:
        """Compute per-gas emissions from fuel input.

        Formulas:
            CO2 (kg) = Fuel_Input (GJ) x CO2_EF (kgCO2/GJ)
            CH4 (kg) = Fuel_Input (GJ) x CH4_EF (kgCH4/GJ)
            N2O (kg) = Fuel_Input (GJ) x N2O_EF (kgN2O/GJ)
            CH4_CO2e (kg) = CH4 (kg) x GWP_CH4
            N2O_CO2e (kg) = N2O (kg) x GWP_N2O

        Args:
            fuel_input_gj: Fuel input in GJ. Must be >= 0.
            fuel_type: Fuel type identifier. Must be a recognized type.
            gwp_source: IPCC GWP source. Default ``AR5``.

        Returns:
            Dictionary with co2_kg, ch4_kg, n2o_kg, ch4_co2e_kg,
            n2o_co2e_kg, total_co2e_kg (all Decimal).

        Raises:
            ValueError: If inputs are invalid.
        """
        fuel_input_gj = _to_decimal(fuel_input_gj)
        fuel_type = fuel_type.strip().lower()

        if fuel_input_gj < _ZERO:
            raise ValueError("fuel_input_gj must be >= 0")
        if fuel_type not in FUEL_EMISSION_FACTORS:
            raise ValueError(
                f"Unknown fuel_type: '{fuel_type}'. "
                f"Valid: {sorted(VALID_FUEL_TYPES)}"
            )
        if gwp_source not in GWP_VALUES:
            raise ValueError(
                f"Unknown gwp_source: {gwp_source}. "
                f"Must be one of {list(GWP_VALUES.keys())}"
            )

        props = FUEL_EMISSION_FACTORS[fuel_type]
        gwp = GWP_VALUES[gwp_source]

        co2_kg = _q(fuel_input_gj * _to_decimal(props["co2_ef"]))
        ch4_kg = _q(fuel_input_gj * _to_decimal(props["ch4_ef"]))
        n2o_kg = _q(fuel_input_gj * _to_decimal(props["n2o_ef"]))

        ch4_co2e = _q(ch4_kg * gwp["CH4"])
        n2o_co2e = _q(n2o_kg * gwp["N2O"])

        # Total includes CO2 at GWP=1 plus CH4 and N2O CO2e
        total_co2e_kg = _q(co2_kg + ch4_co2e + n2o_co2e)

        return {
            "co2_kg": co2_kg,
            "ch4_kg": ch4_kg,
            "n2o_kg": n2o_kg,
            "ch4_co2e_kg": ch4_co2e,
            "n2o_co2e_kg": n2o_co2e,
            "total_co2e_kg": total_co2e_kg,
        }

    # ==================================================================
    # 8. separate_biogenic
    # ==================================================================

    def separate_biogenic(
        self,
        fuel_type: str,
        co2_kg: Decimal,
    ) -> Tuple[Decimal, Decimal]:
        """Separate fossil and biogenic CO2 based on fuel type.

        For biogenic fuels (biomass_wood, biomass_biogas), all CO2 is
        classified as biogenic and excluded from the fossil total.
        Non-biogenic CH4 and N2O still count as fossil GHG.

        Args:
            fuel_type: Fuel type identifier.
            co2_kg: Total CO2 emissions in kg from combustion.

        Returns:
            Tuple of (fossil_co2_kg, biogenic_co2_kg). Both are Decimal.
            For biogenic fuels: (0, co2_kg).
            For fossil fuels: (co2_kg, 0).
        """
        co2_kg = _to_decimal(co2_kg)
        fuel_type = fuel_type.strip().lower()

        if not self._enable_biogenic:
            return co2_kg, _ZERO

        if fuel_type in BIOGENIC_FUEL_TYPES:
            return _ZERO, co2_kg
        else:
            return co2_kg, _ZERO

    # ==================================================================
    # 9. compute_effective_ef
    # ==================================================================

    def compute_effective_ef(
        self,
        total_co2e_kg: Decimal,
        consumption_gj: Decimal,
    ) -> Decimal:
        """Compute the effective emission factor from total CO2e and consumption.

        Formula:
            Effective_EF (kgCO2e/GJ) = Total_CO2e (kg) / Consumption (GJ)

        Args:
            total_co2e_kg: Total CO2e emissions in kg.
            consumption_gj: Steam consumption in GJ. Must be > 0 for a
                meaningful result.

        Returns:
            Effective emission factor in kgCO2e/GJ (Decimal). Returns 0
            if consumption is zero.
        """
        total_co2e_kg = _to_decimal(total_co2e_kg)
        consumption_gj = _to_decimal(consumption_gj)

        if consumption_gj <= _ZERO:
            return _ZERO

        return _q(total_co2e_kg / consumption_gj)

    # ==================================================================
    # 10. validate_request
    # ==================================================================

    def validate_request(
        self,
        request: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Validate a steam emission calculation request.

        Checks:
            - consumption_gj is present and numeric >= 0
            - At least one calculation method specified (supplier_ef,
              fuel_type, or fuel_mix)
            - fuel_type is recognized if provided
            - boiler_efficiency is in valid range if provided
            - condensate_return_pct is in valid range if provided
            - gwp_source is recognized if provided
            - fuel_mix fractions sum to ~1.0 if provided
            - energy_unit is supported if provided

        Args:
            request: Calculation request dictionary.

        Returns:
            Tuple of (is_valid: bool, errors: list[str]).
        """
        errors: List[str] = []

        # Check consumption
        if "consumption_gj" not in request:
            errors.append("Missing required field: consumption_gj")
        else:
            try:
                val = _to_decimal(request["consumption_gj"])
                if val < _ZERO:
                    errors.append(
                        f"consumption_gj must be >= 0, got {val}"
                    )
            except (ValueError, TypeError) as exc:
                errors.append(
                    f"Invalid consumption_gj: {exc}"
                )

        # Check calculation method
        has_method = any(
            k in request for k in ("supplier_ef", "fuel_type", "fuel_mix")
        )
        if not has_method:
            errors.append(
                "Must specify one of: supplier_ef, fuel_type, fuel_mix"
            )

        # Validate supplier_ef
        if "supplier_ef" in request:
            try:
                ef_val = _to_decimal(request["supplier_ef"])
                if ef_val < _ZERO:
                    errors.append(
                        f"supplier_ef must be >= 0, got {ef_val}"
                    )
                if ef_val > Decimal("500"):
                    errors.append(
                        f"WARNING: supplier_ef={ef_val} kgCO2e/GJ exceeds "
                        "500; verify this is correct"
                    )
            except (ValueError, TypeError) as exc:
                errors.append(f"Invalid supplier_ef: {exc}")

        # Validate fuel_type
        if "fuel_type" in request:
            ft = request["fuel_type"].strip().lower()
            if ft not in FUEL_EMISSION_FACTORS:
                errors.append(
                    f"Unknown fuel_type: '{ft}'. "
                    f"Valid: {sorted(VALID_FUEL_TYPES)}"
                )

        # Validate boiler_efficiency
        if "boiler_efficiency" in request:
            try:
                eff_val = _to_decimal(request["boiler_efficiency"])
                if eff_val < _MIN_BOILER_EFFICIENCY:
                    errors.append(
                        f"boiler_efficiency must be >= "
                        f"{_MIN_BOILER_EFFICIENCY}, got {eff_val}"
                    )
                if eff_val > _MAX_BOILER_EFFICIENCY:
                    errors.append(
                        f"boiler_efficiency must be <= "
                        f"{_MAX_BOILER_EFFICIENCY}, got {eff_val}"
                    )
            except (ValueError, TypeError) as exc:
                errors.append(
                    f"Invalid boiler_efficiency: {exc}"
                )

        # Validate condensate_return_pct
        if "condensate_return_pct" in request:
            try:
                cond_val = _to_decimal(request["condensate_return_pct"])
                if cond_val < _MIN_CONDENSATE_RETURN_PCT:
                    errors.append(
                        f"condensate_return_pct must be >= "
                        f"{_MIN_CONDENSATE_RETURN_PCT}, got {cond_val}"
                    )
                if cond_val > _MAX_CONDENSATE_RETURN_PCT:
                    errors.append(
                        f"condensate_return_pct must be <= "
                        f"{_MAX_CONDENSATE_RETURN_PCT}, got {cond_val}"
                    )
            except (ValueError, TypeError) as exc:
                errors.append(
                    f"Invalid condensate_return_pct: {exc}"
                )

        # Validate gwp_source
        if "gwp_source" in request:
            gwp_val = request["gwp_source"]
            if gwp_val not in GWP_VALUES:
                errors.append(
                    f"Unknown gwp_source: '{gwp_val}'. "
                    f"Valid: {list(GWP_VALUES.keys())}"
                )

        # Validate energy_unit
        if "energy_unit" in request:
            eu = request["energy_unit"].strip().upper()
            if eu not in _ENERGY_UNIT_TO_GJ:
                errors.append(
                    f"Unsupported energy_unit: '{request['energy_unit']}'. "
                    f"Valid: {sorted(_ENERGY_UNIT_TO_GJ.keys())}"
                )

        # Validate fuel_mix
        if "fuel_mix" in request:
            fm = request["fuel_mix"]
            if not isinstance(fm, list) or len(fm) == 0:
                errors.append("fuel_mix must be a non-empty list")
            else:
                total_frac = _ZERO
                for i, item in enumerate(fm):
                    if not isinstance(item, dict):
                        errors.append(
                            f"fuel_mix[{i}] must be a dict"
                        )
                        continue
                    if "fuel_type" not in item:
                        errors.append(
                            f"fuel_mix[{i}]: missing fuel_type"
                        )
                    else:
                        ft_val = item["fuel_type"].strip().lower()
                        if ft_val not in FUEL_EMISSION_FACTORS:
                            errors.append(
                                f"fuel_mix[{i}]: unknown fuel_type "
                                f"'{ft_val}'"
                            )
                    if "fraction" not in item:
                        errors.append(
                            f"fuel_mix[{i}]: missing fraction"
                        )
                    else:
                        try:
                            frac_val = _to_decimal(item["fraction"])
                            if frac_val < _ZERO or frac_val > _ONE:
                                errors.append(
                                    f"fuel_mix[{i}]: fraction "
                                    f"must be 0-1, got {frac_val}"
                                )
                            total_frac += frac_val
                        except (ValueError, TypeError) as exc:
                            errors.append(
                                f"fuel_mix[{i}]: invalid fraction: {exc}"
                            )

                if total_frac > _ZERO:
                    tol = Decimal("0.001")
                    if abs(total_frac - _ONE) > tol:
                        errors.append(
                            f"fuel_mix fractions sum to {total_frac}, "
                            f"expected ~1.0"
                        )

        return (len(errors) == 0, errors)

    # ==================================================================
    # 11. batch_calculate
    # ==================================================================

    def batch_calculate(
        self,
        requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Process a batch of steam emission calculations.

        Each request dict must follow the format for
        :meth:`calculate_steam_emissions`.

        Args:
            requests: List of calculation request dictionaries.
                Maximum batch size is 10,000.

        Returns:
            Dictionary with:
                - batch_id (str)
                - results (list of individual result dicts)
                - total_co2e_kg (Decimal): Sum of all successful results
                - total_co2e_tonnes (Decimal)
                - total_biogenic_co2_kg (Decimal)
                - success_count (int)
                - failure_count (int)
                - processing_time_ms (float)
                - provenance_hash (str)
        """
        start = time.monotonic()
        batch_id = f"batch_steam_{uuid.uuid4().hex[:12]}"

        if len(requests) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(requests)} exceeds maximum "
                f"{_MAX_BATCH_SIZE}"
            )

        results: List[Dict[str, Any]] = []
        total_co2e_kg = _ZERO
        total_biogenic = _ZERO
        success_count = 0
        failure_count = 0

        for i, req in enumerate(requests):
            try:
                result = self.calculate_steam_emissions(req)
                result["request_index"] = i
                results.append(result)

                if result.get("status") == "SUCCESS":
                    total_co2e_kg += _to_decimal(
                        result.get("total_co2e_kg", 0)
                    )
                    total_biogenic += _to_decimal(
                        result.get("biogenic_co2_kg", 0)
                    )
                    success_count += 1
                else:
                    failure_count += 1

            except Exception as exc:
                results.append({
                    "request_index": i,
                    "status": "FAILED",
                    "error_message": str(exc),
                    "total_co2e_kg": _ZERO,
                    "total_co2e_tonnes": _ZERO,
                    "biogenic_co2_kg": _ZERO,
                })
                failure_count += 1

        total_co2e_kg = _q(total_co2e_kg)
        total_co2e_tonnes = _q(total_co2e_kg * _KG_TO_TONNES)
        total_biogenic = _q(total_biogenic)

        # Provenance
        prov_data = {
            "method": "batch_calculate",
            "batch_id": batch_id,
            "request_count": len(requests),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_co2e_kg": str(total_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)

        elapsed = time.monotonic() - start
        status = "success" if failure_count == 0 else (
            "partial" if success_count > 0 else "failure"
        )
        _record_batch_metric(status, len(requests), self._tenant_id)

        with self._stats_lock:
            self._total_batches += 1

        if self._provenance is not None:
            self._provenance.record(
                "batch", "batch_calculate", batch_id, prov_data,
            )

        return {
            "batch_id": batch_id,
            "results": results,
            "total_co2e_kg": _q_out(total_co2e_kg),
            "total_co2e_tonnes": _q_out(total_co2e_tonnes),
            "total_biogenic_co2_kg": _q_out(total_biogenic),
            "success_count": success_count,
            "failure_count": failure_count,
            "request_count": len(requests),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
            "timestamp": _utcnow().isoformat(),
        }

    # ==================================================================
    # 12. get_calculation_stats
    # ==================================================================

    def get_calculation_stats(self) -> Dict[str, Any]:
        """Return engine calculation statistics.

        Returns:
            Dictionary with:
                - total_calculations (int)
                - total_batches (int)
                - total_consumption_gj (Decimal)
                - total_co2e_kg_processed (Decimal)
                - total_biogenic_co2_kg (Decimal)
                - total_errors (int)
                - fuel_type_counts (dict): Calculations per fuel type
                - provenance_entry_count (int)
                - supported_fuel_types (list)
                - supported_gwp_sources (list)
                - biogenic_fuel_types (list)
                - zero_emission_fuel_types (list)
                - default_gwp_source (str)
                - default_boiler_efficiency (str)
                - biogenic_separation_enabled (bool)
                - provenance_enabled (bool)
        """
        with self._stats_lock:
            return {
                "total_calculations": self._total_calculations,
                "total_batches": self._total_batches,
                "total_consumption_gj": self._total_consumption_gj,
                "total_co2e_kg_processed": self._total_co2e_kg_processed,
                "total_biogenic_co2_kg": self._total_biogenic_co2_kg,
                "total_errors": self._total_errors,
                "fuel_type_counts": dict(self._fuel_type_counts),
                "provenance_entry_count": (
                    self._provenance.entry_count
                    if self._provenance else 0
                ),
                "supported_fuel_types": sorted(VALID_FUEL_TYPES),
                "supported_gwp_sources": sorted(GWP_VALUES.keys()),
                "biogenic_fuel_types": sorted(BIOGENIC_FUEL_TYPES),
                "zero_emission_fuel_types": sorted(
                    ZERO_EMISSION_FUEL_TYPES
                ),
                "default_gwp_source": self._default_gwp,
                "default_boiler_efficiency": str(
                    self._default_boiler_efficiency
                ),
                "biogenic_separation_enabled": self._enable_biogenic,
                "provenance_enabled": self._enable_provenance,
            }

    # ==================================================================
    # 13. health_check
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the steam emissions calculator engine.

        Validates that the engine is operational by performing a
        lightweight test calculation and checking internal state.

        Returns:
            Dictionary with:
                - status (str): HEALTHY or DEGRADED
                - engine (str): Engine name
                - version (str): Engine version
                - fuel_types_loaded (int)
                - gwp_sources_loaded (int)
                - provenance_enabled (bool)
                - provenance_chain_valid (bool)
                - test_calculation_passed (bool)
                - total_calculations (int)
                - total_errors (int)
                - timestamp (str)
        """
        status = "HEALTHY"
        test_passed = False
        chain_valid = True

        # Test calculation
        try:
            test_result = self.calculate_with_supplier_ef(
                consumption_gj=Decimal("1"),
                supplier_ef=Decimal("66.5"),
                gwp_source="AR5",
            )
            expected = Decimal("66.500")
            if test_result["total_co2e_kg"] == expected:
                test_passed = True
            else:
                status = "DEGRADED"
                logger.warning(
                    "Health check: test calculation returned %s, "
                    "expected %s",
                    test_result["total_co2e_kg"], expected,
                )
        except Exception as exc:
            status = "DEGRADED"
            logger.warning(
                "Health check: test calculation failed: %s", exc,
            )

        # Provenance chain verification
        if self._provenance is not None:
            try:
                chain_valid = self._provenance.verify_chain()
                if not chain_valid:
                    status = "DEGRADED"
                    logger.warning(
                        "Health check: provenance chain verification failed"
                    )
            except Exception as exc:
                chain_valid = False
                status = "DEGRADED"
                logger.warning(
                    "Health check: provenance verification error: %s",
                    exc,
                )

        with self._stats_lock:
            total_calcs = self._total_calculations
            total_errs = self._total_errors

        return {
            "status": status,
            "engine": "SteamEmissionsCalculatorEngine",
            "version": "1.0.0",
            "fuel_types_loaded": len(FUEL_EMISSION_FACTORS),
            "gwp_sources_loaded": len(GWP_VALUES),
            "provenance_enabled": self._enable_provenance,
            "provenance_chain_valid": chain_valid,
            "test_calculation_passed": test_passed,
            "total_calculations": total_calcs,
            "total_errors": total_errs,
            "timestamp": _utcnow().isoformat(),
        }

    # ==================================================================
    # 14. estimate_fuel_consumption
    # ==================================================================

    def estimate_fuel_consumption(
        self,
        steam_gj: Decimal,
        efficiency: Optional[Decimal] = None,
    ) -> Decimal:
        """Estimate fuel consumption required to produce a given steam output.

        Formula:
            Fuel_GJ = Steam_GJ / Boiler_Efficiency

        This is equivalent to :meth:`compute_fuel_input` but provides
        a convenience name for estimation use cases.

        Args:
            steam_gj: Target steam output in GJ. Must be >= 0.
            efficiency: Boiler efficiency (0.50-1.00). If None, uses
                the default boiler efficiency.

        Returns:
            Estimated fuel consumption in GJ (Decimal).

        Raises:
            ValueError: If steam_gj is negative or efficiency is
                out of bounds.
        """
        if efficiency is None:
            efficiency = self._default_boiler_efficiency

        return self.compute_fuel_input(steam_gj, efficiency)

    # ==================================================================
    # 15. compare_fuels
    # ==================================================================

    def compare_fuels(
        self,
        consumption_gj: Decimal,
        fuel_types: List[str],
        efficiency: Optional[Decimal] = None,
        gwp_source: str = "AR5",
    ) -> Dict[str, Any]:
        """Compare emissions across different fuel types for a given consumption.

        Calculates emissions for the same steam consumption using each
        specified fuel type and returns a comparison table sorted by
        total CO2e (ascending).

        Args:
            consumption_gj: Steam consumption in GJ. Must be >= 0.
            fuel_types: List of fuel type identifiers to compare.
            efficiency: Boiler efficiency for all fuels (0.50-1.00).
                If None, uses each fuel's default efficiency.
            gwp_source: IPCC GWP source. Default ``AR5``.

        Returns:
            Dictionary with:
                - consumption_gj (Decimal)
                - gwp_source (str)
                - comparisons (list[dict]): Per-fuel results sorted
                  by total_co2e_kg ascending
                - lowest_emission_fuel (str)
                - highest_emission_fuel (str)
                - provenance_hash (str)
                - processing_time_ms (float)

        Raises:
            ValueError: If fuel_types is empty or contains invalid types.
        """
        start = time.monotonic()
        consumption_gj = _to_decimal(consumption_gj)

        if consumption_gj < _ZERO:
            raise ValueError("consumption_gj must be >= 0")
        if not fuel_types:
            raise ValueError("fuel_types must not be empty")

        comparisons: List[Dict[str, Any]] = []

        for ft in fuel_types:
            ft_lower = ft.strip().lower()
            if ft_lower not in FUEL_EMISSION_FACTORS:
                raise ValueError(
                    f"Unknown fuel_type: '{ft}'. "
                    f"Valid: {sorted(VALID_FUEL_TYPES)}"
                )

            eff = (
                _to_decimal(efficiency)
                if efficiency is not None
                else self._get_default_efficiency(ft_lower)
            )

            try:
                result = self.calculate_with_fuel(
                    consumption_gj=consumption_gj,
                    fuel_type=ft_lower,
                    boiler_efficiency=eff,
                    gwp_source=gwp_source,
                )
                comparisons.append({
                    "fuel_type": ft_lower,
                    "boiler_efficiency": eff,
                    "fuel_input_gj": result["fuel_input_gj"],
                    "co2_kg": result["co2_kg"],
                    "ch4_kg": result["ch4_kg"],
                    "n2o_kg": result["n2o_kg"],
                    "total_co2e_kg": result["total_co2e_kg"],
                    "total_co2e_tonnes": result["total_co2e_tonnes"],
                    "biogenic_co2_kg": result["biogenic_co2_kg"],
                    "effective_ef_kgco2e_gj": result[
                        "effective_ef_kgco2e_gj"
                    ],
                    "is_biogenic": result["is_biogenic"],
                })
            except Exception as exc:
                comparisons.append({
                    "fuel_type": ft_lower,
                    "error": str(exc),
                    "total_co2e_kg": _ZERO,
                })

        # Sort by total_co2e_kg ascending
        comparisons.sort(
            key=lambda x: _to_decimal(x.get("total_co2e_kg", 0))
        )

        lowest = comparisons[0]["fuel_type"] if comparisons else ""
        highest = comparisons[-1]["fuel_type"] if comparisons else ""

        prov_data = {
            "method": "compare_fuels",
            "consumption_gj": str(consumption_gj),
            "fuel_types": fuel_types,
            "lowest": lowest,
            "highest": highest,
        }
        provenance_hash = self._compute_provenance_hash(prov_data)

        elapsed = time.monotonic() - start

        return {
            "consumption_gj": consumption_gj,
            "gwp_source": gwp_source,
            "comparisons": comparisons,
            "fuel_count": len(fuel_types),
            "lowest_emission_fuel": lowest,
            "highest_emission_fuel": highest,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 16. normalize_energy_to_gj
    # ==================================================================

    def normalize_energy_to_gj(
        self,
        quantity: Decimal,
        unit: str,
    ) -> Decimal:
        """Convert an energy quantity from any supported unit to GJ.

        Supported units (case-insensitive): GJ, MWH, KWH, MMBTU,
        THERM, MJ.

        Args:
            quantity: Energy quantity. Must be >= 0.
            unit: Unit string (case-insensitive).

        Returns:
            Energy in GJ (Decimal).

        Raises:
            ValueError: If quantity is negative or unit is unsupported.
        """
        return _normalize_to_gj(_to_decimal(quantity), unit)

    # ==================================================================
    # 17. validate_boiler_efficiency
    # ==================================================================

    def validate_boiler_efficiency(
        self,
        efficiency: Decimal,
    ) -> List[str]:
        """Validate a boiler efficiency value for reasonableness.

        Checks:
            - Must be a valid Decimal
            - Must be >= 0.50 (50%)
            - Must be <= 1.00 (100%)
            - Warns if < 0.65 (unusually low)

        Args:
            efficiency: Boiler efficiency as a fraction (0-1).

        Returns:
            List of validation error/warning strings. Empty if valid.
        """
        errors: List[str] = []
        try:
            val = _to_decimal(efficiency)
        except (ValueError, TypeError):
            errors.append(
                f"Cannot convert efficiency to Decimal: {efficiency!r}"
            )
            return errors

        if val < _MIN_BOILER_EFFICIENCY:
            errors.append(
                f"boiler_efficiency must be >= {_MIN_BOILER_EFFICIENCY}, "
                f"got {val}"
            )
        if val > _MAX_BOILER_EFFICIENCY:
            errors.append(
                f"boiler_efficiency must be <= {_MAX_BOILER_EFFICIENCY}, "
                f"got {val}"
            )
        if _MIN_BOILER_EFFICIENCY <= val < Decimal("0.65"):
            errors.append(
                f"WARNING: boiler_efficiency={val} is below 0.65; "
                "verify this is correct for the equipment type"
            )

        return errors

    # ==================================================================
    # 18. validate_fuel_type
    # ==================================================================

    def validate_fuel_type(
        self,
        fuel_type: str,
    ) -> List[str]:
        """Validate a fuel type identifier.

        Args:
            fuel_type: Fuel type string to validate.

        Returns:
            List of validation error strings. Empty if valid.
        """
        errors: List[str] = []
        ft = fuel_type.strip().lower()

        if ft not in FUEL_EMISSION_FACTORS:
            errors.append(
                f"Unknown fuel_type: '{fuel_type}'. "
                f"Valid: {sorted(VALID_FUEL_TYPES)}"
            )

        return errors

    # ==================================================================
    # 19. get_fuel_emission_factors
    # ==================================================================

    def get_fuel_emission_factors(
        self,
        fuel_type: str,
    ) -> Dict[str, Any]:
        """Retrieve emission factors for a specific fuel type.

        Args:
            fuel_type: Fuel type identifier.

        Returns:
            Dictionary with co2_ef, ch4_ef, n2o_ef, default_efficiency,
            is_biogenic, and description.

        Raises:
            ValueError: If fuel_type is not recognized.
        """
        ft = fuel_type.strip().lower()
        if ft not in FUEL_EMISSION_FACTORS:
            raise ValueError(
                f"Unknown fuel_type: '{fuel_type}'. "
                f"Valid: {sorted(VALID_FUEL_TYPES)}"
            )

        props = FUEL_EMISSION_FACTORS[ft]
        return {
            "fuel_type": ft,
            "co2_ef_kgco2_gj": props["co2_ef"],
            "ch4_ef_kgch4_gj": props["ch4_ef"],
            "n2o_ef_kgn2o_gj": props["n2o_ef"],
            "default_efficiency": props["default_efficiency"],
            "is_biogenic": props["is_biogenic"],
            "description": props.get("description", ""),
        }

    # ==================================================================
    # 20. get_gwp_values
    # ==================================================================

    def get_gwp_values(
        self,
        gwp_source: str = "AR5",
    ) -> Dict[str, Decimal]:
        """Retrieve GWP values for a specific IPCC Assessment Report.

        Args:
            gwp_source: IPCC GWP source (AR4, AR5, AR6, AR6_20YR).
                Default ``AR5``.

        Returns:
            Dictionary with CO2, CH4, N2O GWP multipliers.

        Raises:
            ValueError: If gwp_source is not recognized.
        """
        if gwp_source not in GWP_VALUES:
            raise ValueError(
                f"Unknown gwp_source: '{gwp_source}'. "
                f"Valid: {list(GWP_VALUES.keys())}"
            )

        return dict(GWP_VALUES[gwp_source])

    # ==================================================================
    # 21. aggregate_by_fuel_type
    # ==================================================================

    def aggregate_by_fuel_type(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate calculation results by fuel type.

        Groups results by their ``fuel_type`` field and sums emissions.

        Args:
            results: List of calculation result dictionaries. Each must
                contain ``fuel_type`` and ``total_co2e_kg``.

        Returns:
            Dictionary with per-fuel aggregations and grand total.
        """
        start = time.monotonic()
        by_fuel: Dict[str, Dict[str, Decimal]] = {}

        for r in results:
            ft = r.get("fuel_type", "unknown")
            if ft not in by_fuel:
                by_fuel[ft] = {
                    "total_co2e_kg": _ZERO,
                    "total_co2e_tonnes": _ZERO,
                    "biogenic_co2_kg": _ZERO,
                    "consumption_gj": _ZERO,
                    "count": _ZERO,
                }
            by_fuel[ft]["total_co2e_kg"] += _to_decimal(
                r.get("total_co2e_kg", 0)
            )
            by_fuel[ft]["total_co2e_tonnes"] += _to_decimal(
                r.get("total_co2e_tonnes", 0)
            )
            by_fuel[ft]["biogenic_co2_kg"] += _to_decimal(
                r.get("biogenic_co2_kg", 0)
            )
            by_fuel[ft]["consumption_gj"] += _to_decimal(
                r.get("consumption_gj", 0)
            )
            by_fuel[ft]["count"] += _ONE

        grand_co2e_kg = _ZERO
        grand_biogenic = _ZERO
        fuel_summaries: Dict[str, Dict[str, Any]] = {}

        for ft, agg in sorted(by_fuel.items()):
            co2e_kg = _q(agg["total_co2e_kg"])
            co2e_t = _q(agg["total_co2e_tonnes"])
            bio = _q(agg["biogenic_co2_kg"])
            cons = _q(agg["consumption_gj"])
            fuel_summaries[ft] = {
                "total_co2e_kg": _q_out(co2e_kg),
                "total_co2e_tonnes": _q_out(co2e_t),
                "biogenic_co2_kg": _q_out(bio),
                "consumption_gj": _q_out(cons),
                "calculation_count": int(agg["count"]),
            }
            grand_co2e_kg += co2e_kg
            grand_biogenic += bio

        prov_data = {
            "method": "aggregate_by_fuel_type",
            "fuel_type_count": len(by_fuel),
            "grand_co2e_kg": str(grand_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        elapsed = time.monotonic() - start

        return {
            "by_fuel_type": fuel_summaries,
            "grand_total": {
                "total_co2e_kg": _q_out(_q(grand_co2e_kg)),
                "total_co2e_tonnes": _q_out(
                    _q(grand_co2e_kg * _KG_TO_TONNES)
                ),
                "total_biogenic_co2_kg": _q_out(_q(grand_biogenic)),
            },
            "fuel_type_count": len(by_fuel),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 22. aggregate_by_facility
    # ==================================================================

    def aggregate_by_facility(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate calculation results by facility_id.

        Groups results by their ``facility_id`` and sums emissions.

        Args:
            results: List of calculation result dictionaries. Each must
                contain ``facility_id`` and ``total_co2e_kg``.

        Returns:
            Dictionary mapping facility_id to aggregated totals.
        """
        start = time.monotonic()
        by_facility: Dict[str, Dict[str, Decimal]] = {}

        for r in results:
            fid = r.get("facility_id", "unknown")
            if fid not in by_facility:
                by_facility[fid] = {
                    "total_co2e_kg": _ZERO,
                    "total_co2e_tonnes": _ZERO,
                    "biogenic_co2_kg": _ZERO,
                    "consumption_gj": _ZERO,
                    "count": _ZERO,
                }
            by_facility[fid]["total_co2e_kg"] += _to_decimal(
                r.get("total_co2e_kg", 0)
            )
            by_facility[fid]["total_co2e_tonnes"] += _to_decimal(
                r.get("total_co2e_tonnes", 0)
            )
            by_facility[fid]["biogenic_co2_kg"] += _to_decimal(
                r.get("biogenic_co2_kg", 0)
            )
            by_facility[fid]["consumption_gj"] += _to_decimal(
                r.get("consumption_gj", r.get("effective_consumption_gj", 0))
            )
            by_facility[fid]["count"] += _ONE

        grand_co2e_kg = _ZERO
        facility_summaries: Dict[str, Dict[str, Any]] = {}

        for fid, agg in sorted(by_facility.items()):
            co2e_kg = _q(agg["total_co2e_kg"])
            facility_summaries[fid] = {
                "total_co2e_kg": _q_out(co2e_kg),
                "total_co2e_tonnes": _q_out(
                    _q(agg["total_co2e_tonnes"])
                ),
                "biogenic_co2_kg": _q_out(
                    _q(agg["biogenic_co2_kg"])
                ),
                "consumption_gj": _q_out(
                    _q(agg["consumption_gj"])
                ),
                "calculation_count": int(agg["count"]),
            }
            grand_co2e_kg += co2e_kg

        prov_data = {
            "method": "aggregate_by_facility",
            "facility_count": len(by_facility),
            "grand_co2e_kg": str(grand_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        elapsed = time.monotonic() - start

        return {
            "by_facility": facility_summaries,
            "grand_total": {
                "total_co2e_kg": _q_out(_q(grand_co2e_kg)),
                "total_co2e_tonnes": _q_out(
                    _q(grand_co2e_kg * _KG_TO_TONNES)
                ),
            },
            "facility_count": len(by_facility),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 23. aggregate_by_period
    # ==================================================================

    def aggregate_by_period(
        self,
        results: List[Dict[str, Any]],
        period_type: str = "annual",
    ) -> Dict[str, Any]:
        """Aggregate calculation results by time period.

        Groups results by ``period``, ``timestamp``, or ``date`` field.

        Args:
            results: List of calculation result dictionaries.
            period_type: Aggregation period. One of ``monthly``,
                ``quarterly``, ``annual``. Default ``annual``.

        Returns:
            Dictionary with period-level aggregations and grand total.

        Raises:
            ValueError: If period_type is not valid.
        """
        start = time.monotonic()
        valid_periods = ("monthly", "quarterly", "annual")
        if period_type not in valid_periods:
            raise ValueError(
                f"period_type must be one of {valid_periods}, "
                f"got '{period_type}'"
            )

        by_period: Dict[str, Dict[str, Decimal]] = {}

        for r in results:
            key = self._get_period_key(r, period_type)
            if key not in by_period:
                by_period[key] = {
                    "total_co2e_kg": _ZERO,
                    "total_co2e_tonnes": _ZERO,
                    "biogenic_co2_kg": _ZERO,
                    "consumption_gj": _ZERO,
                    "count": _ZERO,
                }
            by_period[key]["total_co2e_kg"] += _to_decimal(
                r.get("total_co2e_kg", 0)
            )
            by_period[key]["total_co2e_tonnes"] += _to_decimal(
                r.get("total_co2e_tonnes", 0)
            )
            by_period[key]["biogenic_co2_kg"] += _to_decimal(
                r.get("biogenic_co2_kg", 0)
            )
            by_period[key]["consumption_gj"] += _to_decimal(
                r.get("consumption_gj", 0)
            )
            by_period[key]["count"] += _ONE

        grand_co2e_kg = _ZERO
        period_summaries: Dict[str, Dict[str, Any]] = {}

        for key, agg in sorted(by_period.items()):
            co2e_kg = _q(agg["total_co2e_kg"])
            period_summaries[key] = {
                "total_co2e_kg": _q_out(co2e_kg),
                "total_co2e_tonnes": _q_out(
                    _q(agg["total_co2e_tonnes"])
                ),
                "biogenic_co2_kg": _q_out(
                    _q(agg["biogenic_co2_kg"])
                ),
                "consumption_gj": _q_out(
                    _q(agg["consumption_gj"])
                ),
                "calculation_count": int(agg["count"]),
            }
            grand_co2e_kg += co2e_kg

        prov_data = {
            "method": "aggregate_by_period",
            "period_type": period_type,
            "grand_co2e_kg": str(grand_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        elapsed = time.monotonic() - start

        return {
            "period_type": period_type,
            "by_period": period_summaries,
            "grand_total_co2e_kg": _q_out(_q(grand_co2e_kg)),
            "period_count": len(period_summaries),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 24. reset
    # ==================================================================

    def reset(self) -> None:
        """Reset all engine statistics and provenance to initial state.

        Clears all counters and creates a fresh provenance tracker.
        This method is thread-safe.
        """
        with self._stats_lock:
            self._total_calculations = 0
            self._total_batches = 0
            self._total_consumption_gj = _ZERO
            self._total_co2e_kg_processed = _ZERO
            self._total_biogenic_co2_kg = _ZERO
            self._total_errors = 0
            self._fuel_type_counts = {}

        if self._enable_provenance:
            self._provenance = _ProvenanceTracker()
        else:
            self._provenance = None

        logger.info(
            "SteamEmissionsCalculatorEngine reset to initial state"
        )

    # ==================================================================
    # Class-level reset (singleton)
    # ==================================================================

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance.

        **Testing only.** This method is not part of the public API
        and must never be called in production code.
        """
        with cls._singleton_lock:
            cls._instance = None

    # ==================================================================
    # Internal: Fuel input validation
    # ==================================================================

    def _validate_fuel_inputs(
        self,
        consumption_gj: Decimal,
        fuel_type: str,
        boiler_efficiency: Optional[Decimal],
        gwp_source: str,
    ) -> List[str]:
        """Validate inputs for fuel-based emissions calculation.

        Args:
            consumption_gj: Consumption in GJ.
            fuel_type: Fuel type string (already lowercased).
            boiler_efficiency: Optional efficiency override.
            gwp_source: GWP source string.

        Returns:
            List of error strings. Empty if all valid.
        """
        errors: List[str] = []

        if consumption_gj < _ZERO:
            errors.append("consumption_gj must be >= 0")

        if fuel_type not in FUEL_EMISSION_FACTORS:
            errors.append(
                f"Unknown fuel_type: '{fuel_type}'. "
                f"Valid: {sorted(VALID_FUEL_TYPES)}"
            )

        if boiler_efficiency is not None:
            eff = _to_decimal(boiler_efficiency)
            if eff < _MIN_BOILER_EFFICIENCY:
                errors.append(
                    f"boiler_efficiency must be >= "
                    f"{_MIN_BOILER_EFFICIENCY}, got {eff}"
                )
            if eff > _MAX_BOILER_EFFICIENCY:
                errors.append(
                    f"boiler_efficiency must be <= "
                    f"{_MAX_BOILER_EFFICIENCY}, got {eff}"
                )

        if gwp_source not in GWP_VALUES:
            errors.append(
                f"Unknown gwp_source: {gwp_source}. "
                f"Must be one of {list(GWP_VALUES.keys())}"
            )

        return errors

    # ==================================================================
    # Internal: Default efficiency lookup
    # ==================================================================

    def _get_default_efficiency(self, fuel_type: str) -> Decimal:
        """Get the default boiler efficiency for a fuel type.

        Args:
            fuel_type: Fuel type identifier (already lowercased).

        Returns:
            Default efficiency as Decimal. Falls back to 0.80 if
            fuel type is not found.
        """
        props = FUEL_EMISSION_FACTORS.get(fuel_type)
        if props is not None:
            return _to_decimal(props["default_efficiency"])
        return _DEFAULT_BOILER_EFFICIENCY

    # ==================================================================
    # Internal: Period key extraction
    # ==================================================================

    def _get_period_key(
        self,
        result: Dict[str, Any],
        period: str,
    ) -> str:
        """Extract the period key from a result dictionary.

        Uses ``period``, ``timestamp``, or ``date`` fields.

        Args:
            result: Calculation result dictionary.
            period: ``monthly``, ``quarterly``, or ``annual``.

        Returns:
            String key for grouping.
        """
        p = result.get("period")
        ts = result.get("timestamp")
        date_str = result.get("date")

        # If explicit period field exists, use it
        if p and isinstance(p, str) and len(p) >= 4:
            if period == "annual":
                return p[:4]
            if period == "monthly" and len(p) >= 7:
                return p[:7]
            if period == "quarterly" and len(p) >= 7:
                m = int(p[5:7])
                q = (m - 1) // 3 + 1
                return f"{p[:4]}-Q{q}"
            return p

        if period == "annual":
            if isinstance(ts, str) and len(ts) >= 4:
                return ts[:4]
            if date_str and len(str(date_str)) >= 4:
                return str(date_str)[:4]
            return "unknown"

        if period == "monthly":
            if isinstance(ts, str) and len(ts) >= 7:
                return ts[:7]
            return "unknown"

        if period == "quarterly":
            if isinstance(ts, str) and len(ts) >= 7:
                m = int(ts[5:7])
                q = (m - 1) // 3 + 1
                return f"{ts[:4]}-Q{q}"
            return "unknown"

        return "unknown"

    # ==================================================================
    # Internal: Provenance Hash
    # ==================================================================

    def _compute_provenance_hash(
        self,
        data: Dict[str, Any],
    ) -> str:
        """Compute SHA-256 provenance hash of calculation data.

        Serializes the data dictionary to sorted JSON and computes the
        SHA-256 hex digest. Deterministic for identical inputs.

        Args:
            data: Dictionary of calculation data.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # ==================================================================
    # Internal: Statistics
    # ==================================================================

    def _update_stats(
        self,
        co2e_kg: Decimal,
        consumption_gj: Decimal = _ZERO,
        fuel_type: str = "unknown",
        biogenic_co2_kg: Decimal = _ZERO,
    ) -> None:
        """Update running statistics counters (thread-safe).

        Args:
            co2e_kg: Emissions processed in this calculation (kg CO2e).
            consumption_gj: Steam consumption in GJ.
            fuel_type: Fuel type used.
            biogenic_co2_kg: Biogenic CO2 in kg.
        """
        with self._stats_lock:
            self._total_calculations += 1
            self._total_co2e_kg_processed += co2e_kg
            self._total_consumption_gj += consumption_gj
            self._total_biogenic_co2_kg += biogenic_co2_kg
            if fuel_type not in self._fuel_type_counts:
                self._fuel_type_counts[fuel_type] = 0
            self._fuel_type_counts[fuel_type] += 1

    def _update_stats_error(self) -> None:
        """Increment error counter (thread-safe)."""
        with self._stats_lock:
            self._total_errors += 1


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_default_calculator: Optional[SteamEmissionsCalculatorEngine] = None


def get_calculator(
    config: Optional[Dict[str, Any]] = None,
) -> SteamEmissionsCalculatorEngine:
    """Return the module-level default SteamEmissionsCalculatorEngine instance.

    This function provides a convenient module-level accessor that
    lazily creates and caches a :class:`SteamEmissionsCalculatorEngine`
    singleton. It is the recommended entry point for agent code.

    Args:
        config: Optional configuration dictionary. Only honoured on
            the very first call.

    Returns:
        The shared :class:`SteamEmissionsCalculatorEngine` instance.

    Example:
        >>> from greenlang.agents.mrv.steam_heat_purchase.steam_emissions_calculator import (
        ...     get_calculator,
        ... )
        >>> calc = get_calculator()
        >>> r = calc.calculate_with_supplier_ef(
        ...     Decimal("1000"), Decimal("66.5"),
        ... )
    """
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = SteamEmissionsCalculatorEngine(config=config)
    return _default_calculator


def reset() -> None:
    """Reset the module-level calculator singleton.

    **Testing only.** Resets both the module-level cached reference
    and the class-level singleton instance.
    """
    global _default_calculator
    _default_calculator = None
    SteamEmissionsCalculatorEngine.reset_singleton()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["SteamEmissionsCalculatorEngine"]
