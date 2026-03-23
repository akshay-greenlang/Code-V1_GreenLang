# -*- coding: utf-8 -*-
"""
Engine 1: Steam/Heat Database Engine for AGENT-MRV-011.

Stores and retrieves emission factors, efficiency defaults, regional district
heating factors, cooling system parameters, CHP default efficiencies, GWP
values, and energy unit conversions for Scope 2 purchased steam, district
heating, and district cooling emission calculations per GHG Protocol Scope 2
Guidance (2015).

Built-in Data:
- 14 fuel types with combustion emission factors (CO2, CH4, N2O per GJ)
- 13 district heating regional factors (kgCO2e/GJ, distribution loss %)
- 9 cooling system technologies with COP ranges and energy sources
- 5 CHP fuel types with electrical, thermal, and overall efficiencies
- 4 IPCC GWP assessment report editions (AR4, AR5, AR6, AR6_20YR)
- 9 energy unit conversion factors (GJ, MWh, kWh, MMBtu, therm, MJ)

All values as Decimal with ROUND_HALF_UP for zero-hallucination guarantees.
Thread-safe singleton via RLock. SHA-256 provenance on every operation.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-011 Steam/Heat Purchase Agent (GL-MRV-X-022)
Status: Production Ready
"""

from __future__ import annotations

__all__ = ["SteamHeatDatabaseEngine"]

import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.mrv.steam_heat_purchase.models import (
    FUEL_EMISSION_FACTORS,
    DISTRICT_HEATING_FACTORS,
    COOLING_SYSTEM_FACTORS,
    COOLING_ENERGY_SOURCE,
    CHP_DEFAULT_EFFICIENCIES,
    GWP_VALUES,
    UNIT_CONVERSIONS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional metrics import
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


# ---------------------------------------------------------------------------
# Biogenic fuel set
# ---------------------------------------------------------------------------
# Fuel types whose CO2 is biogenic and reported separately from fossil CO2e.
# Based on GHG Protocol Scope 2 Guidance and IPCC 2006 Guidelines Vol 2 Ch 2.

_BIOGENIC_FUELS: frozenset = frozenset({
    "biomass_wood",
    "biomass_biogas",
})

# ---------------------------------------------------------------------------
# Zero-emission fuel set
# ---------------------------------------------------------------------------
# Fuel types with zero direct combustion emissions.

_ZERO_EMISSION_FUELS: frozenset = frozenset({
    "waste_heat",
    "geothermal",
    "solar_thermal",
    "electric",
})


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _canonical_json(data: Dict[str, Any]) -> str:
    """Serialize dictionary to canonical JSON for hashing.

    Uses sort_keys=True and default=str for deterministic output
    regardless of insertion order or non-standard types.

    Args:
        data: Dictionary to serialize.

    Returns:
        Canonical JSON string with sorted keys.
    """
    return json.dumps(data, sort_keys=True, default=str)


def _sha256(payload: str) -> str:
    """Compute SHA-256 hex digest of a string payload.

    Args:
        payload: String to hash.

    Returns:
        64-character lowercase hex digest.
    """
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ===========================================================================
# SteamHeatDatabaseEngine
# ===========================================================================


class SteamHeatDatabaseEngine:
    """Engine 1: Emission factor database for Scope 2 Steam/Heat Purchase
    emission calculations.

    Manages a comprehensive database of fuel emission factors, district
    heating regional factors, cooling system technology parameters, CHP
    default efficiencies, GWP values, and energy unit conversion factors
    for GHG Protocol Scope 2 steam, heat, and cooling accounting.

    Implements the thread-safe singleton pattern using RLock to ensure
    exactly one instance per process. All arithmetic uses Decimal with
    ROUND_HALF_UP for zero-hallucination deterministic calculations.
    Every lookup produces a SHA-256 provenance hash for complete audit
    trails.

    Thread Safety:
        Uses ``threading.RLock`` for singleton creation and all mutable
        state access. Immutable built-in data (FUEL_EMISSION_FACTORS,
        DISTRICT_HEATING_FACTORS, etc.) is inherently thread-safe.

    Data Tables:
        1. Fuel Emission Factors - 14 fuel types (CO2, CH4, N2O per GJ)
        2. District Heating Network Factors - 13 regions
        3. Cooling System Factors - 9 technologies with COP ranges
        4. CHP Default Efficiencies - 5 fuel types
        5. GWP Values - 4 IPCC sources (AR4, AR5, AR6, AR6_20YR)
        6. Unit Conversions - 9 conversion factors

    Attributes:
        ENGINE_ID: Constant identifier for this engine.
        ENGINE_VERSION: Semantic version string.

    Example:
        >>> engine = SteamHeatDatabaseEngine()
        >>> ef = engine.get_fuel_ef("natural_gas")
        >>> assert ef["co2_ef"] == Decimal("56.100")
        >>> dh = engine.get_dh_factor("denmark")
        >>> assert dh["ef_kgco2e_per_gj"] == Decimal("36.0")
        >>> cop = engine.get_cop("centrifugal_chiller")
        >>> assert cop == Decimal("6.0")
    """

    ENGINE_ID: str = "steam_heat_database"
    ENGINE_VERSION: str = "1.0.0"

    _instance: Optional[SteamHeatDatabaseEngine] = None
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton construction
    # ------------------------------------------------------------------

    def __new__(
        cls,
        config: Any = None,
        metrics: Any = None,
        provenance: Any = None,
    ) -> SteamHeatDatabaseEngine:
        """Return the singleton SteamHeatDatabaseEngine instance.

        Uses double-checked locking with an RLock to ensure exactly one
        instance is created even under concurrent first-access.

        Args:
            config: Optional configuration object (ignored after first init).
            metrics: Optional metrics recorder (ignored after first init).
            provenance: Optional provenance tracker (ignored after first init).

        Returns:
            The singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(
        self,
        config: Any = None,
        metrics: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize the steam/heat database engine.

        Idempotent: after the first call, subsequent invocations are
        silently skipped to prevent duplicate initialisation.

        Args:
            config: Optional configuration object for engine tuning.
            metrics: Optional Prometheus metrics recorder. Must expose
                ``record_db_lookup(lookup_type, status)`` method.
            provenance: Optional provenance tracker for chain hashing.
        """
        if self._initialized:
            return

        self._config = config
        self._metrics = metrics
        self._provenance = provenance

        # Mutable state protected by _state_lock
        self._state_lock = threading.RLock()

        # Custom overrides
        self._custom_fuel_efs: Dict[str, Dict[str, Any]] = {}
        self._custom_dh_factors: Dict[str, Dict[str, Any]] = {}
        self._custom_cooling_factors: Dict[str, Dict[str, Any]] = {}
        self._custom_chp_defaults: Dict[str, Dict[str, Any]] = {}

        # Operational counters
        self._lookup_count: int = 0
        self._mutation_count: int = 0
        self._conversion_count: int = 0
        self._provenance_hashes: List[str] = []

        self._initialized = True
        logger.info(
            "SteamHeatDatabaseEngine v%s initialized "
            "(fuel_types=%d, dh_regions=%d, cooling_techs=%d, "
            "chp_fuels=%d, gwp_sources=%d, conversions=%d)",
            self.ENGINE_VERSION,
            len(FUEL_EMISSION_FACTORS),
            len(DISTRICT_HEATING_FACTORS),
            len(COOLING_SYSTEM_FACTORS),
            len(CHP_DEFAULT_EFFICIENCIES),
            len(GWP_VALUES),
            len(UNIT_CONVERSIONS),
        )

    # ------------------------------------------------------------------
    # Provenance helper
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        operation: str,
        data: Dict[str, Any],
    ) -> str:
        """Compute and record a SHA-256 provenance hash for an operation.

        Args:
            operation: Name of the operation (e.g. 'get_fuel_ef').
            data: Dictionary of operation inputs and outputs to hash.

        Returns:
            64-character SHA-256 hex digest.
        """
        payload = {
            "engine": self.ENGINE_ID,
            "operation": operation,
            "timestamp": _utcnow().isoformat(),
            "data": data,
        }
        hash_value = _sha256(_canonical_json(payload))
        with self._state_lock:
            self._provenance_hashes.append(hash_value)
        if self._provenance:
            try:
                self._provenance.record(operation, hash_value, data)
            except Exception:
                pass
        return hash_value

    # ------------------------------------------------------------------
    # Metrics helper
    # ------------------------------------------------------------------

    def _record_metric(self, lookup_type: str, status: str = "hit") -> None:
        """Record a database lookup metric.

        Args:
            lookup_type: The category of data being looked up
                (emission_factor, fuel_property, efficiency_default,
                chp_parameter, grid_factor, conversion_factor).
            status: Lookup outcome (hit, miss, error, timeout).
        """
        if self._metrics:
            try:
                self._metrics.record_db_lookup(lookup_type, status)
            except Exception:
                pass
        elif _METRICS_AVAILABLE and _get_metrics is not None:
            try:
                _get_metrics().record_db_lookup(lookup_type, status)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal computation helper
    # ------------------------------------------------------------------

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for a result dictionary.

        Args:
            data: Dictionary to hash.

        Returns:
            64-character SHA-256 hex digest.
        """
        return _sha256(_canonical_json(data))

    # ==================================================================
    # PUBLIC METHODS: Fuel Emission Factors (Methods 1-5)
    # ==================================================================

    def get_fuel_ef(
        self,
        fuel_type: str,
    ) -> Dict[str, Any]:
        """Get the emission factors for a specific fuel type.

        Returns the CO2, CH4, and N2O emission factors per GJ of fuel
        input, the default boiler efficiency, and the biogenic flag for
        the specified fuel type. Checks custom overrides first, then
        falls back to built-in data.

        Args:
            fuel_type: Fuel type identifier (e.g. 'natural_gas',
                'coal_bituminous', 'biomass_wood'). Must match a key
                in FUEL_EMISSION_FACTORS or a custom override.

        Returns:
            Dictionary with fuel_type, co2_ef, ch4_ef, n2o_ef,
            default_efficiency, is_biogenic, provenance_hash, and
            calculation_trace.

        Raises:
            ValueError: If fuel_type is not recognised.
        """
        with self._state_lock:
            self._lookup_count += 1

        key = fuel_type.lower().strip()
        trace: List[str] = []
        trace.append(f"Fuel EF lookup initiated for fuel_type={key!r}")

        # Check custom overrides first
        with self._state_lock:
            if key in self._custom_fuel_efs:
                custom = self._custom_fuel_efs[key]
                trace.append(f"Custom fuel EF found for {key!r}")
                result = {
                    "fuel_type": key,
                    "co2_ef": custom["co2_ef"],
                    "ch4_ef": custom["ch4_ef"],
                    "n2o_ef": custom["n2o_ef"],
                    "default_efficiency": custom["default_efficiency"],
                    "is_biogenic": custom.get("is_biogenic", False),
                    "source": "custom",
                    "calculation_trace": trace,
                }
                provenance_hash = self._record_provenance(
                    "get_fuel_ef",
                    {"fuel_type": key, "source": "custom"},
                )
                result["provenance_hash"] = provenance_hash
                self._record_metric("emission_factor", "hit")
                logger.debug("Fuel EF lookup (custom): fuel=%s", key)
                return result

        # Built-in data
        if key not in FUEL_EMISSION_FACTORS:
            self._record_metric("emission_factor", "miss")
            trace.append(f"Fuel type {key!r} not found in database")
            raise ValueError(
                f"Unknown fuel type: {fuel_type!r}. "
                f"Valid fuel types: {sorted(FUEL_EMISSION_FACTORS.keys())}"
            )

        data = FUEL_EMISSION_FACTORS[key]
        is_biogenic = data["is_biogenic"] == Decimal("1")
        trace.append(
            f"Built-in fuel EF retrieved: co2={data['co2_ef']}, "
            f"ch4={data['ch4_ef']}, n2o={data['n2o_ef']}, "
            f"efficiency={data['default_efficiency']}, "
            f"biogenic={is_biogenic}"
        )

        result = {
            "fuel_type": key,
            "co2_ef": data["co2_ef"],
            "ch4_ef": data["ch4_ef"],
            "n2o_ef": data["n2o_ef"],
            "default_efficiency": data["default_efficiency"],
            "is_biogenic": is_biogenic,
            "source": "builtin",
            "calculation_trace": trace,
        }

        provenance_hash = self._record_provenance(
            "get_fuel_ef",
            {
                "fuel_type": key,
                "co2_ef": str(data["co2_ef"]),
                "ch4_ef": str(data["ch4_ef"]),
                "n2o_ef": str(data["n2o_ef"]),
                "source": "builtin",
            },
        )
        result["provenance_hash"] = provenance_hash
        self._record_metric("emission_factor", "hit")

        logger.debug(
            "Fuel EF lookup: fuel=%s, co2=%s, ch4=%s, n2o=%s",
            key,
            data["co2_ef"],
            data["ch4_ef"],
            data["n2o_ef"],
        )
        return result

    def get_all_fuel_efs(self) -> Dict[str, Dict[str, Any]]:
        """Get emission factors for all available fuel types.

        Returns a dictionary keyed by fuel type with each value containing
        the complete emission factor record. Includes both built-in and
        custom override factors (custom overrides take precedence).

        Returns:
            Dictionary mapping fuel_type to emission factor dictionaries.
            Each inner dictionary has co2_ef, ch4_ef, n2o_ef,
            default_efficiency, is_biogenic, source, and provenance_hash.
        """
        with self._state_lock:
            self._lookup_count += 1

        results: Dict[str, Dict[str, Any]] = {}

        # Built-in factors
        for fuel_key, data in sorted(FUEL_EMISSION_FACTORS.items()):
            is_biogenic = data["is_biogenic"] == Decimal("1")
            results[fuel_key] = {
                "fuel_type": fuel_key,
                "co2_ef": data["co2_ef"],
                "ch4_ef": data["ch4_ef"],
                "n2o_ef": data["n2o_ef"],
                "default_efficiency": data["default_efficiency"],
                "is_biogenic": is_biogenic,
                "source": "builtin",
            }

        # Apply custom overrides
        with self._state_lock:
            for fuel_key, custom in self._custom_fuel_efs.items():
                results[fuel_key] = {
                    "fuel_type": fuel_key,
                    "co2_ef": custom["co2_ef"],
                    "ch4_ef": custom["ch4_ef"],
                    "n2o_ef": custom["n2o_ef"],
                    "default_efficiency": custom["default_efficiency"],
                    "is_biogenic": custom.get("is_biogenic", False),
                    "source": "custom",
                }

        provenance_hash = self._record_provenance(
            "get_all_fuel_efs",
            {"count": len(results)},
        )
        for entry in results.values():
            entry["provenance_hash"] = provenance_hash

        self._record_metric("emission_factor", "hit")
        logger.debug("All fuel EFs retrieved: count=%d", len(results))
        return results

    def get_fuel_types(self) -> List[str]:
        """List all available fuel types.

        Returns fuel types from both built-in data and any custom
        overrides, sorted alphabetically.

        Returns:
            Sorted list of fuel type identifiers.
        """
        types = set(FUEL_EMISSION_FACTORS.keys())
        with self._state_lock:
            types.update(self._custom_fuel_efs.keys())
        return sorted(types)

    def is_biogenic_fuel(
        self,
        fuel_type: str,
    ) -> bool:
        """Check whether a fuel type is classified as biogenic.

        Biogenic fuels (biomass_wood, biomass_biogas) have their CO2
        emissions reported separately from fossil CO2e per GHG Protocol
        guidance. This method checks both built-in and custom fuel data.

        Args:
            fuel_type: Fuel type identifier.

        Returns:
            True if the fuel is biogenic (biomass), False otherwise.

        Raises:
            ValueError: If fuel_type is not recognised.
        """
        key = fuel_type.lower().strip()

        # Check custom overrides
        with self._state_lock:
            if key in self._custom_fuel_efs:
                return self._custom_fuel_efs[key].get("is_biogenic", False)

        # Check built-in biogenic set (fast path)
        if key in _BIOGENIC_FUELS:
            return True

        # Verify the fuel type exists in built-in data
        if key in FUEL_EMISSION_FACTORS:
            return FUEL_EMISSION_FACTORS[key]["is_biogenic"] == Decimal("1")

        raise ValueError(
            f"Unknown fuel type: {fuel_type!r}. "
            f"Valid fuel types: {sorted(FUEL_EMISSION_FACTORS.keys())}"
        )

    def get_default_efficiency(
        self,
        fuel_type: str,
    ) -> Decimal:
        """Get the default boiler thermal efficiency for a fuel type.

        Returns the typical thermal efficiency (fuel energy to useful
        heat) for the specified fuel type. Used as the default when no
        site-specific boiler efficiency is provided (Tier 1).

        Args:
            fuel_type: Fuel type identifier.

        Returns:
            Default thermal efficiency as Decimal (0 < eta <= 1).

        Raises:
            ValueError: If fuel_type is not recognised.
        """
        key = fuel_type.lower().strip()

        # Check custom overrides
        with self._state_lock:
            if key in self._custom_fuel_efs:
                self._record_metric("efficiency_default", "hit")
                return self._custom_fuel_efs[key]["default_efficiency"]

        if key not in FUEL_EMISSION_FACTORS:
            self._record_metric("efficiency_default", "miss")
            raise ValueError(
                f"Unknown fuel type: {fuel_type!r}. "
                f"Valid fuel types: {sorted(FUEL_EMISSION_FACTORS.keys())}"
            )

        self._record_metric("efficiency_default", "hit")
        return FUEL_EMISSION_FACTORS[key]["default_efficiency"]

    # ==================================================================
    # PUBLIC METHODS: District Heating Factors (Methods 6-9)
    # ==================================================================

    def get_dh_factor(
        self,
        region: str,
    ) -> Dict[str, Any]:
        """Get the district heating network emission factor for a region.

        Returns the composite emission factor (kgCO2e per GJ delivered)
        and distribution loss percentage for the specified region.
        Checks custom overrides first, then built-in data, then falls
        back to global_default.

        Args:
            region: Region identifier (e.g. 'denmark', 'germany', 'us',
                'global_default'). Case-insensitive.

        Returns:
            Dictionary with region, ef_kgco2e_per_gj,
            distribution_loss_pct, source, provenance_hash, and
            calculation_trace.
        """
        with self._state_lock:
            self._lookup_count += 1

        key = region.lower().strip()
        trace: List[str] = []
        trace.append(f"DH factor lookup initiated for region={key!r}")

        # Check custom overrides first
        with self._state_lock:
            if key in self._custom_dh_factors:
                custom = self._custom_dh_factors[key]
                trace.append(f"Custom DH factor found for {key!r}")
                result = {
                    "region": key,
                    "ef_kgco2e_per_gj": custom["ef_kgco2e_per_gj"],
                    "distribution_loss_pct": custom["distribution_loss_pct"],
                    "source": "custom",
                    "calculation_trace": trace,
                }
                provenance_hash = self._record_provenance(
                    "get_dh_factor",
                    {"region": key, "source": "custom"},
                )
                result["provenance_hash"] = provenance_hash
                self._record_metric("grid_factor", "hit")
                return result

        # Built-in data
        if key in DISTRICT_HEATING_FACTORS:
            data = DISTRICT_HEATING_FACTORS[key]
            trace.append(
                f"Built-in DH factor: ef={data['ef_kgco2e_per_gj']} "
                f"kgCO2e/GJ, loss={data['distribution_loss_pct']}"
            )
            result = {
                "region": key,
                "ef_kgco2e_per_gj": data["ef_kgco2e_per_gj"],
                "distribution_loss_pct": data["distribution_loss_pct"],
                "source": "builtin",
                "calculation_trace": trace,
            }
            provenance_hash = self._record_provenance(
                "get_dh_factor",
                {
                    "region": key,
                    "ef_kgco2e_per_gj": str(data["ef_kgco2e_per_gj"]),
                    "distribution_loss_pct": str(data["distribution_loss_pct"]),
                    "source": "builtin",
                },
            )
            result["provenance_hash"] = provenance_hash
            self._record_metric("grid_factor", "hit")
            logger.debug(
                "DH factor lookup: region=%s, ef=%s kgCO2e/GJ",
                key,
                data["ef_kgco2e_per_gj"],
            )
            return result

        # Fallback to global_default
        logger.warning(
            "No DH factor for region %s, using global_default",
            key,
        )
        trace.append(f"Region {key!r} not found, falling back to global_default")
        fallback = DISTRICT_HEATING_FACTORS["global_default"]
        result = {
            "region": key,
            "ef_kgco2e_per_gj": fallback["ef_kgco2e_per_gj"],
            "distribution_loss_pct": fallback["distribution_loss_pct"],
            "source": "global_default_fallback",
            "calculation_trace": trace,
        }
        provenance_hash = self._record_provenance(
            "get_dh_factor",
            {
                "region": key,
                "ef_kgco2e_per_gj": str(fallback["ef_kgco2e_per_gj"]),
                "distribution_loss_pct": str(fallback["distribution_loss_pct"]),
                "source": "global_default_fallback",
            },
        )
        result["provenance_hash"] = provenance_hash
        self._record_metric("grid_factor", "hit")
        return result

    def get_all_dh_factors(self) -> Dict[str, Dict[str, Any]]:
        """Get district heating factors for all available regions.

        Returns a dictionary keyed by region with each value containing
        the emission factor and distribution loss percentage. Custom
        overrides take precedence over built-in data.

        Returns:
            Dictionary mapping region to DH factor dictionaries.
        """
        with self._state_lock:
            self._lookup_count += 1

        results: Dict[str, Dict[str, Any]] = {}

        # Built-in factors
        for region_key, data in sorted(DISTRICT_HEATING_FACTORS.items()):
            results[region_key] = {
                "region": region_key,
                "ef_kgco2e_per_gj": data["ef_kgco2e_per_gj"],
                "distribution_loss_pct": data["distribution_loss_pct"],
                "source": "builtin",
            }

        # Apply custom overrides
        with self._state_lock:
            for region_key, custom in self._custom_dh_factors.items():
                results[region_key] = {
                    "region": region_key,
                    "ef_kgco2e_per_gj": custom["ef_kgco2e_per_gj"],
                    "distribution_loss_pct": custom["distribution_loss_pct"],
                    "source": "custom",
                }

        provenance_hash = self._record_provenance(
            "get_all_dh_factors",
            {"count": len(results)},
        )
        for entry in results.values():
            entry["provenance_hash"] = provenance_hash

        self._record_metric("grid_factor", "hit")
        logger.debug("All DH factors retrieved: count=%d", len(results))
        return results

    def get_dh_regions(self) -> List[str]:
        """List all available district heating regions.

        Returns regions from both built-in data and any custom
        overrides, sorted alphabetically.

        Returns:
            Sorted list of region identifiers.
        """
        regions = set(DISTRICT_HEATING_FACTORS.keys())
        with self._state_lock:
            regions.update(self._custom_dh_factors.keys())
        return sorted(regions)

    def get_distribution_loss_pct(
        self,
        region: str,
    ) -> Decimal:
        """Get the distribution loss percentage for a district heating region.

        Distribution loss is the fraction of thermal energy lost in the
        pipeline network between the heating plant and the consumer's
        meter. Used to adjust consumption for network losses.

        Args:
            region: Region identifier. Case-insensitive.

        Returns:
            Distribution loss as Decimal fraction (0-1).
            Falls back to global_default if region is not found.
        """
        key = region.lower().strip()

        # Check custom overrides
        with self._state_lock:
            if key in self._custom_dh_factors:
                return self._custom_dh_factors[key]["distribution_loss_pct"]

        # Built-in data
        if key in DISTRICT_HEATING_FACTORS:
            return DISTRICT_HEATING_FACTORS[key]["distribution_loss_pct"]

        # Fallback
        logger.warning(
            "No distribution loss for region %s, using global_default",
            key,
        )
        return DISTRICT_HEATING_FACTORS["global_default"]["distribution_loss_pct"]

    # ==================================================================
    # PUBLIC METHODS: Cooling System Factors (Methods 10-14)
    # ==================================================================

    def get_cooling_factor(
        self,
        technology: str,
    ) -> Dict[str, Any]:
        """Get the cooling system parameters for a specific technology.

        Returns COP range (min, max, default), energy source, and
        the technology identifier for the specified cooling technology.

        Args:
            technology: Cooling technology identifier (e.g.
                'centrifugal_chiller', 'absorption_double',
                'free_cooling'). Case-insensitive.

        Returns:
            Dictionary with technology, cop_min, cop_max, cop_default,
            energy_source, source, provenance_hash, and
            calculation_trace.

        Raises:
            ValueError: If technology is not recognised.
        """
        with self._state_lock:
            self._lookup_count += 1

        key = technology.lower().strip()
        trace: List[str] = []
        trace.append(f"Cooling factor lookup for technology={key!r}")

        # Check custom overrides
        with self._state_lock:
            if key in self._custom_cooling_factors:
                custom = self._custom_cooling_factors[key]
                trace.append(f"Custom cooling factor found for {key!r}")
                result = {
                    "technology": key,
                    "cop_min": custom["cop_min"],
                    "cop_max": custom["cop_max"],
                    "cop_default": custom["cop_default"],
                    "energy_source": custom.get("energy_source", "electricity"),
                    "source": "custom",
                    "calculation_trace": trace,
                }
                provenance_hash = self._record_provenance(
                    "get_cooling_factor",
                    {"technology": key, "source": "custom"},
                )
                result["provenance_hash"] = provenance_hash
                self._record_metric("fuel_property", "hit")
                return result

        # Built-in data
        if key not in COOLING_SYSTEM_FACTORS:
            self._record_metric("fuel_property", "miss")
            trace.append(f"Technology {key!r} not found in database")
            raise ValueError(
                f"Unknown cooling technology: {technology!r}. "
                f"Valid technologies: "
                f"{sorted(COOLING_SYSTEM_FACTORS.keys())}"
            )

        data = COOLING_SYSTEM_FACTORS[key]
        energy_source = COOLING_ENERGY_SOURCE.get(key, "electricity")
        trace.append(
            f"Built-in cooling params: COP min={data['cop_min']}, "
            f"max={data['cop_max']}, default={data['cop_default']}, "
            f"energy_source={energy_source}"
        )

        result = {
            "technology": key,
            "cop_min": data["cop_min"],
            "cop_max": data["cop_max"],
            "cop_default": data["cop_default"],
            "energy_source": energy_source,
            "source": "builtin",
            "calculation_trace": trace,
        }

        provenance_hash = self._record_provenance(
            "get_cooling_factor",
            {
                "technology": key,
                "cop_default": str(data["cop_default"]),
                "energy_source": energy_source,
                "source": "builtin",
            },
        )
        result["provenance_hash"] = provenance_hash
        self._record_metric("fuel_property", "hit")

        logger.debug(
            "Cooling factor lookup: tech=%s, COP=%s, source=%s",
            key,
            data["cop_default"],
            energy_source,
        )
        return result

    def get_all_cooling_factors(self) -> Dict[str, Dict[str, Any]]:
        """Get parameters for all available cooling technologies.

        Returns a dictionary keyed by technology with each value
        containing COP ranges, energy source, and source information.
        Custom overrides take precedence.

        Returns:
            Dictionary mapping technology to cooling parameter
            dictionaries.
        """
        with self._state_lock:
            self._lookup_count += 1

        results: Dict[str, Dict[str, Any]] = {}

        # Built-in factors
        for tech_key, data in sorted(COOLING_SYSTEM_FACTORS.items()):
            energy_source = COOLING_ENERGY_SOURCE.get(tech_key, "electricity")
            results[tech_key] = {
                "technology": tech_key,
                "cop_min": data["cop_min"],
                "cop_max": data["cop_max"],
                "cop_default": data["cop_default"],
                "energy_source": energy_source,
                "source": "builtin",
            }

        # Apply custom overrides
        with self._state_lock:
            for tech_key, custom in self._custom_cooling_factors.items():
                results[tech_key] = {
                    "technology": tech_key,
                    "cop_min": custom["cop_min"],
                    "cop_max": custom["cop_max"],
                    "cop_default": custom["cop_default"],
                    "energy_source": custom.get("energy_source", "electricity"),
                    "source": "custom",
                }

        provenance_hash = self._record_provenance(
            "get_all_cooling_factors",
            {"count": len(results)},
        )
        for entry in results.values():
            entry["provenance_hash"] = provenance_hash

        self._record_metric("fuel_property", "hit")
        logger.debug(
            "All cooling factors retrieved: count=%d",
            len(results),
        )
        return results

    def get_cooling_technologies(self) -> List[str]:
        """List all available cooling technologies.

        Returns technologies from both built-in data and any custom
        overrides, sorted alphabetically.

        Returns:
            Sorted list of cooling technology identifiers.
        """
        technologies = set(COOLING_SYSTEM_FACTORS.keys())
        with self._state_lock:
            technologies.update(self._custom_cooling_factors.keys())
        return sorted(technologies)

    def get_cop(
        self,
        technology: str,
    ) -> Decimal:
        """Get the default Coefficient of Performance for a cooling technology.

        The COP is the ratio of cooling output to energy input. Higher
        COP means more efficient cooling. Used to convert cooling
        consumption to energy input for emission factor application.

        Args:
            technology: Cooling technology identifier. Case-insensitive.

        Returns:
            Default COP as Decimal.

        Raises:
            ValueError: If technology is not recognised.
        """
        key = technology.lower().strip()

        # Check custom overrides
        with self._state_lock:
            if key in self._custom_cooling_factors:
                return self._custom_cooling_factors[key]["cop_default"]

        if key not in COOLING_SYSTEM_FACTORS:
            raise ValueError(
                f"Unknown cooling technology: {technology!r}. "
                f"Valid technologies: "
                f"{sorted(COOLING_SYSTEM_FACTORS.keys())}"
            )

        return COOLING_SYSTEM_FACTORS[key]["cop_default"]

    def get_cooling_energy_source(
        self,
        technology: str,
    ) -> str:
        """Get the primary energy source for a cooling technology.

        Returns 'electricity' for electric chillers and storage systems,
        or 'heat' for absorption chillers. Determines which emission
        factor category to apply to the energy input.

        Args:
            technology: Cooling technology identifier. Case-insensitive.

        Returns:
            Energy source string ('electricity' or 'heat').

        Raises:
            ValueError: If technology is not recognised.
        """
        key = technology.lower().strip()

        # Check custom overrides
        with self._state_lock:
            if key in self._custom_cooling_factors:
                return self._custom_cooling_factors[key].get(
                    "energy_source", "electricity",
                )

        if key not in COOLING_ENERGY_SOURCE:
            raise ValueError(
                f"Unknown cooling technology: {technology!r}. "
                f"Valid technologies: "
                f"{sorted(COOLING_ENERGY_SOURCE.keys())}"
            )

        return COOLING_ENERGY_SOURCE[key]

    # ==================================================================
    # PUBLIC METHODS: CHP Default Efficiencies (Methods 15-16)
    # ==================================================================

    def get_chp_defaults(
        self,
        fuel_type: str,
    ) -> Dict[str, Any]:
        """Get default CHP efficiencies for a fuel type.

        Returns electrical, thermal, and overall efficiencies for a
        combined heat and power (CHP) plant burning the specified fuel.
        Used as defaults when site-specific CHP performance data is
        not available (Tier 1).

        Args:
            fuel_type: CHP fuel type identifier (e.g. 'natural_gas',
                'coal', 'biomass', 'fuel_oil', 'municipal_waste').
                Case-insensitive.

        Returns:
            Dictionary with fuel_type, electrical_efficiency,
            thermal_efficiency, overall_efficiency, source,
            provenance_hash, and calculation_trace.

        Raises:
            ValueError: If fuel_type is not recognised as a CHP fuel.
        """
        with self._state_lock:
            self._lookup_count += 1

        key = fuel_type.lower().strip()
        trace: List[str] = []
        trace.append(f"CHP defaults lookup for fuel_type={key!r}")

        # Check custom overrides
        with self._state_lock:
            if key in self._custom_chp_defaults:
                custom = self._custom_chp_defaults[key]
                trace.append(f"Custom CHP defaults found for {key!r}")
                result = {
                    "fuel_type": key,
                    "electrical_efficiency": custom["electrical_efficiency"],
                    "thermal_efficiency": custom["thermal_efficiency"],
                    "overall_efficiency": custom["overall_efficiency"],
                    "source": "custom",
                    "calculation_trace": trace,
                }
                provenance_hash = self._record_provenance(
                    "get_chp_defaults",
                    {"fuel_type": key, "source": "custom"},
                )
                result["provenance_hash"] = provenance_hash
                self._record_metric("chp_parameter", "hit")
                return result

        # Built-in data
        if key not in CHP_DEFAULT_EFFICIENCIES:
            self._record_metric("chp_parameter", "miss")
            trace.append(f"CHP fuel type {key!r} not found in database")
            raise ValueError(
                f"Unknown CHP fuel type: {fuel_type!r}. "
                f"Valid CHP fuel types: "
                f"{sorted(CHP_DEFAULT_EFFICIENCIES.keys())}"
            )

        data = CHP_DEFAULT_EFFICIENCIES[key]
        trace.append(
            f"Built-in CHP defaults: elec={data['electrical_efficiency']}, "
            f"thermal={data['thermal_efficiency']}, "
            f"overall={data['overall_efficiency']}"
        )

        result = {
            "fuel_type": key,
            "electrical_efficiency": data["electrical_efficiency"],
            "thermal_efficiency": data["thermal_efficiency"],
            "overall_efficiency": data["overall_efficiency"],
            "source": "builtin",
            "calculation_trace": trace,
        }

        provenance_hash = self._record_provenance(
            "get_chp_defaults",
            {
                "fuel_type": key,
                "electrical_efficiency": str(data["electrical_efficiency"]),
                "thermal_efficiency": str(data["thermal_efficiency"]),
                "overall_efficiency": str(data["overall_efficiency"]),
                "source": "builtin",
            },
        )
        result["provenance_hash"] = provenance_hash
        self._record_metric("chp_parameter", "hit")

        logger.debug(
            "CHP defaults lookup: fuel=%s, elec=%s, thermal=%s, overall=%s",
            key,
            data["electrical_efficiency"],
            data["thermal_efficiency"],
            data["overall_efficiency"],
        )
        return result

    def get_all_chp_defaults(self) -> Dict[str, Dict[str, Any]]:
        """Get default CHP efficiencies for all available fuel types.

        Returns a dictionary keyed by fuel type with each value
        containing the electrical, thermal, and overall efficiency.
        Custom overrides take precedence.

        Returns:
            Dictionary mapping fuel_type to CHP efficiency dictionaries.
        """
        with self._state_lock:
            self._lookup_count += 1

        results: Dict[str, Dict[str, Any]] = {}

        # Built-in defaults
        for fuel_key, data in sorted(CHP_DEFAULT_EFFICIENCIES.items()):
            results[fuel_key] = {
                "fuel_type": fuel_key,
                "electrical_efficiency": data["electrical_efficiency"],
                "thermal_efficiency": data["thermal_efficiency"],
                "overall_efficiency": data["overall_efficiency"],
                "source": "builtin",
            }

        # Apply custom overrides
        with self._state_lock:
            for fuel_key, custom in self._custom_chp_defaults.items():
                results[fuel_key] = {
                    "fuel_type": fuel_key,
                    "electrical_efficiency": custom["electrical_efficiency"],
                    "thermal_efficiency": custom["thermal_efficiency"],
                    "overall_efficiency": custom["overall_efficiency"],
                    "source": "custom",
                }

        provenance_hash = self._record_provenance(
            "get_all_chp_defaults",
            {"count": len(results)},
        )
        for entry in results.values():
            entry["provenance_hash"] = provenance_hash

        self._record_metric("chp_parameter", "hit")
        logger.debug(
            "All CHP defaults retrieved: count=%d",
            len(results),
        )
        return results

    # ==================================================================
    # PUBLIC METHODS: GWP Values (Methods 17-18)
    # ==================================================================

    def get_gwp_values(
        self,
        source: str,
    ) -> Dict[str, Any]:
        """Get Global Warming Potential values for an IPCC assessment report.

        Returns the 100-year (or 20-year for AR6_20YR) GWP multipliers
        for CO2, CH4, and N2O from the specified IPCC assessment report.

        Args:
            source: IPCC source identifier ('AR4', 'AR5', 'AR6',
                'AR6_20YR'). Case-insensitive.

        Returns:
            Dictionary with source, co2_gwp, ch4_gwp, n2o_gwp,
            provenance_hash, and calculation_trace.

        Raises:
            ValueError: If source is not a recognised GWP source.
        """
        with self._state_lock:
            self._lookup_count += 1

        key = source.upper().strip()
        trace: List[str] = []
        trace.append(f"GWP values lookup for source={key!r}")

        if key not in GWP_VALUES:
            trace.append(f"GWP source {key!r} not recognised")
            raise ValueError(
                f"Unknown GWP source: {source!r}. "
                f"Valid sources: {sorted(GWP_VALUES.keys())}"
            )

        data = GWP_VALUES[key]
        trace.append(
            f"GWP values ({key}): CO2={data['CO2']}, "
            f"CH4={data['CH4']}, N2O={data['N2O']}"
        )

        result = {
            "source": key,
            "co2_gwp": data["CO2"],
            "ch4_gwp": data["CH4"],
            "n2o_gwp": data["N2O"],
            "calculation_trace": trace,
        }

        provenance_hash = self._record_provenance(
            "get_gwp_values",
            {
                "source": key,
                "co2_gwp": str(data["CO2"]),
                "ch4_gwp": str(data["CH4"]),
                "n2o_gwp": str(data["N2O"]),
            },
        )
        result["provenance_hash"] = provenance_hash
        self._record_metric("fuel_property", "hit")

        logger.debug(
            "GWP values lookup: source=%s, CH4=%s, N2O=%s",
            key,
            data["CH4"],
            data["N2O"],
        )
        return result

    def get_gwp(
        self,
        gas: str,
        source: str,
    ) -> Decimal:
        """Get the GWP multiplier for a specific gas and IPCC source.

        Convenience method for looking up a single gas GWP value
        without retrieving the full GWP record.

        Args:
            gas: Gas identifier ('CO2', 'CH4', 'N2O'). Case-insensitive.
            source: IPCC source identifier ('AR4', 'AR5', 'AR6',
                'AR6_20YR'). Case-insensitive.

        Returns:
            GWP multiplier as Decimal (dimensionless).

        Raises:
            ValueError: If gas or source is not recognised.
        """
        source_key = source.upper().strip()
        gas_key = gas.upper().strip()

        if source_key not in GWP_VALUES:
            raise ValueError(
                f"Unknown GWP source: {source!r}. "
                f"Valid sources: {sorted(GWP_VALUES.keys())}"
            )

        data = GWP_VALUES[source_key]
        if gas_key not in data:
            raise ValueError(
                f"Unknown gas: {gas!r}. "
                f"Valid gases: {sorted(data.keys())}"
            )

        return data[gas_key]

    # ==================================================================
    # PUBLIC METHODS: Energy Unit Conversions (Methods 19-20)
    # ==================================================================

    def convert_energy(
        self,
        value: Decimal,
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """Convert a thermal energy quantity between units.

        All conversions route through GJ as the intermediate unit to
        ensure consistency and traceability. Uses deterministic Decimal
        arithmetic for zero-hallucination guarantees.

        Supported units: gj, mwh, kwh, mmbtu, therm, mj.

        Args:
            value: Energy quantity to convert. Must be >= 0.
            from_unit: Source unit identifier. Case-insensitive.
            to_unit: Target unit identifier. Case-insensitive.

        Returns:
            Converted energy quantity as Decimal, rounded to 6 decimal
            places with ROUND_HALF_UP.

        Raises:
            ValueError: If from_unit or to_unit is not recognised, or
                if value is negative.
        """
        if value < Decimal("0"):
            raise ValueError(
                f"Energy value must be >= 0, got {value}"
            )

        with self._state_lock:
            self._conversion_count += 1

        from_key = from_unit.lower().strip()
        to_key = to_unit.lower().strip()

        valid_units = {"gj", "mwh", "kwh", "mmbtu", "therm", "mj"}
        if from_key not in valid_units:
            raise ValueError(
                f"Unknown energy unit: {from_unit!r}. "
                f"Valid units: {sorted(valid_units)}"
            )
        if to_key not in valid_units:
            raise ValueError(
                f"Unknown energy unit: {to_unit!r}. "
                f"Valid units: {sorted(valid_units)}"
            )

        # Same unit: no conversion needed
        if from_key == to_key:
            return value

        # Step 1: Convert from source unit to GJ
        gj_value = self._to_gj(value, from_key)

        # Step 2: Convert from GJ to target unit
        result = self._from_gj(gj_value, to_key)

        # Round to 6 decimal places
        result = result.quantize(Decimal("0.000001"), ROUND_HALF_UP)

        self._record_provenance(
            "convert_energy",
            {
                "value": str(value),
                "from_unit": from_key,
                "to_unit": to_key,
                "result": str(result),
            },
        )
        self._record_metric("conversion_factor", "hit")

        logger.debug(
            "Energy conversion: %s %s -> %s %s",
            value,
            from_key,
            result,
            to_key,
        )
        return result

    def _to_gj(
        self,
        value: Decimal,
        unit: str,
    ) -> Decimal:
        """Convert a value from the specified unit to GJ.

        Args:
            value: Energy quantity.
            unit: Source unit (lowercase).

        Returns:
            Value in GJ.
        """
        if unit == "gj":
            return value
        elif unit == "mwh":
            return value * UNIT_CONVERSIONS["mwh_to_gj"]
        elif unit == "kwh":
            # kWh -> GJ: divide by gj_to_kwh
            return value / UNIT_CONVERSIONS["gj_to_kwh"]
        elif unit == "mmbtu":
            return value * UNIT_CONVERSIONS["mmbtu_to_gj"]
        elif unit == "therm":
            return value * UNIT_CONVERSIONS["therm_to_gj"]
        elif unit == "mj":
            return value * UNIT_CONVERSIONS["mj_to_gj"]
        else:
            raise ValueError(f"Cannot convert unit {unit!r} to GJ")

    def _from_gj(
        self,
        gj_value: Decimal,
        unit: str,
    ) -> Decimal:
        """Convert a GJ value to the specified target unit.

        Args:
            gj_value: Energy quantity in GJ.
            unit: Target unit (lowercase).

        Returns:
            Value in target unit.
        """
        if unit == "gj":
            return gj_value
        elif unit == "mwh":
            return gj_value * UNIT_CONVERSIONS["gj_to_mwh"]
        elif unit == "kwh":
            return gj_value * UNIT_CONVERSIONS["gj_to_kwh"]
        elif unit == "mmbtu":
            return gj_value * UNIT_CONVERSIONS["gj_to_mmbtu"]
        elif unit == "therm":
            return gj_value * UNIT_CONVERSIONS["gj_to_therm"]
        elif unit == "mj":
            return gj_value * UNIT_CONVERSIONS["gj_to_mj"]
        else:
            raise ValueError(f"Cannot convert GJ to unit {unit!r}")

    def get_conversion_factor(
        self,
        conversion: str,
    ) -> Decimal:
        """Get a raw energy unit conversion factor by name.

        Args:
            conversion: Conversion factor name (e.g. 'mwh_to_gj',
                'gj_to_mmbtu', 'therm_to_gj'). Case-insensitive.

        Returns:
            Conversion factor as Decimal.

        Raises:
            ValueError: If conversion name is not recognised.
        """
        key = conversion.lower().strip()
        if key not in UNIT_CONVERSIONS:
            raise ValueError(
                f"Unknown conversion factor: {conversion!r}. "
                f"Valid conversions: {sorted(UNIT_CONVERSIONS.keys())}"
            )

        self._record_metric("conversion_factor", "hit")
        return UNIT_CONVERSIONS[key]

    # ==================================================================
    # PUBLIC METHODS: Blended Emission Factor (Method 21)
    # ==================================================================

    def get_blended_ef(
        self,
        fuel_mix: Dict[str, Decimal],
    ) -> Dict[str, Any]:
        """Calculate a blended emission factor for a multi-fuel mix.

        Given a dictionary of fuel types and their fractional shares
        (summing to approximately 1.0), computes the weighted average
        CO2, CH4, and N2O emission factors. Used when a steam supplier
        burns multiple fuels and provides a fuel mix breakdown.

        Args:
            fuel_mix: Dictionary mapping fuel_type to fractional share
                (Decimal, 0-1). Must sum to approximately 1.0 (within
                0.01 tolerance).
                Example: {"natural_gas": Decimal("0.70"),
                          "fuel_oil_2": Decimal("0.30")}

        Returns:
            Dictionary with blended_co2_ef, blended_ch4_ef,
            blended_n2o_ef, blended_efficiency, biogenic_fraction,
            fuel_mix, provenance_hash, and calculation_trace.

        Raises:
            ValueError: If fuel_mix is empty, fractions do not sum to
                ~1.0, or a fuel type is not recognised.
        """
        if not fuel_mix:
            raise ValueError("fuel_mix must not be empty")

        total_fraction = sum(fuel_mix.values())
        if abs(total_fraction - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(
                f"Fuel mix fractions must sum to ~1.0, got {total_fraction}"
            )

        with self._state_lock:
            self._lookup_count += 1

        trace: List[str] = []
        trace.append(
            f"Blended EF calculation for {len(fuel_mix)} fuel types"
        )

        blended_co2 = Decimal("0")
        blended_ch4 = Decimal("0")
        blended_n2o = Decimal("0")
        blended_efficiency = Decimal("0")
        biogenic_fraction = Decimal("0")

        for fuel_type, fraction in fuel_mix.items():
            key = fuel_type.lower().strip()

            # Check custom overrides first
            with self._state_lock:
                if key in self._custom_fuel_efs:
                    data = self._custom_fuel_efs[key]
                    co2_ef = data["co2_ef"]
                    ch4_ef = data["ch4_ef"]
                    n2o_ef = data["n2o_ef"]
                    efficiency = data["default_efficiency"]
                    is_bio = data.get("is_biogenic", False)
                else:
                    data = None

            if data is None:
                if key not in FUEL_EMISSION_FACTORS:
                    raise ValueError(
                        f"Unknown fuel type in mix: {fuel_type!r}. "
                        f"Valid fuel types: "
                        f"{sorted(FUEL_EMISSION_FACTORS.keys())}"
                    )
                builtin = FUEL_EMISSION_FACTORS[key]
                co2_ef = builtin["co2_ef"]
                ch4_ef = builtin["ch4_ef"]
                n2o_ef = builtin["n2o_ef"]
                efficiency = builtin["default_efficiency"]
                is_bio = builtin["is_biogenic"] == Decimal("1")

            blended_co2 += co2_ef * fraction
            blended_ch4 += ch4_ef * fraction
            blended_n2o += n2o_ef * fraction
            blended_efficiency += efficiency * fraction
            if is_bio:
                biogenic_fraction += fraction

            trace.append(
                f"  {key}: fraction={fraction}, co2={co2_ef}, "
                f"ch4={ch4_ef}, n2o={n2o_ef}, "
                f"efficiency={efficiency}, biogenic={is_bio}"
            )

        # Round results
        blended_co2 = blended_co2.quantize(
            Decimal("0.001"), ROUND_HALF_UP,
        )
        blended_ch4 = blended_ch4.quantize(
            Decimal("0.000001"), ROUND_HALF_UP,
        )
        blended_n2o = blended_n2o.quantize(
            Decimal("0.0000001"), ROUND_HALF_UP,
        )
        blended_efficiency = blended_efficiency.quantize(
            Decimal("0.001"), ROUND_HALF_UP,
        )
        biogenic_fraction = biogenic_fraction.quantize(
            Decimal("0.001"), ROUND_HALF_UP,
        )

        trace.append(
            f"Blended result: co2={blended_co2}, ch4={blended_ch4}, "
            f"n2o={blended_n2o}, efficiency={blended_efficiency}, "
            f"biogenic_fraction={biogenic_fraction}"
        )

        result = {
            "blended_co2_ef": blended_co2,
            "blended_ch4_ef": blended_ch4,
            "blended_n2o_ef": blended_n2o,
            "blended_efficiency": blended_efficiency,
            "biogenic_fraction": biogenic_fraction,
            "fuel_mix": {k: str(v) for k, v in fuel_mix.items()},
            "fuel_count": len(fuel_mix),
            "calculation_trace": trace,
        }

        provenance_hash = self._record_provenance(
            "get_blended_ef",
            {
                "fuel_count": len(fuel_mix),
                "blended_co2_ef": str(blended_co2),
                "blended_ch4_ef": str(blended_ch4),
                "blended_n2o_ef": str(blended_n2o),
                "biogenic_fraction": str(biogenic_fraction),
            },
        )
        result["provenance_hash"] = provenance_hash
        self._record_metric("emission_factor", "hit")

        logger.info(
            "Blended EF calculated: fuels=%d, co2=%s, ch4=%s, n2o=%s",
            len(fuel_mix),
            blended_co2,
            blended_ch4,
            blended_n2o,
        )
        return result

    # ==================================================================
    # PUBLIC METHODS: Search (Method 22)
    # ==================================================================

    def search_factors(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Search across all factor tables by text query.

        Performs case-insensitive substring matching against fuel types,
        district heating regions, cooling technologies, CHP fuel types,
        GWP sources, and conversion factor names.

        Args:
            query: Search string.

        Returns:
            List of matching factor entries, each with type, key,
            summary, and provenance_hash.
        """
        with self._state_lock:
            self._lookup_count += 1

        q = query.lower().strip()
        results: List[Dict[str, Any]] = []

        # Search fuel emission factors
        for fuel_key, data in FUEL_EMISSION_FACTORS.items():
            if q in fuel_key:
                is_biogenic = data["is_biogenic"] == Decimal("1")
                results.append({
                    "type": "fuel_emission_factor",
                    "key": fuel_key,
                    "co2_ef": data["co2_ef"],
                    "ch4_ef": data["ch4_ef"],
                    "n2o_ef": data["n2o_ef"],
                    "default_efficiency": data["default_efficiency"],
                    "is_biogenic": is_biogenic,
                    "source": "builtin",
                })

        # Search custom fuel EFs
        with self._state_lock:
            for fuel_key, custom in self._custom_fuel_efs.items():
                if q in fuel_key:
                    results.append({
                        "type": "fuel_emission_factor",
                        "key": fuel_key,
                        "co2_ef": custom["co2_ef"],
                        "ch4_ef": custom["ch4_ef"],
                        "n2o_ef": custom["n2o_ef"],
                        "default_efficiency": custom["default_efficiency"],
                        "is_biogenic": custom.get("is_biogenic", False),
                        "source": "custom",
                    })

        # Search district heating factors
        for region_key, data in DISTRICT_HEATING_FACTORS.items():
            if q in region_key:
                results.append({
                    "type": "district_heating_factor",
                    "key": region_key,
                    "ef_kgco2e_per_gj": data["ef_kgco2e_per_gj"],
                    "distribution_loss_pct": data["distribution_loss_pct"],
                    "source": "builtin",
                })

        # Search custom DH factors
        with self._state_lock:
            for region_key, custom in self._custom_dh_factors.items():
                if q in region_key:
                    results.append({
                        "type": "district_heating_factor",
                        "key": region_key,
                        "ef_kgco2e_per_gj": custom["ef_kgco2e_per_gj"],
                        "distribution_loss_pct": custom["distribution_loss_pct"],
                        "source": "custom",
                    })

        # Search cooling system factors
        for tech_key, data in COOLING_SYSTEM_FACTORS.items():
            if q in tech_key:
                energy_source = COOLING_ENERGY_SOURCE.get(
                    tech_key, "electricity",
                )
                results.append({
                    "type": "cooling_system_factor",
                    "key": tech_key,
                    "cop_default": data["cop_default"],
                    "cop_min": data["cop_min"],
                    "cop_max": data["cop_max"],
                    "energy_source": energy_source,
                    "source": "builtin",
                })

        # Search custom cooling factors
        with self._state_lock:
            for tech_key, custom in self._custom_cooling_factors.items():
                if q in tech_key:
                    results.append({
                        "type": "cooling_system_factor",
                        "key": tech_key,
                        "cop_default": custom["cop_default"],
                        "cop_min": custom["cop_min"],
                        "cop_max": custom["cop_max"],
                        "energy_source": custom.get(
                            "energy_source", "electricity",
                        ),
                        "source": "custom",
                    })

        # Search CHP defaults
        for fuel_key, data in CHP_DEFAULT_EFFICIENCIES.items():
            if q in fuel_key:
                results.append({
                    "type": "chp_default_efficiency",
                    "key": fuel_key,
                    "electrical_efficiency": data["electrical_efficiency"],
                    "thermal_efficiency": data["thermal_efficiency"],
                    "overall_efficiency": data["overall_efficiency"],
                    "source": "builtin",
                })

        # Search custom CHP defaults
        with self._state_lock:
            for fuel_key, custom in self._custom_chp_defaults.items():
                if q in fuel_key:
                    results.append({
                        "type": "chp_default_efficiency",
                        "key": fuel_key,
                        "electrical_efficiency": custom["electrical_efficiency"],
                        "thermal_efficiency": custom["thermal_efficiency"],
                        "overall_efficiency": custom["overall_efficiency"],
                        "source": "custom",
                    })

        # Search GWP sources
        for source_key in GWP_VALUES:
            if q in source_key.lower():
                gwp_data = GWP_VALUES[source_key]
                results.append({
                    "type": "gwp_values",
                    "key": source_key,
                    "co2_gwp": gwp_data["CO2"],
                    "ch4_gwp": gwp_data["CH4"],
                    "n2o_gwp": gwp_data["N2O"],
                    "source": "builtin",
                })

        # Search unit conversions
        for conv_key, factor in UNIT_CONVERSIONS.items():
            if q in conv_key:
                results.append({
                    "type": "unit_conversion",
                    "key": conv_key,
                    "factor": factor,
                    "source": "builtin",
                })

        provenance_hash = self._record_provenance(
            "search_factors",
            {"query": q, "result_count": len(results)},
        )
        for r in results:
            r["provenance_hash"] = provenance_hash

        logger.debug(
            "Factor search: query=%s, results=%d",
            q,
            len(results),
        )
        return results

    # ==================================================================
    # PUBLIC METHODS: Database Statistics (Method 23)
    # ==================================================================

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics including counts and operational metrics.

        Returns:
            Dictionary with fuel_types, dh_regions, cooling_technologies,
            chp_fuel_types, gwp_sources, conversions, custom counts,
            operational counters, and provenance_hash.
        """
        with self._state_lock:
            stats = {
                "fuel_types": len(FUEL_EMISSION_FACTORS),
                "custom_fuel_types": len(self._custom_fuel_efs),
                "dh_regions": len(DISTRICT_HEATING_FACTORS),
                "custom_dh_regions": len(self._custom_dh_factors),
                "cooling_technologies": len(COOLING_SYSTEM_FACTORS),
                "custom_cooling_technologies": len(
                    self._custom_cooling_factors,
                ),
                "chp_fuel_types": len(CHP_DEFAULT_EFFICIENCIES),
                "custom_chp_fuel_types": len(self._custom_chp_defaults),
                "gwp_sources": len(GWP_VALUES),
                "unit_conversions": len(UNIT_CONVERSIONS),
                "biogenic_fuel_types": len(_BIOGENIC_FUELS),
                "zero_emission_fuel_types": len(_ZERO_EMISSION_FUELS),
                "total_lookups": self._lookup_count,
                "total_mutations": self._mutation_count,
                "total_conversions": self._conversion_count,
                "provenance_hashes_count": len(self._provenance_hashes),
            }

        provenance_hash = self._record_provenance(
            "get_database_stats",
            stats,
        )
        stats["provenance_hash"] = provenance_hash

        logger.debug(
            "Database stats: fuel=%d, dh=%d, cooling=%d, chp=%d, "
            "gwp=%d, conv=%d, lookups=%d",
            stats["fuel_types"],
            stats["dh_regions"],
            stats["cooling_technologies"],
            stats["chp_fuel_types"],
            stats["gwp_sources"],
            stats["unit_conversions"],
            stats["total_lookups"],
        )
        return stats

    # ==================================================================
    # PUBLIC METHODS: Health Check (Method 24)
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the database engine.

        Verifies that all built-in data tables are accessible, have
        expected record counts, and that the engine is operational.

        Returns:
            Dictionary with status ('healthy' or 'unhealthy'),
            checks (list of individual check results), engine_id,
            engine_version, timestamp, and provenance_hash.
        """
        checks: List[Dict[str, Any]] = []
        all_healthy = True

        # Check 1: Fuel emission factors table
        fuel_count = len(FUEL_EMISSION_FACTORS)
        fuel_ok = fuel_count == 14
        checks.append({
            "check": "fuel_emission_factors",
            "expected_count": 14,
            "actual_count": fuel_count,
            "status": "healthy" if fuel_ok else "unhealthy",
        })
        if not fuel_ok:
            all_healthy = False

        # Check 2: District heating factors table
        dh_count = len(DISTRICT_HEATING_FACTORS)
        dh_ok = dh_count == 13
        checks.append({
            "check": "district_heating_factors",
            "expected_count": 13,
            "actual_count": dh_count,
            "status": "healthy" if dh_ok else "unhealthy",
        })
        if not dh_ok:
            all_healthy = False

        # Check 3: Cooling system factors table
        cooling_count = len(COOLING_SYSTEM_FACTORS)
        cooling_ok = cooling_count == 9
        checks.append({
            "check": "cooling_system_factors",
            "expected_count": 9,
            "actual_count": cooling_count,
            "status": "healthy" if cooling_ok else "unhealthy",
        })
        if not cooling_ok:
            all_healthy = False

        # Check 4: CHP default efficiencies table
        chp_count = len(CHP_DEFAULT_EFFICIENCIES)
        chp_ok = chp_count == 5
        checks.append({
            "check": "chp_default_efficiencies",
            "expected_count": 5,
            "actual_count": chp_count,
            "status": "healthy" if chp_ok else "unhealthy",
        })
        if not chp_ok:
            all_healthy = False

        # Check 5: GWP values table
        gwp_count = len(GWP_VALUES)
        gwp_ok = gwp_count == 4
        checks.append({
            "check": "gwp_values",
            "expected_count": 4,
            "actual_count": gwp_count,
            "status": "healthy" if gwp_ok else "unhealthy",
        })
        if not gwp_ok:
            all_healthy = False

        # Check 6: Unit conversions table
        conv_count = len(UNIT_CONVERSIONS)
        conv_ok = conv_count == 9
        checks.append({
            "check": "unit_conversions",
            "expected_count": 9,
            "actual_count": conv_count,
            "status": "healthy" if conv_ok else "unhealthy",
        })
        if not conv_ok:
            all_healthy = False

        # Check 7: Cooling energy source companion table
        ces_count = len(COOLING_ENERGY_SOURCE)
        ces_ok = ces_count == 9
        checks.append({
            "check": "cooling_energy_source",
            "expected_count": 9,
            "actual_count": ces_count,
            "status": "healthy" if ces_ok else "unhealthy",
        })
        if not ces_ok:
            all_healthy = False

        # Check 8: Data integrity - verify a known value
        ng_ok = False
        try:
            ng_data = FUEL_EMISSION_FACTORS.get("natural_gas", {})
            ng_ok = ng_data.get("co2_ef") == Decimal("56.100")
        except Exception:
            pass
        checks.append({
            "check": "data_integrity_natural_gas_co2",
            "expected_value": "56.100",
            "actual_value": str(ng_data.get("co2_ef", "N/A")),
            "status": "healthy" if ng_ok else "unhealthy",
        })
        if not ng_ok:
            all_healthy = False

        # Check 9: Data integrity - verify GWP AR6
        ar6_ok = False
        try:
            ar6_data = GWP_VALUES.get("AR6", {})
            ar6_ok = ar6_data.get("CH4") == Decimal("27.9")
        except Exception:
            pass
        checks.append({
            "check": "data_integrity_gwp_ar6_ch4",
            "expected_value": "27.9",
            "actual_value": str(ar6_data.get("CH4", "N/A")),
            "status": "healthy" if ar6_ok else "unhealthy",
        })
        if not ar6_ok:
            all_healthy = False

        # Check 10: Singleton is initialized
        init_ok = self._initialized
        checks.append({
            "check": "engine_initialized",
            "status": "healthy" if init_ok else "unhealthy",
        })
        if not init_ok:
            all_healthy = False

        overall_status = "healthy" if all_healthy else "unhealthy"

        result = {
            "status": overall_status,
            "engine_id": self.ENGINE_ID,
            "engine_version": self.ENGINE_VERSION,
            "timestamp": _utcnow().isoformat(),
            "checks": checks,
            "checks_passed": sum(
                1 for c in checks if c["status"] == "healthy"
            ),
            "checks_total": len(checks),
        }

        provenance_hash = self._record_provenance(
            "health_check",
            {
                "status": overall_status,
                "checks_passed": result["checks_passed"],
                "checks_total": result["checks_total"],
            },
        )
        result["provenance_hash"] = provenance_hash

        if all_healthy:
            logger.info(
                "Health check PASSED: %d/%d checks healthy",
                result["checks_passed"],
                result["checks_total"],
            )
        else:
            logger.warning(
                "Health check DEGRADED: %d/%d checks healthy",
                result["checks_passed"],
                result["checks_total"],
            )

        return result

    # ==================================================================
    # PUBLIC METHODS: Engine Info
    # ==================================================================

    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine identification and version metadata.

        Returns:
            Dictionary with engine_id, version, description, data_sources,
            capabilities, built_in_data, and provenance_hash.
        """
        info = {
            "engine_id": self.ENGINE_ID,
            "version": self.ENGINE_VERSION,
            "description": (
                "Steam/Heat Database Engine for Scope 2 purchased "
                "steam, district heating, and district cooling emission "
                "calculations per GHG Protocol Scope 2 Guidance (2015)."
            ),
            "data_sources": [
                "IPCC 2006 Guidelines Vol 2, Chapter 2 Tables 2.2/2.4",
                "US EPA AP-42 Compilation of Air Emission Factors",
                "UK BEIS Greenhouse Gas Reporting Conversion Factors",
                "ASHRAE Handbook - HVAC Systems and Equipment (2024)",
                "US EPA CHP Partnership Programme",
                "IEA CHP and District Heating Country Assessments",
                "IPCC AR4/AR5/AR6 GWP Tables",
                "National DH statistics (DEA, SEA, UBA, URE, etc.)",
            ],
            "capabilities": [
                "fuel_ef_lookup",
                "district_heating_factor_lookup",
                "cooling_system_factor_lookup",
                "chp_defaults_lookup",
                "gwp_values_lookup",
                "energy_unit_conversion",
                "blended_ef_calculation",
                "factor_search",
                "custom_factor_management",
                "health_check",
            ],
            "built_in_data": {
                "fuel_types": len(FUEL_EMISSION_FACTORS),
                "dh_regions": len(DISTRICT_HEATING_FACTORS),
                "cooling_technologies": len(COOLING_SYSTEM_FACTORS),
                "chp_fuel_types": len(CHP_DEFAULT_EFFICIENCIES),
                "gwp_sources": len(GWP_VALUES),
                "unit_conversions": len(UNIT_CONVERSIONS),
            },
        }

        provenance_hash = self._record_provenance(
            "get_engine_info",
            {"engine_id": self.ENGINE_ID, "version": self.ENGINE_VERSION},
        )
        info["provenance_hash"] = provenance_hash
        return info

    # ==================================================================
    # PUBLIC METHODS: Custom Factor Management
    # ==================================================================

    def set_custom_fuel_ef(
        self,
        fuel_type: str,
        co2_ef: Decimal,
        ch4_ef: Decimal,
        n2o_ef: Decimal,
        default_efficiency: Decimal,
        is_biogenic: bool = False,
    ) -> str:
        """Register a custom fuel emission factor override.

        Overrides the built-in emission factor for subsequent lookups
        of the specified fuel type.

        Args:
            fuel_type: Fuel type identifier.
            co2_ef: CO2 emission factor in kgCO2 per GJ (>= 0).
            ch4_ef: CH4 emission factor in kgCH4 per GJ (>= 0).
            n2o_ef: N2O emission factor in kgN2O per GJ (>= 0).
            default_efficiency: Boiler thermal efficiency (0 < eta <= 1).
            is_biogenic: Whether the fuel is biogenic.

        Returns:
            Provenance hash of the mutation.

        Raises:
            ValueError: If any emission factor is negative or efficiency
                is out of range.
        """
        if co2_ef < Decimal("0"):
            raise ValueError(
                f"CO2 emission factor must be >= 0, got {co2_ef}"
            )
        if ch4_ef < Decimal("0"):
            raise ValueError(
                f"CH4 emission factor must be >= 0, got {ch4_ef}"
            )
        if n2o_ef < Decimal("0"):
            raise ValueError(
                f"N2O emission factor must be >= 0, got {n2o_ef}"
            )
        if default_efficiency <= Decimal("0") or default_efficiency > Decimal("1"):
            raise ValueError(
                f"Default efficiency must be in (0, 1], got "
                f"{default_efficiency}"
            )

        key = fuel_type.lower().strip()
        entry = {
            "co2_ef": co2_ef.quantize(Decimal("0.001"), ROUND_HALF_UP),
            "ch4_ef": ch4_ef.quantize(Decimal("0.000001"), ROUND_HALF_UP),
            "n2o_ef": n2o_ef.quantize(Decimal("0.0000001"), ROUND_HALF_UP),
            "default_efficiency": default_efficiency.quantize(
                Decimal("0.001"), ROUND_HALF_UP,
            ),
            "is_biogenic": is_biogenic,
            "created_at": _utcnow().isoformat(),
        }

        with self._state_lock:
            self._custom_fuel_efs[key] = entry
            self._mutation_count += 1

        provenance_hash = self._record_provenance(
            "set_custom_fuel_ef",
            {
                "fuel_type": key,
                "co2_ef": str(entry["co2_ef"]),
                "ch4_ef": str(entry["ch4_ef"]),
                "n2o_ef": str(entry["n2o_ef"]),
                "default_efficiency": str(entry["default_efficiency"]),
                "is_biogenic": is_biogenic,
            },
        )

        logger.info(
            "Set custom fuel EF: fuel=%s, co2=%s, ch4=%s, n2o=%s, "
            "efficiency=%s, biogenic=%s",
            key,
            entry["co2_ef"],
            entry["ch4_ef"],
            entry["n2o_ef"],
            entry["default_efficiency"],
            is_biogenic,
        )
        return provenance_hash

    def remove_custom_fuel_ef(
        self,
        fuel_type: str,
    ) -> bool:
        """Remove a custom fuel emission factor override.

        After removal, lookups for this fuel type will revert to the
        built-in emission factor data.

        Args:
            fuel_type: Fuel type identifier to remove.

        Returns:
            True if the custom factor was removed, False if it did
            not exist.
        """
        key = fuel_type.lower().strip()
        with self._state_lock:
            if key in self._custom_fuel_efs:
                del self._custom_fuel_efs[key]
                self._mutation_count += 1
                self._record_provenance(
                    "remove_custom_fuel_ef",
                    {"fuel_type": key, "removed": True},
                )
                logger.info(
                    "Removed custom fuel EF for fuel=%s", key,
                )
                return True

        self._record_provenance(
            "remove_custom_fuel_ef",
            {"fuel_type": key, "removed": False},
        )
        return False

    def set_custom_dh_factor(
        self,
        region: str,
        ef_kgco2e_per_gj: Decimal,
        distribution_loss_pct: Decimal,
    ) -> str:
        """Register a custom district heating factor override.

        Args:
            region: Region identifier.
            ef_kgco2e_per_gj: Emission factor in kgCO2e/GJ (>= 0).
            distribution_loss_pct: Distribution loss fraction (0-1).

        Returns:
            Provenance hash of the mutation.

        Raises:
            ValueError: If ef is negative or loss is out of range.
        """
        if ef_kgco2e_per_gj < Decimal("0"):
            raise ValueError(
                f"DH emission factor must be >= 0, got {ef_kgco2e_per_gj}"
            )
        if distribution_loss_pct < Decimal("0") or distribution_loss_pct > Decimal("1"):
            raise ValueError(
                f"Distribution loss must be in [0, 1], got "
                f"{distribution_loss_pct}"
            )

        key = region.lower().strip()
        entry = {
            "ef_kgco2e_per_gj": ef_kgco2e_per_gj.quantize(
                Decimal("0.1"), ROUND_HALF_UP,
            ),
            "distribution_loss_pct": distribution_loss_pct.quantize(
                Decimal("0.01"), ROUND_HALF_UP,
            ),
            "created_at": _utcnow().isoformat(),
        }

        with self._state_lock:
            self._custom_dh_factors[key] = entry
            self._mutation_count += 1

        provenance_hash = self._record_provenance(
            "set_custom_dh_factor",
            {
                "region": key,
                "ef_kgco2e_per_gj": str(entry["ef_kgco2e_per_gj"]),
                "distribution_loss_pct": str(entry["distribution_loss_pct"]),
            },
        )

        logger.info(
            "Set custom DH factor: region=%s, ef=%s kgCO2e/GJ, loss=%s",
            key,
            entry["ef_kgco2e_per_gj"],
            entry["distribution_loss_pct"],
        )
        return provenance_hash

    def remove_custom_dh_factor(
        self,
        region: str,
    ) -> bool:
        """Remove a custom district heating factor override.

        Args:
            region: Region identifier to remove.

        Returns:
            True if removed, False if it did not exist.
        """
        key = region.lower().strip()
        with self._state_lock:
            if key in self._custom_dh_factors:
                del self._custom_dh_factors[key]
                self._mutation_count += 1
                self._record_provenance(
                    "remove_custom_dh_factor",
                    {"region": key, "removed": True},
                )
                logger.info(
                    "Removed custom DH factor for region=%s", key,
                )
                return True

        self._record_provenance(
            "remove_custom_dh_factor",
            {"region": key, "removed": False},
        )
        return False

    def set_custom_cooling_factor(
        self,
        technology: str,
        cop_min: Decimal,
        cop_max: Decimal,
        cop_default: Decimal,
        energy_source: str = "electricity",
    ) -> str:
        """Register a custom cooling system factor override.

        Args:
            technology: Cooling technology identifier.
            cop_min: Minimum COP (> 0).
            cop_max: Maximum COP (>= cop_min).
            cop_default: Default COP (cop_min <= default <= cop_max).
            energy_source: Energy source ('electricity' or 'heat').

        Returns:
            Provenance hash of the mutation.

        Raises:
            ValueError: If COP values are invalid or energy_source
                is not recognised.
        """
        if cop_min <= Decimal("0"):
            raise ValueError(
                f"COP min must be > 0, got {cop_min}"
            )
        if cop_max < cop_min:
            raise ValueError(
                f"COP max ({cop_max}) must be >= COP min ({cop_min})"
            )
        if cop_default < cop_min or cop_default > cop_max:
            raise ValueError(
                f"COP default ({cop_default}) must be between "
                f"COP min ({cop_min}) and COP max ({cop_max})"
            )
        if energy_source not in ("electricity", "heat"):
            raise ValueError(
                f"Energy source must be 'electricity' or 'heat', "
                f"got {energy_source!r}"
            )

        key = technology.lower().strip()
        entry = {
            "cop_min": cop_min.quantize(Decimal("0.1"), ROUND_HALF_UP),
            "cop_max": cop_max.quantize(Decimal("0.1"), ROUND_HALF_UP),
            "cop_default": cop_default.quantize(
                Decimal("0.1"), ROUND_HALF_UP,
            ),
            "energy_source": energy_source,
            "created_at": _utcnow().isoformat(),
        }

        with self._state_lock:
            self._custom_cooling_factors[key] = entry
            self._mutation_count += 1

        provenance_hash = self._record_provenance(
            "set_custom_cooling_factor",
            {
                "technology": key,
                "cop_default": str(entry["cop_default"]),
                "energy_source": energy_source,
            },
        )

        logger.info(
            "Set custom cooling factor: tech=%s, COP=%s, source=%s",
            key,
            entry["cop_default"],
            energy_source,
        )
        return provenance_hash

    def remove_custom_cooling_factor(
        self,
        technology: str,
    ) -> bool:
        """Remove a custom cooling system factor override.

        Args:
            technology: Cooling technology identifier to remove.

        Returns:
            True if removed, False if it did not exist.
        """
        key = technology.lower().strip()
        with self._state_lock:
            if key in self._custom_cooling_factors:
                del self._custom_cooling_factors[key]
                self._mutation_count += 1
                self._record_provenance(
                    "remove_custom_cooling_factor",
                    {"technology": key, "removed": True},
                )
                logger.info(
                    "Removed custom cooling factor for tech=%s", key,
                )
                return True

        self._record_provenance(
            "remove_custom_cooling_factor",
            {"technology": key, "removed": False},
        )
        return False

    def set_custom_chp_defaults(
        self,
        fuel_type: str,
        electrical_efficiency: Decimal,
        thermal_efficiency: Decimal,
        overall_efficiency: Decimal,
    ) -> str:
        """Register custom CHP default efficiencies.

        Args:
            fuel_type: CHP fuel type identifier.
            electrical_efficiency: Electrical efficiency (0 < eta <= 1).
            thermal_efficiency: Thermal efficiency (0 < eta <= 1).
            overall_efficiency: Overall efficiency (0 < eta <= 1).

        Returns:
            Provenance hash of the mutation.

        Raises:
            ValueError: If any efficiency is out of range.
        """
        for name, val in [
            ("electrical_efficiency", electrical_efficiency),
            ("thermal_efficiency", thermal_efficiency),
            ("overall_efficiency", overall_efficiency),
        ]:
            if val <= Decimal("0") or val > Decimal("1"):
                raise ValueError(
                    f"{name} must be in (0, 1], got {val}"
                )

        key = fuel_type.lower().strip()
        entry = {
            "electrical_efficiency": electrical_efficiency.quantize(
                Decimal("0.001"), ROUND_HALF_UP,
            ),
            "thermal_efficiency": thermal_efficiency.quantize(
                Decimal("0.001"), ROUND_HALF_UP,
            ),
            "overall_efficiency": overall_efficiency.quantize(
                Decimal("0.001"), ROUND_HALF_UP,
            ),
            "created_at": _utcnow().isoformat(),
        }

        with self._state_lock:
            self._custom_chp_defaults[key] = entry
            self._mutation_count += 1

        provenance_hash = self._record_provenance(
            "set_custom_chp_defaults",
            {
                "fuel_type": key,
                "electrical_efficiency": str(
                    entry["electrical_efficiency"],
                ),
                "thermal_efficiency": str(entry["thermal_efficiency"]),
                "overall_efficiency": str(entry["overall_efficiency"]),
            },
        )

        logger.info(
            "Set custom CHP defaults: fuel=%s, elec=%s, thermal=%s, "
            "overall=%s",
            key,
            entry["electrical_efficiency"],
            entry["thermal_efficiency"],
            entry["overall_efficiency"],
        )
        return provenance_hash

    def remove_custom_chp_defaults(
        self,
        fuel_type: str,
    ) -> bool:
        """Remove custom CHP default efficiencies.

        Args:
            fuel_type: CHP fuel type identifier to remove.

        Returns:
            True if removed, False if it did not exist.
        """
        key = fuel_type.lower().strip()
        with self._state_lock:
            if key in self._custom_chp_defaults:
                del self._custom_chp_defaults[key]
                self._mutation_count += 1
                self._record_provenance(
                    "remove_custom_chp_defaults",
                    {"fuel_type": key, "removed": True},
                )
                logger.info(
                    "Removed custom CHP defaults for fuel=%s", key,
                )
                return True

        self._record_provenance(
            "remove_custom_chp_defaults",
            {"fuel_type": key, "removed": False},
        )
        return False

    # ==================================================================
    # PUBLIC METHODS: Utility Queries
    # ==================================================================

    def get_biogenic_fuels(self) -> List[str]:
        """Get a sorted list of all biogenic fuel types.

        Returns:
            Sorted list of fuel type identifiers classified as biogenic.
        """
        return sorted(_BIOGENIC_FUELS)

    def get_zero_emission_fuels(self) -> List[str]:
        """Get a sorted list of all zero-emission fuel types.

        These fuels have zero direct combustion emissions (waste heat,
        geothermal, solar thermal, electric).

        Returns:
            Sorted list of fuel type identifiers with zero emissions.
        """
        return sorted(_ZERO_EMISSION_FUELS)

    def get_all_gwp_sources(self) -> List[str]:
        """Get a sorted list of all available GWP source identifiers.

        Returns:
            Sorted list of GWP source strings (AR4, AR5, AR6, AR6_20YR).
        """
        return sorted(GWP_VALUES.keys())

    def get_all_conversion_names(self) -> List[str]:
        """Get a sorted list of all available unit conversion names.

        Returns:
            Sorted list of conversion factor name strings.
        """
        return sorted(UNIT_CONVERSIONS.keys())

    def get_chp_fuel_types(self) -> List[str]:
        """List all available CHP fuel types.

        Returns fuel types from both built-in data and any custom
        overrides, sorted alphabetically.

        Returns:
            Sorted list of CHP fuel type identifiers.
        """
        types = set(CHP_DEFAULT_EFFICIENCIES.keys())
        with self._state_lock:
            types.update(self._custom_chp_defaults.keys())
        return sorted(types)

    # ==================================================================
    # PUBLIC METHODS: Reset
    # ==================================================================

    def reset(self) -> None:
        """Reset all mutable state to initial values.

        Clears all custom overrides, operational counters, and
        provenance hashes. Built-in data is not affected.

        Primarily used for testing and development.
        """
        with self._state_lock:
            self._custom_fuel_efs.clear()
            self._custom_dh_factors.clear()
            self._custom_cooling_factors.clear()
            self._custom_chp_defaults.clear()
            self._lookup_count = 0
            self._mutation_count = 0
            self._conversion_count = 0
            self._provenance_hashes.clear()

        logger.info(
            "SteamHeatDatabaseEngine reset: all mutable state cleared"
        )

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance for testing.

        After calling this method, the next instantiation will create
        a fresh instance. This is intended for test isolation only.
        """
        with cls._lock:
            cls._instance = None
        logger.info("SteamHeatDatabaseEngine singleton reset")


# ===========================================================================
# Module-level convenience functions
# ===========================================================================

_module_engine: Optional[SteamHeatDatabaseEngine] = None
_module_lock = threading.Lock()


def get_database(
    config: Any = None,
    metrics: Any = None,
    provenance: Any = None,
) -> SteamHeatDatabaseEngine:
    """Get or create the module-level database engine singleton.

    Args:
        config: Optional configuration object.
        metrics: Optional metrics recorder.
        provenance: Optional provenance tracker.

    Returns:
        The SteamHeatDatabaseEngine singleton instance.
    """
    global _module_engine
    if _module_engine is None:
        with _module_lock:
            if _module_engine is None:
                _module_engine = SteamHeatDatabaseEngine(
                    config=config,
                    metrics=metrics,
                    provenance=provenance,
                )
    return _module_engine


def get_fuel_ef(fuel_type: str) -> Dict[str, Any]:
    """Module-level convenience: get fuel emission factor.

    Args:
        fuel_type: Fuel type identifier.

    Returns:
        Fuel emission factor dictionary.
    """
    return get_database().get_fuel_ef(fuel_type)


def get_all_fuel_efs() -> Dict[str, Dict[str, Any]]:
    """Module-level convenience: get all fuel emission factors.

    Returns:
        Dictionary mapping fuel_type to emission factor dictionaries.
    """
    return get_database().get_all_fuel_efs()


def get_fuel_types() -> List[str]:
    """Module-level convenience: list available fuel types.

    Returns:
        Sorted list of fuel type identifiers.
    """
    return get_database().get_fuel_types()


def is_biogenic_fuel(fuel_type: str) -> bool:
    """Module-level convenience: check if fuel is biogenic.

    Args:
        fuel_type: Fuel type identifier.

    Returns:
        True if fuel is biogenic.
    """
    return get_database().is_biogenic_fuel(fuel_type)


def get_default_efficiency(fuel_type: str) -> Decimal:
    """Module-level convenience: get default boiler efficiency.

    Args:
        fuel_type: Fuel type identifier.

    Returns:
        Default thermal efficiency as Decimal.
    """
    return get_database().get_default_efficiency(fuel_type)


def get_dh_factor(region: str) -> Dict[str, Any]:
    """Module-level convenience: get district heating factor.

    Args:
        region: Region identifier.

    Returns:
        District heating factor dictionary.
    """
    return get_database().get_dh_factor(region)


def get_all_dh_factors() -> Dict[str, Dict[str, Any]]:
    """Module-level convenience: get all district heating factors.

    Returns:
        Dictionary mapping region to DH factor dictionaries.
    """
    return get_database().get_all_dh_factors()


def get_dh_regions() -> List[str]:
    """Module-level convenience: list available DH regions.

    Returns:
        Sorted list of region identifiers.
    """
    return get_database().get_dh_regions()


def get_distribution_loss_pct(region: str) -> Decimal:
    """Module-level convenience: get DH distribution loss.

    Args:
        region: Region identifier.

    Returns:
        Distribution loss as Decimal fraction.
    """
    return get_database().get_distribution_loss_pct(region)


def get_cooling_factor(technology: str) -> Dict[str, Any]:
    """Module-level convenience: get cooling system parameters.

    Args:
        technology: Cooling technology identifier.

    Returns:
        Cooling factor dictionary.
    """
    return get_database().get_cooling_factor(technology)


def get_all_cooling_factors() -> Dict[str, Dict[str, Any]]:
    """Module-level convenience: get all cooling factors.

    Returns:
        Dictionary mapping technology to cooling parameter dictionaries.
    """
    return get_database().get_all_cooling_factors()


def get_cooling_technologies() -> List[str]:
    """Module-level convenience: list cooling technologies.

    Returns:
        Sorted list of cooling technology identifiers.
    """
    return get_database().get_cooling_technologies()


def get_cop(technology: str) -> Decimal:
    """Module-level convenience: get default COP.

    Args:
        technology: Cooling technology identifier.

    Returns:
        Default COP as Decimal.
    """
    return get_database().get_cop(technology)


def get_cooling_energy_source(technology: str) -> str:
    """Module-level convenience: get cooling energy source.

    Args:
        technology: Cooling technology identifier.

    Returns:
        Energy source string.
    """
    return get_database().get_cooling_energy_source(technology)


def get_chp_defaults(fuel_type: str) -> Dict[str, Any]:
    """Module-level convenience: get CHP default efficiencies.

    Args:
        fuel_type: CHP fuel type identifier.

    Returns:
        CHP efficiency dictionary.
    """
    return get_database().get_chp_defaults(fuel_type)


def get_all_chp_defaults() -> Dict[str, Dict[str, Any]]:
    """Module-level convenience: get all CHP defaults.

    Returns:
        Dictionary mapping fuel_type to CHP efficiency dictionaries.
    """
    return get_database().get_all_chp_defaults()


def get_gwp_values(source: str) -> Dict[str, Any]:
    """Module-level convenience: get GWP values.

    Args:
        source: IPCC source identifier.

    Returns:
        GWP values dictionary.
    """
    return get_database().get_gwp_values(source)


def get_gwp(gas: str, source: str) -> Decimal:
    """Module-level convenience: get specific GWP.

    Args:
        gas: Gas identifier.
        source: IPCC source identifier.

    Returns:
        GWP multiplier as Decimal.
    """
    return get_database().get_gwp(gas, source)


def convert_energy(
    value: Decimal,
    from_unit: str,
    to_unit: str,
) -> Decimal:
    """Module-level convenience: convert energy units.

    Args:
        value: Energy quantity.
        from_unit: Source unit.
        to_unit: Target unit.

    Returns:
        Converted energy quantity as Decimal.
    """
    return get_database().convert_energy(value, from_unit, to_unit)


def get_conversion_factor(conversion: str) -> Decimal:
    """Module-level convenience: get raw conversion factor.

    Args:
        conversion: Conversion factor name.

    Returns:
        Conversion factor as Decimal.
    """
    return get_database().get_conversion_factor(conversion)


def get_blended_ef(
    fuel_mix: Dict[str, Decimal],
) -> Dict[str, Any]:
    """Module-level convenience: calculate blended emission factor.

    Args:
        fuel_mix: Fuel type to fraction mapping.

    Returns:
        Blended emission factor dictionary.
    """
    return get_database().get_blended_ef(fuel_mix)


def search_factors(query: str) -> List[Dict[str, Any]]:
    """Module-level convenience: search factor databases.

    Args:
        query: Search string.

    Returns:
        List of matching factor entries.
    """
    return get_database().search_factors(query)


def get_database_stats() -> Dict[str, Any]:
    """Module-level convenience: get database statistics.

    Returns:
        Statistics dictionary.
    """
    return get_database().get_database_stats()


def health_check() -> Dict[str, Any]:
    """Module-level convenience: perform health check.

    Returns:
        Health check result dictionary.
    """
    return get_database().health_check()
