# -*- coding: utf-8 -*-
"""
Engine 1: Cooling Database Engine for AGENT-MRV-012.

Stores and retrieves cooling technology specifications, emission factors,
refrigerant GWP data, efficiency conversions, cooling unit conversions,
district cooling regional factors, heat source emission factors, GWP
values, and AHRI part-load weights for Scope 2 purchased cooling emission
calculations per GHG Protocol Scope 2 Guidance (2015).

Built-in Data:
- 18 cooling technologies with COP ranges, IPLV, and energy sources
- 12 district cooling regional emission factors (kgCO2e/GJ)
- 11 heat source emission factors for absorption chillers
- 11 refrigerant GWP values (AR5 and AR6)
- 4 AHRI 550/590 part-load weighting factors for IPLV calculation
- 4 efficiency metric conversion constants (COP/EER/kW_per_ton/SEER)
- 6 cooling unit conversion factors (ton_hour/GJ/MMBTU/MJ/BTU/TR)
- 4 IPCC GWP assessment report editions (AR4, AR5, AR6, AR6_20YR)

All values as Decimal with ROUND_HALF_UP for zero-hallucination guarantees.
Thread-safe singleton via RLock. SHA-256 provenance on every operation.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-012 Cooling Purchase Agent (GL-MRV-X-023)
Status: Production Ready
"""

from __future__ import annotations

__all__ = ["CoolingDatabaseEngine"]

import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import utcnow

from greenlang.agents.mrv.cooling_purchase.models import (
    COOLING_TECHNOLOGY_SPECS,
    DISTRICT_COOLING_FACTORS,
    HEAT_SOURCE_FACTORS,
    REFRIGERANT_GWP,
    GWP_VALUES,
    UNIT_CONVERSIONS,
    AHRI_PART_LOAD_WEIGHTS,
    EFFICIENCY_CONVERSIONS,
    CoolingTechnology,
    CoolingTechnologySpec,
    DistrictCoolingFactor,
    HeatSourceFactor,
    RefrigerantData,
    HeatSource,
    Refrigerant,
    GWPSource,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional metrics import
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.cooling_purchase.metrics import (
        CoolingPurchaseMetrics,
        get_metrics as _get_metrics,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _get_metrics = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Technology classification sets
# ---------------------------------------------------------------------------
# Electric chiller technologies (vapour-compression cycle with compressor).

_ELECTRIC_TECHNOLOGIES: frozenset = frozenset({
    CoolingTechnology.WATER_COOLED_CENTRIFUGAL.value,
    CoolingTechnology.AIR_COOLED_CENTRIFUGAL.value,
    CoolingTechnology.WATER_COOLED_SCREW.value,
    CoolingTechnology.AIR_COOLED_SCREW.value,
    CoolingTechnology.WATER_COOLED_RECIPROCATING.value,
    CoolingTechnology.AIR_COOLED_SCROLL.value,
})

# Absorption chiller technologies (heat-driven cycle).

_ABSORPTION_TECHNOLOGIES: frozenset = frozenset({
    CoolingTechnology.SINGLE_EFFECT_LIBR.value,
    CoolingTechnology.DOUBLE_EFFECT_LIBR.value,
    CoolingTechnology.TRIPLE_EFFECT_LIBR.value,
    CoolingTechnology.AMMONIA_ABSORPTION.value,
})

# Free cooling technologies (natural heat sinks, pump/fan only).

_FREE_COOLING_TECHNOLOGIES: frozenset = frozenset({
    CoolingTechnology.SEAWATER_FREE.value,
    CoolingTechnology.LAKE_FREE.value,
    CoolingTechnology.RIVER_FREE.value,
    CoolingTechnology.AMBIENT_AIR_FREE.value,
})

# Thermal energy storage technologies.

_TES_TECHNOLOGIES: frozenset = frozenset({
    CoolingTechnology.ICE_TES.value,
    CoolingTechnology.CHILLED_WATER_TES.value,
    CoolingTechnology.PCM_TES.value,
})

# Zero-emission heat sources (no direct fossil combustion).

_ZERO_EMISSION_HEAT_SOURCES: frozenset = frozenset({
    HeatSource.WASTE_HEAT.value,
    HeatSource.CHP_EXHAUST.value,
    HeatSource.SOLAR_THERMAL.value,
    HeatSource.GEOTHERMAL.value,
    HeatSource.BIOGAS_STEAM.value,
    HeatSource.ELECTRIC_BOILER.value,
    HeatSource.HEAT_PUMP.value,
})

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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
# CoolingDatabaseEngine
# ===========================================================================

class CoolingDatabaseEngine:
    """Engine 1: Reference data database for Scope 2 Cooling Purchase
    emission calculations.

    Manages a comprehensive database of cooling technology specifications,
    district cooling regional emission factors, heat source emission factors
    for absorption chillers, refrigerant GWP values, AHRI part-load
    weights, efficiency metric conversions, cooling unit conversions, and
    GWP values for GHG Protocol Scope 2 purchased cooling accounting.

    Implements the thread-safe singleton pattern using RLock to ensure
    exactly one instance per process. All arithmetic uses Decimal with
    ROUND_HALF_UP for zero-hallucination deterministic calculations.
    Every lookup produces a SHA-256 provenance hash for complete audit
    trails.

    Thread Safety:
        Uses ``threading.RLock`` for singleton creation and all mutable
        state access. Immutable built-in data (COOLING_TECHNOLOGY_SPECS,
        DISTRICT_COOLING_FACTORS, etc.) is inherently thread-safe.

    Data Tables:
        1. Cooling Technology Specs - 18 technologies with COP ranges
        2. District Cooling Factors - 12 regions
        3. Heat Source Factors - 11 absorption chiller heat sources
        4. Refrigerant GWP Data - 11 refrigerants (AR5 + AR6)
        5. AHRI Part-Load Weights - 4 load points (100/75/50/25%)
        6. Efficiency Conversions - 4 metric conversion constants
        7. Cooling Unit Conversions - 6 unit conversion factors
        8. GWP Values - 4 IPCC sources (AR4, AR5, AR6, AR6_20YR)

    Attributes:
        ENGINE_ID: Constant identifier for this engine.
        ENGINE_VERSION: Semantic version string.

    Example:
        >>> engine = CoolingDatabaseEngine()
        >>> spec = engine.get_technology_spec("water_cooled_centrifugal")
        >>> assert spec.cop_default == Decimal("6.1")
        >>> dc = engine.get_district_cooling_factor("dubai_uae")
        >>> assert dc.ef_kgco2e_per_gj == Decimal("45.0")
        >>> gwp = engine.get_refrigerant_gwp("r_134a", "AR5")
        >>> assert gwp == Decimal("1430")
    """

    ENGINE_ID: str = "cooling_database"
    ENGINE_VERSION: str = "1.0.0"

    _instance: Optional[CoolingDatabaseEngine] = None
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton construction
    # ------------------------------------------------------------------

    def __new__(
        cls,
        config: Any = None,
        metrics: Any = None,
        provenance: Any = None,
    ) -> CoolingDatabaseEngine:
        """Return the singleton CoolingDatabaseEngine instance.

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
        """Initialize the cooling database engine.

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
        self._custom_technologies: Dict[str, CoolingTechnologySpec] = {}
        self._custom_district_factors: Dict[str, DistrictCoolingFactor] = {}
        self._custom_heat_source_factors: Dict[str, HeatSourceFactor] = {}
        self._custom_refrigerants: Dict[str, RefrigerantData] = {}

        # Operational counters
        self._lookup_count: int = 0
        self._mutation_count: int = 0
        self._conversion_count: int = 0
        self._provenance_hashes: List[str] = []

        self._initialized = True
        logger.info(
            "CoolingDatabaseEngine v%s initialized "
            "(technologies=%d, dc_regions=%d, heat_sources=%d, "
            "refrigerants=%d, gwp_sources=%d, unit_conversions=%d)",
            self.ENGINE_VERSION,
            len(COOLING_TECHNOLOGY_SPECS),
            len(DISTRICT_COOLING_FACTORS),
            len(HEAT_SOURCE_FACTORS),
            len(REFRIGERANT_GWP),
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
            operation: Name of the operation (e.g. 'get_technology_spec').
            data: Dictionary of operation inputs and outputs to hash.

        Returns:
            64-character SHA-256 hex digest.
        """
        payload = {
            "engine": self.ENGINE_ID,
            "operation": operation,
            "timestamp": utcnow().isoformat(),
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
                (technology_spec, district_cooling, heat_source,
                refrigerant, efficiency, unit_conversion, gwp).
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
    # PUBLIC METHODS: Technology Lookups (Methods 1-10)
    # ==================================================================

    def get_technology_spec(
        self,
        technology: str,
    ) -> CoolingTechnologySpec:
        """Get the full performance specification for a cooling technology.

        Returns the CoolingTechnologySpec model containing COP range,
        default COP, IPLV, energy source, compressor type, and condenser
        type for the specified cooling technology. Checks custom
        overrides first, then falls back to built-in data.

        Args:
            technology: Cooling technology identifier (e.g.
                'water_cooled_centrifugal', 'single_effect_libr',
                'seawater_free'). Must match a CoolingTechnology enum
                value or a custom override key. Case-insensitive.

        Returns:
            CoolingTechnologySpec with full technology parameters.

        Raises:
            ValueError: If technology is not recognised.
        """
        with self._state_lock:
            self._lookup_count += 1

        key = technology.lower().strip()

        # Check custom overrides first
        with self._state_lock:
            if key in self._custom_technologies:
                spec = self._custom_technologies[key]
                self._record_provenance(
                    "get_technology_spec",
                    {"technology": key, "source": "custom"},
                )
                self._record_metric("technology_spec", "hit")
                logger.debug(
                    "Technology spec lookup (custom): tech=%s, COP=%s",
                    key,
                    spec.cop_default,
                )
                return spec

        # Built-in data
        if key not in COOLING_TECHNOLOGY_SPECS:
            self._record_metric("technology_spec", "miss")
            raise ValueError(
                f"Unknown cooling technology: {technology!r}. "
                f"Valid technologies: "
                f"{sorted(COOLING_TECHNOLOGY_SPECS.keys())}"
            )

        spec = COOLING_TECHNOLOGY_SPECS[key]
        self._record_provenance(
            "get_technology_spec",
            {
                "technology": key,
                "cop_default": str(spec.cop_default),
                "cop_min": str(spec.cop_min),
                "cop_max": str(spec.cop_max),
                "energy_source": spec.energy_source,
                "source": "builtin",
            },
        )
        self._record_metric("technology_spec", "hit")

        logger.debug(
            "Technology spec lookup: tech=%s, COP=%s, source=%s",
            key,
            spec.cop_default,
            spec.energy_source,
        )
        return spec

    def get_all_technologies(
        self,
    ) -> Dict[str, CoolingTechnologySpec]:
        """Get performance specifications for all available technologies.

        Returns a dictionary keyed by technology identifier with each
        value being the full CoolingTechnologySpec. Custom overrides
        take precedence over built-in data.

        Returns:
            Dictionary mapping technology identifier to
            CoolingTechnologySpec instances.
        """
        with self._state_lock:
            self._lookup_count += 1

        results: Dict[str, CoolingTechnologySpec] = {}

        # Built-in data
        for tech_key in sorted(COOLING_TECHNOLOGY_SPECS.keys()):
            results[tech_key] = COOLING_TECHNOLOGY_SPECS[tech_key]

        # Apply custom overrides
        with self._state_lock:
            for tech_key, spec in self._custom_technologies.items():
                results[tech_key] = spec

        self._record_provenance(
            "get_all_technologies",
            {"count": len(results)},
        )
        self._record_metric("technology_spec", "hit")

        logger.debug(
            "All technology specs retrieved: count=%d",
            len(results),
        )
        return results

    def get_default_cop(
        self,
        technology: str,
    ) -> Decimal:
        """Get the default Coefficient of Performance for a technology.

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
        spec = self.get_technology_spec(technology)
        return spec.cop_default

    def get_cop_range(
        self,
        technology: str,
    ) -> Tuple[Decimal, Decimal]:
        """Get the COP range (min, max) for a cooling technology.

        Returns the minimum and maximum COP values that represent the
        expected operating range under typical to optimal conditions
        for the specified technology.

        Args:
            technology: Cooling technology identifier. Case-insensitive.

        Returns:
            Tuple of (cop_min, cop_max) as Decimal values.

        Raises:
            ValueError: If technology is not recognised.
        """
        spec = self.get_technology_spec(technology)
        return (spec.cop_min, spec.cop_max)

    def get_iplv(
        self,
        technology: str,
    ) -> Optional[Decimal]:
        """Get the Integrated Part-Load Value for a cooling technology.

        IPLV represents the weighted average COP across four AHRI
        standard operating conditions. Not all technologies have an
        IPLV (free cooling, TES, and district cooling return None).

        Args:
            technology: Cooling technology identifier. Case-insensitive.

        Returns:
            IPLV as Decimal, or None if not applicable.

        Raises:
            ValueError: If technology is not recognised.
        """
        spec = self.get_technology_spec(technology)
        return spec.iplv

    def get_energy_source(
        self,
        technology: str,
    ) -> str:
        """Get the primary energy source for a cooling technology.

        Returns the type of energy input required by the technology:
        'electricity' for electric chillers, free cooling, and TES;
        'heat' for absorption chillers; 'mixed' for district cooling.

        Args:
            technology: Cooling technology identifier. Case-insensitive.

        Returns:
            Energy source string ('electricity', 'heat', or 'mixed').

        Raises:
            ValueError: If technology is not recognised.
        """
        spec = self.get_technology_spec(technology)
        return spec.energy_source

    def is_electric_technology(
        self,
        technology: str,
    ) -> bool:
        """Check whether a technology is an electric chiller type.

        Electric chiller technologies use a vapour-compression cycle
        with a mechanical compressor driven by electricity. These are
        the six electric chiller types: water-cooled centrifugal,
        air-cooled centrifugal, water-cooled screw, air-cooled screw,
        water-cooled reciprocating, and air-cooled scroll.

        Args:
            technology: Cooling technology identifier. Case-insensitive.

        Returns:
            True if the technology is an electric chiller, False otherwise.

        Raises:
            ValueError: If technology is not recognised.
        """
        key = technology.lower().strip()

        # Validate the technology exists
        self._ensure_technology_exists(key)

        return key in _ELECTRIC_TECHNOLOGIES

    def is_absorption_technology(
        self,
        technology: str,
    ) -> bool:
        """Check whether a technology is an absorption chiller type.

        Absorption chiller technologies use heat energy rather than
        mechanical compression to drive the refrigeration cycle. These
        are the four absorption types: single-effect LiBr, double-effect
        LiBr, triple-effect LiBr, and ammonia absorption.

        Args:
            technology: Cooling technology identifier. Case-insensitive.

        Returns:
            True if the technology is an absorption chiller, False otherwise.

        Raises:
            ValueError: If technology is not recognised.
        """
        key = technology.lower().strip()

        # Validate the technology exists
        self._ensure_technology_exists(key)

        return key in _ABSORPTION_TECHNOLOGIES

    def is_free_cooling_technology(
        self,
        technology: str,
    ) -> bool:
        """Check whether a technology is a free cooling type.

        Free cooling technologies exploit naturally available cold
        sources (seawater, lake water, river water, or ambient air) to
        provide cooling with minimal energy input. Only pump or fan
        electricity is consumed.

        Args:
            technology: Cooling technology identifier. Case-insensitive.

        Returns:
            True if the technology is free cooling, False otherwise.

        Raises:
            ValueError: If technology is not recognised.
        """
        key = technology.lower().strip()

        # Validate the technology exists
        self._ensure_technology_exists(key)

        return key in _FREE_COOLING_TECHNOLOGIES

    def is_tes_technology(
        self,
        technology: str,
    ) -> bool:
        """Check whether a technology is a thermal energy storage type.

        TES technologies enable temporal shifting of cooling production
        from peak to off-peak hours. The three TES types are: ice TES,
        chilled water TES, and PCM (phase-change material) TES.

        Args:
            technology: Cooling technology identifier. Case-insensitive.

        Returns:
            True if the technology is TES, False otherwise.

        Raises:
            ValueError: If technology is not recognised.
        """
        key = technology.lower().strip()

        # Validate the technology exists
        self._ensure_technology_exists(key)

        return key in _TES_TECHNOLOGIES

    def _ensure_technology_exists(self, key: str) -> None:
        """Validate that a technology key exists in built-in or custom data.

        Args:
            key: Lowercase stripped technology identifier.

        Raises:
            ValueError: If the technology does not exist.
        """
        with self._state_lock:
            if key in self._custom_technologies:
                return

        if key not in COOLING_TECHNOLOGY_SPECS:
            raise ValueError(
                f"Unknown cooling technology: {key!r}. "
                f"Valid technologies: "
                f"{sorted(COOLING_TECHNOLOGY_SPECS.keys())}"
            )

    # ==================================================================
    # PUBLIC METHODS: District Cooling Factors (Methods 11-13)
    # ==================================================================

    def get_district_cooling_factor(
        self,
        region: str,
    ) -> DistrictCoolingFactor:
        """Get the district cooling emission factor for a region.

        Returns the DistrictCoolingFactor model containing the composite
        emission factor (kgCO2e per GJ cooling delivered), technology
        mix description, and notes for the specified region. Checks
        custom overrides first, then built-in data, then falls back
        to global_default.

        Args:
            region: Region identifier (e.g. 'dubai_uae', 'singapore',
                'eu_nordic', 'global_default'). Case-insensitive.

        Returns:
            DistrictCoolingFactor with regional emission data.
        """
        with self._state_lock:
            self._lookup_count += 1

        key = region.lower().strip()

        # Check custom overrides first
        with self._state_lock:
            if key in self._custom_district_factors:
                factor = self._custom_district_factors[key]
                self._record_provenance(
                    "get_district_cooling_factor",
                    {"region": key, "source": "custom"},
                )
                self._record_metric("district_cooling", "hit")
                logger.debug(
                    "District cooling factor lookup (custom): "
                    "region=%s, ef=%s kgCO2e/GJ",
                    key,
                    factor.ef_kgco2e_per_gj,
                )
                return factor

        # Built-in data
        if key in DISTRICT_COOLING_FACTORS:
            factor = DISTRICT_COOLING_FACTORS[key]
            self._record_provenance(
                "get_district_cooling_factor",
                {
                    "region": key,
                    "ef_kgco2e_per_gj": str(factor.ef_kgco2e_per_gj),
                    "technology_mix": factor.technology_mix,
                    "source": "builtin",
                },
            )
            self._record_metric("district_cooling", "hit")
            logger.debug(
                "District cooling factor lookup: region=%s, "
                "ef=%s kgCO2e/GJ",
                key,
                factor.ef_kgco2e_per_gj,
            )
            return factor

        # Fallback to global_default
        logger.warning(
            "No district cooling factor for region %s, "
            "using global_default",
            key,
        )
        fallback = DISTRICT_COOLING_FACTORS["global_default"]
        self._record_provenance(
            "get_district_cooling_factor",
            {
                "region": key,
                "ef_kgco2e_per_gj": str(fallback.ef_kgco2e_per_gj),
                "source": "global_default_fallback",
            },
        )
        self._record_metric("district_cooling", "hit")
        return fallback

    def get_all_district_cooling_factors(
        self,
    ) -> Dict[str, DistrictCoolingFactor]:
        """Get district cooling factors for all available regions.

        Returns a dictionary keyed by region with each value being the
        full DistrictCoolingFactor model. Custom overrides take
        precedence over built-in data.

        Returns:
            Dictionary mapping region to DistrictCoolingFactor instances.
        """
        with self._state_lock:
            self._lookup_count += 1

        results: Dict[str, DistrictCoolingFactor] = {}

        # Built-in factors
        for region_key in sorted(DISTRICT_COOLING_FACTORS.keys()):
            results[region_key] = DISTRICT_COOLING_FACTORS[region_key]

        # Apply custom overrides
        with self._state_lock:
            for region_key, factor in self._custom_district_factors.items():
                results[region_key] = factor

        self._record_provenance(
            "get_all_district_cooling_factors",
            {"count": len(results)},
        )
        self._record_metric("district_cooling", "hit")

        logger.debug(
            "All district cooling factors retrieved: count=%d",
            len(results),
        )
        return results

    def get_district_ef(
        self,
        region: str,
    ) -> Decimal:
        """Get the emission factor value for a district cooling region.

        Convenience method returning only the numeric emission factor
        (kgCO2e per GJ) without the full DistrictCoolingFactor model.

        Args:
            region: Region identifier. Case-insensitive.

        Returns:
            Emission factor as Decimal in kgCO2e/GJ.
        """
        factor = self.get_district_cooling_factor(region)
        return factor.ef_kgco2e_per_gj

    # ==================================================================
    # PUBLIC METHODS: Heat Source Factors (Methods 14-17)
    # ==================================================================

    def get_heat_source_factor(
        self,
        source: str,
    ) -> HeatSourceFactor:
        """Get the emission factor for an absorption chiller heat source.

        Returns the HeatSourceFactor model containing the emission
        factor (kgCO2e per GJ heat input) and notes for the specified
        heat source type. Checks custom overrides first, then built-in
        data.

        Args:
            source: Heat source identifier (e.g. 'natural_gas_steam',
                'waste_heat', 'solar_thermal'). Must match a HeatSource
                enum value or a custom override key. Case-insensitive.

        Returns:
            HeatSourceFactor with emission data.

        Raises:
            ValueError: If source is not recognised.
        """
        with self._state_lock:
            self._lookup_count += 1

        key = source.lower().strip()

        # Check custom overrides first
        with self._state_lock:
            if key in self._custom_heat_source_factors:
                factor = self._custom_heat_source_factors[key]
                self._record_provenance(
                    "get_heat_source_factor",
                    {"source": key, "source_type": "custom"},
                )
                self._record_metric("heat_source", "hit")
                logger.debug(
                    "Heat source factor lookup (custom): source=%s, "
                    "ef=%s kgCO2e/GJ",
                    key,
                    factor.ef_kgco2e_per_gj,
                )
                return factor

        # Built-in data
        if key not in HEAT_SOURCE_FACTORS:
            self._record_metric("heat_source", "miss")
            raise ValueError(
                f"Unknown heat source: {source!r}. "
                f"Valid heat sources: "
                f"{sorted(HEAT_SOURCE_FACTORS.keys())}"
            )

        factor = HEAT_SOURCE_FACTORS[key]
        self._record_provenance(
            "get_heat_source_factor",
            {
                "source": key,
                "ef_kgco2e_per_gj": str(factor.ef_kgco2e_per_gj),
                "source_type": "builtin",
            },
        )
        self._record_metric("heat_source", "hit")

        logger.debug(
            "Heat source factor lookup: source=%s, ef=%s kgCO2e/GJ",
            key,
            factor.ef_kgco2e_per_gj,
        )
        return factor

    def get_all_heat_source_factors(
        self,
    ) -> Dict[str, HeatSourceFactor]:
        """Get emission factors for all available heat sources.

        Returns a dictionary keyed by heat source identifier with each
        value being the full HeatSourceFactor model. Custom overrides
        take precedence over built-in data.

        Returns:
            Dictionary mapping heat source to HeatSourceFactor instances.
        """
        with self._state_lock:
            self._lookup_count += 1

        results: Dict[str, HeatSourceFactor] = {}

        # Built-in factors
        for source_key in sorted(HEAT_SOURCE_FACTORS.keys()):
            results[source_key] = HEAT_SOURCE_FACTORS[source_key]

        # Apply custom overrides
        with self._state_lock:
            for source_key, factor in self._custom_heat_source_factors.items():
                results[source_key] = factor

        self._record_provenance(
            "get_all_heat_source_factors",
            {"count": len(results)},
        )
        self._record_metric("heat_source", "hit")

        logger.debug(
            "All heat source factors retrieved: count=%d",
            len(results),
        )
        return results

    def get_heat_source_ef(
        self,
        source: str,
    ) -> Decimal:
        """Get the emission factor value for a heat source.

        Convenience method returning only the numeric emission factor
        (kgCO2e per GJ) without the full HeatSourceFactor model.

        Args:
            source: Heat source identifier. Case-insensitive.

        Returns:
            Emission factor as Decimal in kgCO2e/GJ.

        Raises:
            ValueError: If source is not recognised.
        """
        factor = self.get_heat_source_factor(source)
        return factor.ef_kgco2e_per_gj

    def is_zero_emission_heat_source(
        self,
        source: str,
    ) -> bool:
        """Check whether a heat source has zero direct emissions.

        Zero-emission heat sources include waste heat, CHP exhaust,
        solar thermal, geothermal, biogas steam, electric boiler, and
        heat pump. These sources have zero direct fossil combustion
        emissions, though some may have grid-dependent upstream
        emissions (electric boiler, heat pump).

        Args:
            source: Heat source identifier. Case-insensitive.

        Returns:
            True if the heat source has zero direct emissions.

        Raises:
            ValueError: If source is not recognised.
        """
        key = source.lower().strip()

        # Validate the source exists
        with self._state_lock:
            if key in self._custom_heat_source_factors:
                # For custom sources, check if EF is zero
                custom = self._custom_heat_source_factors[key]
                return custom.ef_kgco2e_per_gj == Decimal("0")

        if key not in HEAT_SOURCE_FACTORS:
            raise ValueError(
                f"Unknown heat source: {source!r}. "
                f"Valid heat sources: "
                f"{sorted(HEAT_SOURCE_FACTORS.keys())}"
            )

        return key in _ZERO_EMISSION_HEAT_SOURCES

    # ==================================================================
    # PUBLIC METHODS: Refrigerant Data (Methods 18-20)
    # ==================================================================

    def get_refrigerant_data(
        self,
        refrigerant: str,
    ) -> RefrigerantData:
        """Get the full GWP and phase-down data for a refrigerant.

        Returns the RefrigerantData model containing GWP values from
        AR5 and AR6, common usage context, and regulatory phase-down
        status for the specified refrigerant. Checks custom overrides
        first, then built-in data.

        Args:
            refrigerant: Refrigerant identifier (e.g. 'r_134a',
                'r_410a', 'r_717'). Must match a Refrigerant enum
                value or a custom override key. Case-insensitive.

        Returns:
            RefrigerantData with GWP and phase-down information.

        Raises:
            ValueError: If refrigerant is not recognised.
        """
        with self._state_lock:
            self._lookup_count += 1

        key = refrigerant.lower().strip()

        # Check custom overrides first
        with self._state_lock:
            if key in self._custom_refrigerants:
                data = self._custom_refrigerants[key]
                self._record_provenance(
                    "get_refrigerant_data",
                    {"refrigerant": key, "source": "custom"},
                )
                self._record_metric("refrigerant", "hit")
                logger.debug(
                    "Refrigerant data lookup (custom): ref=%s, "
                    "gwp_ar5=%s, gwp_ar6=%s",
                    key,
                    data.gwp_ar5,
                    data.gwp_ar6,
                )
                return data

        # Built-in data
        if key not in REFRIGERANT_GWP:
            self._record_metric("refrigerant", "miss")
            raise ValueError(
                f"Unknown refrigerant: {refrigerant!r}. "
                f"Valid refrigerants: "
                f"{sorted(REFRIGERANT_GWP.keys())}"
            )

        data = REFRIGERANT_GWP[key]
        self._record_provenance(
            "get_refrigerant_data",
            {
                "refrigerant": key,
                "gwp_ar5": str(data.gwp_ar5),
                "gwp_ar6": str(data.gwp_ar6),
                "common_use": data.common_use,
                "source": "builtin",
            },
        )
        self._record_metric("refrigerant", "hit")

        logger.debug(
            "Refrigerant data lookup: ref=%s, gwp_ar5=%s, gwp_ar6=%s",
            key,
            data.gwp_ar5,
            data.gwp_ar6,
        )
        return data

    def get_all_refrigerants(
        self,
    ) -> Dict[str, RefrigerantData]:
        """Get GWP and phase-down data for all available refrigerants.

        Returns a dictionary keyed by refrigerant identifier with each
        value being the full RefrigerantData model. Custom overrides
        take precedence over built-in data.

        Returns:
            Dictionary mapping refrigerant to RefrigerantData instances.
        """
        with self._state_lock:
            self._lookup_count += 1

        results: Dict[str, RefrigerantData] = {}

        # Built-in data
        for ref_key in sorted(REFRIGERANT_GWP.keys()):
            results[ref_key] = REFRIGERANT_GWP[ref_key]

        # Apply custom overrides
        with self._state_lock:
            for ref_key, data in self._custom_refrigerants.items():
                results[ref_key] = data

        self._record_provenance(
            "get_all_refrigerants",
            {"count": len(results)},
        )
        self._record_metric("refrigerant", "hit")

        logger.debug(
            "All refrigerant data retrieved: count=%d",
            len(results),
        )
        return results

    def get_refrigerant_gwp(
        self,
        refrigerant: str,
        gwp_source: str,
    ) -> Decimal:
        """Get the GWP value for a refrigerant from a specific IPCC source.

        Convenience method returning only the numeric GWP value for
        the specified refrigerant and IPCC assessment report. Supports
        AR5 and AR6 for refrigerant-specific GWP lookups.

        Args:
            refrigerant: Refrigerant identifier. Case-insensitive.
            gwp_source: IPCC source ('AR5' or 'AR6'). Case-insensitive.

        Returns:
            GWP value as Decimal (dimensionless multiplier).

        Raises:
            ValueError: If refrigerant is not recognised or gwp_source
                is not 'AR5' or 'AR6'.
        """
        source_key = gwp_source.upper().strip()
        if source_key not in ("AR5", "AR6"):
            raise ValueError(
                f"Refrigerant GWP source must be 'AR5' or 'AR6', "
                f"got {gwp_source!r}"
            )

        data = self.get_refrigerant_data(refrigerant)

        if source_key == "AR5":
            return data.gwp_ar5
        else:
            return data.gwp_ar6

    # ==================================================================
    # PUBLIC METHODS: Efficiency Conversions (Methods 21-23)
    # ==================================================================

    def convert_efficiency(
        self,
        value: Decimal,
        from_metric: str,
        to_metric: str,
    ) -> Decimal:
        """Convert between cooling efficiency metrics.

        Supports conversions between COP, EER, kW/ton, and SEER.
        All conversions use deterministic Decimal arithmetic.

        Supported conversions:
        - COP to EER: COP * 3.412
        - EER to COP: EER / 3.412
        - COP to kW/ton: 3.517 / COP
        - kW/ton to COP: 3.517 / kW_per_ton
        - SEER to COP: SEER / 3.412
        - COP to SEER: COP * 3.412 (approximation)
        - EER to kW/ton: 3.517 / (EER / 3.412) = 12.0 / EER
        - kW/ton to EER: 12.0 / kW_per_ton

        Args:
            value: Efficiency value to convert. Must be > 0.
            from_metric: Source efficiency metric ('cop', 'eer',
                'kw_per_ton', 'seer'). Case-insensitive.
            to_metric: Target efficiency metric ('cop', 'eer',
                'kw_per_ton', 'seer'). Case-insensitive.

        Returns:
            Converted efficiency value as Decimal, rounded to 4
            decimal places with ROUND_HALF_UP.

        Raises:
            ValueError: If value is not positive, or if from_metric
                or to_metric is not recognised.
        """
        if value <= Decimal("0"):
            raise ValueError(
                f"Efficiency value must be > 0, got {value}"
            )

        with self._state_lock:
            self._conversion_count += 1

        from_key = from_metric.lower().strip()
        to_key = to_metric.lower().strip()

        valid_metrics = {"cop", "eer", "kw_per_ton", "seer"}
        if from_key not in valid_metrics:
            raise ValueError(
                f"Unknown efficiency metric: {from_metric!r}. "
                f"Valid metrics: {sorted(valid_metrics)}"
            )
        if to_key not in valid_metrics:
            raise ValueError(
                f"Unknown efficiency metric: {to_metric!r}. "
                f"Valid metrics: {sorted(valid_metrics)}"
            )

        # Same metric: no conversion needed
        if from_key == to_key:
            return value

        # Step 1: Convert to COP as intermediate
        cop_value = self._to_cop(value, from_key)

        # Step 2: Convert from COP to target metric
        result = self._from_cop(cop_value, to_key)

        # Round to 4 decimal places
        result = result.quantize(Decimal("0.0001"), ROUND_HALF_UP)

        self._record_provenance(
            "convert_efficiency",
            {
                "value": str(value),
                "from_metric": from_key,
                "to_metric": to_key,
                "result": str(result),
            },
        )
        self._record_metric("efficiency", "hit")

        logger.debug(
            "Efficiency conversion: %s %s -> %s %s",
            value,
            from_key,
            result,
            to_key,
        )
        return result

    def _to_cop(
        self,
        value: Decimal,
        metric: str,
    ) -> Decimal:
        """Convert a value from the specified efficiency metric to COP.

        Args:
            value: Efficiency value (must be > 0).
            metric: Source metric (lowercase).

        Returns:
            Value expressed as COP.
        """
        if metric == "cop":
            return value
        elif metric == "eer":
            # EER / 3.412 = COP
            return value * EFFICIENCY_CONVERSIONS["eer_to_cop"]
        elif metric == "kw_per_ton":
            # 3.517 / kW_per_ton = COP
            return EFFICIENCY_CONVERSIONS["cop_to_kw_per_ton"] / value
        elif metric == "seer":
            # SEER / 3.412 = COP (approximate)
            return value * EFFICIENCY_CONVERSIONS["seer_to_cop"]
        else:
            raise ValueError(
                f"Cannot convert metric {metric!r} to COP"
            )

    def _from_cop(
        self,
        cop_value: Decimal,
        metric: str,
    ) -> Decimal:
        """Convert a COP value to the specified target efficiency metric.

        Args:
            cop_value: COP value.
            metric: Target metric (lowercase).

        Returns:
            Value in target metric.
        """
        if metric == "cop":
            return cop_value
        elif metric == "eer":
            # COP * 3.412 = EER
            return cop_value * EFFICIENCY_CONVERSIONS["cop_to_eer"]
        elif metric == "kw_per_ton":
            # 3.517 / COP = kW/ton
            if cop_value == Decimal("0"):
                raise ValueError("Cannot convert COP=0 to kW/ton")
            return EFFICIENCY_CONVERSIONS["cop_to_kw_per_ton"] / cop_value
        elif metric == "seer":
            # COP * 3.412 = SEER (approximate)
            return cop_value * EFFICIENCY_CONVERSIONS["cop_to_eer"]
        else:
            raise ValueError(
                f"Cannot convert COP to metric {metric!r}"
            )

    def calculate_iplv(
        self,
        cop_100: Decimal,
        cop_75: Decimal,
        cop_50: Decimal,
        cop_25: Decimal,
    ) -> Decimal:
        """Calculate the Integrated Part-Load Value from four COP points.

        Computes IPLV per AHRI Standard 550/590 using the standard
        weighting factors:
            IPLV = 0.01 * COP_100% + 0.42 * COP_75%
                 + 0.45 * COP_50% + 0.12 * COP_25%

        All arithmetic uses Decimal for deterministic results.

        Args:
            cop_100: COP at 100% load. Must be > 0.
            cop_75: COP at 75% load. Must be > 0.
            cop_50: COP at 50% load. Must be > 0.
            cop_25: COP at 25% load. Must be > 0.

        Returns:
            IPLV as Decimal, rounded to 3 decimal places with
            ROUND_HALF_UP.

        Raises:
            ValueError: If any COP value is not positive.
        """
        for name, val in [
            ("cop_100", cop_100),
            ("cop_75", cop_75),
            ("cop_50", cop_50),
            ("cop_25", cop_25),
        ]:
            if val <= Decimal("0"):
                raise ValueError(
                    f"{name} must be > 0, got {val}"
                )

        with self._state_lock:
            self._conversion_count += 1

        w100 = AHRI_PART_LOAD_WEIGHTS["100%"]
        w75 = AHRI_PART_LOAD_WEIGHTS["75%"]
        w50 = AHRI_PART_LOAD_WEIGHTS["50%"]
        w25 = AHRI_PART_LOAD_WEIGHTS["25%"]

        iplv = (
            w100 * cop_100
            + w75 * cop_75
            + w50 * cop_50
            + w25 * cop_25
        )

        iplv = iplv.quantize(Decimal("0.001"), ROUND_HALF_UP)

        self._record_provenance(
            "calculate_iplv",
            {
                "cop_100": str(cop_100),
                "cop_75": str(cop_75),
                "cop_50": str(cop_50),
                "cop_25": str(cop_25),
                "weights": {
                    "100%": str(w100),
                    "75%": str(w75),
                    "50%": str(w50),
                    "25%": str(w25),
                },
                "iplv": str(iplv),
            },
        )
        self._record_metric("efficiency", "hit")

        logger.debug(
            "IPLV calculated: 100%%=%s, 75%%=%s, 50%%=%s, 25%%=%s "
            "-> IPLV=%s",
            cop_100,
            cop_75,
            cop_50,
            cop_25,
            iplv,
        )
        return iplv

    def get_part_load_weights(self) -> Dict[str, Decimal]:
        """Get the AHRI 550/590 part-load weighting factors.

        Returns the four standard AHRI weighting factors used for
        IPLV/NPLV calculation:
        - 100% load: 0.01
        - 75% load: 0.42
        - 50% load: 0.45
        - 25% load: 0.12

        Returns:
            Dictionary mapping load percentage string to Decimal weight.
        """
        with self._state_lock:
            self._lookup_count += 1

        self._record_provenance(
            "get_part_load_weights",
            {"weights": {k: str(v) for k, v in AHRI_PART_LOAD_WEIGHTS.items()}},
        )
        self._record_metric("efficiency", "hit")

        return dict(AHRI_PART_LOAD_WEIGHTS)

    # ==================================================================
    # PUBLIC METHODS: Cooling Unit Conversions (Method 24)
    # ==================================================================

    def convert_cooling_units(
        self,
        value: Decimal,
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """Convert a cooling energy quantity between units.

        All conversions route through kWh_th as the intermediate unit
        to ensure consistency and traceability. Uses deterministic
        Decimal arithmetic for zero-hallucination guarantees.

        Supported units: ton_hour, kwh_th, gj, btu, mmbtu, mj, tr.

        Note: 'tr' is a rate (kW), not energy. Converting TR requires
        context of operating hours. When from_unit or to_unit is 'tr',
        the conversion treats 1 TR-hour = 3.517 kWh_th (same as
        ton_hour).

        Args:
            value: Cooling energy quantity to convert. Must be >= 0.
            from_unit: Source unit identifier. Case-insensitive.
            to_unit: Target unit identifier. Case-insensitive.

        Returns:
            Converted quantity as Decimal, rounded to 6 decimal
            places with ROUND_HALF_UP.

        Raises:
            ValueError: If from_unit or to_unit is not recognised,
                or if value is negative.
        """
        if value < Decimal("0"):
            raise ValueError(
                f"Cooling energy value must be >= 0, got {value}"
            )

        with self._state_lock:
            self._conversion_count += 1

        from_key = from_unit.lower().strip()
        to_key = to_unit.lower().strip()

        valid_units = {"ton_hour", "kwh_th", "gj", "btu", "mmbtu", "mj", "tr"}
        if from_key not in valid_units:
            raise ValueError(
                f"Unknown cooling unit: {from_unit!r}. "
                f"Valid units: {sorted(valid_units)}"
            )
        if to_key not in valid_units:
            raise ValueError(
                f"Unknown cooling unit: {to_unit!r}. "
                f"Valid units: {sorted(valid_units)}"
            )

        # Same unit: no conversion needed
        if from_key == to_key:
            return value

        # Step 1: Convert from source unit to kWh_th
        kwh_th_value = self._to_kwh_th(value, from_key)

        # Step 2: Convert from kWh_th to target unit
        result = self._from_kwh_th(kwh_th_value, to_key)

        # Round to 6 decimal places
        result = result.quantize(Decimal("0.000001"), ROUND_HALF_UP)

        self._record_provenance(
            "convert_cooling_units",
            {
                "value": str(value),
                "from_unit": from_key,
                "to_unit": to_key,
                "result": str(result),
            },
        )
        self._record_metric("unit_conversion", "hit")

        logger.debug(
            "Cooling unit conversion: %s %s -> %s %s",
            value,
            from_key,
            result,
            to_key,
        )
        return result

    def _to_kwh_th(
        self,
        value: Decimal,
        unit: str,
    ) -> Decimal:
        """Convert a value from the specified unit to kWh_th.

        Args:
            value: Cooling energy quantity.
            unit: Source unit (lowercase).

        Returns:
            Value in kWh_th.
        """
        if unit == "kwh_th":
            return value
        elif unit == "ton_hour":
            return value * UNIT_CONVERSIONS["ton_hour_to_kwh_th"]
        elif unit == "gj":
            return value * UNIT_CONVERSIONS["gj_to_kwh_th"]
        elif unit == "mmbtu":
            return value * UNIT_CONVERSIONS["mmbtu_to_kwh_th"]
        elif unit == "mj":
            return value * UNIT_CONVERSIONS["mj_to_kwh_th"]
        elif unit == "btu":
            return value * UNIT_CONVERSIONS["btu_to_kwh_th"]
        elif unit == "tr":
            # TR is a rate unit (1 TR = 3.517 kW).
            # Treating 1 TR-hour = 3.517 kWh_th (same as ton_hour).
            return value * UNIT_CONVERSIONS["tr_to_kw"]
        else:
            raise ValueError(f"Cannot convert unit {unit!r} to kWh_th")

    def _from_kwh_th(
        self,
        kwh_th_value: Decimal,
        unit: str,
    ) -> Decimal:
        """Convert a kWh_th value to the specified target unit.

        Args:
            kwh_th_value: Energy quantity in kWh_th.
            unit: Target unit (lowercase).

        Returns:
            Value in target unit.
        """
        if unit == "kwh_th":
            return kwh_th_value
        elif unit == "ton_hour":
            return kwh_th_value / UNIT_CONVERSIONS["ton_hour_to_kwh_th"]
        elif unit == "gj":
            return kwh_th_value / UNIT_CONVERSIONS["gj_to_kwh_th"]
        elif unit == "mmbtu":
            return kwh_th_value / UNIT_CONVERSIONS["mmbtu_to_kwh_th"]
        elif unit == "mj":
            return kwh_th_value / UNIT_CONVERSIONS["mj_to_kwh_th"]
        elif unit == "btu":
            return kwh_th_value / UNIT_CONVERSIONS["btu_to_kwh_th"]
        elif unit == "tr":
            # Inverse of tr_to_kw for rate conversion
            return kwh_th_value / UNIT_CONVERSIONS["tr_to_kw"]
        else:
            raise ValueError(f"Cannot convert kWh_th to unit {unit!r}")

    # ==================================================================
    # PUBLIC METHODS: GWP Values (Methods 25-26)
    # ==================================================================

    def get_gwp(
        self,
        gas: str,
        source: str,
    ) -> Decimal:
        """Get the GWP multiplier for a specific gas and IPCC source.

        Returns the 100-year (or 20-year for AR6_20YR) Global Warming
        Potential multiplier for the specified greenhouse gas from the
        specified IPCC Assessment Report.

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

        self._record_provenance(
            "get_gwp",
            {
                "gas": gas_key,
                "source": source_key,
                "gwp": str(data[gas_key]),
            },
        )

        return data[gas_key]

    def get_gwp_values(
        self,
        source: str,
    ) -> Dict[str, Any]:
        """Get all GWP values for an IPCC assessment report.

        Returns the CO2, CH4, and N2O GWP multipliers from the
        specified IPCC assessment report edition.

        Args:
            source: IPCC source identifier ('AR4', 'AR5', 'AR6',
                'AR6_20YR'). Case-insensitive.

        Returns:
            Dictionary with source, co2_gwp, ch4_gwp, n2o_gwp,
            and provenance_hash.

        Raises:
            ValueError: If source is not a recognised GWP source.
        """
        with self._state_lock:
            self._lookup_count += 1

        key = source.upper().strip()

        if key not in GWP_VALUES:
            raise ValueError(
                f"Unknown GWP source: {source!r}. "
                f"Valid sources: {sorted(GWP_VALUES.keys())}"
            )

        data = GWP_VALUES[key]

        result = {
            "source": key,
            "co2_gwp": data["CO2"],
            "ch4_gwp": data["CH4"],
            "n2o_gwp": data["N2O"],
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
        self._record_metric("gwp", "hit")

        logger.debug(
            "GWP values lookup: source=%s, CH4=%s, N2O=%s",
            key,
            data["CH4"],
            data["N2O"],
        )
        return result

    def get_all_gwp_sources(self) -> List[str]:
        """Get a sorted list of all available GWP source identifiers.

        Returns:
            Sorted list of GWP source strings (AR4, AR5, AR6, AR6_20YR).
        """
        return sorted(GWP_VALUES.keys())

    # ==================================================================
    # PUBLIC METHODS: Custom Factor Management (Methods 27-32)
    # ==================================================================

    def add_custom_technology(
        self,
        key: str,
        spec: CoolingTechnologySpec,
    ) -> None:
        """Register a custom cooling technology specification.

        Adds or overrides a cooling technology in the database. The
        custom specification takes precedence over any built-in data
        with the same key for all subsequent lookups.

        Args:
            key: Technology identifier string. Will be lowercased
                and stripped. Must be non-empty.
            spec: Complete CoolingTechnologySpec with COP range,
                default COP, IPLV, energy source, and optional
                compressor/condenser types.

        Raises:
            ValueError: If key is empty after stripping.
        """
        normalized_key = key.lower().strip()
        if not normalized_key:
            raise ValueError("Technology key must not be empty")

        with self._state_lock:
            self._custom_technologies[normalized_key] = spec
            self._mutation_count += 1

        self._record_provenance(
            "add_custom_technology",
            {
                "key": normalized_key,
                "cop_default": str(spec.cop_default),
                "cop_min": str(spec.cop_min),
                "cop_max": str(spec.cop_max),
                "energy_source": spec.energy_source,
            },
        )

        logger.info(
            "Added custom technology: key=%s, COP=%s, source=%s",
            normalized_key,
            spec.cop_default,
            spec.energy_source,
        )

    def add_custom_district_factor(
        self,
        key: str,
        factor: DistrictCoolingFactor,
    ) -> None:
        """Register a custom district cooling emission factor.

        Adds or overrides a district cooling region factor. The custom
        factor takes precedence over any built-in data with the same
        key for all subsequent lookups.

        Args:
            key: Region identifier string. Will be lowercased and
                stripped. Must be non-empty.
            factor: Complete DistrictCoolingFactor with emission factor,
                technology mix, and notes.

        Raises:
            ValueError: If key is empty after stripping.
        """
        normalized_key = key.lower().strip()
        if not normalized_key:
            raise ValueError("District cooling region key must not be empty")

        with self._state_lock:
            self._custom_district_factors[normalized_key] = factor
            self._mutation_count += 1

        self._record_provenance(
            "add_custom_district_factor",
            {
                "key": normalized_key,
                "ef_kgco2e_per_gj": str(factor.ef_kgco2e_per_gj),
                "technology_mix": factor.technology_mix,
            },
        )

        logger.info(
            "Added custom district cooling factor: key=%s, "
            "ef=%s kgCO2e/GJ",
            normalized_key,
            factor.ef_kgco2e_per_gj,
        )

    def add_custom_heat_source_factor(
        self,
        source: str,
        factor: HeatSourceFactor,
    ) -> None:
        """Register a custom heat source emission factor.

        Adds or overrides a heat source factor for absorption chiller
        calculations. The custom factor takes precedence over any
        built-in data with the same key for all subsequent lookups.

        Args:
            source: Heat source identifier string. Will be lowercased
                and stripped. Must be non-empty.
            factor: Complete HeatSourceFactor with emission factor
                and notes.

        Raises:
            ValueError: If source is empty after stripping.
        """
        normalized_key = source.lower().strip()
        if not normalized_key:
            raise ValueError("Heat source key must not be empty")

        with self._state_lock:
            self._custom_heat_source_factors[normalized_key] = factor
            self._mutation_count += 1

        self._record_provenance(
            "add_custom_heat_source_factor",
            {
                "source": normalized_key,
                "ef_kgco2e_per_gj": str(factor.ef_kgco2e_per_gj),
            },
        )

        logger.info(
            "Added custom heat source factor: source=%s, "
            "ef=%s kgCO2e/GJ",
            normalized_key,
            factor.ef_kgco2e_per_gj,
        )

    def add_custom_refrigerant(
        self,
        key: str,
        data: RefrigerantData,
    ) -> None:
        """Register a custom refrigerant GWP record.

        Adds or overrides a refrigerant in the database. The custom
        data takes precedence over any built-in data with the same
        key for all subsequent lookups.

        Args:
            key: Refrigerant identifier string. Will be lowercased
                and stripped. Must be non-empty.
            data: Complete RefrigerantData with GWP values, common
                use, and phase-down status.

        Raises:
            ValueError: If key is empty after stripping.
        """
        normalized_key = key.lower().strip()
        if not normalized_key:
            raise ValueError("Refrigerant key must not be empty")

        with self._state_lock:
            self._custom_refrigerants[normalized_key] = data
            self._mutation_count += 1

        self._record_provenance(
            "add_custom_refrigerant",
            {
                "key": normalized_key,
                "gwp_ar5": str(data.gwp_ar5),
                "gwp_ar6": str(data.gwp_ar6),
                "common_use": data.common_use,
            },
        )

        logger.info(
            "Added custom refrigerant: key=%s, gwp_ar5=%s, gwp_ar6=%s",
            normalized_key,
            data.gwp_ar5,
            data.gwp_ar6,
        )

    def remove_custom_technology(
        self,
        key: str,
    ) -> bool:
        """Remove a custom cooling technology override.

        After removal, lookups for this technology will revert to the
        built-in data (if it exists as a built-in technology).

        Args:
            key: Technology identifier to remove.

        Returns:
            True if the custom technology was removed, False if it
            did not exist in the custom overrides.
        """
        normalized_key = key.lower().strip()
        with self._state_lock:
            if normalized_key in self._custom_technologies:
                del self._custom_technologies[normalized_key]
                self._mutation_count += 1
                self._record_provenance(
                    "remove_custom_technology",
                    {"key": normalized_key, "removed": True},
                )
                logger.info(
                    "Removed custom technology: key=%s",
                    normalized_key,
                )
                return True

        self._record_provenance(
            "remove_custom_technology",
            {"key": normalized_key, "removed": False},
        )
        return False

    def remove_custom_district_factor(
        self,
        key: str,
    ) -> bool:
        """Remove a custom district cooling factor override.

        After removal, lookups for this region will revert to the
        built-in data or global_default fallback.

        Args:
            key: Region identifier to remove.

        Returns:
            True if removed, False if it did not exist.
        """
        normalized_key = key.lower().strip()
        with self._state_lock:
            if normalized_key in self._custom_district_factors:
                del self._custom_district_factors[normalized_key]
                self._mutation_count += 1
                self._record_provenance(
                    "remove_custom_district_factor",
                    {"key": normalized_key, "removed": True},
                )
                logger.info(
                    "Removed custom district cooling factor: key=%s",
                    normalized_key,
                )
                return True

        self._record_provenance(
            "remove_custom_district_factor",
            {"key": normalized_key, "removed": False},
        )
        return False

    def remove_custom_heat_source_factor(
        self,
        source: str,
    ) -> bool:
        """Remove a custom heat source factor override.

        After removal, lookups for this heat source will revert to
        the built-in data.

        Args:
            source: Heat source identifier to remove.

        Returns:
            True if removed, False if it did not exist.
        """
        normalized_key = source.lower().strip()
        with self._state_lock:
            if normalized_key in self._custom_heat_source_factors:
                del self._custom_heat_source_factors[normalized_key]
                self._mutation_count += 1
                self._record_provenance(
                    "remove_custom_heat_source_factor",
                    {"source": normalized_key, "removed": True},
                )
                logger.info(
                    "Removed custom heat source factor: source=%s",
                    normalized_key,
                )
                return True

        self._record_provenance(
            "remove_custom_heat_source_factor",
            {"source": normalized_key, "removed": False},
        )
        return False

    def remove_custom_refrigerant(
        self,
        key: str,
    ) -> bool:
        """Remove a custom refrigerant override.

        After removal, lookups for this refrigerant will revert to
        the built-in data.

        Args:
            key: Refrigerant identifier to remove.

        Returns:
            True if removed, False if it did not exist.
        """
        normalized_key = key.lower().strip()
        with self._state_lock:
            if normalized_key in self._custom_refrigerants:
                del self._custom_refrigerants[normalized_key]
                self._mutation_count += 1
                self._record_provenance(
                    "remove_custom_refrigerant",
                    {"key": normalized_key, "removed": True},
                )
                logger.info(
                    "Removed custom refrigerant: key=%s",
                    normalized_key,
                )
                return True

        self._record_provenance(
            "remove_custom_refrigerant",
            {"key": normalized_key, "removed": False},
        )
        return False

    def reset_custom_factors(self) -> None:
        """Clear all custom factor overrides.

        Removes all custom technology specs, district cooling factors,
        heat source factors, and refrigerant data. Built-in data is
        not affected. Subsequent lookups will use only built-in data.

        Primarily used for testing and development.
        """
        with self._state_lock:
            custom_count = (
                len(self._custom_technologies)
                + len(self._custom_district_factors)
                + len(self._custom_heat_source_factors)
                + len(self._custom_refrigerants)
            )
            self._custom_technologies.clear()
            self._custom_district_factors.clear()
            self._custom_heat_source_factors.clear()
            self._custom_refrigerants.clear()
            self._mutation_count += 1

        self._record_provenance(
            "reset_custom_factors",
            {"cleared_count": custom_count},
        )

        logger.info(
            "Custom factors reset: cleared %d custom entries",
            custom_count,
        )

    # ==================================================================
    # PUBLIC METHODS: Search (Method 33)
    # ==================================================================

    def search_factors(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Search across all factor tables by text query.

        Performs case-insensitive substring matching against technology
        identifiers, district cooling regions, heat source names,
        refrigerant identifiers, GWP sources, and unit conversion names.

        Args:
            query: Search string. Case-insensitive.

        Returns:
            List of matching factor entries, each with type, key,
            summary data, and provenance_hash.
        """
        with self._state_lock:
            self._lookup_count += 1

        q = query.lower().strip()
        results: List[Dict[str, Any]] = []

        # Search cooling technology specs
        for tech_key, spec in COOLING_TECHNOLOGY_SPECS.items():
            if q in tech_key:
                results.append({
                    "type": "cooling_technology",
                    "key": tech_key,
                    "cop_default": spec.cop_default,
                    "cop_min": spec.cop_min,
                    "cop_max": spec.cop_max,
                    "energy_source": spec.energy_source,
                    "source": "builtin",
                })

        # Search custom technologies
        with self._state_lock:
            for tech_key, spec in self._custom_technologies.items():
                if q in tech_key:
                    results.append({
                        "type": "cooling_technology",
                        "key": tech_key,
                        "cop_default": spec.cop_default,
                        "cop_min": spec.cop_min,
                        "cop_max": spec.cop_max,
                        "energy_source": spec.energy_source,
                        "source": "custom",
                    })

        # Search district cooling factors
        for region_key, factor in DISTRICT_COOLING_FACTORS.items():
            if q in region_key:
                results.append({
                    "type": "district_cooling_factor",
                    "key": region_key,
                    "ef_kgco2e_per_gj": factor.ef_kgco2e_per_gj,
                    "technology_mix": factor.technology_mix,
                    "source": "builtin",
                })

        # Search custom district cooling factors
        with self._state_lock:
            for region_key, factor in self._custom_district_factors.items():
                if q in region_key:
                    results.append({
                        "type": "district_cooling_factor",
                        "key": region_key,
                        "ef_kgco2e_per_gj": factor.ef_kgco2e_per_gj,
                        "technology_mix": factor.technology_mix,
                        "source": "custom",
                    })

        # Search heat source factors
        for source_key, factor in HEAT_SOURCE_FACTORS.items():
            if q in source_key:
                results.append({
                    "type": "heat_source_factor",
                    "key": source_key,
                    "ef_kgco2e_per_gj": factor.ef_kgco2e_per_gj,
                    "source": "builtin",
                })

        # Search custom heat source factors
        with self._state_lock:
            for source_key, factor in self._custom_heat_source_factors.items():
                if q in source_key:
                    results.append({
                        "type": "heat_source_factor",
                        "key": source_key,
                        "ef_kgco2e_per_gj": factor.ef_kgco2e_per_gj,
                        "source": "custom",
                    })

        # Search refrigerant GWP data
        for ref_key, ref_data in REFRIGERANT_GWP.items():
            if q in ref_key:
                results.append({
                    "type": "refrigerant_gwp",
                    "key": ref_key,
                    "gwp_ar5": ref_data.gwp_ar5,
                    "gwp_ar6": ref_data.gwp_ar6,
                    "common_use": ref_data.common_use,
                    "source": "builtin",
                })

        # Search custom refrigerants
        with self._state_lock:
            for ref_key, ref_data in self._custom_refrigerants.items():
                if q in ref_key:
                    results.append({
                        "type": "refrigerant_gwp",
                        "key": ref_key,
                        "gwp_ar5": ref_data.gwp_ar5,
                        "gwp_ar6": ref_data.gwp_ar6,
                        "common_use": ref_data.common_use,
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
        for conv_key, conv_factor in UNIT_CONVERSIONS.items():
            if q in conv_key:
                results.append({
                    "type": "unit_conversion",
                    "key": conv_key,
                    "factor": conv_factor,
                    "source": "builtin",
                })

        # Search efficiency conversions
        for eff_key, eff_factor in EFFICIENCY_CONVERSIONS.items():
            if q in eff_key:
                results.append({
                    "type": "efficiency_conversion",
                    "key": eff_key,
                    "factor": eff_factor,
                    "source": "builtin",
                })

        # Search AHRI part-load weights
        for weight_key, weight_val in AHRI_PART_LOAD_WEIGHTS.items():
            if q in weight_key.lower():
                results.append({
                    "type": "ahri_part_load_weight",
                    "key": weight_key,
                    "weight": weight_val,
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
    # PUBLIC METHODS: Database Statistics (Method 34)
    # ==================================================================

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics including counts and operational metrics.

        Returns:
            Dictionary with technology_count, dc_region_count,
            heat_source_count, refrigerant_count, gwp_source_count,
            unit_conversion_count, custom counts, operational counters,
            and provenance_hash.
        """
        with self._state_lock:
            stats = {
                "technologies": len(COOLING_TECHNOLOGY_SPECS),
                "custom_technologies": len(self._custom_technologies),
                "dc_regions": len(DISTRICT_COOLING_FACTORS),
                "custom_dc_regions": len(self._custom_district_factors),
                "heat_sources": len(HEAT_SOURCE_FACTORS),
                "custom_heat_sources": len(
                    self._custom_heat_source_factors,
                ),
                "refrigerants": len(REFRIGERANT_GWP),
                "custom_refrigerants": len(self._custom_refrigerants),
                "gwp_sources": len(GWP_VALUES),
                "unit_conversions": len(UNIT_CONVERSIONS),
                "efficiency_conversions": len(EFFICIENCY_CONVERSIONS),
                "ahri_part_load_weights": len(AHRI_PART_LOAD_WEIGHTS),
                "electric_technologies": len(_ELECTRIC_TECHNOLOGIES),
                "absorption_technologies": len(_ABSORPTION_TECHNOLOGIES),
                "free_cooling_technologies": len(_FREE_COOLING_TECHNOLOGIES),
                "tes_technologies": len(_TES_TECHNOLOGIES),
                "zero_emission_heat_sources": len(
                    _ZERO_EMISSION_HEAT_SOURCES,
                ),
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
            "Database stats: techs=%d, dc=%d, hs=%d, ref=%d, "
            "gwp=%d, conv=%d, lookups=%d",
            stats["technologies"],
            stats["dc_regions"],
            stats["heat_sources"],
            stats["refrigerants"],
            stats["gwp_sources"],
            stats["unit_conversions"],
            stats["total_lookups"],
        )
        return stats

    # ==================================================================
    # PUBLIC METHODS: Health Check (Method 35)
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the database engine.

        Verifies that all built-in data tables are accessible, have
        expected record counts, and that the engine is operational.
        Checks data integrity by verifying known reference values.

        Returns:
            Dictionary with status ('healthy' or 'unhealthy'),
            checks (list of individual check results), engine_id,
            engine_version, timestamp, and provenance_hash.
        """
        checks: List[Dict[str, Any]] = []
        all_healthy = True

        # Check 1: Cooling technology specs table
        tech_count = len(COOLING_TECHNOLOGY_SPECS)
        tech_ok = tech_count == 18
        checks.append({
            "check": "cooling_technology_specs",
            "expected_count": 18,
            "actual_count": tech_count,
            "status": "healthy" if tech_ok else "unhealthy",
        })
        if not tech_ok:
            all_healthy = False

        # Check 2: District cooling factors table
        dc_count = len(DISTRICT_COOLING_FACTORS)
        dc_ok = dc_count == 12
        checks.append({
            "check": "district_cooling_factors",
            "expected_count": 12,
            "actual_count": dc_count,
            "status": "healthy" if dc_ok else "unhealthy",
        })
        if not dc_ok:
            all_healthy = False

        # Check 3: Heat source factors table
        hs_count = len(HEAT_SOURCE_FACTORS)
        hs_ok = hs_count == 11
        checks.append({
            "check": "heat_source_factors",
            "expected_count": 11,
            "actual_count": hs_count,
            "status": "healthy" if hs_ok else "unhealthy",
        })
        if not hs_ok:
            all_healthy = False

        # Check 4: Refrigerant GWP table
        ref_count = len(REFRIGERANT_GWP)
        ref_ok = ref_count == 11
        checks.append({
            "check": "refrigerant_gwp",
            "expected_count": 11,
            "actual_count": ref_count,
            "status": "healthy" if ref_ok else "unhealthy",
        })
        if not ref_ok:
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

        # Check 6: AHRI part-load weights
        ahri_count = len(AHRI_PART_LOAD_WEIGHTS)
        ahri_ok = ahri_count == 4
        checks.append({
            "check": "ahri_part_load_weights",
            "expected_count": 4,
            "actual_count": ahri_count,
            "status": "healthy" if ahri_ok else "unhealthy",
        })
        if not ahri_ok:
            all_healthy = False

        # Check 7: Efficiency conversions
        eff_count = len(EFFICIENCY_CONVERSIONS)
        eff_ok = eff_count == 4
        checks.append({
            "check": "efficiency_conversions",
            "expected_count": 4,
            "actual_count": eff_count,
            "status": "healthy" if eff_ok else "unhealthy",
        })
        if not eff_ok:
            all_healthy = False

        # Check 8: Unit conversions
        unit_count = len(UNIT_CONVERSIONS)
        unit_ok = unit_count == 6
        checks.append({
            "check": "unit_conversions",
            "expected_count": 6,
            "actual_count": unit_count,
            "status": "healthy" if unit_ok else "unhealthy",
        })
        if not unit_ok:
            all_healthy = False

        # Check 9: Data integrity - water-cooled centrifugal COP
        wcc_ok = False
        try:
            wcc_key = CoolingTechnology.WATER_COOLED_CENTRIFUGAL.value
            wcc_spec = COOLING_TECHNOLOGY_SPECS.get(wcc_key)
            if wcc_spec is not None:
                wcc_ok = wcc_spec.cop_default == Decimal("6.1")
        except Exception:
            pass
        checks.append({
            "check": "data_integrity_wcc_cop",
            "expected_value": "6.1",
            "actual_value": str(
                wcc_spec.cop_default if wcc_spec else "N/A"
            ),
            "status": "healthy" if wcc_ok else "unhealthy",
        })
        if not wcc_ok:
            all_healthy = False

        # Check 10: Data integrity - R-134a GWP AR5
        r134a_ok = False
        try:
            r134a_key = Refrigerant.R_134A.value
            r134a_data = REFRIGERANT_GWP.get(r134a_key)
            if r134a_data is not None:
                r134a_ok = r134a_data.gwp_ar5 == Decimal("1430")
        except Exception:
            pass
        checks.append({
            "check": "data_integrity_r134a_gwp_ar5",
            "expected_value": "1430",
            "actual_value": str(
                r134a_data.gwp_ar5 if r134a_data else "N/A"
            ),
            "status": "healthy" if r134a_ok else "unhealthy",
        })
        if not r134a_ok:
            all_healthy = False

        # Check 11: Data integrity - AHRI weights sum to 1.0
        ahri_sum_ok = False
        try:
            ahri_sum = sum(AHRI_PART_LOAD_WEIGHTS.values())
            ahri_sum_ok = ahri_sum == Decimal("1.00")
        except Exception:
            pass
        checks.append({
            "check": "data_integrity_ahri_weights_sum",
            "expected_value": "1.00",
            "actual_value": str(ahri_sum) if ahri_sum_ok else "N/A",
            "status": "healthy" if ahri_sum_ok else "unhealthy",
        })
        if not ahri_sum_ok:
            all_healthy = False

        # Check 12: Data integrity - GWP AR6 CH4
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

        # Check 13: Data integrity - Dubai DC EF
        dubai_ok = False
        try:
            dubai_factor = DISTRICT_COOLING_FACTORS.get("dubai_uae")
            if dubai_factor is not None:
                dubai_ok = (
                    dubai_factor.ef_kgco2e_per_gj == Decimal("45.0")
                )
        except Exception:
            pass
        checks.append({
            "check": "data_integrity_dubai_dc_ef",
            "expected_value": "45.0",
            "actual_value": str(
                dubai_factor.ef_kgco2e_per_gj
                if dubai_factor else "N/A"
            ),
            "status": "healthy" if dubai_ok else "unhealthy",
        })
        if not dubai_ok:
            all_healthy = False

        # Check 14: Data integrity - natural gas steam heat source EF
        ng_hs_ok = False
        try:
            ng_key = HeatSource.NATURAL_GAS_STEAM.value
            ng_factor = HEAT_SOURCE_FACTORS.get(ng_key)
            if ng_factor is not None:
                ng_hs_ok = (
                    ng_factor.ef_kgco2e_per_gj == Decimal("70.1")
                )
        except Exception:
            pass
        checks.append({
            "check": "data_integrity_ng_heat_source_ef",
            "expected_value": "70.1",
            "actual_value": str(
                ng_factor.ef_kgco2e_per_gj if ng_factor else "N/A"
            ),
            "status": "healthy" if ng_hs_ok else "unhealthy",
        })
        if not ng_hs_ok:
            all_healthy = False

        # Check 15: Singleton is initialized
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
            "timestamp": utcnow().isoformat(),
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
    # PUBLIC METHODS: Engine Info (Method 36)
    # ==================================================================

    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine identification and version metadata.

        Returns:
            Dictionary with engine_id, version, description,
            data_sources, capabilities, built_in_data, and
            provenance_hash.
        """
        info = {
            "engine_id": self.ENGINE_ID,
            "version": self.ENGINE_VERSION,
            "description": (
                "Cooling Database Engine for Scope 2 purchased cooling "
                "emission calculations per GHG Protocol Scope 2 "
                "Guidance (2015). Provides cooling technology "
                "specifications, district cooling regional emission "
                "factors, heat source factors for absorption chillers, "
                "refrigerant GWP data, efficiency conversions, cooling "
                "unit conversions, and AHRI part-load weights."
            ),
            "data_sources": [
                "ASHRAE Handbook - HVAC Systems and Equipment (2024)",
                "AHRI Standard 550/590 (2023) - Water-chilling packages",
                "AHRI Standard 560 (2023) - Absorption water-chilling",
                "US DOE FEMP chiller benchmarks",
                "IEA District Cooling Report (2024)",
                "European District Energy Association (Euroheat & Power)",
                "IPCC 2006 Guidelines Vol 2, Chapter 2",
                "US EPA AP-42 Compilation of Air Emission Factors",
                "UK DESNZ Greenhouse Gas Reporting Conversion Factors",
                "IPCC AR4/AR5/AR6 GWP Tables",
                "EU F-Gas Regulation (2024/573)",
                "Kigali Amendment to Montreal Protocol",
                "ASHRAE Standard 34 - Refrigerant Safety Classification",
            ],
            "capabilities": [
                "technology_spec_lookup",
                "technology_classification",
                "district_cooling_factor_lookup",
                "heat_source_factor_lookup",
                "refrigerant_gwp_lookup",
                "efficiency_metric_conversion",
                "iplv_calculation",
                "cooling_unit_conversion",
                "gwp_values_lookup",
                "factor_search",
                "custom_factor_management",
                "health_check",
            ],
            "built_in_data": {
                "technologies": len(COOLING_TECHNOLOGY_SPECS),
                "dc_regions": len(DISTRICT_COOLING_FACTORS),
                "heat_sources": len(HEAT_SOURCE_FACTORS),
                "refrigerants": len(REFRIGERANT_GWP),
                "gwp_sources": len(GWP_VALUES),
                "unit_conversions": len(UNIT_CONVERSIONS),
                "efficiency_conversions": len(EFFICIENCY_CONVERSIONS),
                "ahri_weights": len(AHRI_PART_LOAD_WEIGHTS),
            },
        }

        provenance_hash = self._record_provenance(
            "get_engine_info",
            {"engine_id": self.ENGINE_ID, "version": self.ENGINE_VERSION},
        )
        info["provenance_hash"] = provenance_hash
        return info

    # ==================================================================
    # PUBLIC METHODS: Technology List Helpers (Methods 37-40)
    # ==================================================================

    def get_technology_list(self) -> List[str]:
        """Get a sorted list of all available technology identifiers.

        Returns technologies from both built-in data and any custom
        overrides, sorted alphabetically.

        Returns:
            Sorted list of technology identifier strings.
        """
        technologies = set(COOLING_TECHNOLOGY_SPECS.keys())
        with self._state_lock:
            technologies.update(self._custom_technologies.keys())
        return sorted(technologies)

    def get_electric_technologies(self) -> List[str]:
        """Get a sorted list of electric chiller technology identifiers.

        Returns only the six vapour-compression electric chiller
        technologies from the built-in classification set.

        Returns:
            Sorted list of electric chiller technology identifiers.
        """
        return sorted(_ELECTRIC_TECHNOLOGIES)

    def get_absorption_technologies(self) -> List[str]:
        """Get a sorted list of absorption chiller technology identifiers.

        Returns only the four absorption chiller technologies from the
        built-in classification set.

        Returns:
            Sorted list of absorption chiller technology identifiers.
        """
        return sorted(_ABSORPTION_TECHNOLOGIES)

    def get_free_cooling_technologies(self) -> List[str]:
        """Get a sorted list of free cooling technology identifiers.

        Returns only the four free cooling technologies from the
        built-in classification set.

        Returns:
            Sorted list of free cooling technology identifiers.
        """
        return sorted(_FREE_COOLING_TECHNOLOGIES)

    def get_tes_technologies(self) -> List[str]:
        """Get a sorted list of TES technology identifiers.

        Returns only the three thermal energy storage technologies
        from the built-in classification set.

        Returns:
            Sorted list of TES technology identifiers.
        """
        return sorted(_TES_TECHNOLOGIES)

    # ==================================================================
    # PUBLIC METHODS: Region and Source Lists (Methods 41-42)
    # ==================================================================

    def get_district_cooling_regions(self) -> List[str]:
        """Get a sorted list of available district cooling regions.

        Returns regions from both built-in data and any custom
        overrides, sorted alphabetically.

        Returns:
            Sorted list of region identifier strings.
        """
        regions = set(DISTRICT_COOLING_FACTORS.keys())
        with self._state_lock:
            regions.update(self._custom_district_factors.keys())
        return sorted(regions)

    def get_heat_source_list(self) -> List[str]:
        """Get a sorted list of available heat source identifiers.

        Returns heat sources from both built-in data and any custom
        overrides, sorted alphabetically.

        Returns:
            Sorted list of heat source identifier strings.
        """
        sources = set(HEAT_SOURCE_FACTORS.keys())
        with self._state_lock:
            sources.update(self._custom_heat_source_factors.keys())
        return sorted(sources)

    def get_refrigerant_list(self) -> List[str]:
        """Get a sorted list of available refrigerant identifiers.

        Returns refrigerants from both built-in data and any custom
        overrides, sorted alphabetically.

        Returns:
            Sorted list of refrigerant identifier strings.
        """
        refrigerants = set(REFRIGERANT_GWP.keys())
        with self._state_lock:
            refrigerants.update(self._custom_refrigerants.keys())
        return sorted(refrigerants)

    # ==================================================================
    # PUBLIC METHODS: Conversion Factor Access (Methods 43-44)
    # ==================================================================

    def get_unit_conversion_factor(
        self,
        conversion: str,
    ) -> Decimal:
        """Get a raw cooling unit conversion factor by name.

        Args:
            conversion: Conversion factor name (e.g. 'ton_hour_to_kwh_th',
                'gj_to_kwh_th', 'tr_to_kw'). Case-insensitive.

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

        self._record_metric("unit_conversion", "hit")
        return UNIT_CONVERSIONS[key]

    def get_efficiency_conversion_factor(
        self,
        conversion: str,
    ) -> Decimal:
        """Get a raw efficiency conversion factor by name.

        Args:
            conversion: Conversion factor name (e.g. 'cop_to_eer',
                'eer_to_cop', 'cop_to_kw_per_ton', 'seer_to_cop').
                Case-insensitive.

        Returns:
            Conversion factor as Decimal.

        Raises:
            ValueError: If conversion name is not recognised.
        """
        key = conversion.lower().strip()
        if key not in EFFICIENCY_CONVERSIONS:
            raise ValueError(
                f"Unknown efficiency conversion: {conversion!r}. "
                f"Valid conversions: "
                f"{sorted(EFFICIENCY_CONVERSIONS.keys())}"
            )

        self._record_metric("efficiency", "hit")
        return EFFICIENCY_CONVERSIONS[key]

    # ==================================================================
    # PUBLIC METHODS: Reset (Methods 45-46)
    # ==================================================================

    def reset(self) -> None:
        """Reset all mutable state to initial values.

        Clears all custom overrides, operational counters, and
        provenance hashes. Built-in data is not affected.

        Primarily used for testing and development.
        """
        with self._state_lock:
            self._custom_technologies.clear()
            self._custom_district_factors.clear()
            self._custom_heat_source_factors.clear()
            self._custom_refrigerants.clear()
            self._lookup_count = 0
            self._mutation_count = 0
            self._conversion_count = 0
            self._provenance_hashes.clear()

        logger.info(
            "CoolingDatabaseEngine reset: all mutable state cleared"
        )

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance for testing.

        After calling this method, the next instantiation will create
        a fresh instance. This is intended for test isolation only.
        """
        with cls._lock:
            cls._instance = None
        logger.info("CoolingDatabaseEngine singleton reset")

# ===========================================================================
# Module-level convenience functions
# ===========================================================================

_module_engine: Optional[CoolingDatabaseEngine] = None
_module_lock = threading.Lock()

def get_database(
    config: Any = None,
    metrics: Any = None,
    provenance: Any = None,
) -> CoolingDatabaseEngine:
    """Get or create the module-level database engine singleton.

    Args:
        config: Optional configuration object.
        metrics: Optional metrics recorder.
        provenance: Optional provenance tracker.

    Returns:
        The CoolingDatabaseEngine singleton instance.
    """
    global _module_engine
    if _module_engine is None:
        with _module_lock:
            if _module_engine is None:
                _module_engine = CoolingDatabaseEngine(
                    config=config,
                    metrics=metrics,
                    provenance=provenance,
                )
    return _module_engine

def get_technology_spec(technology: str) -> CoolingTechnologySpec:
    """Module-level convenience: get cooling technology specification.

    Args:
        technology: Technology identifier.

    Returns:
        CoolingTechnologySpec with full technology parameters.
    """
    return get_database().get_technology_spec(technology)

def get_all_technologies() -> Dict[str, CoolingTechnologySpec]:
    """Module-level convenience: get all cooling technology specifications.

    Returns:
        Dictionary mapping technology to CoolingTechnologySpec instances.
    """
    return get_database().get_all_technologies()

def get_default_cop(technology: str) -> Decimal:
    """Module-level convenience: get default COP for a technology.

    Args:
        technology: Technology identifier.

    Returns:
        Default COP as Decimal.
    """
    return get_database().get_default_cop(technology)

def get_cop_range(technology: str) -> Tuple[Decimal, Decimal]:
    """Module-level convenience: get COP range for a technology.

    Args:
        technology: Technology identifier.

    Returns:
        Tuple of (cop_min, cop_max).
    """
    return get_database().get_cop_range(technology)

def get_iplv(technology: str) -> Optional[Decimal]:
    """Module-level convenience: get IPLV for a technology.

    Args:
        technology: Technology identifier.

    Returns:
        IPLV as Decimal, or None if not applicable.
    """
    return get_database().get_iplv(technology)

def get_energy_source(technology: str) -> str:
    """Module-level convenience: get energy source for a technology.

    Args:
        technology: Technology identifier.

    Returns:
        Energy source string.
    """
    return get_database().get_energy_source(technology)

def is_electric_technology(technology: str) -> bool:
    """Module-level convenience: check if technology is electric chiller.

    Args:
        technology: Technology identifier.

    Returns:
        True if electric chiller technology.
    """
    return get_database().is_electric_technology(technology)

def is_absorption_technology(technology: str) -> bool:
    """Module-level convenience: check if technology is absorption chiller.

    Args:
        technology: Technology identifier.

    Returns:
        True if absorption chiller technology.
    """
    return get_database().is_absorption_technology(technology)

def is_free_cooling_technology(technology: str) -> bool:
    """Module-level convenience: check if technology is free cooling.

    Args:
        technology: Technology identifier.

    Returns:
        True if free cooling technology.
    """
    return get_database().is_free_cooling_technology(technology)

def is_tes_technology(technology: str) -> bool:
    """Module-level convenience: check if technology is TES.

    Args:
        technology: Technology identifier.

    Returns:
        True if TES technology.
    """
    return get_database().is_tes_technology(technology)

def get_district_cooling_factor(
    region: str,
) -> DistrictCoolingFactor:
    """Module-level convenience: get district cooling factor for a region.

    Args:
        region: Region identifier.

    Returns:
        DistrictCoolingFactor instance.
    """
    return get_database().get_district_cooling_factor(region)

def get_all_district_cooling_factors() -> Dict[str, DistrictCoolingFactor]:
    """Module-level convenience: get all district cooling factors.

    Returns:
        Dictionary mapping region to DistrictCoolingFactor instances.
    """
    return get_database().get_all_district_cooling_factors()

def get_district_ef(region: str) -> Decimal:
    """Module-level convenience: get district cooling emission factor.

    Args:
        region: Region identifier.

    Returns:
        Emission factor as Decimal in kgCO2e/GJ.
    """
    return get_database().get_district_ef(region)

def get_heat_source_factor(source: str) -> HeatSourceFactor:
    """Module-level convenience: get heat source factor.

    Args:
        source: Heat source identifier.

    Returns:
        HeatSourceFactor instance.
    """
    return get_database().get_heat_source_factor(source)

def get_all_heat_source_factors() -> Dict[str, HeatSourceFactor]:
    """Module-level convenience: get all heat source factors.

    Returns:
        Dictionary mapping heat source to HeatSourceFactor instances.
    """
    return get_database().get_all_heat_source_factors()

def get_heat_source_ef(source: str) -> Decimal:
    """Module-level convenience: get heat source emission factor value.

    Args:
        source: Heat source identifier.

    Returns:
        Emission factor as Decimal in kgCO2e/GJ.
    """
    return get_database().get_heat_source_ef(source)

def is_zero_emission_heat_source(source: str) -> bool:
    """Module-level convenience: check if heat source is zero-emission.

    Args:
        source: Heat source identifier.

    Returns:
        True if zero direct emissions.
    """
    return get_database().is_zero_emission_heat_source(source)

def get_refrigerant_data(refrigerant: str) -> RefrigerantData:
    """Module-level convenience: get refrigerant GWP data.

    Args:
        refrigerant: Refrigerant identifier.

    Returns:
        RefrigerantData instance.
    """
    return get_database().get_refrigerant_data(refrigerant)

def get_all_refrigerants() -> Dict[str, RefrigerantData]:
    """Module-level convenience: get all refrigerant data.

    Returns:
        Dictionary mapping refrigerant to RefrigerantData instances.
    """
    return get_database().get_all_refrigerants()

def get_refrigerant_gwp(refrigerant: str, gwp_source: str) -> Decimal:
    """Module-level convenience: get refrigerant GWP for a source.

    Args:
        refrigerant: Refrigerant identifier.
        gwp_source: IPCC source ('AR5' or 'AR6').

    Returns:
        GWP value as Decimal.
    """
    return get_database().get_refrigerant_gwp(refrigerant, gwp_source)

def convert_efficiency(
    value: Decimal,
    from_metric: str,
    to_metric: str,
) -> Decimal:
    """Module-level convenience: convert between efficiency metrics.

    Args:
        value: Efficiency value to convert.
        from_metric: Source efficiency metric.
        to_metric: Target efficiency metric.

    Returns:
        Converted efficiency value as Decimal.
    """
    return get_database().convert_efficiency(value, from_metric, to_metric)

def calculate_iplv(
    cop_100: Decimal,
    cop_75: Decimal,
    cop_50: Decimal,
    cop_25: Decimal,
) -> Decimal:
    """Module-level convenience: calculate IPLV from four COP points.

    Args:
        cop_100: COP at 100% load.
        cop_75: COP at 75% load.
        cop_50: COP at 50% load.
        cop_25: COP at 25% load.

    Returns:
        IPLV as Decimal.
    """
    return get_database().calculate_iplv(cop_100, cop_75, cop_50, cop_25)

def get_part_load_weights() -> Dict[str, Decimal]:
    """Module-level convenience: get AHRI part-load weights.

    Returns:
        Dictionary mapping load percentage to Decimal weight.
    """
    return get_database().get_part_load_weights()

def convert_cooling_units(
    value: Decimal,
    from_unit: str,
    to_unit: str,
) -> Decimal:
    """Module-level convenience: convert cooling energy units.

    Args:
        value: Cooling energy quantity.
        from_unit: Source unit identifier.
        to_unit: Target unit identifier.

    Returns:
        Converted quantity as Decimal.
    """
    return get_database().convert_cooling_units(value, from_unit, to_unit)

def get_gwp(gas: str, source: str) -> Decimal:
    """Module-level convenience: get GWP for a gas and source.

    Args:
        gas: Gas identifier ('CO2', 'CH4', 'N2O').
        source: IPCC source identifier.

    Returns:
        GWP multiplier as Decimal.
    """
    return get_database().get_gwp(gas, source)
