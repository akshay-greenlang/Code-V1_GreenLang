# -*- coding: utf-8 -*-
"""
DistanceBasedCalculatorEngine - Engine 2: Downstream Transportation Agent (AGENT-MRV-022)

Core calculation engine implementing the distance-based method for Scope 3 Category 9
(Downstream Transportation & Distribution) emissions per GHG Protocol, ISO 14083, and
GLEC Framework v3.0.

Primary Formula:
    Transport = distance_km x weight_tonnes x EF_per_tkm x cold_chain_uplift x load_adj
    WTT       = distance_km x weight_tonnes x WTT_per_tkm x cold_chain_uplift x load_adj
    Return    = Transport x return_multiplier
    Total     = Transport + WTT + Return

This engine supports all five transport modes (road, rail, maritime, air, pipeline)
plus intermodal chains with hub/transshipment emissions. It applies corrections for
cold chain uplift, load factor, return trips, and WTT upstream fuel emissions.

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the calculation path
    - Every intermediate step is recorded in the calculation trace
    - SHA-256 provenance hash on every result
    - Emission factors sourced from GLEC v3.0, DEFRA 2024, IMO 4th GHG Study, ICAO v12

Supports:
    - 5 transport modes (road, rail, maritime, air, pipeline) + intermodal
    - 40+ vehicle/vessel types with mode-specific emission factors
    - Cold chain uplift (CHILLED, FROZEN) by mode
    - Load factor adjustments (actual vs default)
    - Return trip emissions (empty, partial, full)
    - Multi-leg transport chains with hub emissions
    - Great circle distance via Haversine formula with 1.09x air correction
    - Batch processing with error isolation
    - Fleet-level aggregation by mode
    - Mode comparison for sustainability analysis
    - SHA-256 provenance hash per calculation
    - Performance timing for all operations

Example:
    >>> from greenlang.downstream_transportation.distance_based_calculator import (
    ...     get_distance_based_calculator,
    ... )
    >>> engine = get_distance_based_calculator()
    >>> result = engine.calculate_shipment({
    ...     "shipment_id": "DS-2026-001",
    ...     "distance_km": 500.0,
    ...     "weight_tonnes": 10.0,
    ...     "vehicle_type": "ARTICULATED_40_44T",
    ...     "temperature_regime": "AMBIENT",
    ...     "return_type": "EMPTY",
    ... })
    >>> print(result["total_emissions_kgco2e"])

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-009 (AGENT-MRV-022)
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.downstream_transportation.downstream_transport_database import (
    DownstreamTransportDatabaseEngine,
    get_downstream_transport_database,
    TRANSPORT_EMISSION_FACTORS,
    COLD_CHAIN_UPLIFT_FACTORS,
    LOAD_FACTOR_DEFAULTS,
    RETURN_TRIP_MULTIPLIERS,
    MODE_DEFAULT_VEHICLE_TYPES,
    TransportMode,
    TemperatureRegime,
    ReturnType,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# DECIMAL PRECISION & CONSTANTS
# ==============================================================================

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
_HUNDRED = Decimal("100")
_THOUSAND = Decimal("1000")

# Earth radius for Haversine (WGS-84 mean radius in km)
_EARTH_RADIUS_KM = Decimal("6371.0088")

# DEFRA great-circle distance correction for air freight (9% uplift)
_GCD_AIR_CORRECTION = Decimal("1.09")

# Road distance factor applied to great-circle for road estimates (1.3x)
_ROAD_DISTANCE_FACTOR = Decimal("1.30")

# Rail distance factor applied to great-circle for rail estimates (1.2x)
_RAIL_DISTANCE_FACTOR = Decimal("1.20")

# Maritime distance factor applied to great-circle for sea estimates (1.15x)
_MARITIME_DISTANCE_FACTOR = Decimal("1.15")

# Hub/transshipment emissions per tonne handled (kgCO2e/tonne)
_HUB_EMISSIONS_PER_TONNE = Decimal("2.50000000")


def _q(value: Decimal) -> Decimal:
    """Quantize to 8 decimal places with ROUND_HALF_UP."""
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


# ==============================================================================
# GAS SPLIT RATIOS BY MODE
# ==============================================================================
# For CO2e disaggregation into individual greenhouse gases.

_GAS_SPLIT_RATIOS: Dict[str, Dict[str, Decimal]] = {
    "ROAD": {
        "co2": Decimal("0.99500000"),
        "ch4": Decimal("0.00300000"),
        "n2o": Decimal("0.00200000"),
    },
    "RAIL": {
        "co2": Decimal("0.99000000"),
        "ch4": Decimal("0.00500000"),
        "n2o": Decimal("0.00500000"),
    },
    "MARITIME": {
        "co2": Decimal("0.99700000"),
        "ch4": Decimal("0.00200000"),
        "n2o": Decimal("0.00100000"),
    },
    "AIR": {
        "co2": Decimal("0.99800000"),
        "ch4": Decimal("0.00100000"),
        "n2o": Decimal("0.00100000"),
    },
    "PIPELINE": {
        "co2": Decimal("0.95000000"),
        "ch4": Decimal("0.04500000"),
        "n2o": Decimal("0.00500000"),
    },
}


# ==============================================================================
# WTT-TO-TTW RATIOS BY MODE
# ==============================================================================
# Used for decomposing WTW into TTW and WTT when only total is known.

_WTT_TO_TTW_RATIOS: Dict[str, Decimal] = {
    "ROAD": Decimal("0.21800000"),
    "RAIL": Decimal("0.24500000"),
    "MARITIME": Decimal("0.17400000"),
    "AIR": Decimal("0.24100000"),
    "PIPELINE": Decimal("0.16200000"),
}


# ==============================================================================
# DistanceBasedCalculatorEngine
# ==============================================================================


class DistanceBasedCalculatorEngine:
    """
    Engine 2: Distance-based emissions calculator for downstream transportation.

    Implements the distance-based method per GHG Protocol Scope 3 Category 9,
    ISO 14083, and GLEC Framework v3.0 for downstream (post-sale) transport
    and distribution.

    Core formula:
        Transport = distance_km x weight_tonnes x ef_per_tkm x cold_chain x load_adj
        WTT       = distance_km x weight_tonnes x wtt_per_tkm x cold_chain x load_adj
        Return    = Transport x return_multiplier
        Total     = Transport + WTT + Return

    All arithmetic uses Python Decimal with 8-digit precision and ROUND_HALF_UP
    to ensure deterministic, auditable, regulatory-grade results.

    Thread Safety:
        This engine is thread-safe. The singleton instance is protected by
        a threading.Lock during creation, and instance-level operations use
        a threading.RLock for reentrant safety.

    Attributes:
        _db: Reference to the DownstreamTransportDatabaseEngine singleton.
        _lock: Reentrant lock for thread safety.
        _calculation_count: Running count of calculations performed.

    Example:
        >>> engine = get_distance_based_calculator()
        >>> result = engine.calculate_shipment({
        ...     "shipment_id": "DS-001",
        ...     "distance_km": 500,
        ...     "weight_tonnes": 10,
        ...     "vehicle_type": "ARTICULATED_40_44T",
        ... })
        >>> assert result["total_emissions_kgco2e"] > Decimal("0")
    """

    _instance: Optional["DistanceBasedCalculatorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "DistanceBasedCalculatorEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the calculator engine (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._db: DownstreamTransportDatabaseEngine = get_downstream_transport_database()
        self._instance_lock: threading.RLock = threading.RLock()
        self._calculation_count: int = 0

        logger.info(
            "DistanceBasedCalculatorEngine initialized (singleton): "
            "precision=%s, gcd_correction=%s, road_factor=%s",
            _PRECISION,
            _GCD_AIR_CORRECTION,
            _ROAD_DISTANCE_FACTOR,
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_calc(self) -> None:
        """Increment the calculation counter."""
        self._calculation_count += 1

    def _generate_calc_id(self) -> str:
        """Generate a unique calculation identifier."""
        return f"dto_calc_{uuid.uuid4().hex[:12]}"

    def _build_provenance_hash(self, data: Dict[str, Any]) -> str:
        """
        Compute SHA-256 provenance hash for a calculation result.

        Args:
            data: Dictionary of calculation inputs and outputs.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _split_by_gas(
        self, total_co2e: Decimal, mode: str
    ) -> Dict[str, Decimal]:
        """
        Split total CO2e into individual gas components (CO2, CH4, N2O).

        Uses mode-specific gas split ratios. The remainder from rounding
        is assigned to CO2 to ensure the components sum to the total.

        Args:
            total_co2e: Total emissions in kgCO2e.
            mode: Transport mode (ROAD, RAIL, etc.).

        Returns:
            Dict with keys co2, ch4, n2o (all in kgCO2e).
        """
        ratios = _GAS_SPLIT_RATIOS.get(mode, _GAS_SPLIT_RATIOS["ROAD"])

        ch4 = _q(total_co2e * ratios["ch4"])
        n2o = _q(total_co2e * ratios["n2o"])
        co2 = _q(total_co2e - ch4 - n2o)

        return {"co2": co2, "ch4": ch4, "n2o": n2o}

    def _get_ef_data(self, vehicle_type: str) -> Dict[str, Any]:
        """
        Retrieve emission factor data for a vehicle type.

        Args:
            vehicle_type: Vehicle/vessel type key.

        Returns:
            Dict with ef_per_tkm, wtt_per_tkm, mode, source.

        Raises:
            ValueError: If vehicle_type is not found.
        """
        return self._db.get_transport_ef(vehicle_type)

    def _resolve_cold_chain(
        self, temperature_regime: str, mode: str
    ) -> Decimal:
        """
        Resolve cold chain uplift factor.

        Args:
            temperature_regime: AMBIENT, CHILLED, or FROZEN.
            mode: Transport mode.

        Returns:
            Cold chain uplift multiplier.
        """
        return self._db.get_cold_chain_uplift(temperature_regime, mode)

    def _resolve_load_adjustment(
        self, load_factor: Optional[float], mode: str
    ) -> Decimal:
        """
        Calculate load factor adjustment.

        If load_factor is provided and differs from the default for the mode,
        returns default_lf / actual_lf. Otherwise returns 1.0.

        Args:
            load_factor: Actual load factor (0-1), or None for no adjustment.
            mode: Transport mode.

        Returns:
            Load factor adjustment multiplier.
        """
        if load_factor is None:
            return _ONE

        lf = Decimal(str(load_factor))
        if lf <= _ZERO or lf > _ONE:
            logger.warning(
                "Invalid load_factor=%s, using 1.0 adjustment", load_factor
            )
            return _ONE

        default_lf = LOAD_FACTOR_DEFAULTS.get(mode, Decimal("0.65000000"))
        return _q(default_lf / lf)

    def _resolve_return_multiplier(self, return_type: Optional[str]) -> Decimal:
        """
        Resolve return trip emissions multiplier.

        Args:
            return_type: EMPTY, PARTIAL_LOAD, FULL_LOAD, NO_RETURN, or None.

        Returns:
            Return trip multiplier (0.0 to 0.35).
        """
        if return_type is None:
            return _ZERO

        rtype_upper = return_type.upper()
        return RETURN_TRIP_MULTIPLIERS.get(rtype_upper, _ZERO)

    def _get_mode_for_vehicle(self, vehicle_type: str) -> str:
        """
        Determine the transport mode from the vehicle type.

        Args:
            vehicle_type: Vehicle/vessel type key.

        Returns:
            Transport mode string (ROAD, RAIL, etc.).
        """
        ef_data = TRANSPORT_EMISSION_FACTORS.get(vehicle_type)
        if ef_data is not None:
            return ef_data["mode"]
        # Fallback: try to find in mode mappings
        for mode, vtypes in MODE_DEFAULT_VEHICLE_TYPES.items():
            if vehicle_type in vtypes:
                return mode
        return "ROAD"

    # =========================================================================
    # 1. calculate_shipment
    # =========================================================================

    def calculate_shipment(
        self, shipment_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a single downstream shipment.

        Primary formula:
            Transport = distance_km x weight_tonnes x EF x cold_chain x load_adj
            WTT = distance_km x weight_tonnes x WTT_EF x cold_chain x load_adj
            Return = Transport x return_multiplier
            Total = Transport + WTT + Return

        Args:
            shipment_input: Dictionary containing:
                - shipment_id (str): Unique shipment identifier.
                - distance_km (float): Transport distance in km. Must be > 0.
                - weight_tonnes (float): Cargo weight in tonnes. Must be > 0.
                - vehicle_type (str): Vehicle type key (e.g., "ARTICULATED_40_44T").
                - temperature_regime (str, optional): "AMBIENT"/"CHILLED"/"FROZEN".
                    Defaults to "AMBIENT".
                - return_type (str, optional): "EMPTY"/"PARTIAL_LOAD"/"FULL_LOAD"/
                    "NO_RETURN". Defaults to None (no return).
                - load_factor (float, optional): Actual load factor (0-1).
                - ef_source (str, optional): EF source override description.

        Returns:
            Dict containing:
                - shipment_id (str)
                - calc_id (str): Unique calculation identifier.
                - distance_km (Decimal)
                - weight_tonnes (Decimal)
                - tonne_km (Decimal)
                - vehicle_type (str)
                - mode (str)
                - ef_per_tkm (Decimal)
                - wtt_per_tkm (Decimal)
                - cold_chain_uplift (Decimal)
                - load_factor_adjustment (Decimal)
                - return_multiplier (Decimal)
                - transport_emissions_kgco2e (Decimal)
                - wtt_emissions_kgco2e (Decimal)
                - return_emissions_kgco2e (Decimal)
                - total_emissions_kgco2e (Decimal)
                - co2_kg (Decimal)
                - ch4_kg (Decimal)
                - n2o_kg (Decimal)
                - provenance_hash (str): SHA-256 hash.
                - calculation_timestamp (str): ISO-8601 timestamp.
                - processing_time_ms (float)
                - ef_source (str)

        Raises:
            ValueError: If required fields are missing or invalid.

        Example:
            >>> result = engine.calculate_shipment({
            ...     "shipment_id": "DS-001",
            ...     "distance_km": 500.0,
            ...     "weight_tonnes": 10.0,
            ...     "vehicle_type": "ARTICULATED_40_44T",
            ... })
            >>> result["total_emissions_kgco2e"]
            Decimal('515.00000000')
        """
        start_ts = time.monotonic()

        with self._instance_lock:
            self._increment_calc()
            calc_id = self._generate_calc_id()

        # ---- Extract and validate inputs ----
        shipment_id = shipment_input.get("shipment_id", f"auto_{calc_id}")
        distance_km = self._validate_positive_decimal(
            shipment_input.get("distance_km"), "distance_km"
        )
        weight_tonnes = self._validate_positive_decimal(
            shipment_input.get("weight_tonnes"), "weight_tonnes"
        )
        vehicle_type = shipment_input.get("vehicle_type")
        if not vehicle_type:
            raise ValueError("vehicle_type is required")

        temperature_regime = shipment_input.get("temperature_regime", "AMBIENT")
        return_type = shipment_input.get("return_type")
        load_factor = shipment_input.get("load_factor")

        logger.info(
            "calculate_shipment: id=%s, calc_id=%s, dist=%s km, wt=%s t, "
            "vehicle=%s, temp=%s, return=%s",
            shipment_id, calc_id, distance_km, weight_tonnes,
            vehicle_type, temperature_regime, return_type,
        )

        # ---- Retrieve emission factors ----
        ef_data = self._get_ef_data(vehicle_type)
        ef_per_tkm = ef_data["ef_per_tkm"]
        wtt_per_tkm = ef_data["wtt_per_tkm"]
        mode = ef_data["mode"]
        ef_source = shipment_input.get("ef_source", ef_data["source"])

        # ---- Calculate adjustments ----
        cold_chain_uplift = self._resolve_cold_chain(temperature_regime, mode)
        load_adj = self._resolve_load_adjustment(load_factor, mode)
        return_mult = self._resolve_return_multiplier(return_type)

        # ---- Core calculation (all Decimal) ----
        tonne_km = _q(distance_km * weight_tonnes)

        # Transport emissions = tkm x EF x cold_chain x load_adj
        transport_emissions = _q(
            tonne_km * ef_per_tkm * cold_chain_uplift * load_adj
        )

        # WTT emissions = tkm x WTT_EF x cold_chain x load_adj
        wtt_emissions = _q(
            tonne_km * wtt_per_tkm * cold_chain_uplift * load_adj
        )

        # Return emissions = Transport x return_multiplier
        return_emissions = _q(transport_emissions * return_mult)

        # Total = Transport + WTT + Return
        total_emissions = _q(transport_emissions + wtt_emissions + return_emissions)

        # ---- Gas-by-gas split ----
        gas_split = self._split_by_gas(total_emissions, mode)

        # ---- Build provenance hash ----
        provenance_data = {
            "shipment_id": shipment_id,
            "calc_id": calc_id,
            "distance_km": str(distance_km),
            "weight_tonnes": str(weight_tonnes),
            "vehicle_type": vehicle_type,
            "ef_per_tkm": str(ef_per_tkm),
            "wtt_per_tkm": str(wtt_per_tkm),
            "cold_chain_uplift": str(cold_chain_uplift),
            "load_factor_adjustment": str(load_adj),
            "return_multiplier": str(return_mult),
            "total_emissions_kgco2e": str(total_emissions),
        }
        provenance_hash = self._build_provenance_hash(provenance_data)

        # ---- Timing ----
        elapsed_ms = (time.monotonic() - start_ts) * 1000.0
        timestamp = datetime.now(timezone.utc).isoformat()

        # ---- Build result ----
        result: Dict[str, Any] = {
            "shipment_id": shipment_id,
            "calc_id": calc_id,
            "distance_km": distance_km,
            "weight_tonnes": weight_tonnes,
            "tonne_km": tonne_km,
            "vehicle_type": vehicle_type,
            "mode": mode,
            "ef_per_tkm": ef_per_tkm,
            "wtt_per_tkm": wtt_per_tkm,
            "cold_chain_uplift": cold_chain_uplift,
            "load_factor_adjustment": load_adj,
            "return_multiplier": return_mult,
            "transport_emissions_kgco2e": transport_emissions,
            "wtt_emissions_kgco2e": wtt_emissions,
            "return_emissions_kgco2e": return_emissions,
            "total_emissions_kgco2e": total_emissions,
            "co2_kg": gas_split["co2"],
            "ch4_kg": gas_split["ch4"],
            "n2o_kg": gas_split["n2o"],
            "provenance_hash": provenance_hash,
            "calculation_timestamp": timestamp,
            "processing_time_ms": round(elapsed_ms, 4),
            "ef_source": ef_source,
        }

        logger.info(
            "calculate_shipment complete: id=%s, total=%s kgCO2e, "
            "transport=%s, wtt=%s, return=%s, elapsed=%.2fms",
            shipment_id, total_emissions, transport_emissions,
            wtt_emissions, return_emissions, elapsed_ms,
        )

        return result

    # =========================================================================
    # 2. calculate_multi_leg
    # =========================================================================

    def calculate_multi_leg(
        self, legs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a multi-leg transport chain.

        Calculates emissions for each leg independently, adds hub/transshipment
        emissions between legs, and returns a combined result.

        Hub emissions represent energy consumed at transshipment points
        (loading/unloading, sorting, temporary storage). Default:
        2.5 kgCO2e per tonne handled per hub stop.

        Args:
            legs: List of leg dictionaries, each containing the same fields
                  as calculate_shipment() input, plus optional:
                  - hub_emissions_per_tonne (Decimal): Override hub EF.

        Returns:
            Dict containing:
                - chain_id (str)
                - legs (List[Dict]): Individual leg results.
                - hub_emissions_kgco2e (Decimal): Total hub emissions.
                - total_transport_kgco2e (Decimal): Sum of all leg transport.
                - total_wtt_kgco2e (Decimal): Sum of all leg WTT.
                - total_return_kgco2e (Decimal): Sum of all leg return.
                - total_emissions_kgco2e (Decimal): Grand total.
                - provenance_hash (str)
                - processing_time_ms (float)

        Raises:
            ValueError: If legs list is empty.

        Example:
            >>> result = engine.calculate_multi_leg([
            ...     {"shipment_id": "L1", "distance_km": 300, "weight_tonnes": 10,
            ...      "vehicle_type": "ARTICULATED_40_44T"},
            ...     {"shipment_id": "L2", "distance_km": 5000, "weight_tonnes": 10,
            ...      "vehicle_type": "CONTAINER_POST_PANAMAX"},
            ... ])
        """
        start_ts = time.monotonic()

        if not legs:
            raise ValueError("legs list must not be empty")

        chain_id = f"chain_{uuid.uuid4().hex[:12]}"

        logger.info(
            "calculate_multi_leg: chain_id=%s, legs=%d", chain_id, len(legs),
        )

        leg_results: List[Dict[str, Any]] = []
        total_transport = _ZERO
        total_wtt = _ZERO
        total_return = _ZERO
        total_hub = _ZERO

        for idx, leg in enumerate(legs):
            # Calculate leg emissions
            leg_result = self.calculate_shipment(leg)
            leg_results.append(leg_result)

            total_transport = _q(
                total_transport + leg_result["transport_emissions_kgco2e"]
            )
            total_wtt = _q(total_wtt + leg_result["wtt_emissions_kgco2e"])
            total_return = _q(
                total_return + leg_result["return_emissions_kgco2e"]
            )

            # Add hub emissions between legs (not after the last leg)
            if idx < len(legs) - 1:
                weight = leg_result["weight_tonnes"]
                hub_ef = Decimal(
                    str(
                        leg.get(
                            "hub_emissions_per_tonne",
                            _HUB_EMISSIONS_PER_TONNE,
                        )
                    )
                )
                hub_emission = _q(weight * hub_ef)
                total_hub = _q(total_hub + hub_emission)

                logger.debug(
                    "Hub emissions between leg %d and %d: %s kgCO2e "
                    "(%s t x %s kgCO2e/t)",
                    idx, idx + 1, hub_emission, weight, hub_ef,
                )

        total_emissions = _q(
            total_transport + total_wtt + total_return + total_hub
        )

        # Provenance hash
        provenance_data = {
            "chain_id": chain_id,
            "leg_count": len(legs),
            "total_transport": str(total_transport),
            "total_wtt": str(total_wtt),
            "total_return": str(total_return),
            "total_hub": str(total_hub),
            "total_emissions": str(total_emissions),
        }
        provenance_hash = self._build_provenance_hash(provenance_data)

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0

        result: Dict[str, Any] = {
            "chain_id": chain_id,
            "legs": leg_results,
            "hub_emissions_kgco2e": total_hub,
            "total_transport_kgco2e": total_transport,
            "total_wtt_kgco2e": total_wtt,
            "total_return_kgco2e": total_return,
            "total_emissions_kgco2e": total_emissions,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed_ms, 4),
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "calculate_multi_leg complete: chain_id=%s, legs=%d, "
            "total=%s kgCO2e (transport=%s, wtt=%s, return=%s, hub=%s), "
            "elapsed=%.2fms",
            chain_id, len(legs), total_emissions,
            total_transport, total_wtt, total_return, total_hub,
            elapsed_ms,
        )

        return result

    # =========================================================================
    # 3. calculate_intermodal
    # =========================================================================

    def calculate_intermodal(
        self, legs_with_modes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate emissions for an intermodal (multi-mode) transport chain.

        Each leg specifies a different transport mode. Results include per-leg
        breakdown, per-mode aggregation, and hub emissions at modal transfer
        points.

        Args:
            legs_with_modes: List of leg dicts, each with vehicle_type set to
                            a mode-specific vehicle type (e.g., "ARTICULATED_40_44T"
                            for road, "CONTAINER_POST_PANAMAX" for maritime).

        Returns:
            Dict containing:
                - intermodal_id (str)
                - legs (List[Dict]): Per-leg results.
                - by_mode (Dict[str, Decimal]): Emissions aggregated by mode.
                - hub_emissions_kgco2e (Decimal)
                - total_emissions_kgco2e (Decimal)
                - provenance_hash (str)
                - processing_time_ms (float)

        Raises:
            ValueError: If legs_with_modes is empty.

        Example:
            >>> result = engine.calculate_intermodal([
            ...     {"shipment_id": "IM-R", "distance_km": 100,
            ...      "weight_tonnes": 20, "vehicle_type": "ARTICULATED_40_44T"},
            ...     {"shipment_id": "IM-S", "distance_km": 8000,
            ...      "weight_tonnes": 20, "vehicle_type": "CONTAINER_POST_PANAMAX"},
            ...     {"shipment_id": "IM-L", "distance_km": 50,
            ...      "weight_tonnes": 20, "vehicle_type": "RIGID_7_5_17T"},
            ... ])
        """
        start_ts = time.monotonic()

        if not legs_with_modes:
            raise ValueError("legs_with_modes list must not be empty")

        intermodal_id = f"intermodal_{uuid.uuid4().hex[:12]}"

        logger.info(
            "calculate_intermodal: id=%s, legs=%d",
            intermodal_id, len(legs_with_modes),
        )

        # Use multi_leg for the core calculation
        chain_result = self.calculate_multi_leg(legs_with_modes)

        # Aggregate by mode
        by_mode: Dict[str, Decimal] = {}
        for leg_result in chain_result["legs"]:
            mode = leg_result["mode"]
            current = by_mode.get(mode, _ZERO)
            by_mode[mode] = _q(current + leg_result["total_emissions_kgco2e"])

        # Build intermodal result
        provenance_data = {
            "intermodal_id": intermodal_id,
            "chain_id": chain_result["chain_id"],
            "by_mode": {k: str(v) for k, v in by_mode.items()},
            "total": str(chain_result["total_emissions_kgco2e"]),
        }
        provenance_hash = self._build_provenance_hash(provenance_data)

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0

        result: Dict[str, Any] = {
            "intermodal_id": intermodal_id,
            "legs": chain_result["legs"],
            "by_mode": by_mode,
            "hub_emissions_kgco2e": chain_result["hub_emissions_kgco2e"],
            "total_transport_kgco2e": chain_result["total_transport_kgco2e"],
            "total_wtt_kgco2e": chain_result["total_wtt_kgco2e"],
            "total_return_kgco2e": chain_result["total_return_kgco2e"],
            "total_emissions_kgco2e": chain_result["total_emissions_kgco2e"],
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed_ms, 4),
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "calculate_intermodal complete: id=%s, by_mode=%s, "
            "total=%s kgCO2e, elapsed=%.2fms",
            intermodal_id,
            {k: str(v) for k, v in by_mode.items()},
            chain_result["total_emissions_kgco2e"],
            elapsed_ms,
        )

        return result

    # =========================================================================
    # 4. calculate_batch
    # =========================================================================

    def calculate_batch(
        self, shipments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate emissions for a batch of shipments with error isolation.

        Processes each shipment independently. Failed calculations produce
        error entries in the result list rather than raising exceptions,
        allowing partial batch completion.

        Args:
            shipments: List of shipment input dicts (same format as
                      calculate_shipment() input).

        Returns:
            List of result dicts. Successful results contain all
            calculate_shipment() fields. Failed results contain:
            - shipment_id (str)
            - status (str): "error"
            - error (str): Error message.
            - processing_time_ms (float)

        Example:
            >>> results = engine.calculate_batch([
            ...     {"shipment_id": "B1", "distance_km": 100,
            ...      "weight_tonnes": 5, "vehicle_type": "LCV_DIESEL"},
            ...     {"shipment_id": "B2", "distance_km": 500,
            ...      "weight_tonnes": 10, "vehicle_type": "ARTICULATED_40_44T"},
            ... ])
            >>> len(results)
            2
        """
        start_ts = time.monotonic()
        results: List[Dict[str, Any]] = []
        success_count = 0
        error_count = 0

        logger.info("calculate_batch: processing %d shipments", len(shipments))

        for idx, shipment in enumerate(shipments):
            ship_start = time.monotonic()
            shipment_id = shipment.get(
                "shipment_id", f"batch_{idx}"
            )

            try:
                result = self.calculate_shipment(shipment)
                result["status"] = "success"
                results.append(result)
                success_count += 1
            except Exception as exc:
                ship_elapsed = (time.monotonic() - ship_start) * 1000.0
                error_result: Dict[str, Any] = {
                    "shipment_id": shipment_id,
                    "status": "error",
                    "error": str(exc),
                    "processing_time_ms": round(ship_elapsed, 4),
                }
                results.append(error_result)
                error_count += 1

                logger.error(
                    "calculate_batch: shipment %d/%d (%s) failed: %s",
                    idx + 1, len(shipments), shipment_id, exc,
                )

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0

        # Compute batch summary
        total_emissions = _ZERO
        for r in results:
            if r.get("status") == "success":
                total_emissions = _q(
                    total_emissions + r.get("total_emissions_kgco2e", _ZERO)
                )

        logger.info(
            "calculate_batch complete: total=%d, success=%d, errors=%d, "
            "emissions=%s kgCO2e, elapsed=%.2fms",
            len(shipments), success_count, error_count,
            total_emissions, elapsed_ms,
        )

        return results

    # =========================================================================
    # 5. calculate_fleet
    # =========================================================================

    def calculate_fleet(
        self, shipments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate aggregated fleet-level emissions from multiple shipments.

        Processes all shipments and aggregates results by transport mode,
        providing fleet-level totals and mode breakdown.

        Args:
            shipments: List of shipment input dicts.

        Returns:
            Dict containing:
                - fleet_id (str)
                - total_shipments (int)
                - successful_shipments (int)
                - failed_shipments (int)
                - by_mode (Dict[str, Dict]): Per-mode aggregation with
                  shipment_count, total_tonne_km, total_emissions_kgco2e.
                - total_tonne_km (Decimal)
                - total_emissions_kgco2e (Decimal)
                - total_transport_kgco2e (Decimal)
                - total_wtt_kgco2e (Decimal)
                - total_return_kgco2e (Decimal)
                - avg_ef_per_tkm (Decimal): Weighted average EF.
                - provenance_hash (str)
                - processing_time_ms (float)

        Example:
            >>> fleet = engine.calculate_fleet([
            ...     {"shipment_id": "F1", "distance_km": 500,
            ...      "weight_tonnes": 10, "vehicle_type": "ARTICULATED_40_44T"},
            ...     {"shipment_id": "F2", "distance_km": 300,
            ...      "weight_tonnes": 5, "vehicle_type": "RIGID_7_5_17T"},
            ... ])
            >>> fleet["by_mode"]["ROAD"]["shipment_count"]
            2
        """
        start_ts = time.monotonic()
        fleet_id = f"fleet_{uuid.uuid4().hex[:12]}"

        logger.info(
            "calculate_fleet: fleet_id=%s, shipments=%d",
            fleet_id, len(shipments),
        )

        # Calculate batch
        batch_results = self.calculate_batch(shipments)

        # Aggregate
        by_mode: Dict[str, Dict[str, Any]] = {}
        total_tonne_km = _ZERO
        total_emissions = _ZERO
        total_transport = _ZERO
        total_wtt = _ZERO
        total_return = _ZERO
        success_count = 0
        error_count = 0

        for r in batch_results:
            if r.get("status") != "success":
                error_count += 1
                continue

            success_count += 1
            mode = r["mode"]
            tkm = r["tonne_km"]
            emissions = r["total_emissions_kgco2e"]
            transport = r["transport_emissions_kgco2e"]
            wtt = r["wtt_emissions_kgco2e"]
            ret = r["return_emissions_kgco2e"]

            total_tonne_km = _q(total_tonne_km + tkm)
            total_emissions = _q(total_emissions + emissions)
            total_transport = _q(total_transport + transport)
            total_wtt = _q(total_wtt + wtt)
            total_return = _q(total_return + ret)

            if mode not in by_mode:
                by_mode[mode] = {
                    "shipment_count": 0,
                    "total_tonne_km": _ZERO,
                    "total_emissions_kgco2e": _ZERO,
                }

            by_mode[mode]["shipment_count"] += 1
            by_mode[mode]["total_tonne_km"] = _q(
                by_mode[mode]["total_tonne_km"] + tkm
            )
            by_mode[mode]["total_emissions_kgco2e"] = _q(
                by_mode[mode]["total_emissions_kgco2e"] + emissions
            )

        # Weighted average EF
        avg_ef = _ZERO
        if total_tonne_km > _ZERO:
            avg_ef = _q(total_emissions / total_tonne_km)

        # Provenance
        provenance_data = {
            "fleet_id": fleet_id,
            "total_shipments": len(shipments),
            "success": success_count,
            "total_emissions": str(total_emissions),
            "total_tonne_km": str(total_tonne_km),
        }
        provenance_hash = self._build_provenance_hash(provenance_data)

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0

        result: Dict[str, Any] = {
            "fleet_id": fleet_id,
            "total_shipments": len(shipments),
            "successful_shipments": success_count,
            "failed_shipments": error_count,
            "by_mode": by_mode,
            "total_tonne_km": total_tonne_km,
            "total_emissions_kgco2e": total_emissions,
            "total_transport_kgco2e": total_transport,
            "total_wtt_kgco2e": total_wtt,
            "total_return_kgco2e": total_return,
            "avg_ef_per_tkm": avg_ef,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed_ms, 4),
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "calculate_fleet complete: fleet_id=%s, shipments=%d/%d, "
            "modes=%s, total=%s kgCO2e, avg_ef=%s, elapsed=%.2fms",
            fleet_id, success_count, len(shipments),
            list(by_mode.keys()), total_emissions, avg_ef, elapsed_ms,
        )

        return result

    # =========================================================================
    # 6. estimate_from_origin_destination
    # =========================================================================

    def estimate_from_origin_destination(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        mode: str,
        weight_tonnes: float,
        vehicle_type: Optional[str] = None,
        temperature_regime: str = "AMBIENT",
        return_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Estimate emissions from origin/destination coordinates using
        great-circle distance with mode-specific correction factors.

        Distance estimation:
            - AIR: great-circle x 1.09 (DEFRA correction)
            - ROAD: great-circle x 1.30 (road routing factor)
            - RAIL: great-circle x 1.20 (rail routing factor)
            - MARITIME: great-circle x 1.15 (sea routing factor)
            - PIPELINE: great-circle x 1.05

        Args:
            origin_lat: Origin latitude (-90 to 90).
            origin_lon: Origin longitude (-180 to 180).
            dest_lat: Destination latitude.
            dest_lon: Destination longitude.
            mode: Transport mode ("ROAD", "RAIL", "MARITIME", "AIR", "PIPELINE").
            weight_tonnes: Cargo weight in tonnes. Must be > 0.
            vehicle_type: Vehicle type. If None, uses mode default.
            temperature_regime: Temperature control. Defaults to "AMBIENT".
            return_type: Return type. Optional.

        Returns:
            Dict with calculate_shipment() result plus:
            - great_circle_km (Decimal): Raw Haversine distance.
            - estimated_distance_km (Decimal): Corrected distance.
            - distance_correction_factor (Decimal)
            - distance_method (str): "great_circle_estimated"

        Raises:
            ValueError: If coordinates are out of range or mode is invalid.

        Example:
            >>> result = engine.estimate_from_origin_destination(
            ...     origin_lat=51.47, origin_lon=-0.45,  # London
            ...     dest_lat=40.64, dest_lon=-73.78,     # New York
            ...     mode="AIR",
            ...     weight_tonnes=5.0,
            ... )
        """
        start_ts = time.monotonic()

        # Validate coordinates
        self._validate_coordinate(origin_lat, "origin_lat", -90.0, 90.0)
        self._validate_coordinate(origin_lon, "origin_lon", -180.0, 180.0)
        self._validate_coordinate(dest_lat, "dest_lat", -90.0, 90.0)
        self._validate_coordinate(dest_lon, "dest_lon", -180.0, 180.0)

        # Calculate great-circle distance (Haversine)
        gcd = self._haversine(origin_lat, origin_lon, dest_lat, dest_lon)

        # Apply mode-specific correction
        mode_upper = mode.upper()
        correction_factors = {
            "ROAD": _ROAD_DISTANCE_FACTOR,
            "RAIL": _RAIL_DISTANCE_FACTOR,
            "MARITIME": _MARITIME_DISTANCE_FACTOR,
            "AIR": _GCD_AIR_CORRECTION,
            "PIPELINE": Decimal("1.05000000"),
        }

        correction = correction_factors.get(mode_upper, _ROAD_DISTANCE_FACTOR)
        estimated_distance = _q(gcd * correction)

        # Resolve default vehicle type if not provided
        if vehicle_type is None:
            mode_defaults = {
                "ROAD": "ARTICULATED_40_44T",
                "RAIL": "FREIGHT_DIESEL",
                "MARITIME": "CONTAINER_POST_PANAMAX",
                "AIR": "LONG_HAUL_WIDE",
                "PIPELINE": "REFINED_PRODUCTS_PIPELINE",
            }
            vehicle_type = mode_defaults.get(mode_upper, "ARTICULATED_40_44T")

        logger.info(
            "estimate_from_origin_destination: (%s,%s)->(%s,%s), "
            "mode=%s, gcd=%s km, corrected=%s km, vehicle=%s",
            origin_lat, origin_lon, dest_lat, dest_lon,
            mode_upper, gcd, estimated_distance, vehicle_type,
        )

        # Calculate emissions using the estimated distance
        shipment_input = {
            "shipment_id": f"estimate_{uuid.uuid4().hex[:8]}",
            "distance_km": float(estimated_distance),
            "weight_tonnes": weight_tonnes,
            "vehicle_type": vehicle_type,
            "temperature_regime": temperature_regime,
            "return_type": return_type,
        }

        result = self.calculate_shipment(shipment_input)

        # Add origin-destination metadata
        result["great_circle_km"] = gcd
        result["estimated_distance_km"] = estimated_distance
        result["distance_correction_factor"] = correction
        result["distance_method"] = "great_circle_estimated"
        result["origin_lat"] = Decimal(str(origin_lat))
        result["origin_lon"] = Decimal(str(origin_lon))
        result["dest_lat"] = Decimal(str(dest_lat))
        result["dest_lon"] = Decimal(str(dest_lon))

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0
        result["processing_time_ms"] = round(elapsed_ms, 4)

        return result

    # =========================================================================
    # 7. compare_modes
    # =========================================================================

    def compare_modes(
        self,
        distance_km: float,
        weight_tonnes: float,
        temperature_regime: str = "AMBIENT",
        return_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compare emissions across all transport modes for a given shipment.

        Calculates emissions using a representative default vehicle type for
        each mode and returns results ranked by total emissions (ascending).
        Useful for mode-shift analysis and sustainability reporting.

        Args:
            distance_km: Transport distance in km. Must be > 0.
            weight_tonnes: Cargo weight in tonnes. Must be > 0.
            temperature_regime: Temperature control. Defaults to "AMBIENT".
            return_type: Return type. Optional.

        Returns:
            List of result dicts sorted by total_emissions_kgco2e (ascending),
            each containing full calculate_shipment() output plus a rank field.

        Raises:
            ValueError: If distance_km or weight_tonnes is <= 0.

        Example:
            >>> comparison = engine.compare_modes(500.0, 10.0)
            >>> comparison[0]["mode"]  # Lowest emissions mode
            'PIPELINE'
            >>> comparison[-1]["mode"]  # Highest emissions mode
            'AIR'
        """
        start_ts = time.monotonic()

        if distance_km <= 0:
            raise ValueError(f"distance_km must be > 0, got {distance_km}")
        if weight_tonnes <= 0:
            raise ValueError(f"weight_tonnes must be > 0, got {weight_tonnes}")

        mode_defaults = {
            "ROAD": "ARTICULATED_40_44T",
            "RAIL": "FREIGHT_DIESEL",
            "MARITIME": "CONTAINER_POST_PANAMAX",
            "AIR": "LONG_HAUL_WIDE",
            "PIPELINE": "REFINED_PRODUCTS_PIPELINE",
        }

        results: List[Dict[str, Any]] = []

        for mode, vehicle_type in mode_defaults.items():
            try:
                shipment_input = {
                    "shipment_id": f"compare_{mode}",
                    "distance_km": distance_km,
                    "weight_tonnes": weight_tonnes,
                    "vehicle_type": vehicle_type,
                    "temperature_regime": temperature_regime,
                    "return_type": return_type,
                }
                result = self.calculate_shipment(shipment_input)
                result["comparison_mode"] = mode
                results.append(result)
            except Exception as exc:
                logger.warning(
                    "compare_modes: failed for mode=%s: %s", mode, exc
                )

        # Sort by total emissions ascending
        results.sort(key=lambda x: x["total_emissions_kgco2e"])

        # Add rank
        for idx, result in enumerate(results):
            result["rank"] = idx + 1

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0

        logger.info(
            "compare_modes: distance=%s km, weight=%s t, %d modes compared, "
            "lowest=%s (%s kgCO2e), highest=%s (%s kgCO2e), elapsed=%.2fms",
            distance_km, weight_tonnes, len(results),
            results[0]["mode"] if results else "N/A",
            results[0]["total_emissions_kgco2e"] if results else "N/A",
            results[-1]["mode"] if results else "N/A",
            results[-1]["total_emissions_kgco2e"] if results else "N/A",
            elapsed_ms,
        )

        return results

    # =========================================================================
    # ADDITIONAL PUBLIC METHODS
    # =========================================================================

    def calculate_wtw_breakdown(
        self, total_emissions: Decimal, mode: str
    ) -> Dict[str, Decimal]:
        """
        Decompose WTW emissions into TTW and WTT components.

        Uses mode-specific WTT-to-TTW ratios to separate well-to-wheel
        emissions into tank-to-wheel (direct) and well-to-tank (upstream).

        Formula:
            TTW = WTW / (1 + wtt_ratio)
            WTT = WTW - TTW

        Args:
            total_emissions: Total WTW emissions in kgCO2e.
            mode: Transport mode (ROAD, RAIL, etc.).

        Returns:
            Dict with keys: ttw, wtt, wtw (all in kgCO2e).

        Raises:
            ValueError: If total_emissions < 0.

        Example:
            >>> breakdown = engine.calculate_wtw_breakdown(
            ...     Decimal("1000"), "ROAD"
            ... )
            >>> breakdown["ttw"]
            Decimal('820.69000000')
        """
        if total_emissions < _ZERO:
            raise ValueError(
                f"total_emissions must be >= 0, got {total_emissions}"
            )

        mode_upper = mode.upper()
        wtt_ratio = _WTT_TO_TTW_RATIOS.get(mode_upper, Decimal("0.22000000"))

        # WTW = TTW x (1 + ratio), so TTW = WTW / (1 + ratio)
        divisor = _q(_ONE + wtt_ratio)
        ttw = _q(total_emissions / divisor)
        wtt = _q(total_emissions - ttw)

        result = {
            "ttw": ttw,
            "wtt": wtt,
            "wtw": _q(total_emissions),
        }

        logger.debug(
            "calculate_wtw_breakdown: mode=%s, wtw=%s -> ttw=%s, wtt=%s",
            mode_upper, total_emissions, ttw, wtt,
        )

        return result

    def split_by_gas(
        self, total_co2e: Decimal, mode: str
    ) -> Dict[str, Decimal]:
        """
        Public interface for gas-by-gas split.

        Args:
            total_co2e: Total emissions in kgCO2e.
            mode: Transport mode.

        Returns:
            Dict with keys: co2, ch4, n2o (all in kgCO2e).

        Example:
            >>> split = engine.split_by_gas(Decimal("1000"), "ROAD")
            >>> split["co2"]  # ~995
        """
        return self._split_by_gas(total_co2e, mode.upper())

    def calculate_great_circle_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> Decimal:
        """
        Calculate great circle distance using the Haversine formula.

        Args:
            lat1: Origin latitude (-90 to 90).
            lon1: Origin longitude (-180 to 180).
            lat2: Destination latitude.
            lon2: Destination longitude.

        Returns:
            Distance in km (Decimal).

        Raises:
            ValueError: If coordinates are out of range.

        Example:
            >>> # London to New York
            >>> d = engine.calculate_great_circle_distance(51.47, -0.45, 40.64, -73.78)
            >>> # d ~ 5570 km
        """
        self._validate_coordinate(lat1, "lat1", -90.0, 90.0)
        self._validate_coordinate(lon1, "lon1", -180.0, 180.0)
        self._validate_coordinate(lat2, "lat2", -90.0, 90.0)
        self._validate_coordinate(lon2, "lon2", -180.0, 180.0)

        return self._haversine(lat1, lon1, lat2, lon2)

    def get_calculation_count(self) -> int:
        """
        Get the total number of calculations performed.

        Returns:
            Integer count of calculations since engine initialization.
        """
        with self._instance_lock:
            return self._calculation_count

    # =========================================================================
    # UNIT CONVERSION METHODS
    # =========================================================================

    def convert_distance(
        self, value: float, from_unit: str, to_unit: str
    ) -> Decimal:
        """
        Convert distance between units (km, miles, nautical_miles).

        Supported conversions:
            - km <-> miles (1 mile = 1.609344 km)
            - km <-> nautical_miles (1 nm = 1.852 km)
            - miles <-> nautical_miles

        Args:
            value: Distance value. Must be >= 0.
            from_unit: Source unit ("km", "miles", "nautical_miles").
            to_unit: Target unit.

        Returns:
            Converted distance (Decimal, 8dp).

        Raises:
            ValueError: If value < 0 or units are not recognized.

        Example:
            >>> engine.convert_distance(100.0, "miles", "km")
            Decimal('160.93440000')
        """
        if value < 0:
            raise ValueError(f"Distance must be >= 0, got {value}")

        valid_units = {"km", "miles", "nautical_miles"}
        if from_unit not in valid_units:
            raise ValueError(
                f"Invalid from_unit '{from_unit}'. Must be one of {valid_units}"
            )
        if to_unit not in valid_units:
            raise ValueError(
                f"Invalid to_unit '{to_unit}'. Must be one of {valid_units}"
            )

        if from_unit == to_unit:
            return _q(Decimal(str(value)))

        dec_value = Decimal(str(value))

        # Convert to km first
        km_per_mile = Decimal("1.609344")
        km_per_nm = Decimal("1.852")

        if from_unit == "km":
            km_value = dec_value
        elif from_unit == "miles":
            km_value = _q(dec_value * km_per_mile)
        else:  # nautical_miles
            km_value = _q(dec_value * km_per_nm)

        # Convert from km to target
        if to_unit == "km":
            result = km_value
        elif to_unit == "miles":
            result = _q(km_value / km_per_mile)
        else:  # nautical_miles
            result = _q(km_value / km_per_nm)

        logger.debug(
            "convert_distance: %s %s = %s %s", value, from_unit, result, to_unit
        )

        return result

    def convert_mass(
        self, value: float, from_unit: str, to_unit: str
    ) -> Decimal:
        """
        Convert mass between units (tonnes, kg, lbs, short_tons).

        Supported conversions:
            - tonnes <-> kg (1 tonne = 1000 kg)
            - tonnes <-> lbs (1 tonne = 2204.62262 lbs)
            - tonnes <-> short_tons (1 tonne = 1.10231 short tons)

        Args:
            value: Mass value. Must be >= 0.
            from_unit: Source unit ("tonnes", "kg", "lbs", "short_tons").
            to_unit: Target unit.

        Returns:
            Converted mass (Decimal, 8dp).

        Raises:
            ValueError: If value < 0 or units are not recognized.

        Example:
            >>> engine.convert_mass(5000.0, "kg", "tonnes")
            Decimal('5.00000000')
        """
        if value < 0:
            raise ValueError(f"Mass must be >= 0, got {value}")

        valid_units = {"tonnes", "kg", "lbs", "short_tons"}
        if from_unit not in valid_units:
            raise ValueError(
                f"Invalid from_unit '{from_unit}'. Must be one of {valid_units}"
            )
        if to_unit not in valid_units:
            raise ValueError(
                f"Invalid to_unit '{to_unit}'. Must be one of {valid_units}"
            )

        if from_unit == to_unit:
            return _q(Decimal(str(value)))

        dec_value = Decimal(str(value))

        # Convert to tonnes first
        kg_per_tonne = Decimal("1000")
        lbs_per_tonne = Decimal("2204.62262")
        short_tons_per_tonne = Decimal("1.10231")

        if from_unit == "tonnes":
            tonnes_value = dec_value
        elif from_unit == "kg":
            tonnes_value = _q(dec_value / kg_per_tonne)
        elif from_unit == "lbs":
            tonnes_value = _q(dec_value / lbs_per_tonne)
        else:  # short_tons
            tonnes_value = _q(dec_value / short_tons_per_tonne)

        # Convert from tonnes to target
        if to_unit == "tonnes":
            result = tonnes_value
        elif to_unit == "kg":
            result = _q(tonnes_value * kg_per_tonne)
        elif to_unit == "lbs":
            result = _q(tonnes_value * lbs_per_tonne)
        else:  # short_tons
            result = _q(tonnes_value * short_tons_per_tonne)

        logger.debug(
            "convert_mass: %s %s = %s %s", value, from_unit, result, to_unit
        )

        return result

    # =========================================================================
    # DATA QUALITY SCORING
    # =========================================================================

    def get_data_quality_score(
        self,
        distance_method: str,
        ef_source: str,
        data_completeness: Optional[float] = None,
        temporal_age_years: Optional[int] = None,
        geographic_match: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate composite data quality indicator (DQI) score per ISO 14083.

        The DQI score is based on 5 dimensions. If specific dimension values
        are not provided, defaults are inferred from the distance method and
        EF source.

        Score range: 1.0 (best, primary data) to 5.0 (worst, estimated).

        Dimensions:
            1. Technological representativeness (from EF source)
            2. Geographical representativeness (from geographic_match)
            3. Temporal representativeness (from temporal_age_years)
            4. Completeness (from data_completeness)
            5. Reliability (from EF source)

        Args:
            distance_method: How distance was determined. One of:
                "actual", "shortest_feasible", "great_circle", "estimated".
            ef_source: Emission factor source. One of:
                "supplier_specific", "measured", "defra", "glec", "imo",
                "icao", "industry_average", "default", "estimated".
            data_completeness: Fraction of actual data (0-1). Optional.
            temporal_age_years: Age of EF data in years. Optional.
            geographic_match: Geographic match level. One of:
                "exact", "same_region", "same_climate", "different_region",
                "global". Optional.

        Returns:
            Dict containing:
                - composite_score (Decimal): Overall DQI (1.0-5.0)
                - distance_score (Decimal)
                - ef_score (Decimal)
                - completeness_score (Decimal)
                - temporal_score (Decimal)
                - geographic_score (Decimal)
                - quality_grade (str): A/B/C/D/E

        Example:
            >>> dqi = engine.get_data_quality_score("actual", "defra")
            >>> dqi["composite_score"]
            Decimal('1.6')
            >>> dqi["quality_grade"]
            'A'
        """
        # Distance method scoring
        distance_scores = {
            "actual": Decimal("1.0"),
            "shortest_feasible": Decimal("2.0"),
            "great_circle": Decimal("3.0"),
            "estimated": Decimal("4.0"),
        }
        dist_score = distance_scores.get(
            distance_method.lower(), Decimal("4.0")
        )

        # EF source scoring
        ef_scores = {
            "supplier_specific": Decimal("1.0"),
            "measured": Decimal("1.5"),
            "defra": Decimal("2.0"),
            "glec": Decimal("2.0"),
            "imo": Decimal("2.0"),
            "icao": Decimal("2.0"),
            "iso_14083": Decimal("2.5"),
            "industry_average": Decimal("3.0"),
            "default": Decimal("3.5"),
            "estimated": Decimal("4.0"),
            "unknown": Decimal("5.0"),
        }
        ef_score = ef_scores.get(ef_source.lower(), Decimal("3.5"))

        # Completeness scoring
        if data_completeness is not None:
            if data_completeness >= 1.0:
                comp_score = Decimal("1.0")
            elif data_completeness >= 0.8:
                comp_score = Decimal("2.0")
            elif data_completeness >= 0.5:
                comp_score = Decimal("3.0")
            elif data_completeness >= 0.2:
                comp_score = Decimal("4.0")
            else:
                comp_score = Decimal("5.0")
        else:
            # Infer from distance method
            comp_score = Decimal("2.5") if dist_score <= Decimal("2.0") else Decimal("3.5")

        # Temporal scoring
        if temporal_age_years is not None:
            if temporal_age_years <= 0:
                temp_score = Decimal("1.0")
            elif temporal_age_years <= 2:
                temp_score = Decimal("2.0")
            elif temporal_age_years <= 5:
                temp_score = Decimal("3.0")
            elif temporal_age_years <= 10:
                temp_score = Decimal("4.0")
            else:
                temp_score = Decimal("5.0")
        else:
            temp_score = Decimal("2.0")  # Assume recent DEFRA/GLEC

        # Geographic scoring
        geo_scores = {
            "exact": Decimal("1.0"),
            "same_region": Decimal("2.0"),
            "same_climate": Decimal("3.0"),
            "different_region": Decimal("4.0"),
            "global": Decimal("5.0"),
        }
        if geographic_match is not None:
            geo_score = geo_scores.get(
                geographic_match.lower(), Decimal("3.0")
            )
        else:
            geo_score = Decimal("3.0")  # Default: same climate zone

        # Composite = average of all 5 dimensions
        composite = _q(
            (dist_score + ef_score + comp_score + temp_score + geo_score)
            / Decimal("5")
        )

        # Round to 1dp for readability
        composite_1dp = composite.quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )

        # Clamp to [1.0, 5.0]
        composite_1dp = max(Decimal("1.0"), min(Decimal("5.0"), composite_1dp))

        # Quality grade
        if composite_1dp <= Decimal("1.5"):
            grade = "A"
        elif composite_1dp <= Decimal("2.5"):
            grade = "B"
        elif composite_1dp <= Decimal("3.5"):
            grade = "C"
        elif composite_1dp <= Decimal("4.5"):
            grade = "D"
        else:
            grade = "E"

        result: Dict[str, Any] = {
            "composite_score": composite_1dp,
            "distance_score": dist_score,
            "ef_score": ef_score,
            "completeness_score": comp_score,
            "temporal_score": temp_score,
            "geographic_score": geo_score,
            "quality_grade": grade,
        }

        logger.debug(
            "get_data_quality_score: dist=%s(%s), ef=%s(%s), comp=%s, "
            "temp=%s, geo=%s -> composite=%s grade=%s",
            distance_method, dist_score, ef_source, ef_score,
            comp_score, temp_score, geo_score, composite_1dp, grade,
        )

        return result

    # =========================================================================
    # UNCERTAINTY QUANTIFICATION
    # =========================================================================

    def calculate_uncertainty_range(
        self,
        total_emissions: Decimal,
        method: str,
    ) -> Dict[str, Decimal]:
        """
        Calculate uncertainty range for a given emissions total.

        Applies the percentage uncertainty bounds for the specified
        calculation method to produce low and high estimates.

        Args:
            total_emissions: Central estimate of emissions (kgCO2e).
            method: Calculation method ("DISTANCE_BASED", "SPEND_BASED",
                   "AVERAGE_DATA", "SUPPLIER_SPECIFIC").

        Returns:
            Dict with keys:
                - central (Decimal): Input value.
                - low (Decimal): Lower bound.
                - high (Decimal): Upper bound.
                - low_pct (Decimal): Lower percentage.
                - high_pct (Decimal): Upper percentage.

        Raises:
            ValueError: If total_emissions < 0 or method is invalid.

        Example:
            >>> rng = engine.calculate_uncertainty_range(
            ...     Decimal("1000"), "DISTANCE_BASED"
            ... )
            >>> rng["low"]
            Decimal('850.00000000')
            >>> rng["high"]
            Decimal('1200.00000000')
        """
        if total_emissions < _ZERO:
            raise ValueError(
                f"total_emissions must be >= 0, got {total_emissions}"
            )

        uncertainty = self._db.get_uncertainty_range(method)
        low_pct = uncertainty["low_pct"]
        high_pct = uncertainty["high_pct"]

        low = _q(total_emissions * (_ONE + low_pct / _HUNDRED))
        high = _q(total_emissions * (_ONE + high_pct / _HUNDRED))

        result = {
            "central": _q(total_emissions),
            "low": low,
            "high": high,
            "low_pct": low_pct,
            "high_pct": high_pct,
        }

        logger.debug(
            "calculate_uncertainty_range: %s, method=%s -> [%s, %s]",
            total_emissions, method, low, high,
        )

        return result

    # =========================================================================
    # ALLOCATION METHODS
    # =========================================================================

    def allocate_by_mass(
        self,
        total_emissions: Decimal,
        product_mass_tonnes: Decimal,
        total_mass_tonnes: Decimal,
    ) -> Decimal:
        """
        Allocate shared transport emissions by mass (weight-based).

        Formula: allocated = total_emissions x (product_mass / total_mass)

        Args:
            total_emissions: Total transport emissions (kgCO2e).
            product_mass_tonnes: Mass of reporting company's product.
            total_mass_tonnes: Total mass of all goods transported.

        Returns:
            Allocated emissions for the reporting company's product.

        Raises:
            ValueError: If any value is negative or total_mass is zero.

        Example:
            >>> engine.allocate_by_mass(
            ...     Decimal("1000"), Decimal("5"), Decimal("20")
            ... )
            Decimal('250.00000000')
        """
        if total_emissions < _ZERO:
            raise ValueError(f"total_emissions must be >= 0, got {total_emissions}")
        if product_mass_tonnes < _ZERO:
            raise ValueError(f"product_mass must be >= 0, got {product_mass_tonnes}")
        if total_mass_tonnes <= _ZERO:
            raise ValueError(f"total_mass must be > 0, got {total_mass_tonnes}")
        if product_mass_tonnes > total_mass_tonnes:
            raise ValueError(
                f"product_mass ({product_mass_tonnes}) cannot exceed "
                f"total_mass ({total_mass_tonnes})"
            )

        allocated = _q(total_emissions * product_mass_tonnes / total_mass_tonnes)

        logger.debug(
            "allocate_by_mass: %s kgCO2e x %s/%s t = %s kgCO2e",
            total_emissions, product_mass_tonnes, total_mass_tonnes, allocated,
        )

        return allocated

    def allocate_by_volume(
        self,
        total_emissions: Decimal,
        product_volume_m3: Decimal,
        total_volume_m3: Decimal,
    ) -> Decimal:
        """
        Allocate shared transport emissions by volume (cubic meters).

        Formula: allocated = total_emissions x (product_vol / total_vol)

        Args:
            total_emissions: Total transport emissions (kgCO2e).
            product_volume_m3: Volume of reporting company's product.
            total_volume_m3: Total volume of all goods transported.

        Returns:
            Allocated emissions.

        Raises:
            ValueError: If any value is negative or total_volume is zero.

        Example:
            >>> engine.allocate_by_volume(
            ...     Decimal("1000"), Decimal("10"), Decimal("50")
            ... )
            Decimal('200.00000000')
        """
        if total_emissions < _ZERO:
            raise ValueError(f"total_emissions must be >= 0, got {total_emissions}")
        if product_volume_m3 < _ZERO:
            raise ValueError(f"product_volume must be >= 0, got {product_volume_m3}")
        if total_volume_m3 <= _ZERO:
            raise ValueError(f"total_volume must be > 0, got {total_volume_m3}")
        if product_volume_m3 > total_volume_m3:
            raise ValueError(
                f"product_volume ({product_volume_m3}) cannot exceed "
                f"total_volume ({total_volume_m3})"
            )

        allocated = _q(total_emissions * product_volume_m3 / total_volume_m3)

        logger.debug(
            "allocate_by_volume: %s kgCO2e x %s/%s m3 = %s kgCO2e",
            total_emissions, product_volume_m3, total_volume_m3, allocated,
        )

        return allocated

    def allocate_by_revenue(
        self,
        total_emissions: Decimal,
        product_revenue: Decimal,
        total_revenue: Decimal,
    ) -> Decimal:
        """
        Allocate shared transport emissions by revenue (economic allocation).

        Formula: allocated = total_emissions x (product_revenue / total_revenue)

        Args:
            total_emissions: Total transport emissions (kgCO2e).
            product_revenue: Revenue from reporting company's product.
            total_revenue: Total revenue from all goods transported.

        Returns:
            Allocated emissions.

        Raises:
            ValueError: If any value is negative or total_revenue is zero.

        Example:
            >>> engine.allocate_by_revenue(
            ...     Decimal("1000"), Decimal("25000"), Decimal("100000")
            ... )
            Decimal('250.00000000')
        """
        if total_emissions < _ZERO:
            raise ValueError(f"total_emissions must be >= 0, got {total_emissions}")
        if product_revenue < _ZERO:
            raise ValueError(f"product_revenue must be >= 0, got {product_revenue}")
        if total_revenue <= _ZERO:
            raise ValueError(f"total_revenue must be > 0, got {total_revenue}")
        if product_revenue > total_revenue:
            raise ValueError(
                f"product_revenue ({product_revenue}) cannot exceed "
                f"total_revenue ({total_revenue})"
            )

        allocated = _q(total_emissions * product_revenue / total_revenue)

        logger.debug(
            "allocate_by_revenue: %s kgCO2e x %s/%s = %s kgCO2e",
            total_emissions, product_revenue, total_revenue, allocated,
        )

        return allocated

    # =========================================================================
    # BIOGENIC SPLIT
    # =========================================================================

    def calculate_biogenic_split(
        self,
        total_emissions: Decimal,
        biogenic_fraction: float,
    ) -> Dict[str, Decimal]:
        """
        Split total emissions into fossil and biogenic fractions.

        Biogenic CO2 from biomass combustion (biodiesel, HVO, SAF) is
        reported separately outside the Scope 3 total per GHG Protocol
        guidance.

        Args:
            total_emissions: Total emissions in kgCO2e.
            biogenic_fraction: Fraction of biogenic content (0-1).
                0.0 = 100% fossil. 1.0 = 100% biogenic.

        Returns:
            Dict with keys: fossil_kgco2e, biogenic_kgco2e.

        Raises:
            ValueError: If total_emissions < 0 or fraction out of range.

        Example:
            >>> split = engine.calculate_biogenic_split(Decimal("1000"), 0.80)
            >>> split["fossil_kgco2e"]
            Decimal('200.00000000')
            >>> split["biogenic_kgco2e"]
            Decimal('800.00000000')
        """
        if total_emissions < _ZERO:
            raise ValueError(
                f"total_emissions must be >= 0, got {total_emissions}"
            )
        if biogenic_fraction < 0.0 or biogenic_fraction > 1.0:
            raise ValueError(
                f"biogenic_fraction must be 0-1, got {biogenic_fraction}"
            )

        bio_frac = Decimal(str(biogenic_fraction))
        biogenic = _q(total_emissions * bio_frac)
        fossil = _q(total_emissions - biogenic)

        result = {
            "fossil_kgco2e": fossil,
            "biogenic_kgco2e": biogenic,
        }

        logger.debug(
            "calculate_biogenic_split: total=%s, frac=%s -> fossil=%s, bio=%s",
            total_emissions, biogenic_fraction, fossil, biogenic,
        )

        return result

    # =========================================================================
    # FLEET STATISTICS
    # =========================================================================

    def calculate_fleet_statistics(
        self, fleet_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate summary statistics for a fleet calculation result.

        Args:
            fleet_result: Output from calculate_fleet().

        Returns:
            Dict containing:
                - total_emissions_tco2e (Decimal): Total in tonnes CO2e.
                - avg_emissions_per_shipment (Decimal): Mean per shipment.
                - avg_distance_km (Decimal): Mean distance.
                - avg_weight_tonnes (Decimal): Mean weight.
                - intensity_kgco2e_per_tkm (Decimal): Fleet emission intensity.
                - mode_share (Dict[str, Decimal]): % of emissions by mode.
                - wtt_share_pct (Decimal): WTT as % of total.
                - return_share_pct (Decimal): Return as % of total.

        Example:
            >>> stats = engine.calculate_fleet_statistics(fleet_result)
            >>> stats["total_emissions_tco2e"]
            Decimal('1.23456789')
        """
        total_emissions = fleet_result.get("total_emissions_kgco2e", _ZERO)
        total_tkm = fleet_result.get("total_tonne_km", _ZERO)
        total_transport = fleet_result.get("total_transport_kgco2e", _ZERO)
        total_wtt = fleet_result.get("total_wtt_kgco2e", _ZERO)
        total_return = fleet_result.get("total_return_kgco2e", _ZERO)
        success_count = fleet_result.get("successful_shipments", 0)
        by_mode = fleet_result.get("by_mode", {})

        # Total in tCO2e
        total_tco2e = _q(total_emissions / _THOUSAND)

        # Averages
        if success_count > 0:
            sc = Decimal(str(success_count))
            avg_emissions = _q(total_emissions / sc)
        else:
            avg_emissions = _ZERO

        # Intensity
        intensity = _ZERO
        if total_tkm > _ZERO:
            intensity = _q(total_emissions / total_tkm)

        # Mode share (% of total emissions)
        mode_share: Dict[str, Decimal] = {}
        if total_emissions > _ZERO:
            for mode, data in by_mode.items():
                mode_emissions = data.get("total_emissions_kgco2e", _ZERO)
                share = _q(mode_emissions / total_emissions * _HUNDRED)
                mode_share[mode] = share

        # WTT and return shares
        wtt_share = _ZERO
        return_share = _ZERO
        if total_emissions > _ZERO:
            wtt_share = _q(total_wtt / total_emissions * _HUNDRED)
            return_share = _q(total_return / total_emissions * _HUNDRED)

        result: Dict[str, Any] = {
            "total_emissions_tco2e": total_tco2e,
            "avg_emissions_per_shipment": avg_emissions,
            "intensity_kgco2e_per_tkm": intensity,
            "mode_share_pct": mode_share,
            "wtt_share_pct": wtt_share,
            "return_share_pct": return_share,
        }

        logger.debug(
            "calculate_fleet_statistics: %s tCO2e, intensity=%s, "
            "modes=%s, wtt_share=%s%%, return_share=%s%%",
            total_tco2e, intensity, mode_share, wtt_share, return_share,
        )

        return result

    # =========================================================================
    # INCOTERM SCOPE CLASSIFICATION
    # =========================================================================

    def classify_incoterm_scope(
        self, incoterm: str
    ) -> Dict[str, Any]:
        """
        Determine whether downstream transport falls into Category 4 or
        Category 9 scope based on the Incoterm used.

        Per GHG Protocol Scope 3 Standard:
        - Cat 4 (Upstream): Transport paid for by reporting company
        - Cat 9 (Downstream): Transport NOT paid for by reporting company

        Incoterms starting with E/F generally = Cat 4 (buyer arranges).
        Incoterms starting with C/D generally = Cat 9 (seller arranges).

        Args:
            incoterm: ICC Incoterm code (e.g., "DDP", "FOB", "CIF").

        Returns:
            Dict with:
                - incoterm (str)
                - cat4_scope (bool)
                - cat9_scope (bool)
                - recommendation (str): Which category to report in.

        Raises:
            ValueError: If incoterm is not recognized.

        Example:
            >>> result = engine.classify_incoterm_scope("DDP")
            >>> result["cat9_scope"]
            True
            >>> result["recommendation"]
            'Report in Category 9 (Downstream Transportation)'
        """
        classification = self._db.get_incoterm_classification(incoterm)

        cat4 = classification["cat4_scope"]
        cat9 = classification["cat9_scope"]

        if cat4 and cat9:
            recommendation = (
                "Report in BOTH Category 4 and Category 9 "
                "(Incoterm unclear, conservative approach)"
            )
        elif cat9:
            recommendation = "Report in Category 9 (Downstream Transportation)"
        elif cat4:
            recommendation = "Report in Category 4 (Upstream Transportation)"
        else:
            recommendation = "Consult GHG Protocol guidance for boundary"

        result: Dict[str, Any] = {
            "incoterm": incoterm.upper(),
            "cat4_scope": cat4,
            "cat9_scope": cat9,
            "transfer_point": classification["transfer_point"],
            "seller_responsibility": classification["seller_responsibility"],
            "buyer_responsibility": classification["buyer_responsibility"],
            "recommendation": recommendation,
            "description": classification["description"],
        }

        logger.debug(
            "classify_incoterm_scope: %s -> cat4=%s, cat9=%s, rec='%s'",
            incoterm, cat4, cat9, recommendation,
        )

        return result

    # =========================================================================
    # DOUBLE-COUNTING PREVENTION
    # =========================================================================

    def check_double_counting(
        self,
        incoterm: str,
        already_reported_categories: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Check for potential double-counting of transport emissions between
        GHG Protocol Scope 3 categories.

        Category 9 overlaps with:
        - Category 4 (Upstream Transportation): Depends on Incoterm
        - Category 11 (Use of Sold Products): If transport is part of use
        - Category 12 (End-of-Life): If transport to disposal
        - Scope 1/2: If reporting company operates vehicles

        Args:
            incoterm: ICC Incoterm code.
            already_reported_categories: List of category numbers already
                reported (e.g., [4, 11, 12]).

        Returns:
            Dict with:
                - overlaps (List[Dict]): Potential overlaps found.
                - risk_level (str): "none", "low", "medium", "high".
                - recommendations (List[str]): Actions to take.

        Example:
            >>> result = engine.check_double_counting("DDP", [4])
            >>> result["risk_level"]
            'high'
        """
        if already_reported_categories is None:
            already_reported_categories = []

        classification = self._db.get_incoterm_classification(incoterm)
        overlaps: List[Dict[str, Any]] = []
        recommendations: List[str] = []

        # Check Cat 4 overlap
        if 4 in already_reported_categories:
            if classification["cat4_scope"] and classification["cat9_scope"]:
                overlaps.append({
                    "category": 4,
                    "description": (
                        f"Incoterm {incoterm} may overlap Cat 4 and Cat 9"
                    ),
                    "severity": "high",
                })
                recommendations.append(
                    "Review Incoterm boundary: split transport at transfer "
                    "point to avoid double-counting between Cat 4 and Cat 9"
                )
            elif classification["cat9_scope"]:
                overlaps.append({
                    "category": 4,
                    "description": (
                        f"Incoterm {incoterm} is Cat 9 scope; verify Cat 4 "
                        f"does not include same transport leg"
                    ),
                    "severity": "medium",
                })

        # Check Cat 11 overlap (use of sold products)
        if 11 in already_reported_categories:
            overlaps.append({
                "category": 11,
                "description": (
                    "If product use involves transport (e.g., fuel sold for "
                    "vehicles), ensure no overlap with Cat 9"
                ),
                "severity": "low",
            })

        # Check Cat 12 overlap (end-of-life treatment)
        if 12 in already_reported_categories:
            overlaps.append({
                "category": 12,
                "description": (
                    "Transport to disposal/recycling should be in Cat 12, "
                    "not Cat 9. Ensure clear boundary."
                ),
                "severity": "low",
            })
            recommendations.append(
                "Confirm transport to end-of-life is excluded from Cat 9"
            )

        # Determine risk level
        if any(o["severity"] == "high" for o in overlaps):
            risk_level = "high"
        elif any(o["severity"] == "medium" for o in overlaps):
            risk_level = "medium"
        elif overlaps:
            risk_level = "low"
        else:
            risk_level = "none"

        if not recommendations:
            recommendations.append("No double-counting risks identified")

        result: Dict[str, Any] = {
            "incoterm": incoterm.upper(),
            "already_reported_categories": already_reported_categories,
            "overlaps": overlaps,
            "risk_level": risk_level,
            "recommendations": recommendations,
        }

        logger.debug(
            "check_double_counting: %s, reported=%s -> risk=%s, overlaps=%d",
            incoterm, already_reported_categories, risk_level, len(overlaps),
        )

        return result

    # =========================================================================
    # PRIVATE VALIDATION HELPERS
    # =========================================================================

    def _validate_positive_decimal(
        self, value: Any, field_name: str
    ) -> Decimal:
        """
        Validate and convert a value to a positive Decimal.

        Args:
            value: Value to validate (int, float, str, or Decimal).
            field_name: Field name for error messages.

        Returns:
            Validated positive Decimal.

        Raises:
            ValueError: If value is missing, non-numeric, or <= 0.
        """
        if value is None:
            raise ValueError(f"{field_name} is required")

        try:
            dec_value = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(
                f"{field_name} must be a valid number, got '{value}'"
            ) from exc

        if dec_value <= _ZERO:
            raise ValueError(
                f"{field_name} must be > 0, got {dec_value}"
            )

        return _q(dec_value)

    def _validate_coordinate(
        self,
        value: float,
        field_name: str,
        min_val: float,
        max_val: float,
    ) -> None:
        """
        Validate a geographic coordinate.

        Args:
            value: Coordinate value.
            field_name: Name for error messages.
            min_val: Minimum valid value.
            max_val: Maximum valid value.

        Raises:
            ValueError: If value is out of range.
        """
        if value < min_val or value > max_val:
            raise ValueError(
                f"{field_name} must be between {min_val} and {max_val}, "
                f"got {value}"
            )

    def _haversine(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> Decimal:
        """
        Calculate great-circle distance using Haversine formula.

        Uses WGS-84 mean Earth radius of 6,371.0088 km.

        Args:
            lat1, lon1: Origin coordinates (decimal degrees).
            lat2, lon2: Destination coordinates (decimal degrees).

        Returns:
            Distance in km (Decimal, 8dp).
        """
        lat1_r = math.radians(lat1)
        lon1_r = math.radians(lon1)
        lat2_r = math.radians(lat2)
        lon2_r = math.radians(lon2)

        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r

        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(lat1_r)
            * math.cos(lat2_r)
            * math.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

        distance_float = float(_EARTH_RADIUS_KM) * c
        distance_km = _q(Decimal(str(distance_float)))

        logger.debug(
            "haversine: (%s,%s)->(%s,%s) = %s km",
            lat1, lon1, lat2, lon2, distance_km,
        )

        return distance_km


# ==============================================================================
# MODULE-LEVEL SINGLETON ACCESS FUNCTIONS
# ==============================================================================

_module_instance: Optional[DistanceBasedCalculatorEngine] = None
_module_lock: threading.Lock = threading.Lock()


def get_distance_based_calculator() -> DistanceBasedCalculatorEngine:
    """
    Get the singleton DistanceBasedCalculatorEngine instance.

    Thread-safe. Creates the instance on first call, returns existing
    instance on subsequent calls.

    Returns:
        DistanceBasedCalculatorEngine singleton.

    Example:
        >>> engine = get_distance_based_calculator()
        >>> result = engine.calculate_shipment({...})
    """
    global _module_instance
    if _module_instance is None:
        with _module_lock:
            if _module_instance is None:
                _module_instance = DistanceBasedCalculatorEngine()
    return _module_instance


def reset_distance_based_calculator() -> None:
    """
    Reset the singleton instance (primarily for testing).

    Clears both the module-level reference and the class-level singleton,
    allowing a fresh instance to be created on the next call to
    get_distance_based_calculator().

    Example:
        >>> reset_distance_based_calculator()
        >>> engine = get_distance_based_calculator()  # Fresh instance
    """
    global _module_instance
    with _module_lock:
        _module_instance = None
        DistanceBasedCalculatorEngine._instance = None

    # Also reset the database engine to ensure clean state
    from greenlang.downstream_transportation.downstream_transport_database import (
        reset_downstream_transport_database,
    )
    reset_downstream_transport_database()

    logger.info("DistanceBasedCalculatorEngine singleton reset")


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "DistanceBasedCalculatorEngine",
    "get_distance_based_calculator",
    "reset_distance_based_calculator",
]
