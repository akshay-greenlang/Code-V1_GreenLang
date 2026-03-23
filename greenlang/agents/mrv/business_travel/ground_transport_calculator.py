# -*- coding: utf-8 -*-
"""
GroundTransportCalculatorEngine - Engine 3: Business Travel Agent (AGENT-MRV-019)

Core calculation engine for ground transport emissions covering rail, road
(distance-based and fuel-based), taxi, bus, ferry, and motorcycle modes.

This engine implements deterministic Decimal-based emissions calculations
for all non-aviation transport modes used in business travel, following
DEFRA 2024 emission factors and the GHG Protocol Scope 3 Category 6
methodology.

Primary Formulae:
    Rail / Bus / Ferry (per-pkm):
        co2e      = distance_km x passengers x ttw_ef_per_pkm
        wtt_co2e  = distance_km x passengers x wtt_ef_per_pkm
        total     = co2e + wtt_co2e

    Road / Taxi / Motorcycle (per-vkm):
        co2e      = distance_km x ef_per_vkm
        wtt_co2e  = distance_km x wtt_per_vkm
        total     = co2e + wtt_co2e

    Road Fuel-Based:
        co2e      = litres x ef_per_litre
        wtt_co2e  = litres x wtt_per_litre
        total     = co2e + wtt_co2e

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the calculation path
    - Every intermediate value is deterministic and reproducible
    - SHA-256 provenance hash on every result
    - Emission factors sourced from DEFRA 2024 conversion factors

Supports:
    - 8 rail types (national, international, light rail, underground,
      Eurostar, high-speed, US intercity, US commuter)
    - 13 road vehicle types (average, small/medium/large petrol/diesel,
      hybrid, plug-in hybrid, BEV, taxi regular/black cab, motorcycle)
    - 5 fuel types (petrol, diesel, LPG, CNG, E85)
    - 2 bus types (local, coach)
    - 2 ferry types (foot passenger, car passenger)
    - Miles-to-km and gallons-to-litres unit conversion
    - Batch processing for multiple ground trips
    - Input validation with detailed error messages
    - Provenance hash integration for audit trails
    - Prometheus metrics integration

Example:
    >>> from greenlang.agents.mrv.business_travel.ground_transport_calculator import (
    ...     GroundTransportCalculatorEngine,
    ... )
    >>> from greenlang.agents.mrv.business_travel.models import (
    ...     RailInput, RailType, RoadDistanceInput, RoadVehicleType,
    ... )
    >>> from decimal import Decimal
    >>> engine = GroundTransportCalculatorEngine.get_instance()
    >>> rail_input = RailInput(
    ...     rail_type=RailType.EUROSTAR,
    ...     distance_km=Decimal("340"),
    ...     passengers=2,
    ... )
    >>> result = engine.calculate_rail(rail_input)
    >>> assert result.total_co2e > Decimal("0")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-019 Business Travel (GL-MRV-S3-006)
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Union

from greenlang.agents.mrv.business_travel.models import (
    AGENT_COMPONENT,
    AGENT_ID,
    VERSION,
    RailInput,
    RailResult,
    RoadDistanceInput,
    RoadDistanceResult,
    RoadFuelInput,
    RoadFuelResult,
    TaxiInput,
    BusInput,
    FerryInput,
    RailType,
    RoadVehicleType,
    FuelType,
    BusType,
    FerryType,
    EFSource,
    GWPVersion,
    TransportMode,
    RAIL_EMISSION_FACTORS,
    ROAD_VEHICLE_EMISSION_FACTORS,
    FUEL_EMISSION_FACTORS,
    BUS_EMISSION_FACTORS,
    FERRY_EMISSION_FACTORS,
    calculate_provenance_hash,
)
from greenlang.agents.mrv.business_travel.metrics import BusinessTravelMetrics, get_metrics
from greenlang.agents.mrv.business_travel.config import get_config
from greenlang.agents.mrv.business_travel.provenance import get_provenance_tracker

logger = logging.getLogger(__name__)

# ==============================================================================
# DECIMAL PRECISION & CONSTANTS
# ==============================================================================

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")

# Unit conversion constants
_MILES_TO_KM = Decimal("1.60934")
_GALLONS_TO_LITRES = Decimal("3.78541")

# Batch processing limits
_MAX_BATCH_SIZE = 10000

# ==============================================================================
# SINGLETON INSTANCE MANAGEMENT
# ==============================================================================

_instance: Optional["GroundTransportCalculatorEngine"] = None
_instance_lock: threading.Lock = threading.Lock()


# ==============================================================================
# HELPER: Quantize a Decimal to 8 decimal places
# ==============================================================================

def _q(value: Decimal) -> Decimal:
    """
    Quantize a Decimal value to 8 decimal places with ROUND_HALF_UP.

    Args:
        value: The Decimal value to quantize.

    Returns:
        Quantized Decimal with exactly 8 decimal places.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


# ==============================================================================
# GroundTransportCalculatorEngine
# ==============================================================================


class GroundTransportCalculatorEngine:
    """
    Engine 3: Ground transport emissions calculator for all non-aviation modes.

    Implements deterministic emissions calculations for rail, road (distance
    and fuel-based), taxi, bus, ferry, and motorcycle transport using DEFRA
    2024 emission factors.

    The engine follows GreenLang's zero-hallucination principle by using only
    deterministic Decimal arithmetic with DEFRA/IPCC-sourced parameters. No
    LLM calls are made anywhere in the calculation pipeline.

    Thread Safety:
        This engine is fully thread-safe. A reentrant lock protects shared
        state during calculations. The singleton instance is created lazily
        with double-checked locking.

    Attributes:
        _config: Business travel configuration singleton.
        _metrics: Prometheus metrics collector for monitoring.
        _provenance: SHA-256 provenance tracker for audit trails.
        _lock: Reentrant lock for thread safety.
        _calculation_count: Running count of calculations performed.

    Example:
        >>> engine = GroundTransportCalculatorEngine.get_instance()
        >>> result = engine.calculate_rail(rail_input)
        >>> assert result.total_co2e > Decimal("0")
    """

    # ------------------------------------------------------------------
    # Singleton Access
    # ------------------------------------------------------------------

    @staticmethod
    def get_instance(
        metrics: Optional[BusinessTravelMetrics] = None,
    ) -> "GroundTransportCalculatorEngine":
        """
        Get or create the singleton GroundTransportCalculatorEngine instance.

        Thread-safe lazy initialization using double-checked locking.

        Args:
            metrics: Optional Prometheus metrics collector. A default
                     instance is obtained via get_metrics() if None.

        Returns:
            Singleton GroundTransportCalculatorEngine instance.
        """
        global _instance
        if _instance is None:
            with _instance_lock:
                if _instance is None:
                    _instance = GroundTransportCalculatorEngine(
                        metrics=metrics,
                    )
        return _instance

    @staticmethod
    def reset_instance() -> None:
        """
        Reset the singleton instance (for testing only).

        This method is intended exclusively for unit tests that need
        a fresh engine instance. It should never be called in production.
        """
        global _instance
        with _instance_lock:
            _instance = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        metrics: Optional[BusinessTravelMetrics] = None,
    ) -> None:
        """
        Initialise the GroundTransportCalculatorEngine.

        Args:
            metrics: Optional Prometheus metrics collector. A default
                     instance is obtained via get_metrics() if None.
        """
        self._config = get_config()
        self._metrics: BusinessTravelMetrics = metrics or get_metrics()
        self._provenance = get_provenance_tracker()
        self._lock: threading.RLock = threading.RLock()
        self._calculation_count: int = 0

        logger.info(
            "GroundTransportCalculatorEngine initialised: agent=%s, version=%s",
            AGENT_ID,
            VERSION,
        )

    # ==================================================================
    # PROPERTY: calculation_count
    # ==================================================================

    @property
    def calculation_count(self) -> int:
        """Return the total number of calculations performed by this engine."""
        return self._calculation_count

    # ==================================================================
    # 1. calculate_rail - Rail emissions (per-pkm)
    # ==================================================================

    def calculate_rail(
        self,
        rail_input: RailInput,
    ) -> RailResult:
        """
        Calculate rail travel emissions using distance-based method.

        Formula:
            co2e     = distance_km x passengers x ttw_ef_per_pkm
            wtt_co2e = distance_km x passengers x wtt_ef_per_pkm
            total    = co2e + wtt_co2e

        Args:
            rail_input: Validated RailInput with rail_type, distance_km,
                        and passengers.

        Returns:
            RailResult with co2e, wtt_co2e, total_co2e, and provenance hash.

        Raises:
            ValueError: If distance_km <= 0 or passengers < 1.
            KeyError: If rail_type not found in RAIL_EMISSION_FACTORS.

        Example:
            >>> result = engine.calculate_rail(RailInput(
            ...     rail_type=RailType.EUROSTAR,
            ...     distance_km=Decimal("340"),
            ...     passengers=2,
            ... ))
            >>> assert result.total_co2e > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate input
            self._validate_rail_input(rail_input)

            # Step 2: Resolve emission factor
            rail_ef = self._resolve_rail_ef(rail_input.rail_type)
            ttw_ef = rail_ef["ttw"]
            wtt_ef = rail_ef["wtt"]

            # Step 3: Calculate emissions (ZERO HALLUCINATION - Decimal only)
            distance = rail_input.distance_km
            passengers = Decimal(str(rail_input.passengers))

            co2e = _q(distance * passengers * ttw_ef)
            wtt_co2e = _q(distance * passengers * wtt_ef)
            total_co2e = _q(co2e + wtt_co2e)

            # Step 4: Provenance hash
            provenance_hash = calculate_provenance_hash(
                rail_input,
                co2e,
                wtt_co2e,
                total_co2e,
                "rail",
                AGENT_ID,
            )

            # Step 5: Build result
            result = RailResult(
                rail_type=rail_input.rail_type,
                distance_km=distance,
                passengers=rail_input.passengers,
                co2e=co2e,
                wtt_co2e=wtt_co2e,
                total_co2e=total_co2e,
                ef_source=EFSource.DEFRA,
                provenance_hash=provenance_hash,
            )

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._record_ground_metrics(
                mode="rail",
                vehicle_type=rail_input.rail_type.value,
                distance_km=float(distance),
                co2e=float(total_co2e),
                duration=duration,
            )
            self._calculation_count += 1

            logger.debug(
                "Rail calculation complete: type=%s, dist=%s km, pax=%d, "
                "co2e=%s kg, wtt=%s kg, total=%s kg",
                rail_input.rail_type.value,
                distance,
                rail_input.passengers,
                co2e,
                wtt_co2e,
                total_co2e,
            )

            return result

    def _validate_rail_input(self, rail_input: RailInput) -> None:
        """
        Validate rail input parameters.

        Args:
            rail_input: RailInput to validate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if rail_input.distance_km <= _ZERO:
            raise ValueError(
                f"Rail distance_km must be positive, got {rail_input.distance_km}"
            )
        if rail_input.passengers < 1:
            raise ValueError(
                f"Rail passengers must be >= 1, got {rail_input.passengers}"
            )
        if rail_input.rail_type not in RAIL_EMISSION_FACTORS:
            raise ValueError(
                f"Unknown rail_type '{rail_input.rail_type.value}'. "
                f"Available: {list(RAIL_EMISSION_FACTORS.keys())}"
            )

    def _resolve_rail_ef(self, rail_type: RailType) -> Dict[str, Decimal]:
        """
        Resolve rail emission factors for the given rail type.

        Args:
            rail_type: The RailType enum value.

        Returns:
            Dict with 'ttw' and 'wtt' Decimal emission factors (kgCO2e/pkm).

        Raises:
            KeyError: If rail_type is not in RAIL_EMISSION_FACTORS.
        """
        ef = RAIL_EMISSION_FACTORS.get(rail_type)
        if ef is None:
            raise KeyError(
                f"Rail emission factor not found for rail_type '{rail_type.value}'"
            )

        logger.debug(
            "Resolved rail EF: type=%s, ttw=%s, wtt=%s",
            rail_type.value,
            ef["ttw"],
            ef["wtt"],
        )
        return ef

    # ==================================================================
    # 2. calculate_road_distance - Road distance-based (per-vkm)
    # ==================================================================

    def calculate_road_distance(
        self,
        road_input: RoadDistanceInput,
    ) -> RoadDistanceResult:
        """
        Calculate road vehicle emissions using distance-based method.

        Uses per-vehicle-km emission factors (not per-pkm) since rental
        cars and personal vehicles are typically single-occupancy for
        business travel.

        Formula:
            co2e     = distance_km x ef_per_vkm
            wtt_co2e = distance_km x wtt_per_vkm
            total    = co2e + wtt_co2e

        Args:
            road_input: Validated RoadDistanceInput with vehicle_type
                        and distance_km.

        Returns:
            RoadDistanceResult with co2e, wtt_co2e, total_co2e, and
            provenance hash.

        Raises:
            ValueError: If distance_km <= 0.
            KeyError: If vehicle_type not found in ROAD_VEHICLE_EMISSION_FACTORS.

        Example:
            >>> result = engine.calculate_road_distance(RoadDistanceInput(
            ...     vehicle_type=RoadVehicleType.CAR_MEDIUM_PETROL,
            ...     distance_km=Decimal("250"),
            ... ))
            >>> assert result.total_co2e > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate input
            self._validate_road_distance_input(road_input)

            # Step 2: Resolve emission factor
            vehicle_ef = self._resolve_road_vehicle_ef(road_input.vehicle_type)
            ef_per_vkm = vehicle_ef["ef_per_vkm"]
            wtt_per_vkm = vehicle_ef["wtt_per_vkm"]

            # Step 3: Calculate emissions (ZERO HALLUCINATION - Decimal only)
            distance = road_input.distance_km

            co2e = _q(distance * ef_per_vkm)
            wtt_co2e = _q(distance * wtt_per_vkm)
            total_co2e = _q(co2e + wtt_co2e)

            # Step 4: Provenance hash
            provenance_hash = calculate_provenance_hash(
                road_input,
                co2e,
                wtt_co2e,
                total_co2e,
                "road_distance",
                AGENT_ID,
            )

            # Step 5: Build result
            result = RoadDistanceResult(
                vehicle_type=road_input.vehicle_type,
                distance_km=distance,
                co2e=co2e,
                wtt_co2e=wtt_co2e,
                total_co2e=total_co2e,
                ef_source=EFSource.DEFRA,
                provenance_hash=provenance_hash,
            )

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._record_ground_metrics(
                mode="road",
                vehicle_type=road_input.vehicle_type.value,
                distance_km=float(distance),
                co2e=float(total_co2e),
                duration=duration,
            )
            self._calculation_count += 1

            logger.debug(
                "Road distance calculation complete: type=%s, dist=%s km, "
                "co2e=%s kg, wtt=%s kg, total=%s kg",
                road_input.vehicle_type.value,
                distance,
                co2e,
                wtt_co2e,
                total_co2e,
            )

            return result

    def _validate_road_distance_input(
        self, road_input: RoadDistanceInput
    ) -> None:
        """
        Validate road distance input parameters.

        Args:
            road_input: RoadDistanceInput to validate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if road_input.distance_km <= _ZERO:
            raise ValueError(
                f"Road distance_km must be positive, got {road_input.distance_km}"
            )
        if road_input.vehicle_type not in ROAD_VEHICLE_EMISSION_FACTORS:
            raise ValueError(
                f"Unknown vehicle_type '{road_input.vehicle_type.value}'. "
                f"Available: {list(ROAD_VEHICLE_EMISSION_FACTORS.keys())}"
            )

    def _resolve_road_vehicle_ef(
        self, vehicle_type: RoadVehicleType
    ) -> Dict[str, Decimal]:
        """
        Resolve road vehicle emission factors for the given vehicle type.

        Args:
            vehicle_type: The RoadVehicleType enum value.

        Returns:
            Dict with ef_per_vkm, ef_per_pkm, wtt_per_vkm, occupancy
            as Decimal values.

        Raises:
            KeyError: If vehicle_type is not in ROAD_VEHICLE_EMISSION_FACTORS.
        """
        ef = ROAD_VEHICLE_EMISSION_FACTORS.get(vehicle_type)
        if ef is None:
            raise KeyError(
                f"Road vehicle EF not found for type '{vehicle_type.value}'"
            )

        logger.debug(
            "Resolved road vehicle EF: type=%s, ef_vkm=%s, wtt_vkm=%s",
            vehicle_type.value,
            ef["ef_per_vkm"],
            ef["wtt_per_vkm"],
        )
        return ef

    # ==================================================================
    # 3. calculate_road_fuel - Road fuel-based
    # ==================================================================

    def calculate_road_fuel(
        self,
        fuel_input: RoadFuelInput,
    ) -> RoadFuelResult:
        """
        Calculate road vehicle emissions using fuel-based method.

        Formula:
            co2e     = litres x ef_per_litre
            wtt_co2e = litres x wtt_per_litre
            total    = co2e + wtt_co2e

        For CNG, the unit is kg rather than litres but the factor table
        keys are still named *_per_litre for consistency.

        Args:
            fuel_input: Validated RoadFuelInput with fuel_type and litres.

        Returns:
            RoadFuelResult with co2e, wtt_co2e, total_co2e, and provenance
            hash.

        Raises:
            ValueError: If litres <= 0.
            KeyError: If fuel_type not found in FUEL_EMISSION_FACTORS.

        Example:
            >>> result = engine.calculate_road_fuel(RoadFuelInput(
            ...     fuel_type=FuelType.DIESEL,
            ...     litres=Decimal("45.0"),
            ... ))
            >>> assert result.total_co2e > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate input
            self._validate_road_fuel_input(fuel_input)

            # Step 2: Resolve emission factor
            fuel_ef = self._resolve_fuel_ef(fuel_input.fuel_type)
            ef_per_litre = fuel_ef["ef_per_litre"]
            wtt_per_litre = fuel_ef["wtt_per_litre"]

            # Step 3: Calculate emissions (ZERO HALLUCINATION - Decimal only)
            litres = fuel_input.litres

            co2e = _q(litres * ef_per_litre)
            wtt_co2e = _q(litres * wtt_per_litre)
            total_co2e = _q(co2e + wtt_co2e)

            # Step 4: Provenance hash
            provenance_hash = calculate_provenance_hash(
                fuel_input,
                co2e,
                wtt_co2e,
                total_co2e,
                "road_fuel",
                AGENT_ID,
            )

            # Step 5: Build result
            result = RoadFuelResult(
                fuel_type=fuel_input.fuel_type,
                litres=litres,
                co2e=co2e,
                wtt_co2e=wtt_co2e,
                total_co2e=total_co2e,
                ef_source=EFSource.DEFRA,
                provenance_hash=provenance_hash,
            )

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._record_ground_metrics(
                mode="road",
                vehicle_type=f"fuel_{fuel_input.fuel_type.value}",
                distance_km=0.0,
                co2e=float(total_co2e),
                duration=duration,
            )
            self._calculation_count += 1

            logger.debug(
                "Road fuel calculation complete: fuel=%s, litres=%s, "
                "co2e=%s kg, wtt=%s kg, total=%s kg",
                fuel_input.fuel_type.value,
                litres,
                co2e,
                wtt_co2e,
                total_co2e,
            )

            return result

    def _validate_road_fuel_input(self, fuel_input: RoadFuelInput) -> None:
        """
        Validate road fuel input parameters.

        Args:
            fuel_input: RoadFuelInput to validate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if fuel_input.litres <= _ZERO:
            raise ValueError(
                f"Fuel litres must be positive, got {fuel_input.litres}"
            )
        if fuel_input.fuel_type not in FUEL_EMISSION_FACTORS:
            raise ValueError(
                f"Unknown fuel_type '{fuel_input.fuel_type.value}'. "
                f"Available: {list(FUEL_EMISSION_FACTORS.keys())}"
            )

    def _resolve_fuel_ef(self, fuel_type: FuelType) -> Dict[str, Decimal]:
        """
        Resolve fuel emission factors for the given fuel type.

        Args:
            fuel_type: The FuelType enum value.

        Returns:
            Dict with ef_per_litre and wtt_per_litre Decimal values.

        Raises:
            KeyError: If fuel_type is not in FUEL_EMISSION_FACTORS.
        """
        ef = FUEL_EMISSION_FACTORS.get(fuel_type)
        if ef is None:
            raise KeyError(
                f"Fuel EF not found for type '{fuel_type.value}'"
            )

        logger.debug(
            "Resolved fuel EF: type=%s, ef=%s, wtt=%s",
            fuel_type.value,
            ef["ef_per_litre"],
            ef["wtt_per_litre"],
        )
        return ef

    # ==================================================================
    # 4. calculate_taxi - Taxi / ride-hailing (per-vkm)
    # ==================================================================

    def calculate_taxi(
        self,
        taxi_input: TaxiInput,
    ) -> RoadDistanceResult:
        """
        Calculate taxi / ride-hailing emissions using distance-based method.

        Uses per-vehicle-km emission factors from the ROAD_VEHICLE_EMISSION_FACTORS
        table for TAXI_REGULAR or TAXI_BLACK_CAB types.

        Formula:
            co2e     = distance_km x ef_per_vkm
            wtt_co2e = distance_km x wtt_per_vkm
            total    = co2e + wtt_co2e

        Args:
            taxi_input: Validated TaxiInput with taxi_type and distance_km.

        Returns:
            RoadDistanceResult with vehicle_type set to the taxi type,
            co2e, wtt_co2e, total_co2e, and provenance hash.

        Raises:
            ValueError: If distance_km <= 0.
            KeyError: If taxi_type not found in ROAD_VEHICLE_EMISSION_FACTORS.

        Example:
            >>> result = engine.calculate_taxi(TaxiInput(
            ...     taxi_type=RoadVehicleType.TAXI_REGULAR,
            ...     distance_km=Decimal("15.5"),
            ... ))
            >>> assert result.total_co2e > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate input
            self._validate_taxi_input(taxi_input)

            # Step 2: Resolve emission factor
            taxi_ef = self._resolve_road_vehicle_ef(taxi_input.taxi_type)
            ef_per_vkm = taxi_ef["ef_per_vkm"]
            wtt_per_vkm = taxi_ef["wtt_per_vkm"]

            # Step 3: Calculate emissions (ZERO HALLUCINATION - Decimal only)
            distance = taxi_input.distance_km

            co2e = _q(distance * ef_per_vkm)
            wtt_co2e = _q(distance * wtt_per_vkm)
            total_co2e = _q(co2e + wtt_co2e)

            # Step 4: Provenance hash
            provenance_hash = calculate_provenance_hash(
                taxi_input,
                co2e,
                wtt_co2e,
                total_co2e,
                "taxi",
                AGENT_ID,
            )

            # Step 5: Build result
            result = RoadDistanceResult(
                vehicle_type=taxi_input.taxi_type,
                distance_km=distance,
                co2e=co2e,
                wtt_co2e=wtt_co2e,
                total_co2e=total_co2e,
                ef_source=EFSource.DEFRA,
                provenance_hash=provenance_hash,
            )

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._record_ground_metrics(
                mode="taxi",
                vehicle_type=taxi_input.taxi_type.value,
                distance_km=float(distance),
                co2e=float(total_co2e),
                duration=duration,
            )
            self._calculation_count += 1

            logger.debug(
                "Taxi calculation complete: type=%s, dist=%s km, "
                "co2e=%s kg, wtt=%s kg, total=%s kg",
                taxi_input.taxi_type.value,
                distance,
                co2e,
                wtt_co2e,
                total_co2e,
            )

            return result

    def _validate_taxi_input(self, taxi_input: TaxiInput) -> None:
        """
        Validate taxi input parameters.

        Args:
            taxi_input: TaxiInput to validate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if taxi_input.distance_km <= _ZERO:
            raise ValueError(
                f"Taxi distance_km must be positive, got {taxi_input.distance_km}"
            )
        valid_taxi_types = {
            RoadVehicleType.TAXI_REGULAR,
            RoadVehicleType.TAXI_BLACK_CAB,
        }
        if taxi_input.taxi_type not in valid_taxi_types:
            raise ValueError(
                f"Taxi type must be TAXI_REGULAR or TAXI_BLACK_CAB, "
                f"got '{taxi_input.taxi_type.value}'"
            )
        if taxi_input.taxi_type not in ROAD_VEHICLE_EMISSION_FACTORS:
            raise ValueError(
                f"No emission factor found for taxi type "
                f"'{taxi_input.taxi_type.value}'"
            )

    # ==================================================================
    # 5. calculate_bus - Bus emissions (per-pkm)
    # ==================================================================

    def calculate_bus(
        self,
        bus_input: BusInput,
    ) -> RailResult:
        """
        Calculate bus travel emissions using distance-based method.

        Uses per-passenger-km emission factors from BUS_EMISSION_FACTORS.
        Returns a RailResult since bus per-pkm has the same result shape
        (distance, passengers, co2e, wtt, total).

        Formula:
            co2e     = distance_km x passengers x ef_per_pkm
            wtt_co2e = distance_km x passengers x wtt_per_pkm
            total    = co2e + wtt_co2e

        Args:
            bus_input: Validated BusInput with bus_type, distance_km,
                       and passengers.

        Returns:
            RailResult (reused model) with co2e, wtt_co2e, total_co2e,
            and provenance hash. The rail_type field is set to
            RailType.NATIONAL as a placeholder; callers should use the
            transport mode context to identify this as a bus result.

        Raises:
            ValueError: If distance_km <= 0 or passengers < 1.
            KeyError: If bus_type not found in BUS_EMISSION_FACTORS.

        Example:
            >>> result = engine.calculate_bus(BusInput(
            ...     bus_type=BusType.COACH,
            ...     distance_km=Decimal("200"),
            ...     passengers=1,
            ... ))
            >>> assert result.total_co2e > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate input
            self._validate_bus_input(bus_input)

            # Step 2: Resolve emission factor
            bus_ef = self._resolve_bus_ef(bus_input.bus_type)
            ef_per_pkm = bus_ef["ef"]
            wtt_per_pkm = bus_ef["wtt"]

            # Step 3: Calculate emissions (ZERO HALLUCINATION - Decimal only)
            distance = bus_input.distance_km
            passengers = Decimal(str(bus_input.passengers))

            co2e = _q(distance * passengers * ef_per_pkm)
            wtt_co2e = _q(distance * passengers * wtt_per_pkm)
            total_co2e = _q(co2e + wtt_co2e)

            # Step 4: Provenance hash
            provenance_hash = calculate_provenance_hash(
                bus_input,
                co2e,
                wtt_co2e,
                total_co2e,
                "bus",
                AGENT_ID,
            )

            # Step 5: Build result (reuse RailResult shape for per-pkm)
            # Map bus_type to a rail_type placeholder for the model
            rail_type_map: Dict[BusType, RailType] = {
                BusType.LOCAL: RailType.NATIONAL,
                BusType.COACH: RailType.NATIONAL,
            }
            result = RailResult(
                rail_type=rail_type_map.get(
                    bus_input.bus_type, RailType.NATIONAL
                ),
                distance_km=distance,
                passengers=bus_input.passengers,
                co2e=co2e,
                wtt_co2e=wtt_co2e,
                total_co2e=total_co2e,
                ef_source=EFSource.DEFRA,
                provenance_hash=provenance_hash,
            )

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            vehicle_label = (
                "bus_local" if bus_input.bus_type == BusType.LOCAL
                else "bus_coach"
            )
            self._record_ground_metrics(
                mode="bus",
                vehicle_type=vehicle_label,
                distance_km=float(distance),
                co2e=float(total_co2e),
                duration=duration,
            )
            self._calculation_count += 1

            logger.debug(
                "Bus calculation complete: type=%s, dist=%s km, pax=%d, "
                "co2e=%s kg, wtt=%s kg, total=%s kg",
                bus_input.bus_type.value,
                distance,
                bus_input.passengers,
                co2e,
                wtt_co2e,
                total_co2e,
            )

            return result

    def _validate_bus_input(self, bus_input: BusInput) -> None:
        """
        Validate bus input parameters.

        Args:
            bus_input: BusInput to validate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if bus_input.distance_km <= _ZERO:
            raise ValueError(
                f"Bus distance_km must be positive, got {bus_input.distance_km}"
            )
        if bus_input.passengers < 1:
            raise ValueError(
                f"Bus passengers must be >= 1, got {bus_input.passengers}"
            )
        if bus_input.bus_type not in BUS_EMISSION_FACTORS:
            raise ValueError(
                f"Unknown bus_type '{bus_input.bus_type.value}'. "
                f"Available: {list(BUS_EMISSION_FACTORS.keys())}"
            )

    def _resolve_bus_ef(self, bus_type: BusType) -> Dict[str, Decimal]:
        """
        Resolve bus emission factors for the given bus type.

        Args:
            bus_type: The BusType enum value.

        Returns:
            Dict with 'ef' and 'wtt' Decimal emission factors (kgCO2e/pkm).

        Raises:
            KeyError: If bus_type is not in BUS_EMISSION_FACTORS.
        """
        ef = BUS_EMISSION_FACTORS.get(bus_type)
        if ef is None:
            raise KeyError(
                f"Bus EF not found for type '{bus_type.value}'"
            )

        logger.debug(
            "Resolved bus EF: type=%s, ef=%s, wtt=%s",
            bus_type.value,
            ef["ef"],
            ef["wtt"],
        )
        return ef

    # ==================================================================
    # 6. calculate_ferry - Ferry emissions (per-pkm)
    # ==================================================================

    def calculate_ferry(
        self,
        ferry_input: FerryInput,
    ) -> RailResult:
        """
        Calculate ferry travel emissions using distance-based method.

        Uses per-passenger-km emission factors from FERRY_EMISSION_FACTORS.
        Returns a RailResult since ferry per-pkm has the same result shape.

        Formula:
            co2e     = distance_km x passengers x ef_per_pkm
            wtt_co2e = distance_km x passengers x wtt_per_pkm
            total    = co2e + wtt_co2e

        Args:
            ferry_input: Validated FerryInput with ferry_type, distance_km,
                         and passengers.

        Returns:
            RailResult (reused model) with co2e, wtt_co2e, total_co2e,
            and provenance hash.

        Raises:
            ValueError: If distance_km <= 0 or passengers < 1.
            KeyError: If ferry_type not found in FERRY_EMISSION_FACTORS.

        Example:
            >>> result = engine.calculate_ferry(FerryInput(
            ...     ferry_type=FerryType.FOOT_PASSENGER,
            ...     distance_km=Decimal("35"),
            ...     passengers=2,
            ... ))
            >>> assert result.total_co2e > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate input
            self._validate_ferry_input(ferry_input)

            # Step 2: Resolve emission factor
            ferry_ef = self._resolve_ferry_ef(ferry_input.ferry_type)
            ef_per_pkm = ferry_ef["ef"]
            wtt_per_pkm = ferry_ef["wtt"]

            # Step 3: Calculate emissions (ZERO HALLUCINATION - Decimal only)
            distance = ferry_input.distance_km
            passengers = Decimal(str(ferry_input.passengers))

            co2e = _q(distance * passengers * ef_per_pkm)
            wtt_co2e = _q(distance * passengers * wtt_per_pkm)
            total_co2e = _q(co2e + wtt_co2e)

            # Step 4: Provenance hash
            provenance_hash = calculate_provenance_hash(
                ferry_input,
                co2e,
                wtt_co2e,
                total_co2e,
                "ferry",
                AGENT_ID,
            )

            # Step 5: Build result (reuse RailResult for per-pkm shape)
            result = RailResult(
                rail_type=RailType.NATIONAL,  # placeholder for ferry
                distance_km=distance,
                passengers=ferry_input.passengers,
                co2e=co2e,
                wtt_co2e=wtt_co2e,
                total_co2e=total_co2e,
                ef_source=EFSource.DEFRA,
                provenance_hash=provenance_hash,
            )

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._record_ground_metrics(
                mode="ferry",
                vehicle_type=ferry_input.ferry_type.value,
                distance_km=float(distance),
                co2e=float(total_co2e),
                duration=duration,
            )
            self._calculation_count += 1

            logger.debug(
                "Ferry calculation complete: type=%s, dist=%s km, pax=%d, "
                "co2e=%s kg, wtt=%s kg, total=%s kg",
                ferry_input.ferry_type.value,
                distance,
                ferry_input.passengers,
                co2e,
                wtt_co2e,
                total_co2e,
            )

            return result

    def _validate_ferry_input(self, ferry_input: FerryInput) -> None:
        """
        Validate ferry input parameters.

        Args:
            ferry_input: FerryInput to validate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if ferry_input.distance_km <= _ZERO:
            raise ValueError(
                f"Ferry distance_km must be positive, got {ferry_input.distance_km}"
            )
        if ferry_input.passengers < 1:
            raise ValueError(
                f"Ferry passengers must be >= 1, got {ferry_input.passengers}"
            )
        if ferry_input.ferry_type not in FERRY_EMISSION_FACTORS:
            raise ValueError(
                f"Unknown ferry_type '{ferry_input.ferry_type.value}'. "
                f"Available: {list(FERRY_EMISSION_FACTORS.keys())}"
            )

    def _resolve_ferry_ef(self, ferry_type: FerryType) -> Dict[str, Decimal]:
        """
        Resolve ferry emission factors for the given ferry type.

        Args:
            ferry_type: The FerryType enum value.

        Returns:
            Dict with 'ef' and 'wtt' Decimal emission factors (kgCO2e/pkm).

        Raises:
            KeyError: If ferry_type is not in FERRY_EMISSION_FACTORS.
        """
        ef = FERRY_EMISSION_FACTORS.get(ferry_type)
        if ef is None:
            raise KeyError(
                f"Ferry EF not found for type '{ferry_type.value}'"
            )

        logger.debug(
            "Resolved ferry EF: type=%s, ef=%s, wtt=%s",
            ferry_type.value,
            ef["ef"],
            ef["wtt"],
        )
        return ef

    # ==================================================================
    # 7. calculate_motorcycle - Motorcycle emissions (per-vkm)
    # ==================================================================

    def calculate_motorcycle(
        self,
        distance_km: Decimal,
    ) -> RoadDistanceResult:
        """
        Calculate motorcycle emissions using distance-based method.

        Uses ROAD_VEHICLE_EMISSION_FACTORS[MOTORCYCLE] with per-vehicle-km
        factors. Occupancy is 1.0 (single rider), so ef_per_vkm equals
        ef_per_pkm.

        Formula:
            co2e     = distance_km x ef_per_vkm
            wtt_co2e = distance_km x wtt_per_vkm
            total    = co2e + wtt_co2e

        Args:
            distance_km: Distance travelled in kilometres. Must be > 0.

        Returns:
            RoadDistanceResult with vehicle_type=MOTORCYCLE, co2e,
            wtt_co2e, total_co2e, and provenance hash.

        Raises:
            ValueError: If distance_km <= 0.

        Example:
            >>> result = engine.calculate_motorcycle(Decimal("50"))
            >>> assert result.vehicle_type == RoadVehicleType.MOTORCYCLE
            >>> assert result.total_co2e > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate input
            if distance_km <= _ZERO:
                raise ValueError(
                    f"Motorcycle distance_km must be positive, got {distance_km}"
                )

            # Step 2: Resolve emission factor
            moto_ef = self._resolve_road_vehicle_ef(RoadVehicleType.MOTORCYCLE)
            ef_per_vkm = moto_ef["ef_per_vkm"]
            wtt_per_vkm = moto_ef["wtt_per_vkm"]

            # Step 3: Calculate emissions (ZERO HALLUCINATION - Decimal only)
            co2e = _q(distance_km * ef_per_vkm)
            wtt_co2e = _q(distance_km * wtt_per_vkm)
            total_co2e = _q(co2e + wtt_co2e)

            # Step 4: Provenance hash
            provenance_hash = calculate_provenance_hash(
                distance_km,
                RoadVehicleType.MOTORCYCLE.value,
                co2e,
                wtt_co2e,
                total_co2e,
                "motorcycle",
                AGENT_ID,
            )

            # Step 5: Build result
            result = RoadDistanceResult(
                vehicle_type=RoadVehicleType.MOTORCYCLE,
                distance_km=distance_km,
                co2e=co2e,
                wtt_co2e=wtt_co2e,
                total_co2e=total_co2e,
                ef_source=EFSource.DEFRA,
                provenance_hash=provenance_hash,
            )

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._record_ground_metrics(
                mode="motorcycle",
                vehicle_type="motorcycle",
                distance_km=float(distance_km),
                co2e=float(total_co2e),
                duration=duration,
            )
            self._calculation_count += 1

            logger.debug(
                "Motorcycle calculation complete: dist=%s km, "
                "co2e=%s kg, wtt=%s kg, total=%s kg",
                distance_km,
                co2e,
                wtt_co2e,
                total_co2e,
            )

            return result

    # ==================================================================
    # 8. calculate_batch_ground - Batch ground trip processing
    # ==================================================================

    def calculate_batch_ground(
        self,
        inputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Process multiple ground transport trip calculations in a single batch.

        Each input dict must contain a 'mode' key indicating the transport
        type (rail, road_distance, road_fuel, taxi, bus, ferry, motorcycle)
        plus the mode-specific parameters.

        The method processes each input independently, collecting results and
        errors. Failed calculations do not halt the batch.

        Args:
            inputs: List of dicts, each containing 'mode' and mode-specific
                    parameters.

        Returns:
            List of result dicts. Each dict contains either 'result' (on
            success) or 'error' (on failure) with the original 'mode' and
            'index'.

        Raises:
            ValueError: If inputs list exceeds _MAX_BATCH_SIZE.

        Example:
            >>> results = engine.calculate_batch_ground([
            ...     {"mode": "rail", "rail_type": "eurostar",
            ...      "distance_km": "340", "passengers": 2},
            ...     {"mode": "motorcycle", "distance_km": "50"},
            ... ])
            >>> assert len(results) == 2
        """
        if len(inputs) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(inputs)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        results: List[Dict[str, Any]] = []
        start_time = time.monotonic()

        for idx, inp in enumerate(inputs):
            mode = inp.get("mode", "unknown")
            try:
                result = self._dispatch_ground_calculation(inp)
                results.append({
                    "index": idx,
                    "mode": mode,
                    "status": "success",
                    "result": result,
                })
            except Exception as exc:
                logger.warning(
                    "Batch item %d (%s) failed: %s",
                    idx, mode, str(exc),
                )
                results.append({
                    "index": idx,
                    "mode": mode,
                    "status": "error",
                    "error": str(exc),
                })

        duration = time.monotonic() - start_time
        logger.info(
            "Batch ground calculation complete: total=%d, success=%d, "
            "errors=%d, duration=%.4fs",
            len(inputs),
            sum(1 for r in results if r["status"] == "success"),
            sum(1 for r in results if r["status"] == "error"),
            duration,
        )

        return results

    def _dispatch_ground_calculation(
        self, inp: Dict[str, Any]
    ) -> Any:
        """
        Dispatch a single ground transport calculation based on mode.

        Args:
            inp: Dict with 'mode' and mode-specific parameters.

        Returns:
            Mode-specific result object.

        Raises:
            ValueError: If mode is unknown or parameters are invalid.
        """
        mode = inp.get("mode", "").lower()

        if mode == "rail":
            rail_input = RailInput(
                rail_type=RailType(inp["rail_type"]),
                distance_km=Decimal(str(inp["distance_km"])),
                passengers=int(inp.get("passengers", 1)),
                tenant_id=inp.get("tenant_id"),
            )
            return self.calculate_rail(rail_input)

        elif mode == "road_distance":
            road_input = RoadDistanceInput(
                vehicle_type=RoadVehicleType(inp["vehicle_type"]),
                distance_km=Decimal(str(inp["distance_km"])),
                tenant_id=inp.get("tenant_id"),
            )
            return self.calculate_road_distance(road_input)

        elif mode == "road_fuel":
            fuel_input = RoadFuelInput(
                fuel_type=FuelType(inp["fuel_type"]),
                litres=Decimal(str(inp["litres"])),
                tenant_id=inp.get("tenant_id"),
            )
            return self.calculate_road_fuel(fuel_input)

        elif mode == "taxi":
            taxi_input = TaxiInput(
                taxi_type=RoadVehicleType(
                    inp.get("taxi_type", "taxi_regular")
                ),
                distance_km=Decimal(str(inp["distance_km"])),
                tenant_id=inp.get("tenant_id"),
            )
            return self.calculate_taxi(taxi_input)

        elif mode == "bus":
            bus_input = BusInput(
                bus_type=BusType(inp["bus_type"]),
                distance_km=Decimal(str(inp["distance_km"])),
                passengers=int(inp.get("passengers", 1)),
                tenant_id=inp.get("tenant_id"),
            )
            return self.calculate_bus(bus_input)

        elif mode == "ferry":
            ferry_input = FerryInput(
                ferry_type=FerryType(inp["ferry_type"]),
                distance_km=Decimal(str(inp["distance_km"])),
                passengers=int(inp.get("passengers", 1)),
                tenant_id=inp.get("tenant_id"),
            )
            return self.calculate_ferry(ferry_input)

        elif mode == "motorcycle":
            distance = Decimal(str(inp["distance_km"]))
            return self.calculate_motorcycle(distance)

        else:
            raise ValueError(
                f"Unknown ground transport mode '{mode}'. "
                f"Supported modes: rail, road_distance, road_fuel, taxi, "
                f"bus, ferry, motorcycle"
            )

    # ==================================================================
    # 9. Unit Conversion Helpers
    # ==================================================================

    @staticmethod
    def convert_miles_to_km(miles: Decimal) -> Decimal:
        """
        Convert miles to kilometres.

        Uses the standard conversion factor 1 mile = 1.60934 km.

        Args:
            miles: Distance in miles.

        Returns:
            Distance in kilometres, quantized to 8 decimal places.

        Raises:
            ValueError: If miles is negative.

        Example:
            >>> GroundTransportCalculatorEngine.convert_miles_to_km(Decimal("100"))
            Decimal('160.93400000')
        """
        if miles < _ZERO:
            raise ValueError(f"Miles must be non-negative, got {miles}")
        return _q(miles * _MILES_TO_KM)

    @staticmethod
    def convert_gallons_to_litres(gallons: Decimal) -> Decimal:
        """
        Convert US gallons to litres.

        Uses the standard conversion factor 1 US gallon = 3.78541 litres.

        Args:
            gallons: Volume in US gallons.

        Returns:
            Volume in litres, quantized to 8 decimal places.

        Raises:
            ValueError: If gallons is negative.

        Example:
            >>> GroundTransportCalculatorEngine.convert_gallons_to_litres(Decimal("10"))
            Decimal('37.85410000')
        """
        if gallons < _ZERO:
            raise ValueError(f"Gallons must be non-negative, got {gallons}")
        return _q(gallons * _GALLONS_TO_LITRES)

    # ==================================================================
    # 10. Emission Factor Accessors (read-only)
    # ==================================================================

    @staticmethod
    def get_rail_emission_factors() -> Dict[str, Dict[str, Decimal]]:
        """
        Return all rail emission factors as a dict keyed by rail type value.

        Returns:
            Dict mapping rail type string to its ttw and wtt factors.
        """
        return {
            rt.value: {"ttw": ef["ttw"], "wtt": ef["wtt"]}
            for rt, ef in RAIL_EMISSION_FACTORS.items()
        }

    @staticmethod
    def get_road_vehicle_emission_factors() -> Dict[str, Dict[str, Decimal]]:
        """
        Return all road vehicle emission factors as a dict keyed by type value.

        Returns:
            Dict mapping vehicle type string to its emission factor dict.
        """
        return {
            vt.value: dict(ef)
            for vt, ef in ROAD_VEHICLE_EMISSION_FACTORS.items()
        }

    @staticmethod
    def get_fuel_emission_factors() -> Dict[str, Dict[str, Decimal]]:
        """
        Return all fuel emission factors as a dict keyed by fuel type value.

        Returns:
            Dict mapping fuel type string to ef_per_litre and wtt_per_litre.
        """
        return {
            ft.value: dict(ef)
            for ft, ef in FUEL_EMISSION_FACTORS.items()
        }

    @staticmethod
    def get_bus_emission_factors() -> Dict[str, Dict[str, Decimal]]:
        """
        Return all bus emission factors as a dict keyed by bus type value.

        Returns:
            Dict mapping bus type string to ef and wtt factors.
        """
        return {
            bt.value: dict(ef)
            for bt, ef in BUS_EMISSION_FACTORS.items()
        }

    @staticmethod
    def get_ferry_emission_factors() -> Dict[str, Dict[str, Decimal]]:
        """
        Return all ferry emission factors as a dict keyed by ferry type value.

        Returns:
            Dict mapping ferry type string to ef and wtt factors.
        """
        return {
            ft.value: dict(ef)
            for ft, ef in FERRY_EMISSION_FACTORS.items()
        }

    # ==================================================================
    # Internal: Metrics Recording
    # ==================================================================

    def _record_ground_metrics(
        self,
        mode: str,
        vehicle_type: str,
        distance_km: float,
        co2e: float,
        duration: float,
    ) -> None:
        """
        Record ground transport calculation metrics to Prometheus.

        Wraps calls to the BusinessTravelMetrics singleton to record
        calculation throughput, emissions, duration, and ground trip
        counters. All calls are wrapped in try/except to prevent
        metrics failures from disrupting calculations.

        Args:
            mode: Transport mode label (rail/road/bus/taxi/ferry/motorcycle).
            vehicle_type: Vehicle type label for ground_trips_total counter.
            distance_km: Trip distance in km for distance counter.
            co2e: Emissions in kgCO2e for emissions counter.
            duration: Calculation duration in seconds.
        """
        try:
            # Record the primary calculation counter
            self._metrics.record_calculation(
                method="distance_based",
                mode=mode,
                status="success",
                duration=duration,
                co2e=co2e,
                rf_option="without_rf",
            )

            # Record ground trip counter
            self._metrics.record_ground_trip(
                mode=mode,
                vehicle_type=vehicle_type,
                distance_km=distance_km,
            )
        except Exception as exc:
            logger.warning(
                "Failed to record ground transport metrics: %s", str(exc)
            )


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "GroundTransportCalculatorEngine",
]
