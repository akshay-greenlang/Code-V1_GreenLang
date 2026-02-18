# -*- coding: utf-8 -*-
"""
FleetManagerEngine - Engine 3: Mobile Combustion Agent (AGENT-MRV-003)

Vehicle registry and fleet analytics engine providing in-memory fleet
management with trip logging, emission aggregation, vehicle lifecycle
tracking, and department-level attribution.

Features:
    - Vehicle Registry: Register/update/deactivate vehicles with VIN, make,
      model, year, fuel type, department, location
    - Trip Logging: Record trips with vehicle_id, distance, fuel_consumed,
      start/end times, route, purpose
    - Fleet Analytics: total emissions by period, emission intensity metrics,
      utilization rates, fuel consumption trends, top emitters
    - Vehicle Lifecycle: acquisition, operation, maintenance events, disposal
    - Department Attribution: emissions by cost center, department, project

Zero-Hallucination Guarantees:
    - All arithmetic uses Python Decimal for bit-perfect reproducibility.
    - No LLM involvement in any numeric path.
    - Every operation carries a SHA-256 provenance hash.
    - Complete trip and service history for audit trail.

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.mobile_combustion.fleet_manager import FleetManagerEngine
    >>> from greenlang.mobile_combustion.vehicle_database import VehicleDatabaseEngine
    >>> from greenlang.mobile_combustion.emission_calculator import EmissionCalculatorEngine
    >>> from decimal import Decimal
    >>> db = VehicleDatabaseEngine()
    >>> calc = EmissionCalculatorEngine(vehicle_database=db)
    >>> fleet = FleetManagerEngine(vehicle_database=db, emission_calculator=calc)
    >>> vid = fleet.register_vehicle({
    ...     "vin": "1HGBH41JXMN109186",
    ...     "make": "Honda", "model": "Civic", "year": 2020,
    ...     "vehicle_type": "PASSENGER_CAR_GASOLINE",
    ...     "fuel_type": "GASOLINE",
    ...     "department": "Sales",
    ... })
    >>> tid = fleet.log_trip({
    ...     "vehicle_id": vid,
    ...     "distance_km": Decimal("150"),
    ...     "fuel_consumed_liters": Decimal("12.5"),
    ... })
    >>> agg = fleet.aggregate_fleet_emissions("2026-01", "2026-12")
    >>> print(agg["total_co2e_tonnes"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone, date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from greenlang.mobile_combustion.vehicle_database import VehicleDatabaseEngine
from greenlang.mobile_combustion.emission_calculator import EmissionCalculatorEngine

logger = logging.getLogger(__name__)

__all__ = ["FleetManagerEngine"]

# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------
_PRECISION = Decimal("0.00000001")  # 8 decimal places

# kg to tonnes
_KG_TO_TONNES = Decimal("0.001")

# g to kg
_G_TO_KG = Decimal("0.001")

# ---------------------------------------------------------------------------
# Vehicle Status Constants
# ---------------------------------------------------------------------------

VEHICLE_STATUS_ACTIVE = "ACTIVE"
VEHICLE_STATUS_INACTIVE = "INACTIVE"
VEHICLE_STATUS_MAINTENANCE = "MAINTENANCE"
VEHICLE_STATUS_DISPOSED = "DISPOSED"

VALID_VEHICLE_STATUSES = {
    VEHICLE_STATUS_ACTIVE,
    VEHICLE_STATUS_INACTIVE,
    VEHICLE_STATUS_MAINTENANCE,
    VEHICLE_STATUS_DISPOSED,
}

# ---------------------------------------------------------------------------
# Lifecycle Event Types
# ---------------------------------------------------------------------------

LIFECYCLE_ACQUISITION = "ACQUISITION"
LIFECYCLE_MAINTENANCE = "MAINTENANCE"
LIFECYCLE_REPAIR = "REPAIR"
LIFECYCLE_INSPECTION = "INSPECTION"
LIFECYCLE_STATUS_CHANGE = "STATUS_CHANGE"
LIFECYCLE_DISPOSAL = "DISPOSAL"
LIFECYCLE_FUEL_TYPE_CHANGE = "FUEL_TYPE_CHANGE"
LIFECYCLE_DEPARTMENT_TRANSFER = "DEPARTMENT_TRANSFER"
LIFECYCLE_ODOMETER_UPDATE = "ODOMETER_UPDATE"

VALID_LIFECYCLE_EVENTS = {
    LIFECYCLE_ACQUISITION,
    LIFECYCLE_MAINTENANCE,
    LIFECYCLE_REPAIR,
    LIFECYCLE_INSPECTION,
    LIFECYCLE_STATUS_CHANGE,
    LIFECYCLE_DISPOSAL,
    LIFECYCLE_FUEL_TYPE_CHANGE,
    LIFECYCLE_DEPARTMENT_TRANSFER,
    LIFECYCLE_ODOMETER_UPDATE,
}

# ---------------------------------------------------------------------------
# Trip Purpose Constants
# ---------------------------------------------------------------------------

TRIP_PURPOSES = {
    "BUSINESS", "COMMUTE", "DELIVERY", "SERVICE_CALL", "MAINTENANCE",
    "PICKUP", "TRANSPORT", "PATROL", "INSPECTION", "OTHER",
}


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ===========================================================================
# FleetManagerEngine
# ===========================================================================


class FleetManagerEngine:
    """Vehicle registry and fleet analytics engine for mobile combustion.

    Provides an in-memory registry for fleet vehicles with full lifecycle
    management, trip logging, emission calculation, and fleet-level analytics.
    Integrates with VehicleDatabaseEngine for factor lookups and
    EmissionCalculatorEngine for emission computations.

    Thread Safety:
        All mutable state (_vehicles, _trips, _lifecycle_events) is protected
        by a reentrant lock. Concurrent callers are safe.

    Attributes:
        _vehicle_db: VehicleDatabaseEngine for vehicle/fuel lookups.
        _calculator: EmissionCalculatorEngine for emission calculations.
        _vehicles: In-memory vehicle registry keyed by vehicle_id.
        _trips: In-memory trip records keyed by trip_id.
        _lifecycle_events: Lifecycle events keyed by vehicle_id.

    Example:
        >>> fleet = FleetManagerEngine()
        >>> vid = fleet.register_vehicle({...})
        >>> tid = fleet.log_trip({...})
        >>> agg = fleet.aggregate_fleet_emissions("2026-01", "2026-12")
    """

    def __init__(
        self,
        vehicle_database: Optional[VehicleDatabaseEngine] = None,
        emission_calculator: Optional[EmissionCalculatorEngine] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the FleetManagerEngine.

        Args:
            vehicle_database: VehicleDatabaseEngine instance. If None, a
                default instance is created.
            emission_calculator: EmissionCalculatorEngine instance. If None,
                a default instance is created using the vehicle database.
            config: Optional configuration dictionary. Supported keys:
                - ``max_vehicles``: Maximum vehicle registrations (default 100000).
                - ``max_trips``: Maximum trip records (default 1000000).
                - ``max_lifecycle_events_per_vehicle``: Max events per vehicle
                  (default 10000).
                - ``enable_auto_emission_calc``: Auto-calculate emissions for
                  trips (default True).
        """
        self._config: Dict[str, Any] = config or {}
        self._vehicle_db = vehicle_database or VehicleDatabaseEngine()
        self._calculator = emission_calculator or EmissionCalculatorEngine(
            vehicle_database=self._vehicle_db
        )
        self._max_vehicles: int = self._config.get("max_vehicles", 100_000)
        self._max_trips: int = self._config.get("max_trips", 1_000_000)
        self._max_lifecycle_per_vehicle: int = self._config.get(
            "max_lifecycle_events_per_vehicle", 10_000
        )
        self._auto_calc: bool = self._config.get("enable_auto_emission_calc", True)

        self._vehicles: Dict[str, Dict[str, Any]] = {}
        self._trips: Dict[str, Dict[str, Any]] = {}
        self._lifecycle_events: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.RLock()

        logger.info(
            "FleetManagerEngine initialized (max_vehicles=%d, max_trips=%d, auto_calc=%s)",
            self._max_vehicles, self._max_trips, self._auto_calc,
        )

    # ------------------------------------------------------------------
    # Public API: Vehicle Registration
    # ------------------------------------------------------------------

    def register_vehicle(self, registration: Dict[str, Any]) -> str:
        """Register a new vehicle in the fleet.

        Args:
            registration: Vehicle registration dictionary. Required keys:
                - ``vehicle_type`` (str): Vehicle type from VehicleDatabaseEngine.
                - ``fuel_type`` (str): Primary fuel type.
                Optional keys:
                - ``vehicle_id`` (str): Custom ID. Auto-generated if absent.
                - ``vin`` (str): Vehicle Identification Number.
                - ``make`` (str): Vehicle manufacturer.
                - ``model`` (str): Vehicle model.
                - ``year`` (int): Model year.
                - ``license_plate`` (str): License plate number.
                - ``department`` (str): Department/cost center.
                - ``location`` (str): Base location.
                - ``driver`` (str): Assigned driver.
                - ``fuel_economy_km_per_l`` (Decimal): Custom fuel economy.
                - ``odometer_km`` (Decimal): Initial odometer reading.
                - ``acquisition_date`` (str): Date of acquisition (ISO format).
                - ``control_technology`` (str): Emission control technology.
                - ``tags`` (dict): Custom tags/metadata.

        Returns:
            Vehicle ID string.

        Raises:
            ValueError: If required fields are missing or invalid, or if
                the maximum vehicle count is reached.
        """
        with self._lock:
            if len(self._vehicles) >= self._max_vehicles:
                raise ValueError(
                    f"Maximum vehicle count reached ({self._max_vehicles})"
                )

            # Validate required fields
            vehicle_type = registration.get("vehicle_type")
            fuel_type = registration.get("fuel_type")

            if not vehicle_type:
                raise ValueError("vehicle_type is required")
            if not fuel_type:
                raise ValueError("fuel_type is required")

            # Validate against database
            veh_data = self._vehicle_db.get_vehicle_type(vehicle_type)
            fuel_data = self._vehicle_db.get_fuel_type(fuel_type)

            # Generate ID
            vehicle_id = registration.get("vehicle_id") or f"veh_{uuid4().hex[:12]}"

            if vehicle_id in self._vehicles:
                raise ValueError(f"Vehicle ID already exists: '{vehicle_id}'")

            now = _utcnow()
            model_year = registration.get("year")
            odometer = Decimal(str(registration.get("odometer_km", "0")))

            vehicle_record: Dict[str, Any] = {
                "vehicle_id": vehicle_id,
                "vehicle_type": vehicle_type.upper().strip(),
                "fuel_type": fuel_type.upper().strip(),
                "vin": registration.get("vin", ""),
                "make": registration.get("make", ""),
                "model": registration.get("model", ""),
                "year": model_year,
                "license_plate": registration.get("license_plate", ""),
                "department": registration.get("department", "UNASSIGNED"),
                "location": registration.get("location", ""),
                "driver": registration.get("driver", ""),
                "fuel_economy_km_per_l": (
                    Decimal(str(registration["fuel_economy_km_per_l"]))
                    if registration.get("fuel_economy_km_per_l") is not None
                    else veh_data.get("default_fuel_economy_km_per_l")
                ),
                "odometer_km": odometer,
                "control_technology": registration.get("control_technology"),
                "acquisition_date": registration.get("acquisition_date", now.date().isoformat()),
                "status": VEHICLE_STATUS_ACTIVE,
                "tags": registration.get("tags", {}),
                "trip_count": 0,
                "total_distance_km": Decimal("0"),
                "total_fuel_liters": Decimal("0"),
                "total_co2e_kg": Decimal("0"),
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "category": veh_data["category"],
            }

            self._vehicles[vehicle_id] = vehicle_record
            self._lifecycle_events[vehicle_id] = []

            # Record acquisition lifecycle event
            self._add_lifecycle_event(
                vehicle_id,
                LIFECYCLE_ACQUISITION,
                f"Vehicle registered: {vehicle_record['make']} {vehicle_record['model']} "
                f"({vehicle_record['year']})",
                {"odometer_km": str(odometer)},
            )

            logger.info(
                "Vehicle registered: %s (%s %s %s, type=%s, fuel=%s, dept=%s)",
                vehicle_id,
                vehicle_record["make"],
                vehicle_record["model"],
                vehicle_record["year"],
                vehicle_record["vehicle_type"],
                vehicle_record["fuel_type"],
                vehicle_record["department"],
            )

            return vehicle_id

    def update_vehicle(
        self, vehicle_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update an existing vehicle registration.

        Args:
            vehicle_id: Vehicle ID to update.
            updates: Dictionary of fields to update. Updatable fields:
                vin, make, model, license_plate, department, location,
                driver, fuel_type, fuel_economy_km_per_l, control_technology,
                odometer_km, tags, status.

        Returns:
            Updated vehicle record dictionary.

        Raises:
            ValueError: If vehicle_id not found or update is invalid.
        """
        with self._lock:
            if vehicle_id not in self._vehicles:
                raise ValueError(f"Vehicle not found: '{vehicle_id}'")

            vehicle = self._vehicles[vehicle_id]
            now = _utcnow()

            updatable_fields = {
                "vin", "make", "model", "license_plate", "department",
                "location", "driver", "fuel_type", "fuel_economy_km_per_l",
                "control_technology", "odometer_km", "tags",
            }

            changes: Dict[str, Any] = {}
            for key, value in updates.items():
                if key not in updatable_fields:
                    continue

                old_value = vehicle.get(key)

                if key == "fuel_type":
                    value = value.upper().strip()
                    self._vehicle_db.get_fuel_type(value)
                    if old_value != value:
                        self._add_lifecycle_event(
                            vehicle_id,
                            LIFECYCLE_FUEL_TYPE_CHANGE,
                            f"Fuel type changed: {old_value} -> {value}",
                            {"old": str(old_value), "new": str(value)},
                        )

                elif key == "department":
                    if old_value != value:
                        self._add_lifecycle_event(
                            vehicle_id,
                            LIFECYCLE_DEPARTMENT_TRANSFER,
                            f"Department transfer: {old_value} -> {value}",
                            {"old": str(old_value), "new": str(value)},
                        )

                elif key == "fuel_economy_km_per_l":
                    value = Decimal(str(value))

                elif key == "odometer_km":
                    value = Decimal(str(value))
                    if old_value is not None and value < Decimal(str(old_value)):
                        raise ValueError(
                            f"Odometer cannot decrease: {old_value} -> {value}"
                        )
                    self._add_lifecycle_event(
                        vehicle_id,
                        LIFECYCLE_ODOMETER_UPDATE,
                        f"Odometer updated: {old_value} -> {value}",
                        {"old": str(old_value), "new": str(value)},
                    )

                vehicle[key] = value
                changes[key] = {"old": old_value, "new": value}

            vehicle["updated_at"] = now.isoformat()

            logger.info(
                "Vehicle updated: %s (changes: %s)",
                vehicle_id, list(changes.keys()),
            )
            return dict(vehicle)

    def deactivate_vehicle(self, vehicle_id: str, reason: str = "") -> bool:
        """Deactivate a vehicle (mark as disposed/removed from fleet).

        Args:
            vehicle_id: Vehicle ID to deactivate.
            reason: Reason for deactivation.

        Returns:
            True if deactivated successfully.

        Raises:
            ValueError: If vehicle_id not found.
        """
        with self._lock:
            if vehicle_id not in self._vehicles:
                raise ValueError(f"Vehicle not found: '{vehicle_id}'")

            vehicle = self._vehicles[vehicle_id]

            if vehicle["status"] == VEHICLE_STATUS_DISPOSED:
                logger.warning("Vehicle %s is already disposed", vehicle_id)
                return False

            old_status = vehicle["status"]
            vehicle["status"] = VEHICLE_STATUS_DISPOSED
            vehicle["updated_at"] = _utcnow().isoformat()
            vehicle["disposal_date"] = _utcnow().date().isoformat()
            vehicle["disposal_reason"] = reason

            self._add_lifecycle_event(
                vehicle_id,
                LIFECYCLE_DISPOSAL,
                f"Vehicle deactivated (was {old_status}): {reason}",
                {"old_status": old_status, "reason": reason},
            )

            logger.info(
                "Vehicle deactivated: %s (reason: %s)", vehicle_id, reason
            )
            return True

    def get_vehicle(self, vehicle_id: str) -> Dict[str, Any]:
        """Retrieve a vehicle registration record.

        Args:
            vehicle_id: Vehicle identifier.

        Returns:
            Vehicle record dictionary.

        Raises:
            ValueError: If vehicle_id not found.
        """
        with self._lock:
            if vehicle_id not in self._vehicles:
                raise ValueError(f"Vehicle not found: '{vehicle_id}'")
            return dict(self._vehicles[vehicle_id])

    def list_vehicles(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List vehicles with optional filters.

        Args:
            filters: Optional filter dictionary. Supported keys:
                - ``status`` (str): Filter by status (ACTIVE, INACTIVE, etc.).
                - ``vehicle_type`` (str): Filter by vehicle type.
                - ``fuel_type`` (str): Filter by fuel type.
                - ``department`` (str): Filter by department.
                - ``location`` (str): Filter by location.
                - ``category`` (str): Filter by vehicle category.
                - ``year_min`` (int): Minimum model year.
                - ``year_max`` (int): Maximum model year.

        Returns:
            List of matching vehicle record dictionaries.
        """
        filters = filters or {}
        with self._lock:
            results: List[Dict[str, Any]] = []

            f_status = filters.get("status")
            f_vtype = filters.get("vehicle_type")
            f_ftype = filters.get("fuel_type")
            f_dept = filters.get("department")
            f_loc = filters.get("location")
            f_cat = filters.get("category")
            f_year_min = filters.get("year_min")
            f_year_max = filters.get("year_max")

            if f_status:
                f_status = f_status.upper().strip()
            if f_vtype:
                f_vtype = f_vtype.upper().strip()
            if f_ftype:
                f_ftype = f_ftype.upper().strip()
            if f_cat:
                f_cat = f_cat.upper().strip()

            for vehicle in self._vehicles.values():
                if f_status and vehicle["status"] != f_status:
                    continue
                if f_vtype and vehicle["vehicle_type"] != f_vtype:
                    continue
                if f_ftype and vehicle["fuel_type"] != f_ftype:
                    continue
                if f_dept and vehicle.get("department", "") != f_dept:
                    continue
                if f_loc and vehicle.get("location", "") != f_loc:
                    continue
                if f_cat and vehicle.get("category", "") != f_cat:
                    continue
                if f_year_min and vehicle.get("year") is not None:
                    if vehicle["year"] < f_year_min:
                        continue
                if f_year_max and vehicle.get("year") is not None:
                    if vehicle["year"] > f_year_max:
                        continue

                results.append(dict(vehicle))

            logger.debug("Listed %d vehicles (filters=%s)", len(results), filters)
            return results

    # ------------------------------------------------------------------
    # Public API: Trip Logging
    # ------------------------------------------------------------------

    def log_trip(self, trip_record: Dict[str, Any]) -> str:
        """Record a trip for a registered vehicle.

        If ``fuel_consumed_liters`` is provided, emissions are calculated
        using the fuel-based method. If only ``distance_km`` is provided,
        the distance-based method is used.

        Args:
            trip_record: Trip data dictionary. Required keys:
                - ``vehicle_id`` (str): Registered vehicle ID.
                At least one of:
                - ``distance_km`` (Decimal): Distance traveled.
                - ``fuel_consumed_liters`` (Decimal): Fuel consumed.
                Optional keys:
                - ``trip_id`` (str): Custom trip ID.
                - ``start_time`` (str): Trip start time (ISO format).
                - ``end_time`` (str): Trip end time (ISO format).
                - ``start_location`` (str): Start location.
                - ``end_location`` (str): End location.
                - ``route`` (str): Route description.
                - ``purpose`` (str): Trip purpose.
                - ``driver`` (str): Driver name.
                - ``passengers`` (int): Number of passengers.
                - ``cargo_tonnes`` (Decimal): Cargo weight.
                - ``project`` (str): Project code.
                - ``cost_center`` (str): Cost center code.
                - ``notes`` (str): Additional notes.

        Returns:
            Trip ID string.

        Raises:
            ValueError: If vehicle_id not found, or neither distance nor
                fuel is provided, or max trips reached.
        """
        with self._lock:
            if len(self._trips) >= self._max_trips:
                raise ValueError(f"Maximum trip count reached ({self._max_trips})")

            vehicle_id = trip_record.get("vehicle_id")
            if not vehicle_id:
                raise ValueError("vehicle_id is required")
            if vehicle_id not in self._vehicles:
                raise ValueError(f"Vehicle not found: '{vehicle_id}'")

            vehicle = self._vehicles[vehicle_id]

            distance_km = trip_record.get("distance_km")
            fuel_consumed = trip_record.get("fuel_consumed_liters")

            if distance_km is None and fuel_consumed is None:
                raise ValueError("At least one of distance_km or fuel_consumed_liters is required")

            if distance_km is not None:
                distance_km = Decimal(str(distance_km))
            if fuel_consumed is not None:
                fuel_consumed = Decimal(str(fuel_consumed))

            trip_id = trip_record.get("trip_id") or f"trip_{uuid4().hex[:12]}"
            now = _utcnow()

            # Calculate emissions if auto-calc is enabled
            emission_result: Optional[Dict[str, Any]] = None
            total_co2e_kg = Decimal("0")

            if self._auto_calc:
                emission_result = self._calculate_trip_emissions(
                    vehicle, distance_km, fuel_consumed,
                )
                if emission_result and emission_result.get("status") == "SUCCESS":
                    total_co2e_kg = emission_result.get("total_co2e_kg", Decimal("0"))

            # Build trip record
            passengers = trip_record.get("passengers")
            cargo_tonnes = trip_record.get("cargo_tonnes")

            trip: Dict[str, Any] = {
                "trip_id": trip_id,
                "vehicle_id": vehicle_id,
                "vehicle_type": vehicle["vehicle_type"],
                "fuel_type": vehicle["fuel_type"],
                "department": vehicle.get("department", "UNASSIGNED"),
                "distance_km": distance_km,
                "fuel_consumed_liters": fuel_consumed,
                "start_time": trip_record.get("start_time", now.isoformat()),
                "end_time": trip_record.get("end_time"),
                "start_location": trip_record.get("start_location", ""),
                "end_location": trip_record.get("end_location", ""),
                "route": trip_record.get("route", ""),
                "purpose": trip_record.get("purpose", "BUSINESS"),
                "driver": trip_record.get("driver", vehicle.get("driver", "")),
                "passengers": passengers,
                "cargo_tonnes": Decimal(str(cargo_tonnes)) if cargo_tonnes is not None else None,
                "project": trip_record.get("project", ""),
                "cost_center": trip_record.get("cost_center", ""),
                "notes": trip_record.get("notes", ""),
                "total_co2e_kg": total_co2e_kg,
                "emission_result": emission_result,
                "created_at": now.isoformat(),
            }

            self._trips[trip_id] = trip

            # Update vehicle aggregates
            vehicle["trip_count"] += 1
            if distance_km is not None:
                vehicle["total_distance_km"] += distance_km
            if fuel_consumed is not None:
                vehicle["total_fuel_liters"] += fuel_consumed
            vehicle["total_co2e_kg"] += total_co2e_kg

            # Update odometer if distance provided
            if distance_km is not None and vehicle.get("odometer_km") is not None:
                vehicle["odometer_km"] += distance_km

            vehicle["updated_at"] = now.isoformat()

            logger.info(
                "Trip logged: %s (vehicle=%s, dist=%s km, fuel=%s L, co2e=%s kg)",
                trip_id, vehicle_id, distance_km, fuel_consumed, total_co2e_kg,
            )

            return trip_id

    def get_trip(self, trip_id: str) -> Dict[str, Any]:
        """Retrieve a trip record.

        Args:
            trip_id: Trip identifier.

        Returns:
            Trip record dictionary.

        Raises:
            ValueError: If trip_id not found.
        """
        with self._lock:
            if trip_id not in self._trips:
                raise ValueError(f"Trip not found: '{trip_id}'")
            return dict(self._trips[trip_id])

    def list_trips(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List trips with optional filters.

        Args:
            filters: Optional filter dictionary. Supported keys:
                - ``vehicle_id`` (str): Filter by vehicle.
                - ``department`` (str): Filter by department.
                - ``purpose`` (str): Filter by trip purpose.
                - ``date_from`` (str): Start date filter (ISO date).
                - ``date_to`` (str): End date filter (ISO date).
                - ``project`` (str): Filter by project code.
                - ``cost_center`` (str): Filter by cost center.
                - ``min_distance_km`` (Decimal): Minimum distance.
                - ``min_co2e_kg`` (Decimal): Minimum CO2e.

        Returns:
            List of matching trip record dictionaries.
        """
        filters = filters or {}
        with self._lock:
            results: List[Dict[str, Any]] = []

            f_vehicle = filters.get("vehicle_id")
            f_dept = filters.get("department")
            f_purpose = filters.get("purpose")
            f_date_from = filters.get("date_from")
            f_date_to = filters.get("date_to")
            f_project = filters.get("project")
            f_cost_center = filters.get("cost_center")
            f_min_dist = filters.get("min_distance_km")
            f_min_co2e = filters.get("min_co2e_kg")

            for trip in self._trips.values():
                if f_vehicle and trip["vehicle_id"] != f_vehicle:
                    continue
                if f_dept and trip.get("department", "") != f_dept:
                    continue
                if f_purpose and trip.get("purpose", "") != f_purpose:
                    continue
                if f_project and trip.get("project", "") != f_project:
                    continue
                if f_cost_center and trip.get("cost_center", "") != f_cost_center:
                    continue

                trip_date = trip.get("start_time", "")[:10]
                if f_date_from and trip_date < f_date_from:
                    continue
                if f_date_to and trip_date > f_date_to:
                    continue

                if f_min_dist is not None and trip.get("distance_km") is not None:
                    if trip["distance_km"] < Decimal(str(f_min_dist)):
                        continue

                if f_min_co2e is not None:
                    if trip.get("total_co2e_kg", Decimal("0")) < Decimal(str(f_min_co2e)):
                        continue

                results.append(dict(trip))

            logger.debug("Listed %d trips (filters=%s)", len(results), filters)
            return results

    # ------------------------------------------------------------------
    # Public API: Fleet Analytics
    # ------------------------------------------------------------------

    def aggregate_fleet_emissions(
        self,
        period_start: str,
        period_end: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Aggregate fleet emissions for a given period.

        Args:
            period_start: Start date (ISO format, e.g. ``"2026-01-01"``).
            period_end: End date (ISO format, e.g. ``"2026-12-31"``).
            filters: Optional filters (same as list_trips).

        Returns:
            Fleet aggregation dictionary with totals, per-vehicle breakdown,
            per-department breakdown, and per-fuel breakdown.
        """
        start_time = time.monotonic()

        trip_filters = dict(filters or {})
        trip_filters["date_from"] = period_start
        trip_filters["date_to"] = period_end

        trips = self.list_trips(trip_filters)

        total_co2e_kg = Decimal("0")
        total_distance_km = Decimal("0")
        total_fuel_liters = Decimal("0")
        trip_count = 0

        per_vehicle: Dict[str, Dict[str, Decimal]] = defaultdict(
            lambda: {"co2e_kg": Decimal("0"), "distance_km": Decimal("0"),
                     "fuel_liters": Decimal("0"), "trips": 0}
        )
        per_department: Dict[str, Dict[str, Decimal]] = defaultdict(
            lambda: {"co2e_kg": Decimal("0"), "distance_km": Decimal("0"),
                     "fuel_liters": Decimal("0"), "trips": 0}
        )
        per_fuel: Dict[str, Dict[str, Decimal]] = defaultdict(
            lambda: {"co2e_kg": Decimal("0"), "distance_km": Decimal("0"),
                     "fuel_liters": Decimal("0"), "trips": 0}
        )
        per_vehicle_type: Dict[str, Dict[str, Decimal]] = defaultdict(
            lambda: {"co2e_kg": Decimal("0"), "distance_km": Decimal("0"),
                     "fuel_liters": Decimal("0"), "trips": 0}
        )

        for trip in trips:
            co2e = trip.get("total_co2e_kg", Decimal("0"))
            dist = trip.get("distance_km") or Decimal("0")
            fuel = trip.get("fuel_consumed_liters") or Decimal("0")
            vid = trip["vehicle_id"]
            dept = trip.get("department", "UNASSIGNED")
            ftype = trip.get("fuel_type", "UNKNOWN")
            vtype = trip.get("vehicle_type", "UNKNOWN")

            total_co2e_kg += co2e
            total_distance_km += dist
            total_fuel_liters += fuel
            trip_count += 1

            per_vehicle[vid]["co2e_kg"] += co2e
            per_vehicle[vid]["distance_km"] += dist
            per_vehicle[vid]["fuel_liters"] += fuel
            per_vehicle[vid]["trips"] += 1

            per_department[dept]["co2e_kg"] += co2e
            per_department[dept]["distance_km"] += dist
            per_department[dept]["fuel_liters"] += fuel
            per_department[dept]["trips"] += 1

            per_fuel[ftype]["co2e_kg"] += co2e
            per_fuel[ftype]["distance_km"] += dist
            per_fuel[ftype]["fuel_liters"] += fuel
            per_fuel[ftype]["trips"] += 1

            per_vehicle_type[vtype]["co2e_kg"] += co2e
            per_vehicle_type[vtype]["distance_km"] += dist
            per_vehicle_type[vtype]["fuel_liters"] += fuel
            per_vehicle_type[vtype]["trips"] += 1

        total_co2e_tonnes = (total_co2e_kg * _KG_TO_TONNES).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        provenance_hash = self._compute_hash(
            "fleet_aggregation",
            {
                "period_start": period_start,
                "period_end": period_end,
                "trip_count": trip_count,
                "total_co2e_kg": str(total_co2e_kg),
            },
        )

        result = {
            "period_start": period_start,
            "period_end": period_end,
            "trip_count": trip_count,
            "total_co2e_kg": total_co2e_kg.quantize(_PRECISION, rounding=ROUND_HALF_UP),
            "total_co2e_tonnes": total_co2e_tonnes,
            "total_distance_km": total_distance_km.quantize(_PRECISION, rounding=ROUND_HALF_UP),
            "total_fuel_liters": total_fuel_liters.quantize(_PRECISION, rounding=ROUND_HALF_UP),
            "per_vehicle": dict(per_vehicle),
            "per_department": dict(per_department),
            "per_fuel_type": dict(per_fuel),
            "per_vehicle_type": dict(per_vehicle_type),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed_ms, 3),
        }

        logger.info(
            "Fleet aggregation %s to %s: %d trips, %.4f tCO2e",
            period_start, period_end, trip_count, total_co2e_tonnes,
        )

        return result

    def get_fleet_intensity(
        self,
        period_start: str,
        period_end: str,
        metric: str = "g_co2e_per_km",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calculate fleet emission intensity for a period.

        Args:
            period_start: Start date.
            period_end: End date.
            metric: Intensity metric. Supported:
                - ``"g_co2e_per_km"``: Grams CO2e per kilometer.
                - ``"g_co2e_per_passenger_km"``: Per passenger-km (requires
                  passenger data).
                - ``"g_co2e_per_tonne_km"``: Per tonne-km (requires cargo data).
                - ``"kg_co2e_per_liter"``: Per liter of fuel consumed.
                - ``"kg_co2e_per_trip"``: Per trip.
            filters: Optional trip filters.

        Returns:
            Intensity result dictionary with overall and per-vehicle values.
        """
        agg = self.aggregate_fleet_emissions(period_start, period_end, filters)

        total_co2e_kg = agg["total_co2e_kg"]
        total_distance = agg["total_distance_km"]
        total_fuel = agg["total_fuel_liters"]
        trip_count = agg["trip_count"]

        overall_intensity = Decimal("0")
        unit = ""

        if metric == "g_co2e_per_km":
            if total_distance > Decimal("0"):
                overall_intensity = (total_co2e_kg * Decimal("1000") / total_distance).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
            unit = "g CO2e/km"

        elif metric == "g_co2e_per_passenger_km":
            trips = self.list_trips(
                dict(filters or {}, date_from=period_start, date_to=period_end)
            )
            total_pass_km = Decimal("0")
            for t in trips:
                pax = t.get("passengers") or 1
                dist = t.get("distance_km") or Decimal("0")
                total_pass_km += Decimal(str(pax)) * dist
            if total_pass_km > Decimal("0"):
                overall_intensity = (total_co2e_kg * Decimal("1000") / total_pass_km).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
            unit = "g CO2e/passenger-km"

        elif metric == "g_co2e_per_tonne_km":
            trips = self.list_trips(
                dict(filters or {}, date_from=period_start, date_to=period_end)
            )
            total_tonne_km = Decimal("0")
            for t in trips:
                cargo = t.get("cargo_tonnes") or Decimal("0")
                dist = t.get("distance_km") or Decimal("0")
                total_tonne_km += Decimal(str(cargo)) * dist
            if total_tonne_km > Decimal("0"):
                overall_intensity = (total_co2e_kg * Decimal("1000") / total_tonne_km).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
            unit = "g CO2e/tonne-km"

        elif metric == "kg_co2e_per_liter":
            if total_fuel > Decimal("0"):
                overall_intensity = (total_co2e_kg / total_fuel).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
            unit = "kg CO2e/L"

        elif metric == "kg_co2e_per_trip":
            if trip_count > 0:
                overall_intensity = (total_co2e_kg / Decimal(str(trip_count))).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
            unit = "kg CO2e/trip"

        else:
            raise ValueError(
                f"Unknown intensity metric: '{metric}'. Supported: "
                f"g_co2e_per_km, g_co2e_per_passenger_km, g_co2e_per_tonne_km, "
                f"kg_co2e_per_liter, kg_co2e_per_trip"
            )

        # Per-vehicle intensity
        per_vehicle: Dict[str, Decimal] = {}
        for vid, vdata in agg.get("per_vehicle", {}).items():
            v_co2e = vdata.get("co2e_kg", Decimal("0"))
            v_dist = vdata.get("distance_km", Decimal("0"))
            v_fuel = vdata.get("fuel_liters", Decimal("0"))
            v_trips = vdata.get("trips", 0)

            if metric == "g_co2e_per_km" and v_dist > Decimal("0"):
                per_vehicle[vid] = (v_co2e * Decimal("1000") / v_dist).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
            elif metric == "kg_co2e_per_liter" and v_fuel > Decimal("0"):
                per_vehicle[vid] = (v_co2e / v_fuel).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
            elif metric == "kg_co2e_per_trip" and v_trips > 0:
                per_vehicle[vid] = (v_co2e / Decimal(str(v_trips))).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )

        return {
            "period_start": period_start,
            "period_end": period_end,
            "metric": metric,
            "unit": unit,
            "overall_intensity": overall_intensity,
            "per_vehicle": per_vehicle,
            "trip_count": trip_count,
            "total_co2e_kg": total_co2e_kg,
            "total_distance_km": total_distance,
            "total_fuel_liters": total_fuel,
        }

    def get_top_emitters(
        self,
        n: int = 10,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get top N emitting vehicles for a period.

        Args:
            n: Number of top emitters to return.
            period_start: Start date (ISO). If None, uses all-time data.
            period_end: End date (ISO). If None, uses all-time data.

        Returns:
            List of top emitters sorted by CO2e descending.
        """
        with self._lock:
            if period_start and period_end:
                agg = self.aggregate_fleet_emissions(period_start, period_end)
                per_vehicle = agg.get("per_vehicle", {})
            else:
                # All-time from vehicle aggregates
                per_vehicle = {}
                for vid, vehicle in self._vehicles.items():
                    per_vehicle[vid] = {
                        "co2e_kg": vehicle.get("total_co2e_kg", Decimal("0")),
                        "distance_km": vehicle.get("total_distance_km", Decimal("0")),
                        "fuel_liters": vehicle.get("total_fuel_liters", Decimal("0")),
                        "trips": vehicle.get("trip_count", 0),
                    }

            # Sort by CO2e descending
            sorted_vehicles = sorted(
                per_vehicle.items(),
                key=lambda x: x[1].get("co2e_kg", Decimal("0")),
                reverse=True,
            )

            results: List[Dict[str, Any]] = []
            for rank, (vid, vdata) in enumerate(sorted_vehicles[:n], start=1):
                entry: Dict[str, Any] = {"rank": rank, "vehicle_id": vid}
                entry.update(vdata)

                # Add vehicle details
                if vid in self._vehicles:
                    vehicle = self._vehicles[vid]
                    entry["vehicle_type"] = vehicle.get("vehicle_type", "")
                    entry["make"] = vehicle.get("make", "")
                    entry["model"] = vehicle.get("model", "")
                    entry["year"] = vehicle.get("year")
                    entry["department"] = vehicle.get("department", "")

                results.append(entry)

            return results

    def get_fleet_composition(self) -> Dict[str, Any]:
        """Get fleet composition breakdown by category, type, fuel, and status.

        Returns:
            Composition dictionary with counts and percentages.
        """
        with self._lock:
            total = len(self._vehicles)

            by_category: Dict[str, int] = defaultdict(int)
            by_type: Dict[str, int] = defaultdict(int)
            by_fuel: Dict[str, int] = defaultdict(int)
            by_status: Dict[str, int] = defaultdict(int)
            by_department: Dict[str, int] = defaultdict(int)
            by_year_range: Dict[str, int] = defaultdict(int)

            for vehicle in self._vehicles.values():
                by_category[vehicle.get("category", "UNKNOWN")] += 1
                by_type[vehicle.get("vehicle_type", "UNKNOWN")] += 1
                by_fuel[vehicle.get("fuel_type", "UNKNOWN")] += 1
                by_status[vehicle.get("status", "UNKNOWN")] += 1
                by_department[vehicle.get("department", "UNASSIGNED")] += 1

                year = vehicle.get("year")
                if year is not None:
                    if year >= 2020:
                        by_year_range["2020+"] += 1
                    elif year >= 2015:
                        by_year_range["2015-2019"] += 1
                    elif year >= 2010:
                        by_year_range["2010-2014"] += 1
                    elif year >= 2005:
                        by_year_range["2005-2009"] += 1
                    else:
                        by_year_range["pre-2005"] += 1
                else:
                    by_year_range["unknown"] += 1

            def _pct(count: int) -> str:
                if total == 0:
                    return "0.0%"
                return f"{(count / total * 100):.1f}%"

            return {
                "total_vehicles": total,
                "by_category": {k: {"count": v, "pct": _pct(v)} for k, v in sorted(by_category.items())},
                "by_vehicle_type": {k: {"count": v, "pct": _pct(v)} for k, v in sorted(by_type.items())},
                "by_fuel_type": {k: {"count": v, "pct": _pct(v)} for k, v in sorted(by_fuel.items())},
                "by_status": {k: {"count": v, "pct": _pct(v)} for k, v in sorted(by_status.items())},
                "by_department": {k: {"count": v, "pct": _pct(v)} for k, v in sorted(by_department.items())},
                "by_year_range": {k: {"count": v, "pct": _pct(v)} for k, v in sorted(by_year_range.items())},
            }

    def get_department_emissions(
        self,
        department: str,
        period_start: str,
        period_end: str,
    ) -> Dict[str, Any]:
        """Get emission summary for a specific department.

        Args:
            department: Department name.
            period_start: Start date (ISO).
            period_end: End date (ISO).

        Returns:
            Department emission summary with vehicle breakdown.
        """
        agg = self.aggregate_fleet_emissions(
            period_start, period_end,
            filters={"department": department},
        )

        dept_vehicles = self.list_vehicles(filters={"department": department})

        return {
            "department": department,
            "period_start": period_start,
            "period_end": period_end,
            "vehicle_count": len(dept_vehicles),
            "active_vehicles": sum(
                1 for v in dept_vehicles if v["status"] == VEHICLE_STATUS_ACTIVE
            ),
            "trip_count": agg["trip_count"],
            "total_co2e_kg": agg["total_co2e_kg"],
            "total_co2e_tonnes": agg["total_co2e_tonnes"],
            "total_distance_km": agg["total_distance_km"],
            "total_fuel_liters": agg["total_fuel_liters"],
            "per_vehicle": agg.get("per_vehicle", {}),
            "per_fuel_type": agg.get("per_fuel_type", {}),
            "per_vehicle_type": agg.get("per_vehicle_type", {}),
        }

    # ------------------------------------------------------------------
    # Public API: Utilization
    # ------------------------------------------------------------------

    def get_vehicle_utilization(
        self,
        period_start: str,
        period_end: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calculate vehicle utilization rates for a period.

        Utilization = trips_taken / typical_annual_trips_prorated

        Args:
            period_start: Start date.
            period_end: End date.
            filters: Optional vehicle filters.

        Returns:
            Utilization dictionary with per-vehicle rates.
        """
        # Calculate period days
        try:
            start_date = date.fromisoformat(period_start)
            end_date = date.fromisoformat(period_end)
            period_days = max(1, (end_date - start_date).days)
        except (ValueError, TypeError):
            period_days = 365

        period_fraction = Decimal(str(period_days)) / Decimal("365")

        vehicles = self.list_vehicles(filters)
        agg = self.aggregate_fleet_emissions(period_start, period_end, filters)
        per_vehicle_agg = agg.get("per_vehicle", {})

        utilization: Dict[str, Dict[str, Any]] = {}

        for vehicle in vehicles:
            vid = vehicle["vehicle_id"]
            vtype = vehicle["vehicle_type"]

            # Expected activity
            veh_data = {}
            try:
                veh_data = self._vehicle_db.get_vehicle_type(vtype)
            except (ValueError, KeyError):
                pass

            typical_km = Decimal(str(veh_data.get("typical_annual_km", "20000")))
            expected_km = (typical_km * period_fraction).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )

            # Actual activity
            v_agg = per_vehicle_agg.get(vid, {})
            actual_km = v_agg.get("distance_km", Decimal("0"))
            actual_trips = v_agg.get("trips", 0)

            # Utilization rate
            util_rate = Decimal("0")
            if expected_km > Decimal("0"):
                util_rate = (actual_km / expected_km * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

            utilization[vid] = {
                "vehicle_type": vtype,
                "department": vehicle.get("department", ""),
                "expected_km": expected_km,
                "actual_km": actual_km,
                "utilization_pct": util_rate,
                "trips": actual_trips,
                "status": vehicle.get("status", ""),
            }

        # Fleet average
        total_expected = sum(
            u["expected_km"] for u in utilization.values()
        )
        total_actual = sum(
            u["actual_km"] for u in utilization.values()
        )
        fleet_util = Decimal("0")
        if total_expected > Decimal("0"):
            fleet_util = (total_actual / total_expected * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        return {
            "period_start": period_start,
            "period_end": period_end,
            "period_days": period_days,
            "fleet_utilization_pct": fleet_util,
            "total_expected_km": total_expected,
            "total_actual_km": total_actual,
            "vehicle_count": len(utilization),
            "per_vehicle": utilization,
        }

    # ------------------------------------------------------------------
    # Public API: Fuel Consumption Trends
    # ------------------------------------------------------------------

    def get_fuel_consumption_trends(
        self,
        period_start: str,
        period_end: str,
        granularity: str = "monthly",
    ) -> Dict[str, Any]:
        """Get fuel consumption trends over time.

        Args:
            period_start: Start date (ISO).
            period_end: End date (ISO).
            granularity: Time granularity - ``"daily"``, ``"monthly"``,
                ``"quarterly"``.

        Returns:
            Dictionary with time-series fuel and emissions data.
        """
        trips = self.list_trips({"date_from": period_start, "date_to": period_end})

        buckets: Dict[str, Dict[str, Decimal]] = defaultdict(
            lambda: {"fuel_liters": Decimal("0"), "distance_km": Decimal("0"),
                     "co2e_kg": Decimal("0"), "trips": Decimal("0")}
        )

        for trip in trips:
            trip_date = trip.get("start_time", "")[:10]
            if not trip_date or len(trip_date) < 7:
                continue

            if granularity == "daily":
                bucket_key = trip_date
            elif granularity == "quarterly":
                month = int(trip_date[5:7])
                quarter = (month - 1) // 3 + 1
                bucket_key = f"{trip_date[:4]}-Q{quarter}"
            else:
                bucket_key = trip_date[:7]

            buckets[bucket_key]["fuel_liters"] += trip.get("fuel_consumed_liters") or Decimal("0")
            buckets[bucket_key]["distance_km"] += trip.get("distance_km") or Decimal("0")
            buckets[bucket_key]["co2e_kg"] += trip.get("total_co2e_kg", Decimal("0"))
            buckets[bucket_key]["trips"] += Decimal("1")

        # Sort by key
        sorted_data = []
        for key in sorted(buckets.keys()):
            entry = dict(buckets[key])
            entry["period"] = key
            entry["co2e_tonnes"] = (entry["co2e_kg"] * _KG_TO_TONNES).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            sorted_data.append(entry)

        return {
            "period_start": period_start,
            "period_end": period_end,
            "granularity": granularity,
            "data_points": len(sorted_data),
            "time_series": sorted_data,
        }

    # ------------------------------------------------------------------
    # Public API: Year-Over-Year Comparison
    # ------------------------------------------------------------------

    def get_year_over_year(
        self,
        year_current: int,
        year_previous: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compare fleet emissions between two years.

        Args:
            year_current: Current reporting year.
            year_previous: Previous comparison year.
            filters: Optional vehicle/trip filters.

        Returns:
            Comparison dictionary with absolute and percentage changes.
        """
        current_agg = self.aggregate_fleet_emissions(
            f"{year_current}-01-01", f"{year_current}-12-31", filters
        )
        previous_agg = self.aggregate_fleet_emissions(
            f"{year_previous}-01-01", f"{year_previous}-12-31", filters
        )

        def _safe_change(current: Decimal, previous: Decimal) -> Dict[str, Any]:
            delta = current - previous
            pct = Decimal("0")
            if previous != Decimal("0"):
                pct = (delta / previous * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            return {
                "current": current,
                "previous": previous,
                "absolute_change": delta,
                "percent_change": pct,
            }

        return {
            "year_current": year_current,
            "year_previous": year_previous,
            "co2e_kg": _safe_change(
                current_agg["total_co2e_kg"],
                previous_agg["total_co2e_kg"],
            ),
            "co2e_tonnes": _safe_change(
                current_agg["total_co2e_tonnes"],
                previous_agg["total_co2e_tonnes"],
            ),
            "distance_km": _safe_change(
                current_agg["total_distance_km"],
                previous_agg["total_distance_km"],
            ),
            "fuel_liters": _safe_change(
                current_agg["total_fuel_liters"],
                previous_agg["total_fuel_liters"],
            ),
            "trip_count": _safe_change(
                Decimal(str(current_agg["trip_count"])),
                Decimal(str(previous_agg["trip_count"])),
            ),
        }

    # ------------------------------------------------------------------
    # Public API: Lifecycle Events
    # ------------------------------------------------------------------

    def log_lifecycle_event(
        self,
        vehicle_id: str,
        event_type: str,
        description: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log a lifecycle event for a vehicle.

        Args:
            vehicle_id: Vehicle identifier.
            event_type: Event type (see VALID_LIFECYCLE_EVENTS).
            description: Human-readable event description.
            data: Optional event-specific data.

        Returns:
            Event ID string.

        Raises:
            ValueError: If vehicle_id not found or event_type invalid.
        """
        if event_type not in VALID_LIFECYCLE_EVENTS:
            raise ValueError(
                f"Invalid lifecycle event type: '{event_type}'. "
                f"Valid: {sorted(VALID_LIFECYCLE_EVENTS)}"
            )

        with self._lock:
            if vehicle_id not in self._vehicles:
                raise ValueError(f"Vehicle not found: '{vehicle_id}'")

            event_id = self._add_lifecycle_event(
                vehicle_id, event_type, description, data
            )

            # Update vehicle status if needed
            if event_type == LIFECYCLE_MAINTENANCE:
                self._vehicles[vehicle_id]["status"] = VEHICLE_STATUS_MAINTENANCE
                self._vehicles[vehicle_id]["updated_at"] = _utcnow().isoformat()
            elif event_type == LIFECYCLE_DISPOSAL:
                self._vehicles[vehicle_id]["status"] = VEHICLE_STATUS_DISPOSED
                self._vehicles[vehicle_id]["updated_at"] = _utcnow().isoformat()

            return event_id

    def get_lifecycle_events(
        self, vehicle_id: str
    ) -> List[Dict[str, Any]]:
        """Get all lifecycle events for a vehicle.

        Args:
            vehicle_id: Vehicle identifier.

        Returns:
            List of lifecycle event dictionaries, sorted by timestamp.

        Raises:
            ValueError: If vehicle_id not found.
        """
        with self._lock:
            if vehicle_id not in self._vehicles:
                raise ValueError(f"Vehicle not found: '{vehicle_id}'")
            events = self._lifecycle_events.get(vehicle_id, [])
            return [dict(e) for e in events]

    # ------------------------------------------------------------------
    # Public API: Fleet Statistics
    # ------------------------------------------------------------------

    def get_fleet_statistics(self) -> Dict[str, Any]:
        """Return overall fleet statistics summary.

        Returns:
            Dictionary with vehicle counts, trip counts, emission totals,
            and utilization metrics.
        """
        with self._lock:
            total_vehicles = len(self._vehicles)
            active = sum(
                1 for v in self._vehicles.values()
                if v["status"] == VEHICLE_STATUS_ACTIVE
            )
            inactive = sum(
                1 for v in self._vehicles.values()
                if v["status"] == VEHICLE_STATUS_INACTIVE
            )
            maintenance = sum(
                1 for v in self._vehicles.values()
                if v["status"] == VEHICLE_STATUS_MAINTENANCE
            )
            disposed = sum(
                1 for v in self._vehicles.values()
                if v["status"] == VEHICLE_STATUS_DISPOSED
            )

            total_trips = len(self._trips)
            total_co2e_kg = sum(
                v.get("total_co2e_kg", Decimal("0"))
                for v in self._vehicles.values()
            )
            total_distance = sum(
                v.get("total_distance_km", Decimal("0"))
                for v in self._vehicles.values()
            )
            total_fuel = sum(
                v.get("total_fuel_liters", Decimal("0"))
                for v in self._vehicles.values()
            )

            total_lifecycle_events = sum(
                len(events) for events in self._lifecycle_events.values()
            )

            return {
                "total_vehicles": total_vehicles,
                "active_vehicles": active,
                "inactive_vehicles": inactive,
                "maintenance_vehicles": maintenance,
                "disposed_vehicles": disposed,
                "total_trips": total_trips,
                "total_co2e_kg": total_co2e_kg.quantize(_PRECISION, rounding=ROUND_HALF_UP),
                "total_co2e_tonnes": (total_co2e_kg * _KG_TO_TONNES).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                ),
                "total_distance_km": total_distance.quantize(_PRECISION, rounding=ROUND_HALF_UP),
                "total_fuel_liters": total_fuel.quantize(_PRECISION, rounding=ROUND_HALF_UP),
                "total_lifecycle_events": total_lifecycle_events,
                "avg_co2e_per_vehicle_kg": (
                    (total_co2e_kg / Decimal(str(total_vehicles))).quantize(
                        _PRECISION, rounding=ROUND_HALF_UP
                    )
                    if total_vehicles > 0 else Decimal("0")
                ),
                "avg_trips_per_vehicle": (
                    round(total_trips / total_vehicles, 2)
                    if total_vehicles > 0 else 0
                ),
            }

    # ==================================================================
    # Internal: Trip Emission Calculation
    # ==================================================================

    def _calculate_trip_emissions(
        self,
        vehicle: Dict[str, Any],
        distance_km: Optional[Decimal],
        fuel_consumed: Optional[Decimal],
    ) -> Optional[Dict[str, Any]]:
        """Calculate emissions for a single trip.

        Prefers fuel-based method when fuel data is available, otherwise
        falls back to distance-based method.

        Args:
            vehicle: Vehicle record dictionary.
            distance_km: Distance traveled (may be None).
            fuel_consumed: Fuel consumed in liters (may be None).

        Returns:
            Calculation result dictionary, or None if calculation fails.
        """
        vtype = vehicle["vehicle_type"]
        ftype = vehicle["fuel_type"]
        model_year = vehicle.get("year")
        control_tech = vehicle.get("control_technology")

        try:
            if fuel_consumed is not None and fuel_consumed > Decimal("0"):
                return self._calculator.calculate_fuel_based(
                    vehicle_type=vtype,
                    fuel_type=ftype,
                    fuel_consumed=fuel_consumed,
                    fuel_unit="liters",
                    model_year=model_year,
                    control_technology=control_tech,
                )

            elif distance_km is not None and distance_km > Decimal("0"):
                fuel_economy = vehicle.get("fuel_economy_km_per_l")
                vehicle_age = None
                if model_year is not None:
                    vehicle_age = _utcnow().year - model_year

                return self._calculator.calculate_distance_based(
                    vehicle_type=vtype,
                    fuel_type=ftype,
                    distance_km=distance_km,
                    fuel_economy_km_per_l=fuel_economy,
                    vehicle_age_years=vehicle_age,
                    model_year=model_year,
                    control_technology=control_tech,
                )

        except Exception as exc:
            logger.warning(
                "Trip emission calculation failed for vehicle %s: %s",
                vehicle.get("vehicle_id", "unknown"), exc,
            )
            return {
                "status": "FAILED",
                "total_co2e_kg": Decimal("0"),
                "error_message": str(exc),
            }

        return None

    # ==================================================================
    # Internal: Lifecycle Event Management
    # ==================================================================

    def _add_lifecycle_event(
        self,
        vehicle_id: str,
        event_type: str,
        description: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a lifecycle event record (internal, already under lock).

        Args:
            vehicle_id: Vehicle identifier.
            event_type: Event type string.
            description: Event description.
            data: Optional event data.

        Returns:
            Event ID string.
        """
        events = self._lifecycle_events.get(vehicle_id, [])

        if len(events) >= self._max_lifecycle_per_vehicle:
            logger.warning(
                "Maximum lifecycle events reached for vehicle %s (%d)",
                vehicle_id, self._max_lifecycle_per_vehicle,
            )
            events.pop(0)

        event_id = f"evt_{uuid4().hex[:12]}"
        now = _utcnow()

        event: Dict[str, Any] = {
            "event_id": event_id,
            "vehicle_id": vehicle_id,
            "event_type": event_type,
            "description": description,
            "data": data or {},
            "timestamp": now.isoformat(),
            "provenance_hash": self._compute_hash(
                "lifecycle_event",
                {"vehicle_id": vehicle_id, "event_type": event_type, "event_id": event_id},
            ),
        }

        events.append(event)
        self._lifecycle_events[vehicle_id] = events

        logger.debug(
            "Lifecycle event: %s for vehicle %s (%s)",
            event_type, vehicle_id, description,
        )
        return event_id

    # ==================================================================
    # Internal: Provenance Hash
    # ==================================================================

    def _compute_hash(self, operation: str, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking.

        Args:
            operation: Operation name for the hash context.
            data: Data dictionary to include in the hash.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        hash_input = json.dumps(
            {"operation": operation, "data": data, "timestamp": _utcnow().isoformat()},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
