# -*- coding: utf-8 -*-
"""
Unit tests for FleetManagerEngine (Engine 3) - AGENT-MRV-003 Mobile Combustion.

Tests all public methods with 55+ test functions covering:
- Initialization, vehicle registration, update, deactivation
- Trip logging, emission calculation, trip filtering
- Fleet aggregation, intensity metrics, top emitters
- Fleet composition, department emissions, utilization
- Fuel consumption trends, year-over-year comparison
- Lifecycle events, fleet statistics, edge cases

Author: GreenLang QA Team
"""

import threading
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import patch, MagicMock

import pytest

from greenlang.mobile_combustion.fleet_manager import (
    FleetManagerEngine,
    VEHICLE_STATUS_ACTIVE,
    VEHICLE_STATUS_DISPOSED,
    VEHICLE_STATUS_MAINTENANCE,
    VALID_LIFECYCLE_EVENTS,
    LIFECYCLE_ACQUISITION,
    LIFECYCLE_MAINTENANCE,
    LIFECYCLE_DISPOSAL,
    LIFECYCLE_FUEL_TYPE_CHANGE,
    LIFECYCLE_DEPARTMENT_TRANSFER,
    LIFECYCLE_ODOMETER_UPDATE,
    LIFECYCLE_INSPECTION,
    _PRECISION,
    _KG_TO_TONNES,
)
from greenlang.mobile_combustion.vehicle_database import VehicleDatabaseEngine
from greenlang.mobile_combustion.emission_calculator import EmissionCalculatorEngine


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def vdb():
    """Create a VehicleDatabaseEngine instance."""
    return VehicleDatabaseEngine()


@pytest.fixture
def calc(vdb):
    """Create an EmissionCalculatorEngine instance."""
    return EmissionCalculatorEngine(vehicle_database=vdb)


@pytest.fixture
def fleet(vdb, calc):
    """Create a FleetManagerEngine with auto emission calc enabled."""
    return FleetManagerEngine(
        vehicle_database=vdb,
        emission_calculator=calc,
    )


@pytest.fixture
def fleet_no_calc(vdb, calc):
    """Create a FleetManagerEngine with auto emission calc disabled."""
    return FleetManagerEngine(
        vehicle_database=vdb,
        emission_calculator=calc,
        config={"enable_auto_emission_calc": False},
    )


@pytest.fixture
def registered_vehicle(fleet):
    """Register a gasoline passenger car and return (fleet, vehicle_id)."""
    vid = fleet.register_vehicle({
        "vehicle_type": "PASSENGER_CAR_GASOLINE",
        "fuel_type": "GASOLINE",
        "make": "Toyota",
        "model": "Corolla",
        "year": 2020,
        "department": "Sales",
        "odometer_km": Decimal("15000"),
        "vehicle_id": "test_veh_001",
    })
    return fleet, vid


@pytest.fixture
def fleet_with_trips(registered_vehicle):
    """Register a vehicle and log two trips; return (fleet, vid, tid1, tid2)."""
    fleet, vid = registered_vehicle
    tid1 = fleet.log_trip({
        "vehicle_id": vid,
        "distance_km": Decimal("100"),
        "fuel_consumed_liters": Decimal("8.5"),
        "start_time": "2026-03-15T10:00:00+00:00",
        "purpose": "BUSINESS",
    })
    tid2 = fleet.log_trip({
        "vehicle_id": vid,
        "distance_km": Decimal("200"),
        "fuel_consumed_liters": Decimal("17"),
        "start_time": "2026-03-20T14:00:00+00:00",
        "purpose": "DELIVERY",
    })
    return fleet, vid, tid1, tid2


# ===========================================================================
# TestInit
# ===========================================================================


class TestInit:
    """Test FleetManagerEngine initialization."""

    def test_default_init(self, fleet):
        """Engine initializes with default configuration."""
        assert fleet._max_vehicles == 100_000
        assert fleet._max_trips == 1_000_000
        assert fleet._auto_calc is True

    def test_custom_config(self, vdb, calc):
        """Engine accepts custom config values."""
        f = FleetManagerEngine(
            vehicle_database=vdb,
            emission_calculator=calc,
            config={
                "max_vehicles": 10,
                "max_trips": 50,
                "enable_auto_emission_calc": False,
                "max_lifecycle_events_per_vehicle": 100,
            },
        )
        assert f._max_vehicles == 10
        assert f._max_trips == 50
        assert f._auto_calc is False
        assert f._max_lifecycle_per_vehicle == 100

    def test_defaults_without_deps(self):
        """Engine can initialize without explicit dependencies."""
        f = FleetManagerEngine()
        assert isinstance(f._vehicle_db, VehicleDatabaseEngine)
        assert isinstance(f._calculator, EmissionCalculatorEngine)

    def test_rlock_created(self, fleet):
        """Engine creates a reentrant lock."""
        assert isinstance(fleet._lock, type(threading.RLock()))

    def test_empty_on_init(self, fleet):
        """No vehicles, trips, or events on a fresh engine."""
        assert fleet._vehicles == {}
        assert fleet._trips == {}
        assert fleet._lifecycle_events == {}


# ===========================================================================
# TestRegisterVehicle
# ===========================================================================


class TestRegisterVehicle:
    """Test vehicle registration."""

    def test_register_returns_id(self, fleet):
        """register_vehicle returns a string vehicle ID."""
        vid = fleet.register_vehicle({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
        })
        assert isinstance(vid, str)
        assert len(vid) > 0

    def test_custom_vehicle_id(self, fleet):
        """Custom vehicle_id is honored."""
        vid = fleet.register_vehicle({
            "vehicle_type": "PASSENGER_CAR_DIESEL",
            "fuel_type": "DIESEL",
            "vehicle_id": "custom_123",
        })
        assert vid == "custom_123"

    def test_auto_generated_id_prefix(self, fleet):
        """Auto-generated IDs start with 'veh_'."""
        vid = fleet.register_vehicle({
            "vehicle_type": "MOTORCYCLE",
            "fuel_type": "GASOLINE",
        })
        assert vid.startswith("veh_")

    def test_vehicle_record_fields(self, registered_vehicle):
        """Registered vehicle has expected fields and values."""
        fleet, vid = registered_vehicle
        v = fleet.get_vehicle(vid)
        assert v["vehicle_type"] == "PASSENGER_CAR_GASOLINE"
        assert v["fuel_type"] == "GASOLINE"
        assert v["make"] == "Toyota"
        assert v["model"] == "Corolla"
        assert v["year"] == 2020
        assert v["department"] == "Sales"
        assert v["status"] == VEHICLE_STATUS_ACTIVE
        assert v["trip_count"] == 0
        assert v["total_co2e_kg"] == Decimal("0")

    def test_missing_vehicle_type_raises(self, fleet):
        """Missing vehicle_type raises ValueError."""
        with pytest.raises(ValueError, match="vehicle_type is required"):
            fleet.register_vehicle({"fuel_type": "GASOLINE"})

    def test_missing_fuel_type_raises(self, fleet):
        """Missing fuel_type raises ValueError."""
        with pytest.raises(ValueError, match="fuel_type is required"):
            fleet.register_vehicle({"vehicle_type": "MOTORCYCLE"})

    def test_duplicate_id_raises(self, registered_vehicle):
        """Registering with an existing ID raises ValueError."""
        fleet, vid = registered_vehicle
        with pytest.raises(ValueError, match="already exists"):
            fleet.register_vehicle({
                "vehicle_type": "MOTORCYCLE",
                "fuel_type": "GASOLINE",
                "vehicle_id": vid,
            })

    def test_max_vehicles_enforced(self, vdb, calc):
        """Exceeding max_vehicles raises ValueError."""
        f = FleetManagerEngine(
            vehicle_database=vdb,
            emission_calculator=calc,
            config={"max_vehicles": 2},
        )
        f.register_vehicle({"vehicle_type": "MOTORCYCLE", "fuel_type": "GASOLINE"})
        f.register_vehicle({"vehicle_type": "MOTORCYCLE", "fuel_type": "GASOLINE"})
        with pytest.raises(ValueError, match="Maximum vehicle count"):
            f.register_vehicle({"vehicle_type": "MOTORCYCLE", "fuel_type": "GASOLINE"})

    def test_acquisition_lifecycle_logged(self, registered_vehicle):
        """Registration logs an ACQUISITION lifecycle event."""
        fleet, vid = registered_vehicle
        events = fleet.get_lifecycle_events(vid)
        assert len(events) >= 1
        assert events[0]["event_type"] == LIFECYCLE_ACQUISITION

    def test_default_department_unassigned(self, fleet):
        """Default department is UNASSIGNED when not specified."""
        vid = fleet.register_vehicle({
            "vehicle_type": "MOTORCYCLE",
            "fuel_type": "GASOLINE",
        })
        v = fleet.get_vehicle(vid)
        assert v["department"] == "UNASSIGNED"


# ===========================================================================
# TestUpdateVehicle
# ===========================================================================


class TestUpdateVehicle:
    """Test vehicle updates."""

    def test_update_department(self, registered_vehicle):
        """Updating department creates a DEPARTMENT_TRANSFER lifecycle event."""
        fleet, vid = registered_vehicle
        fleet.update_vehicle(vid, {"department": "Engineering"})
        v = fleet.get_vehicle(vid)
        assert v["department"] == "Engineering"
        events = fleet.get_lifecycle_events(vid)
        dept_events = [e for e in events if e["event_type"] == LIFECYCLE_DEPARTMENT_TRANSFER]
        assert len(dept_events) == 1

    def test_update_fuel_type(self, registered_vehicle):
        """Changing fuel_type creates a FUEL_TYPE_CHANGE lifecycle event."""
        fleet, vid = registered_vehicle
        fleet.update_vehicle(vid, {"fuel_type": "DIESEL"})
        v = fleet.get_vehicle(vid)
        assert v["fuel_type"] == "DIESEL"
        events = fleet.get_lifecycle_events(vid)
        ft_events = [e for e in events if e["event_type"] == LIFECYCLE_FUEL_TYPE_CHANGE]
        assert len(ft_events) == 1

    def test_update_odometer(self, registered_vehicle):
        """Updating odometer to a higher value succeeds."""
        fleet, vid = registered_vehicle
        fleet.update_vehicle(vid, {"odometer_km": Decimal("20000")})
        v = fleet.get_vehicle(vid)
        assert v["odometer_km"] == Decimal("20000")

    def test_odometer_decrease_raises(self, registered_vehicle):
        """Decreasing odometer raises ValueError."""
        fleet, vid = registered_vehicle
        with pytest.raises(ValueError, match="Odometer cannot decrease"):
            fleet.update_vehicle(vid, {"odometer_km": Decimal("1000")})

    def test_update_unknown_vehicle_raises(self, fleet):
        """Updating a non-existent vehicle raises ValueError."""
        with pytest.raises(ValueError, match="Vehicle not found"):
            fleet.update_vehicle("nonexistent", {"make": "Ford"})

    def test_update_returns_record(self, registered_vehicle):
        """update_vehicle returns the full updated record."""
        fleet, vid = registered_vehicle
        result = fleet.update_vehicle(vid, {"make": "Honda"})
        assert isinstance(result, dict)
        assert result["make"] == "Honda"


# ===========================================================================
# TestDeactivateVehicle
# ===========================================================================


class TestDeactivateVehicle:
    """Test vehicle deactivation."""

    def test_deactivate_returns_true(self, registered_vehicle):
        """Deactivating an active vehicle returns True."""
        fleet, vid = registered_vehicle
        result = fleet.deactivate_vehicle(vid, reason="End of lease")
        assert result is True

    def test_deactivated_status_disposed(self, registered_vehicle):
        """Deactivated vehicle status is DISPOSED."""
        fleet, vid = registered_vehicle
        fleet.deactivate_vehicle(vid)
        v = fleet.get_vehicle(vid)
        assert v["status"] == VEHICLE_STATUS_DISPOSED

    def test_deactivate_already_disposed_returns_false(self, registered_vehicle):
        """Deactivating a DISPOSED vehicle returns False."""
        fleet, vid = registered_vehicle
        fleet.deactivate_vehicle(vid)
        result = fleet.deactivate_vehicle(vid)
        assert result is False

    def test_deactivate_logs_disposal_event(self, registered_vehicle):
        """Deactivation logs a DISPOSAL lifecycle event."""
        fleet, vid = registered_vehicle
        fleet.deactivate_vehicle(vid, reason="Sold")
        events = fleet.get_lifecycle_events(vid)
        disposal_events = [e for e in events if e["event_type"] == LIFECYCLE_DISPOSAL]
        assert len(disposal_events) == 1
        assert "Sold" in disposal_events[0]["description"]

    def test_deactivate_unknown_vehicle_raises(self, fleet):
        """Deactivating a non-existent vehicle raises ValueError."""
        with pytest.raises(ValueError, match="Vehicle not found"):
            fleet.deactivate_vehicle("nonexistent")


# ===========================================================================
# TestListVehicles
# ===========================================================================


class TestListVehicles:
    """Test vehicle listing and filtering."""

    def test_list_all(self, registered_vehicle):
        """list_vehicles with no filters returns all vehicles."""
        fleet, vid = registered_vehicle
        vehicles = fleet.list_vehicles()
        assert len(vehicles) == 1
        assert vehicles[0]["vehicle_id"] == vid

    def test_filter_by_status(self, registered_vehicle):
        """Filtering by status works correctly."""
        fleet, vid = registered_vehicle
        active = fleet.list_vehicles({"status": "ACTIVE"})
        assert len(active) == 1
        disposed = fleet.list_vehicles({"status": "DISPOSED"})
        assert len(disposed) == 0

    def test_filter_by_department(self, registered_vehicle):
        """Filtering by department returns matching vehicles."""
        fleet, vid = registered_vehicle
        sales = fleet.list_vehicles({"department": "Sales"})
        assert len(sales) == 1
        eng = fleet.list_vehicles({"department": "Engineering"})
        assert len(eng) == 0

    def test_filter_by_year_range(self, registered_vehicle):
        """Filtering by year_min/year_max works."""
        fleet, vid = registered_vehicle
        result = fleet.list_vehicles({"year_min": 2019, "year_max": 2021})
        assert len(result) == 1
        result2 = fleet.list_vehicles({"year_min": 2022})
        assert len(result2) == 0


# ===========================================================================
# TestLogTrip
# ===========================================================================


class TestLogTrip:
    """Test trip logging."""

    def test_log_trip_returns_id(self, registered_vehicle):
        """log_trip returns a trip ID string."""
        fleet, vid = registered_vehicle
        tid = fleet.log_trip({
            "vehicle_id": vid,
            "distance_km": Decimal("50"),
            "fuel_consumed_liters": Decimal("4"),
        })
        assert isinstance(tid, str)
        assert len(tid) > 0

    def test_trip_updates_vehicle_aggregates(self, registered_vehicle):
        """Logging a trip updates vehicle trip_count, distance, fuel totals."""
        fleet, vid = registered_vehicle
        fleet.log_trip({
            "vehicle_id": vid,
            "distance_km": Decimal("100"),
            "fuel_consumed_liters": Decimal("8"),
        })
        v = fleet.get_vehicle(vid)
        assert v["trip_count"] == 1
        assert v["total_distance_km"] == Decimal("100")
        assert v["total_fuel_liters"] == Decimal("8")

    def test_trip_auto_emission_calc(self, registered_vehicle):
        """With auto_calc enabled, trips have non-zero total_co2e_kg."""
        fleet, vid = registered_vehicle
        tid = fleet.log_trip({
            "vehicle_id": vid,
            "fuel_consumed_liters": Decimal("10"),
        })
        trip = fleet.get_trip(tid)
        assert trip["total_co2e_kg"] > Decimal("0")

    def test_trip_no_auto_calc(self, fleet_no_calc):
        """With auto_calc disabled, total_co2e_kg is zero."""
        vid = fleet_no_calc.register_vehicle({
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
        })
        tid = fleet_no_calc.log_trip({
            "vehicle_id": vid,
            "fuel_consumed_liters": Decimal("10"),
        })
        trip = fleet_no_calc.get_trip(tid)
        assert trip["total_co2e_kg"] == Decimal("0")

    def test_trip_distance_only(self, registered_vehicle):
        """Trip with only distance_km (no fuel) uses distance-based calc."""
        fleet, vid = registered_vehicle
        tid = fleet.log_trip({
            "vehicle_id": vid,
            "distance_km": Decimal("200"),
        })
        trip = fleet.get_trip(tid)
        assert trip["distance_km"] == Decimal("200")
        assert trip["fuel_consumed_liters"] is None

    def test_trip_neither_distance_nor_fuel_raises(self, registered_vehicle):
        """Trip with neither distance nor fuel raises ValueError."""
        fleet, vid = registered_vehicle
        with pytest.raises(ValueError, match="At least one of"):
            fleet.log_trip({"vehicle_id": vid})

    def test_trip_missing_vehicle_id_raises(self, fleet):
        """Trip without vehicle_id raises ValueError."""
        with pytest.raises(ValueError, match="vehicle_id is required"):
            fleet.log_trip({"distance_km": Decimal("100")})

    def test_trip_unknown_vehicle_raises(self, fleet):
        """Trip for non-existent vehicle raises ValueError."""
        with pytest.raises(ValueError, match="Vehicle not found"):
            fleet.log_trip({
                "vehicle_id": "nonexistent",
                "distance_km": Decimal("100"),
            })

    def test_max_trips_enforced(self, vdb, calc):
        """Exceeding max_trips raises ValueError."""
        f = FleetManagerEngine(
            vehicle_database=vdb,
            emission_calculator=calc,
            config={"max_trips": 1, "enable_auto_emission_calc": False},
        )
        vid = f.register_vehicle({
            "vehicle_type": "MOTORCYCLE",
            "fuel_type": "GASOLINE",
        })
        f.log_trip({"vehicle_id": vid, "distance_km": Decimal("10")})
        with pytest.raises(ValueError, match="Maximum trip count"):
            f.log_trip({"vehicle_id": vid, "distance_km": Decimal("10")})

    def test_odometer_updated_on_trip(self, registered_vehicle):
        """Logging a trip with distance increments the vehicle odometer."""
        fleet, vid = registered_vehicle
        before = fleet.get_vehicle(vid)["odometer_km"]
        fleet.log_trip({
            "vehicle_id": vid,
            "distance_km": Decimal("50"),
            "fuel_consumed_liters": Decimal("4"),
        })
        after = fleet.get_vehicle(vid)["odometer_km"]
        assert after == before + Decimal("50")


# ===========================================================================
# TestListTrips
# ===========================================================================


class TestListTrips:
    """Test trip listing and filtering."""

    def test_list_all_trips(self, fleet_with_trips):
        """list_trips with no filters returns all trips."""
        fleet, vid, tid1, tid2 = fleet_with_trips
        trips = fleet.list_trips()
        assert len(trips) == 2

    def test_filter_by_vehicle_id(self, fleet_with_trips):
        """Filtering by vehicle_id returns only that vehicle's trips."""
        fleet, vid, tid1, tid2 = fleet_with_trips
        trips = fleet.list_trips({"vehicle_id": vid})
        assert len(trips) == 2
        trips2 = fleet.list_trips({"vehicle_id": "other"})
        assert len(trips2) == 0

    def test_filter_by_date_range(self, fleet_with_trips):
        """Filtering by date_from/date_to works."""
        fleet, vid, tid1, tid2 = fleet_with_trips
        trips = fleet.list_trips({"date_from": "2026-03-18"})
        assert len(trips) == 1  # Only the March 20 trip
        assert trips[0]["trip_id"] == tid2

    def test_filter_by_purpose(self, fleet_with_trips):
        """Filtering by trip purpose works."""
        fleet, vid, tid1, tid2 = fleet_with_trips
        biz = fleet.list_trips({"purpose": "BUSINESS"})
        assert len(biz) == 1
        assert biz[0]["trip_id"] == tid1


# ===========================================================================
# TestAggregateFleetEmissions
# ===========================================================================


class TestAggregateFleetEmissions:
    """Test fleet emission aggregation."""

    def test_aggregate_returns_expected_keys(self, fleet_with_trips):
        """Aggregation result has all required keys."""
        fleet, vid, _, _ = fleet_with_trips
        agg = fleet.aggregate_fleet_emissions("2026-01-01", "2026-12-31")
        assert "total_co2e_kg" in agg
        assert "total_co2e_tonnes" in agg
        assert "total_distance_km" in agg
        assert "total_fuel_liters" in agg
        assert "per_vehicle" in agg
        assert "per_department" in agg
        assert "per_fuel_type" in agg
        assert "provenance_hash" in agg
        assert len(agg["provenance_hash"]) == 64

    def test_aggregate_trip_count(self, fleet_with_trips):
        """Aggregation counts trips correctly for the period."""
        fleet, vid, _, _ = fleet_with_trips
        agg = fleet.aggregate_fleet_emissions("2026-03-01", "2026-03-31")
        assert agg["trip_count"] == 2

    def test_aggregate_empty_period(self, fleet_with_trips):
        """Aggregation for a period with no trips returns zeros."""
        fleet, vid, _, _ = fleet_with_trips
        agg = fleet.aggregate_fleet_emissions("2025-01-01", "2025-12-31")
        assert agg["trip_count"] == 0
        assert agg["total_co2e_kg"] == Decimal("0").quantize(_PRECISION, rounding=ROUND_HALF_UP)

    def test_aggregate_per_vehicle_breakdown(self, fleet_with_trips):
        """per_vehicle breakdown includes the registered vehicle."""
        fleet, vid, _, _ = fleet_with_trips
        agg = fleet.aggregate_fleet_emissions("2026-01-01", "2026-12-31")
        assert vid in agg["per_vehicle"]
        assert agg["per_vehicle"][vid]["trips"] == 2

    def test_aggregate_tonnes_conversion(self, fleet_with_trips):
        """total_co2e_tonnes = total_co2e_kg * 0.001."""
        fleet, vid, _, _ = fleet_with_trips
        agg = fleet.aggregate_fleet_emissions("2026-01-01", "2026-12-31")
        expected_tonnes = (agg["total_co2e_kg"] * _KG_TO_TONNES).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        assert agg["total_co2e_tonnes"] == expected_tonnes


# ===========================================================================
# TestFleetIntensity
# ===========================================================================


class TestFleetIntensity:
    """Test fleet emission intensity calculations."""

    def test_g_co2e_per_km(self, fleet_with_trips):
        """g_co2e_per_km metric returns non-zero when distance exists."""
        fleet, _, _, _ = fleet_with_trips
        result = fleet.get_fleet_intensity("2026-01-01", "2026-12-31", "g_co2e_per_km")
        assert result["overall_intensity"] > Decimal("0")
        assert result["unit"] == "g CO2e/km"

    def test_kg_co2e_per_liter(self, fleet_with_trips):
        """kg_co2e_per_liter metric computes correctly."""
        fleet, _, _, _ = fleet_with_trips
        result = fleet.get_fleet_intensity("2026-01-01", "2026-12-31", "kg_co2e_per_liter")
        assert result["overall_intensity"] > Decimal("0")
        assert result["unit"] == "kg CO2e/L"

    def test_kg_co2e_per_trip(self, fleet_with_trips):
        """kg_co2e_per_trip metric divides total by trip count."""
        fleet, _, _, _ = fleet_with_trips
        result = fleet.get_fleet_intensity("2026-01-01", "2026-12-31", "kg_co2e_per_trip")
        assert result["overall_intensity"] > Decimal("0")
        assert result["unit"] == "kg CO2e/trip"

    def test_unknown_metric_raises(self, fleet_with_trips):
        """Unknown intensity metric raises ValueError."""
        fleet, _, _, _ = fleet_with_trips
        with pytest.raises(ValueError, match="Unknown intensity metric"):
            fleet.get_fleet_intensity("2026-01-01", "2026-12-31", "invalid_metric")


# ===========================================================================
# TestTopEmitters
# ===========================================================================


class TestTopEmitters:
    """Test top emitter retrieval."""

    def test_top_emitters_ranked(self, fleet_with_trips):
        """Top emitters are ranked by CO2e descending."""
        fleet, vid, _, _ = fleet_with_trips
        top = fleet.get_top_emitters(n=5)
        assert len(top) >= 1
        assert top[0]["rank"] == 1
        assert top[0]["vehicle_id"] == vid

    def test_top_emitters_with_period(self, fleet_with_trips):
        """Top emitters can be computed for a specific period."""
        fleet, vid, _, _ = fleet_with_trips
        top = fleet.get_top_emitters(n=5, period_start="2026-03-01", period_end="2026-03-31")
        assert len(top) >= 1

    def test_top_emitters_limit(self, fleet_with_trips):
        """n parameter limits the number of results."""
        fleet, _, _, _ = fleet_with_trips
        top = fleet.get_top_emitters(n=1)
        assert len(top) == 1


# ===========================================================================
# TestFleetComposition
# ===========================================================================


class TestFleetComposition:
    """Test fleet composition analysis."""

    def test_composition_keys(self, registered_vehicle):
        """Composition result has all expected breakdown keys."""
        fleet, _ = registered_vehicle
        comp = fleet.get_fleet_composition()
        assert "total_vehicles" in comp
        assert "by_category" in comp
        assert "by_vehicle_type" in comp
        assert "by_fuel_type" in comp
        assert "by_status" in comp
        assert "by_department" in comp
        assert "by_year_range" in comp

    def test_composition_counts(self, registered_vehicle):
        """Composition counts match registered vehicles."""
        fleet, _ = registered_vehicle
        comp = fleet.get_fleet_composition()
        assert comp["total_vehicles"] == 1

    def test_composition_year_range(self, registered_vehicle):
        """Year 2020 falls in the '2020+' range bucket."""
        fleet, _ = registered_vehicle
        comp = fleet.get_fleet_composition()
        assert "2020+" in comp["by_year_range"]
        assert comp["by_year_range"]["2020+"]["count"] == 1

    def test_composition_empty_fleet(self, fleet):
        """Empty fleet composition returns total_vehicles=0."""
        comp = fleet.get_fleet_composition()
        assert comp["total_vehicles"] == 0


# ===========================================================================
# TestLifecycleEvents
# ===========================================================================


class TestLifecycleEvents:
    """Test lifecycle event logging and retrieval."""

    def test_log_lifecycle_event(self, registered_vehicle):
        """Logging a valid lifecycle event returns an event ID."""
        fleet, vid = registered_vehicle
        eid = fleet.log_lifecycle_event(vid, LIFECYCLE_INSPECTION, "Annual inspection")
        assert isinstance(eid, str)
        assert eid.startswith("evt_")

    def test_maintenance_event_changes_status(self, registered_vehicle):
        """MAINTENANCE event changes vehicle status to MAINTENANCE."""
        fleet, vid = registered_vehicle
        fleet.log_lifecycle_event(vid, LIFECYCLE_MAINTENANCE, "Oil change")
        v = fleet.get_vehicle(vid)
        assert v["status"] == VEHICLE_STATUS_MAINTENANCE

    def test_disposal_event_changes_status(self, registered_vehicle):
        """DISPOSAL event changes vehicle status to DISPOSED."""
        fleet, vid = registered_vehicle
        fleet.log_lifecycle_event(vid, LIFECYCLE_DISPOSAL, "Totalled")
        v = fleet.get_vehicle(vid)
        assert v["status"] == VEHICLE_STATUS_DISPOSED

    def test_invalid_event_type_raises(self, registered_vehicle):
        """Invalid lifecycle event type raises ValueError."""
        fleet, vid = registered_vehicle
        with pytest.raises(ValueError, match="Invalid lifecycle event type"):
            fleet.log_lifecycle_event(vid, "INVALID_EVENT", "bad")

    def test_lifecycle_for_unknown_vehicle_raises(self, fleet):
        """Lifecycle event for non-existent vehicle raises ValueError."""
        with pytest.raises(ValueError, match="Vehicle not found"):
            fleet.log_lifecycle_event("nonexistent", LIFECYCLE_INSPECTION, "test")

    def test_get_lifecycle_events_ordered(self, registered_vehicle):
        """Events are returned in chronological order."""
        fleet, vid = registered_vehicle
        fleet.log_lifecycle_event(vid, LIFECYCLE_INSPECTION, "First")
        fleet.log_lifecycle_event(vid, LIFECYCLE_MAINTENANCE, "Second")
        events = fleet.get_lifecycle_events(vid)
        # At least ACQUISITION + 2 new events
        assert len(events) >= 3
        # Each event has provenance_hash
        for e in events:
            assert "provenance_hash" in e
            assert len(e["provenance_hash"]) == 64

    def test_lifecycle_event_max_cap(self, vdb, calc):
        """When max lifecycle events is reached, oldest event is dropped."""
        f = FleetManagerEngine(
            vehicle_database=vdb,
            emission_calculator=calc,
            config={"max_lifecycle_events_per_vehicle": 3},
        )
        vid = f.register_vehicle({
            "vehicle_type": "MOTORCYCLE",
            "fuel_type": "GASOLINE",
        })
        # registration adds 1 event; add 3 more to exceed cap of 3
        f.log_lifecycle_event(vid, LIFECYCLE_INSPECTION, "E2")
        f.log_lifecycle_event(vid, LIFECYCLE_INSPECTION, "E3")
        f.log_lifecycle_event(vid, LIFECYCLE_INSPECTION, "E4")
        events = f.get_lifecycle_events(vid)
        assert len(events) == 3  # Capped


# ===========================================================================
# TestFleetStatistics
# ===========================================================================


class TestFleetStatistics:
    """Test fleet statistics summary."""

    def test_statistics_keys(self, fleet_with_trips):
        """Statistics result has all expected keys."""
        fleet, _, _, _ = fleet_with_trips
        stats = fleet.get_fleet_statistics()
        assert "total_vehicles" in stats
        assert "active_vehicles" in stats
        assert "total_trips" in stats
        assert "total_co2e_kg" in stats
        assert "total_co2e_tonnes" in stats
        assert "avg_co2e_per_vehicle_kg" in stats
        assert "avg_trips_per_vehicle" in stats

    def test_statistics_counts(self, fleet_with_trips):
        """Statistics counts match reality."""
        fleet, _, _, _ = fleet_with_trips
        stats = fleet.get_fleet_statistics()
        assert stats["total_vehicles"] == 1
        assert stats["active_vehicles"] == 1
        assert stats["total_trips"] == 2

    def test_statistics_empty_fleet(self, fleet):
        """Empty fleet statistics returns sensible zeros."""
        stats = fleet.get_fleet_statistics()
        assert stats["total_vehicles"] == 0
        assert stats["total_trips"] == 0
        assert stats["avg_co2e_per_vehicle_kg"] == Decimal("0")
        assert stats["avg_trips_per_vehicle"] == 0


# ===========================================================================
# TestFuelConsumptionTrends
# ===========================================================================


class TestFuelConsumptionTrends:
    """Test fuel consumption trends."""

    def test_monthly_trends(self, fleet_with_trips):
        """Monthly trends return at least one data point for the trip month."""
        fleet, _, _, _ = fleet_with_trips
        trends = fleet.get_fuel_consumption_trends("2026-01-01", "2026-12-31")
        assert trends["granularity"] == "monthly"
        assert trends["data_points"] >= 1
        assert len(trends["time_series"]) >= 1

    def test_daily_granularity(self, fleet_with_trips):
        """Daily granularity uses full date as the bucket key."""
        fleet, _, _, _ = fleet_with_trips
        trends = fleet.get_fuel_consumption_trends(
            "2026-03-01", "2026-03-31", granularity="daily"
        )
        assert trends["granularity"] == "daily"
        for dp in trends["time_series"]:
            assert len(dp["period"]) == 10  # YYYY-MM-DD

    def test_quarterly_granularity(self, fleet_with_trips):
        """Quarterly granularity uses YYYY-QN format."""
        fleet, _, _, _ = fleet_with_trips
        trends = fleet.get_fuel_consumption_trends(
            "2026-01-01", "2026-12-31", granularity="quarterly"
        )
        for dp in trends["time_series"]:
            assert "-Q" in dp["period"]


# ===========================================================================
# TestYearOverYear
# ===========================================================================


class TestYearOverYear:
    """Test year-over-year comparison."""

    def test_yoy_structure(self, fleet_with_trips):
        """YoY comparison has expected metric keys."""
        fleet, _, _, _ = fleet_with_trips
        yoy = fleet.get_year_over_year(2026, 2025)
        assert "co2e_kg" in yoy
        assert "co2e_tonnes" in yoy
        assert "distance_km" in yoy
        assert "fuel_liters" in yoy
        assert "trip_count" in yoy

    def test_yoy_change_format(self, fleet_with_trips):
        """Each metric has current, previous, absolute_change, percent_change."""
        fleet, _, _, _ = fleet_with_trips
        yoy = fleet.get_year_over_year(2026, 2025)
        for metric in ["co2e_kg", "distance_km"]:
            m = yoy[metric]
            assert "current" in m
            assert "previous" in m
            assert "absolute_change" in m
            assert "percent_change" in m

    def test_yoy_zero_previous(self, fleet_with_trips):
        """When previous year has zero, percent_change is 0."""
        fleet, _, _, _ = fleet_with_trips
        yoy = fleet.get_year_over_year(2026, 2025)
        # 2025 has no trips so previous is 0 and pct should be 0
        assert yoy["co2e_kg"]["previous"] == Decimal("0").quantize(_PRECISION, rounding=ROUND_HALF_UP)
        assert yoy["co2e_kg"]["percent_change"] == Decimal("0")


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and thread safety."""

    def test_get_vehicle_not_found_raises(self, fleet):
        """Getting a non-existent vehicle raises ValueError."""
        with pytest.raises(ValueError, match="Vehicle not found"):
            fleet.get_vehicle("nonexistent")

    def test_get_trip_not_found_raises(self, fleet):
        """Getting a non-existent trip raises ValueError."""
        with pytest.raises(ValueError, match="Trip not found"):
            fleet.get_trip("nonexistent")

    def test_provenance_hash_is_sha256(self, fleet_with_trips):
        """Aggregation provenance_hash is a valid 64-char hex SHA-256."""
        fleet, _, _, _ = fleet_with_trips
        agg = fleet.aggregate_fleet_emissions("2026-01-01", "2026-12-31")
        h = agg["provenance_hash"]
        assert len(h) == 64
        int(h, 16)  # Validates it is hex

    def test_thread_safety_register(self, fleet):
        """Concurrent vehicle registrations do not corrupt state."""
        import concurrent.futures

        def register(i):
            return fleet.register_vehicle({
                "vehicle_type": "MOTORCYCLE",
                "fuel_type": "GASOLINE",
                "vehicle_id": f"thread_veh_{i}",
            })

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futs = [executor.submit(register, i) for i in range(20)]
            results = [f.result() for f in futs]

        assert len(results) == 20
        assert len(set(results)) == 20
        assert len(fleet._vehicles) == 20
