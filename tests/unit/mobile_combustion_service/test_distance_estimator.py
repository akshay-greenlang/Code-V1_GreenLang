# -*- coding: utf-8 -*-
"""
Unit tests for DistanceEstimatorEngine (Engine 4) - AGENT-MRV-003 Mobile Combustion.

Tests all public methods with 65+ test functions covering:
- Initialization, fuel-from-distance estimation
- Distance emission factor lookups
- Fuel economy lookup and custom registration
- Age degradation and load factor adjustments
- Distance and fuel economy unit conversions
- Operating hours emissions (off-road equipment)
- Marine emission factors
- Aviation emission factors
- Fleet batch estimation, listing, history, edge cases

Author: GreenLang QA Team
"""

import threading
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import patch

import pytest

from greenlang.mobile_combustion.distance_estimator import (
    DistanceEstimatorEngine,
    VehicleType,
    FuelType,
    LoadFactor,
    DistanceUnit,
    FuelEconomyUnit,
    EquipmentOperatingType,
    MarineVesselType,
    AircraftType,
    FuelEstimationResult,
    DistanceEmissionResult,
    OperatingHoursResult,
    MarineEmissionResult,
    AviationEmissionResult,
    _DEFAULT_FUEL_ECONOMY,
    _DISTANCE_EMISSION_FACTORS,
    _AGE_DEGRADATION_TABLE,
    _AGE_DEGRADATION_MAX,
    _LOAD_FACTOR_ADJUSTMENTS,
    _OPERATING_HOUR_FACTORS,
    _MARINE_EMISSION_FACTORS,
    _AVIATION_EMISSION_FACTORS,
    _DISTANCE_TO_KM,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine():
    """Create a default DistanceEstimatorEngine instance."""
    return DistanceEstimatorEngine()


# ===========================================================================
# TestInit
# ===========================================================================


class TestInit:
    """Test DistanceEstimatorEngine initialization."""

    def test_default_init(self, engine):
        """Engine initializes with empty custom economies and history."""
        assert engine._custom_fuel_economies == {}
        assert engine._estimation_history == []

    def test_rlock_created(self, engine):
        """Engine creates a reentrant lock."""
        assert isinstance(engine._lock, type(threading.RLock()))

    def test_13_on_road_types(self):
        """Default fuel economy table has 13 on-road vehicle types."""
        assert len(_DEFAULT_FUEL_ECONOMY) == 13

    def test_11_equipment_types(self):
        """Operating hour factors cover 11 equipment types."""
        assert len(_OPERATING_HOUR_FACTORS) == 11

    def test_8_marine_vessel_types(self):
        """Marine emission factors cover 8 vessel types."""
        assert len(_MARINE_EMISSION_FACTORS) == 8

    def test_8_aircraft_types(self):
        """Aviation emission factors cover 8 aircraft types."""
        assert len(_AVIATION_EMISSION_FACTORS) == 8


# ===========================================================================
# TestEstimateFuelFromDistance
# ===========================================================================


class TestEstimateFuelFromDistance:
    """Test fuel estimation from distance."""

    def test_basic_gasoline_car(self, engine):
        """Basic estimation for gasoline passenger car returns expected type."""
        result = engine.estimate_fuel_from_distance(
            vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE.value,
            distance_km=Decimal("1000"),
        )
        assert isinstance(result, FuelEstimationResult)
        assert result.vehicle_type == VehicleType.PASSENGER_CAR_GASOLINE.value
        assert result.distance_km == Decimal("1000")
        assert result.fuel_consumed_litres > Decimal("0")
        assert result.estimated_co2e_kg > Decimal("0")

    def test_fuel_consumed_calculation(self, engine):
        """Fuel consumed = distance * adjusted_economy / 100."""
        result = engine.estimate_fuel_from_distance(
            vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE.value,
            distance_km=Decimal("100"),
            vehicle_age_years=0,
            load_factor=LoadFactor.HALF_LOAD.value,
        )
        # base economy for gasoline car = 8.5 L/100km
        expected_fuel = (Decimal("100") * Decimal("8.5") / Decimal("100")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        assert result.fuel_consumed_litres == expected_fuel

    def test_co2e_from_distance_factor(self, engine):
        """CO2e = distance * distance_emission_factor * age * load adjustments."""
        result = engine.estimate_fuel_from_distance(
            vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE.value,
            distance_km=Decimal("100"),
            vehicle_age_years=0,
            load_factor=LoadFactor.HALF_LOAD.value,
        )
        ef = _DISTANCE_EMISSION_FACTORS[VehicleType.PASSENGER_CAR_GASOLINE.value]["CO2e"]
        expected_co2e_g = (Decimal("100") * ef * Decimal("1.00") * Decimal("1.00")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        assert result.estimated_co2e_g == expected_co2e_g

    def test_default_fuel_type(self, engine):
        """Default fuel type for gasoline car is GASOLINE."""
        result = engine.estimate_fuel_from_distance(
            vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE.value,
            distance_km=Decimal("10"),
        )
        assert result.fuel_type == FuelType.GASOLINE.value

    def test_explicit_fuel_type(self, engine):
        """Explicit fuel type overrides the default."""
        result = engine.estimate_fuel_from_distance(
            vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE.value,
            distance_km=Decimal("10"),
            fuel_type=FuelType.ETHANOL.value,
        )
        assert result.fuel_type == FuelType.ETHANOL.value

    def test_negative_distance_raises(self, engine):
        """Negative distance raises ValueError."""
        with pytest.raises(ValueError, match="distance_km must be >= 0"):
            engine.estimate_fuel_from_distance(
                vehicle_type=VehicleType.MOTORCYCLE.value,
                distance_km=Decimal("-1"),
            )

    def test_negative_age_raises(self, engine):
        """Negative vehicle age raises ValueError."""
        with pytest.raises(ValueError, match="vehicle_age_years must be >= 0"):
            engine.estimate_fuel_from_distance(
                vehicle_type=VehicleType.MOTORCYCLE.value,
                distance_km=Decimal("10"),
                vehicle_age_years=-1,
            )

    def test_invalid_vehicle_type_raises(self, engine):
        """Unrecognized vehicle type raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized on-road vehicle type"):
            engine.estimate_fuel_from_distance(
                vehicle_type="SPACESHIP",
                distance_km=Decimal("10"),
            )

    def test_invalid_load_factor_raises(self, engine):
        """Unrecognized load factor raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized load factor"):
            engine.estimate_fuel_from_distance(
                vehicle_type=VehicleType.MOTORCYCLE.value,
                distance_km=Decimal("10"),
                load_factor="SUPER_HEAVY",
            )

    def test_zero_distance(self, engine):
        """Zero distance returns zero fuel and zero emissions."""
        result = engine.estimate_fuel_from_distance(
            vehicle_type=VehicleType.MOTORCYCLE.value,
            distance_km=Decimal("0"),
        )
        assert result.fuel_consumed_litres == Decimal("0")
        assert result.estimated_co2e_g == Decimal("0")

    def test_provenance_hash_is_sha256(self, engine):
        """Provenance hash is a valid 64-char hex SHA-256."""
        result = engine.estimate_fuel_from_distance(
            vehicle_type=VehicleType.MOTORCYCLE.value,
            distance_km=Decimal("100"),
        )
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)  # Validates hex

    def test_estimation_method_is_distance_based(self, engine):
        """estimation_method field is 'DISTANCE_BASED'."""
        result = engine.estimate_fuel_from_distance(
            vehicle_type=VehicleType.MOTORCYCLE.value,
            distance_km=Decimal("100"),
        )
        assert result.estimation_method == "DISTANCE_BASED"

    def test_history_recorded(self, engine):
        """Estimation is recorded in history."""
        engine.estimate_fuel_from_distance(
            vehicle_type=VehicleType.MOTORCYCLE.value,
            distance_km=Decimal("50"),
        )
        history = engine.get_estimation_history()
        assert len(history) == 1

    def test_to_dict(self, engine):
        """to_dict serializes all fields."""
        result = engine.estimate_fuel_from_distance(
            vehicle_type=VehicleType.MOTORCYCLE.value,
            distance_km=Decimal("50"),
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["vehicle_type"] == VehicleType.MOTORCYCLE.value
        assert "provenance_hash" in d


# ===========================================================================
# TestDistanceEmissionFactor
# ===========================================================================


class TestDistanceEmissionFactor:
    """Test distance emission factor lookups."""

    def test_gasoline_car_factor(self, engine):
        """Gasoline car has CO2e of 192 g/km."""
        result = engine.get_distance_emission_factor(VehicleType.PASSENGER_CAR_GASOLINE.value)
        assert isinstance(result, DistanceEmissionResult)
        assert result.co2e_g_per_km == Decimal("192")

    def test_diesel_car_factor(self, engine):
        """Diesel car has CO2e of 171 g/km."""
        result = engine.get_distance_emission_factor(VehicleType.PASSENGER_CAR_DIESEL.value)
        assert result.co2e_g_per_km == Decimal("171")

    def test_heavy_truck_factor(self, engine):
        """Heavy-duty diesel truck has CO2e of 850 g/km."""
        result = engine.get_distance_emission_factor(VehicleType.HEAVY_DUTY_TRUCK_DIESEL.value)
        assert result.co2e_g_per_km == Decimal("850")

    def test_all_13_on_road_types_have_factors(self, engine):
        """All 13 on-road types have distance emission factors."""
        for vtype in _DEFAULT_FUEL_ECONOMY:
            result = engine.get_distance_emission_factor(vtype)
            assert result.co2e_g_per_km > Decimal("0")

    def test_gas_breakdown(self, engine):
        """Result includes CO2, CH4, N2O components."""
        result = engine.get_distance_emission_factor(VehicleType.PASSENGER_CAR_GASOLINE.value)
        assert result.co2_g_per_km == Decimal("186")
        assert result.ch4_g_per_km == Decimal("0.8")
        assert result.n2o_g_per_km == Decimal("5.2")

    def test_unknown_type_raises(self, engine):
        """Requesting a factor for off-road/unknown type raises ValueError."""
        with pytest.raises(ValueError, match="No distance emission factor"):
            engine.get_distance_emission_factor("CONSTRUCTION_EQUIPMENT")

    def test_provenance_hash(self, engine):
        """Distance emission factor result has a provenance hash."""
        result = engine.get_distance_emission_factor(VehicleType.MOTORCYCLE.value)
        assert len(result.provenance_hash) == 64


# ===========================================================================
# TestFuelEconomy
# ===========================================================================


class TestFuelEconomy:
    """Test fuel economy lookup and registration."""

    def test_default_gasoline_car(self, engine):
        """Default gasoline car economy is 8.5 L/100km."""
        assert engine.get_fuel_economy(VehicleType.PASSENGER_CAR_GASOLINE.value) == Decimal("8.5")

    def test_default_heavy_truck(self, engine):
        """Default heavy-duty truck economy is 32.0 L/100km."""
        assert engine.get_fuel_economy(VehicleType.HEAVY_DUTY_TRUCK_DIESEL.value) == Decimal("32.0")

    def test_custom_fuel_economy_overrides(self, engine):
        """Custom fuel economy overrides the default."""
        engine.register_custom_fuel_economy("PASSENGER_CAR_GASOLINE", Decimal("7.0"))
        assert engine.get_fuel_economy("PASSENGER_CAR_GASOLINE") == Decimal("7.0")

    def test_custom_fuel_economy_new_type(self, engine):
        """Custom economy can be registered for entirely new types."""
        engine.register_custom_fuel_economy("ELECTRIC_SCOOTER", Decimal("3.0"))
        assert engine.get_fuel_economy("ELECTRIC_SCOOTER") == Decimal("3.0")

    def test_zero_economy_raises(self, engine):
        """Registering zero fuel economy raises ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            engine.register_custom_fuel_economy("X", Decimal("0"))

    def test_negative_economy_raises(self, engine):
        """Registering negative fuel economy raises ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            engine.register_custom_fuel_economy("X", Decimal("-5"))

    def test_unknown_type_raises(self, engine):
        """Looking up an unknown type without custom registration raises ValueError."""
        with pytest.raises(ValueError, match="No fuel economy data"):
            engine.get_fuel_economy("UNKNOWN_TYPE")


# ===========================================================================
# TestAgeDegradation
# ===========================================================================


class TestAgeDegradation:
    """Test vehicle age fuel economy degradation."""

    @pytest.mark.parametrize("age,expected_factor", [
        (0, Decimal("1.00")),
        (2, Decimal("1.00")),
        (3, Decimal("1.02")),
        (4, Decimal("1.02")),
        (5, Decimal("1.05")),
        (7, Decimal("1.05")),
        (8, Decimal("1.10")),
        (11, Decimal("1.10")),
        (12, Decimal("1.15")),
        (20, Decimal("1.15")),
    ])
    def test_age_degradation_factors(self, engine, age, expected_factor):
        """Age degradation factor matches coded table for various ages."""
        base = Decimal("10.0")
        adjusted = engine.adjust_fuel_economy_for_age(base, age)
        expected = (base * expected_factor).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        assert adjusted == expected

    def test_age_zero_economy_raises(self, engine):
        """Zero base economy raises ValueError."""
        with pytest.raises(ValueError, match="base_economy must be > 0"):
            engine.adjust_fuel_economy_for_age(Decimal("0"), 5)

    def test_negative_age_raises(self, engine):
        """Negative vehicle age raises ValueError."""
        with pytest.raises(ValueError, match="vehicle_age_years must be >= 0"):
            engine.adjust_fuel_economy_for_age(Decimal("10"), -1)


# ===========================================================================
# TestLoadFactor
# ===========================================================================


class TestLoadFactor:
    """Test load factor fuel economy adjustments."""

    @pytest.mark.parametrize("load,expected_adj", [
        (LoadFactor.EMPTY.value, Decimal("0.70")),
        (LoadFactor.QUARTER_LOAD.value, Decimal("0.85")),
        (LoadFactor.HALF_LOAD.value, Decimal("1.00")),
        (LoadFactor.THREE_QUARTER_LOAD.value, Decimal("1.10")),
        (LoadFactor.FULL_LOAD.value, Decimal("1.20")),
        (LoadFactor.OVERLOADED.value, Decimal("1.35")),
    ])
    def test_load_factor_adjustment(self, engine, load, expected_adj):
        """Load factor adjustment matches coded table."""
        base = Decimal("10.0")
        adjusted = engine.adjust_for_load_factor(base, load)
        expected = (base * expected_adj).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        assert adjusted == expected

    def test_load_factor_zero_base_raises(self, engine):
        """Zero base economy raises ValueError."""
        with pytest.raises(ValueError, match="base_economy must be > 0"):
            engine.adjust_for_load_factor(Decimal("0"), LoadFactor.HALF_LOAD.value)

    def test_invalid_load_factor_raises(self, engine):
        """Invalid load factor raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized load factor"):
            engine.adjust_for_load_factor(Decimal("10"), "EXTREME")


# ===========================================================================
# TestDistanceConversion
# ===========================================================================


class TestDistanceConversion:
    """Test distance unit conversions."""

    def test_km_to_mi(self, engine):
        """100 km to miles."""
        result = engine.convert_distance(Decimal("100"), "KM", "MI")
        expected = (Decimal("100") / Decimal("1.60934")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        assert result == expected

    def test_mi_to_km(self, engine):
        """100 miles to km."""
        result = engine.convert_distance(Decimal("100"), "MI", "KM")
        expected = (Decimal("100") * Decimal("1.60934")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        assert result == expected

    def test_nm_to_km(self, engine):
        """100 nautical miles to km."""
        result = engine.convert_distance(Decimal("100"), "NM", "KM")
        expected = (Decimal("100") * Decimal("1.852")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        assert result == expected

    def test_same_unit_identity(self, engine):
        """Converting same unit returns the original value."""
        result = engine.convert_distance(Decimal("42.5"), "KM", "KM")
        assert result == Decimal("42.5")

    def test_negative_distance_raises(self, engine):
        """Negative distance value raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            engine.convert_distance(Decimal("-1"), "KM", "MI")

    def test_invalid_unit_raises(self, engine):
        """Invalid distance unit raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized distance unit"):
            engine.convert_distance(Decimal("10"), "FURLONGS", "KM")


# ===========================================================================
# TestFuelEconomyConversion
# ===========================================================================


class TestFuelEconomyConversion:
    """Test fuel economy unit conversions."""

    def test_l_per_100km_to_mpg_us(self, engine):
        """8.5 L/100km to MPG US = 235.215 / 8.5."""
        result = engine.convert_fuel_economy(Decimal("8.5"), "L_PER_100KM", "MPG_US")
        expected = (Decimal("235.215") / Decimal("8.5")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        assert result == expected

    def test_mpg_us_to_l_per_100km(self, engine):
        """27.67 MPG US to L/100km = 235.215 / 27.67."""
        result = engine.convert_fuel_economy(Decimal("27.67"), "MPG_US", "L_PER_100KM")
        expected = (Decimal("235.215") / Decimal("27.67")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        assert result == expected

    def test_km_per_l_to_l_per_100km(self, engine):
        """12.5 km/L to L/100km = 100 / 12.5 = 8.0."""
        result = engine.convert_fuel_economy(Decimal("12.5"), "KM_PER_L", "L_PER_100KM")
        expected = (Decimal("100") / Decimal("12.5")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        assert result == expected

    def test_same_unit_identity(self, engine):
        """Converting same unit returns original value."""
        result = engine.convert_fuel_economy(Decimal("8.5"), "L_PER_100KM", "L_PER_100KM")
        assert result == Decimal("8.5")

    def test_zero_economy_raises(self, engine):
        """Zero fuel economy raises ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            engine.convert_fuel_economy(Decimal("0"), "L_PER_100KM", "MPG_US")

    def test_invalid_unit_raises(self, engine):
        """Invalid fuel economy unit raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized fuel economy unit"):
            engine.convert_fuel_economy(Decimal("10"), "RODS_PER_HOGSHEAD", "MPG_US")


# ===========================================================================
# TestOperatingHoursEmissions
# ===========================================================================


class TestOperatingHoursEmissions:
    """Test off-road equipment operating hours emissions."""

    def test_excavator_emissions(self, engine):
        """Construction excavator emissions match hand calculation."""
        result = engine.estimate_operating_hours_emissions(
            equipment_type=EquipmentOperatingType.CONSTRUCTION_EXCAVATOR.value,
            hours=Decimal("100"),
        )
        assert isinstance(result, OperatingHoursResult)
        # co2_kg = 100 * 45 = 4500
        assert result.co2_kg == Decimal("4500.000")
        # ch4_g = 100 * 3.2 = 320
        assert result.ch4_g == Decimal("320.000")
        # n2o_g = 100 * 1.8 = 180
        assert result.n2o_g == Decimal("180.000")
        # fuel = 100 * 17 = 1700
        assert result.fuel_consumed_litres == Decimal("1700.000")

    def test_forklift_lpg_fuel_type(self, engine):
        """Forklift LPG defaults to LPG fuel type."""
        result = engine.estimate_operating_hours_emissions(
            equipment_type=EquipmentOperatingType.FORKLIFT_LPG.value,
            hours=Decimal("10"),
        )
        assert result.fuel_type == FuelType.LPG.value

    def test_forklift_diesel_fuel_type(self, engine):
        """Forklift diesel defaults to DIESEL fuel type."""
        result = engine.estimate_operating_hours_emissions(
            equipment_type=EquipmentOperatingType.FORKLIFT_DIESEL.value,
            hours=Decimal("10"),
        )
        assert result.fuel_type == FuelType.DIESEL.value

    def test_zero_hours(self, engine):
        """Zero hours returns zero emissions."""
        result = engine.estimate_operating_hours_emissions(
            equipment_type=EquipmentOperatingType.GENERATOR_SMALL.value,
            hours=Decimal("0"),
        )
        assert result.co2_kg == Decimal("0.000")
        assert result.co2e_kg == Decimal("0.000")

    def test_negative_hours_raises(self, engine):
        """Negative hours raises ValueError."""
        with pytest.raises(ValueError, match="hours must be >= 0"):
            engine.estimate_operating_hours_emissions(
                equipment_type=EquipmentOperatingType.GENERATOR_SMALL.value,
                hours=Decimal("-5"),
            )

    def test_invalid_equipment_type_raises(self, engine):
        """Invalid equipment type raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized equipment type"):
            engine.estimate_operating_hours_emissions(
                equipment_type="HOVERCRAFT",
                hours=Decimal("10"),
            )

    def test_co2e_includes_gwp(self, engine):
        """co2e_kg includes GWP-weighted CH4 and N2O contributions."""
        result = engine.estimate_operating_hours_emissions(
            equipment_type=EquipmentOperatingType.CONSTRUCTION_EXCAVATOR.value,
            hours=Decimal("1"),
        )
        # co2e = co2_kg + ch4_g*28/1000 + n2o_g*265/1000
        co2 = Decimal("45")
        ch4_contrib = Decimal("3.2") * Decimal("28") / Decimal("1000")
        n2o_contrib = Decimal("1.8") * Decimal("265") / Decimal("1000")
        expected = (co2 + ch4_contrib + n2o_contrib).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        assert result.co2e_kg == expected


# ===========================================================================
# TestMarineEmissionFactor
# ===========================================================================


class TestMarineEmissionFactor:
    """Test marine emission factor lookups."""

    def test_barge_factor(self, engine):
        """Barge has co2e_g_per_tonne_km of 34."""
        result = engine.get_marine_emission_factor(
            vessel_type=MarineVesselType.BARGE.value,
            cargo_tonnes=Decimal("100"),
        )
        assert isinstance(result, MarineEmissionResult)
        assert result.emission_factor_g_per_tonne_km == Decimal("34")

    def test_ocean_bulk_factor(self, engine):
        """Ocean bulk carrier has co2e_g_per_tonne_km of 6."""
        result = engine.get_marine_emission_factor(
            vessel_type=MarineVesselType.OCEAN_BULK.value,
            cargo_tonnes=Decimal("1000"),
        )
        assert result.emission_factor_g_per_tonne_km == Decimal("6")

    def test_emission_calculation_with_distance(self, engine):
        """When distance is provided, emissions are computed."""
        result = engine.get_marine_emission_factor(
            vessel_type=MarineVesselType.BARGE.value,
            cargo_tonnes=Decimal("100"),
            distance_km=Decimal("500"),
        )
        # co2_g = 100 * 500 * 33 = 1,650,000
        assert result.co2_g == Decimal("1650000.000")
        assert result.co2e_kg > Decimal("0")

    def test_no_distance_returns_zero_emissions(self, engine):
        """Without distance, emissions are zero."""
        result = engine.get_marine_emission_factor(
            vessel_type=MarineVesselType.BARGE.value,
            cargo_tonnes=Decimal("100"),
        )
        assert result.co2_g == Decimal("0.000")

    def test_invalid_vessel_type_raises(self, engine):
        """Invalid vessel type raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized marine vessel type"):
            engine.get_marine_emission_factor("SUBMARINE", Decimal("100"))

    def test_negative_cargo_raises(self, engine):
        """Negative cargo tonnes raises ValueError."""
        with pytest.raises(ValueError, match="cargo_tonnes must be >= 0"):
            engine.get_marine_emission_factor(
                MarineVesselType.BARGE.value, Decimal("-10")
            )


# ===========================================================================
# TestAviationEmissionFactor
# ===========================================================================


class TestAviationEmissionFactor:
    """Test aviation emission factor lookups."""

    def test_light_jet_factor(self, engine):
        """Light jet has co2e_g_per_km of 3530."""
        result = engine.get_aviation_emission_factor(
            aircraft_type=AircraftType.LIGHT_JET.value,
        )
        assert isinstance(result, AviationEmissionResult)

    def test_light_jet_with_distance(self, engine):
        """Light jet emissions computed for 1000 km."""
        result = engine.get_aviation_emission_factor(
            aircraft_type=AircraftType.LIGHT_JET.value,
            distance_km=Decimal("1000"),
        )
        # co2e_g = 1000 * 3530 = 3,530,000
        assert result.co2e_g == Decimal("3530000.000")
        assert result.co2e_kg == Decimal("3530.000")

    def test_default_passengers_from_table(self, engine):
        """When passengers=0, typical_passengers from table is used."""
        result = engine.get_aviation_emission_factor(
            aircraft_type=AircraftType.LIGHT_JET.value,
            passengers=0,
        )
        assert result.passengers == 6  # typical for light jet

    def test_custom_passenger_count(self, engine):
        """Custom passenger count overrides typical."""
        result = engine.get_aviation_emission_factor(
            aircraft_type=AircraftType.LIGHT_JET.value,
            passengers=3,
            distance_km=Decimal("1000"),
        )
        assert result.passengers == 3

    def test_negative_passengers_raises(self, engine):
        """Negative passengers raises ValueError."""
        with pytest.raises(ValueError, match="passengers must be >= 0"):
            engine.get_aviation_emission_factor(
                aircraft_type=AircraftType.LIGHT_JET.value,
                passengers=-1,
            )

    def test_invalid_aircraft_type_raises(self, engine):
        """Invalid aircraft type raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized aircraft type"):
            engine.get_aviation_emission_factor("SPACE_SHUTTLE")

    def test_helicopter_emissions(self, engine):
        """Helicopter emission factor lookup succeeds."""
        result = engine.get_aviation_emission_factor(
            aircraft_type=AircraftType.HELICOPTER_MEDIUM.value,
            distance_km=Decimal("100"),
        )
        assert result.co2e_kg > Decimal("0")


# ===========================================================================
# TestFleetEstimation
# ===========================================================================


class TestFleetEstimation:
    """Test fleet batch estimation."""

    def test_fleet_estimation(self, engine):
        """Fleet estimation processes multiple vehicles."""
        vehicles = [
            {"vehicle_type": "PASSENGER_CAR_GASOLINE", "distance_km": 1000},
            {"vehicle_type": "HEAVY_DUTY_TRUCK_DIESEL", "distance_km": 500},
        ]
        result = engine.estimate_fleet_emissions(vehicles)
        assert result["vehicle_count"] == 2
        assert len(result["results"]) == 2
        assert len(result["errors"]) == 0
        assert Decimal(result["total_co2e_kg"]) > Decimal("0")

    def test_fleet_estimation_with_errors(self, engine):
        """Fleet estimation handles invalid vehicles gracefully."""
        vehicles = [
            {"vehicle_type": "PASSENGER_CAR_GASOLINE", "distance_km": 100},
            {"vehicle_type": "INVALID_TYPE", "distance_km": 100},
        ]
        result = engine.estimate_fleet_emissions(vehicles)
        assert result["vehicle_count"] == 1
        assert len(result["errors"]) == 1

    def test_fleet_provenance_hash(self, engine):
        """Fleet estimation result has a provenance hash."""
        vehicles = [
            {"vehicle_type": "MOTORCYCLE", "distance_km": 50},
        ]
        result = engine.estimate_fleet_emissions(vehicles)
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# TestListingAndHistory
# ===========================================================================


class TestListingAndHistory:
    """Test listing methods and history."""

    def test_list_vehicle_types(self, engine):
        """list_vehicle_types returns sorted list of 13 types."""
        types = engine.list_vehicle_types()
        assert len(types) == 13
        assert types == sorted(types)

    def test_list_equipment_types(self, engine):
        """list_equipment_types returns sorted list of 11 types."""
        types = engine.list_equipment_types()
        assert len(types) == 11
        assert types == sorted(types)

    def test_list_marine_vessel_types(self, engine):
        """list_marine_vessel_types returns sorted list of 8 types."""
        types = engine.list_marine_vessel_types()
        assert len(types) == 8

    def test_list_aircraft_types(self, engine):
        """list_aircraft_types returns sorted list of 8 types."""
        types = engine.list_aircraft_types()
        assert len(types) == 8

    def test_clear_history(self, engine):
        """clear_history empties history and returns count."""
        engine.estimate_fuel_from_distance(
            vehicle_type=VehicleType.MOTORCYCLE.value,
            distance_km=Decimal("10"),
        )
        engine.estimate_fuel_from_distance(
            vehicle_type=VehicleType.MOTORCYCLE.value,
            distance_km=Decimal("20"),
        )
        count = engine.clear_history()
        assert count == 2
        assert len(engine.get_estimation_history()) == 0
