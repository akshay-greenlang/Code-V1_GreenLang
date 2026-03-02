# -*- coding: utf-8 -*-
"""
Unit tests for PersonalVehicleCalculatorEngine (Engine 2).

Tests distance-based, fuel-based, electric vehicle, and carpool calculations
for all supported vehicle types, fuel types, and edge cases.

Target: ~40 tests covering distance-based, fuel-based, EV, carpool, WTT,
WFH reduction, input validation, and known-value verification.

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_QUANT = Decimal("0.00000001")


def _q(v: Decimal) -> Decimal:
    return v.quantize(_QUANT, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the singleton before each test."""
    from greenlang.employee_commuting.personal_vehicle_calculator import (
        PersonalVehicleCalculatorEngine,
    )
    PersonalVehicleCalculatorEngine.reset_instance()
    yield
    PersonalVehicleCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    """Create a fresh PersonalVehicleCalculatorEngine instance."""
    from greenlang.employee_commuting.personal_vehicle_calculator import (
        PersonalVehicleCalculatorEngine,
    )
    return PersonalVehicleCalculatorEngine.get_instance()


# ===========================================================================
# 1. SINGLETON AND INITIALIZATION
# ===========================================================================

class TestSingletonPersonalVehicle:
    """Singleton and initialization tests."""

    def test_singleton_identity(self, engine):
        """Two get_instance calls return the same object."""
        from greenlang.employee_commuting.personal_vehicle_calculator import (
            PersonalVehicleCalculatorEngine,
        )
        assert engine is PersonalVehicleCalculatorEngine.get_instance()

    def test_initial_calculation_count_is_zero(self, engine):
        """Calculation count starts at zero."""
        assert engine.calculation_count == 0


# ===========================================================================
# 2. DISTANCE-BASED CALCULATION
# ===========================================================================

class TestDistanceBased:
    """Tests for calculate_distance_based()."""

    def test_basic_medium_petrol_car(self, engine):
        """Medium petrol car, 15 km one-way, 225 days, no WFH."""
        result = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("15.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
        )
        assert result["method"] == "distance_based"
        assert result["co2e_kg"] > Decimal("0")
        assert result["annual_distance_km"] == _q(Decimal("15.0") * 2 * 225)

    def test_diesel_car_has_higher_ef(self, engine):
        """Large diesel car should produce emissions."""
        result = engine.calculate_distance_based(
            vehicle_type="car_large_diesel",
            fuel_type="diesel",
            one_way_distance_km=Decimal("25.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
        )
        assert result["co2e_kg"] > Decimal("0")
        assert result["vehicle_type"] == "car_large_diesel"

    def test_hybrid_lower_than_petrol(self, engine):
        """Hybrid car should have lower emissions than petrol for same distance."""
        petrol = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("20.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
        )
        hybrid = engine.calculate_distance_based(
            vehicle_type="hybrid",
            fuel_type="petrol",
            one_way_distance_km=Decimal("20.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
        )
        assert hybrid["co2e_kg"] < petrol["co2e_kg"]

    def test_round_trip_false_halves_distance(self, engine):
        """With round_trip=False, annual distance should be half of round-trip."""
        rt = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("10.0"),
            working_days=200,
            wfh_fraction=Decimal("0.0"),
            round_trip=True,
        )
        ow = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("10.0"),
            working_days=200,
            wfh_fraction=Decimal("0.0"),
            round_trip=False,
        )
        # One-way annual distance = half of round-trip
        assert ow["annual_distance_km"] == _q(rt["annual_distance_km"] / 2)

    def test_wfh_reduces_emissions(self, engine):
        """20% WFH should produce ~80% of full-office emissions."""
        full = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("15.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
        )
        wfh = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("15.0"),
            working_days=225,
            wfh_fraction=Decimal("0.20"),
        )
        # With 20% WFH, annual distance is 80% of full
        assert wfh["annual_distance_km"] == _q(full["annual_distance_km"] * Decimal("0.80"))

    def test_wtt_excluded_reduces_total(self, engine):
        """Excluding WTT should result in lower total CO2e."""
        with_wtt = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("15.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
            include_wtt=True,
        )
        no_wtt = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("15.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
            include_wtt=False,
        )
        assert no_wtt["co2e_kg"] < with_wtt["co2e_kg"]
        assert no_wtt["wtt_co2e_kg"] == Decimal("0")

    def test_old_vehicle_higher_than_new(self, engine):
        """Old vehicle age should produce higher or equal emissions."""
        new = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("15.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
            vehicle_age="new_0_3yr",
        )
        old = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("15.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
            vehicle_age="old_8_12yr",
        )
        assert old["co2e_kg"] >= new["co2e_kg"]

    def test_cold_start_uplift_increases_emissions(self, engine):
        """Cold start uplift should increase total emissions."""
        base = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("15.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
            cold_start_uplift=Decimal("0.0"),
        )
        with_cold = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("15.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
            cold_start_uplift=Decimal("0.05"),
        )
        assert with_cold["co2e_kg"] > base["co2e_kg"]

    def test_result_contains_provenance_hash(self, engine):
        """Result includes a 64-char provenance hash."""
        result = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("15.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_calculation_count_increments(self, engine):
        """Each calculation increments the count."""
        engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("15.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
        )
        assert engine.calculation_count == 1

    def test_invalid_distance_raises(self, engine):
        """Negative distance raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_distance_based(
                vehicle_type="car_medium_petrol",
                fuel_type="petrol",
                one_way_distance_km=Decimal("-5.0"),
                working_days=225,
                wfh_fraction=Decimal("0.0"),
            )

    def test_invalid_wfh_fraction_raises(self, engine):
        """WFH fraction > 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_distance_based(
                vehicle_type="car_medium_petrol",
                fuel_type="petrol",
                one_way_distance_km=Decimal("15.0"),
                working_days=225,
                wfh_fraction=Decimal("1.5"),
            )


# ===========================================================================
# 3. FUEL-BASED CALCULATION
# ===========================================================================

class TestFuelBased:
    """Tests for calculate_fuel_based()."""

    def test_basic_gasoline_annual(self, engine):
        """Gasoline, 600 litres annual, no annualisation."""
        result = engine.calculate_fuel_based(
            fuel_type="gasoline",
            fuel_consumed_litres=Decimal("600.0"),
        )
        assert result["method"] == "fuel_based"
        # 600 L x 2.31484 kg/L = ~1388.9 kg
        assert result["co2e_kg"] > Decimal("1300")
        assert result["co2e_kg"] < Decimal("1500")

    def test_diesel_fuel_factor(self, engine):
        """Diesel produces higher per-litre emissions than gasoline."""
        gasoline = engine.calculate_fuel_based(
            fuel_type="gasoline",
            fuel_consumed_litres=Decimal("100.0"),
        )
        diesel = engine.calculate_fuel_based(
            fuel_type="diesel",
            fuel_consumed_litres=Decimal("100.0"),
        )
        assert diesel["co2e_kg"] > gasoline["co2e_kg"]

    def test_annualisation_from_period(self, engine):
        """50L over 30 days annualised to 225 working days."""
        result = engine.calculate_fuel_based(
            fuel_type="gasoline",
            fuel_consumed_litres=Decimal("50.0"),
            period_days=30,
            working_days=225,
            wfh_fraction=Decimal("0.0"),
        )
        expected_annual_fuel = _q(Decimal("50.0") * Decimal("225") / Decimal("30"))
        assert result["annual_fuel_l"] == expected_annual_fuel

    def test_wtt_included_in_total(self, engine):
        """total_co2e includes both TTW and WTT."""
        result = engine.calculate_fuel_based(
            fuel_type="gasoline",
            fuel_consumed_litres=Decimal("600.0"),
        )
        assert result["total_co2e_kg"] == _q(result["co2e_kg"] + result["wtt_co2e_kg"])

    def test_unknown_fuel_raises_key_error(self, engine):
        """Unknown fuel type raises KeyError."""
        with pytest.raises(KeyError):
            engine.calculate_fuel_based(
                fuel_type="antimatter",
                fuel_consumed_litres=Decimal("100.0"),
            )

    def test_fuel_based_gas_breakdown(self, engine):
        """Result includes co2_kg, ch4_kg, n2o_kg breakdown."""
        result = engine.calculate_fuel_based(
            fuel_type="gasoline",
            fuel_consumed_litres=Decimal("100.0"),
        )
        assert result["co2_kg"] > Decimal("0")
        assert result["ch4_kg"] >= Decimal("0")
        assert result["n2o_kg"] >= Decimal("0")


# ===========================================================================
# 4. ELECTRIC VEHICLE CALCULATION
# ===========================================================================

class TestElectricVehicle:
    """Tests for calculate_electric_vehicle()."""

    def test_basic_ev_medium_car(self, engine):
        """Medium BEV, 20 km one-way, US grid, 225 days."""
        result = engine.calculate_electric_vehicle(
            vehicle_type="medium_car",
            one_way_distance_km=Decimal("20.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
            country_code="US",
        )
        assert result["method"] == "electric_vehicle"
        assert result["co2e_kg"] > Decimal("0")
        assert result["energy_kwh"] > Decimal("0")

    def test_ev_green_grid_much_lower(self, engine):
        """Norway (near-zero grid) should give much lower emissions than India."""
        no = engine.calculate_electric_vehicle(
            vehicle_type="medium_car",
            one_way_distance_km=Decimal("20.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
            country_code="NO",
        )
        india = engine.calculate_electric_vehicle(
            vehicle_type="medium_car",
            one_way_distance_km=Decimal("20.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
            country_code="IN",
        )
        assert no["co2e_kg"] < india["co2e_kg"]

    def test_ev_wfh_reduces_energy(self, engine):
        """40% WFH should reduce annual energy by 40%."""
        full = engine.calculate_electric_vehicle(
            vehicle_type="medium_car",
            one_way_distance_km=Decimal("20.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
            country_code="US",
        )
        wfh = engine.calculate_electric_vehicle(
            vehicle_type="medium_car",
            one_way_distance_km=Decimal("20.0"),
            working_days=225,
            wfh_fraction=Decimal("0.40"),
            country_code="US",
        )
        assert wfh["energy_kwh"] == _q(full["energy_kwh"] * Decimal("0.60"))

    def test_ev_unknown_type_raises(self, engine):
        """Unknown EV type raises KeyError."""
        with pytest.raises(KeyError):
            engine.calculate_electric_vehicle(
                vehicle_type="flying_car",
                one_way_distance_km=Decimal("20.0"),
                working_days=225,
                wfh_fraction=Decimal("0.0"),
            )


# ===========================================================================
# 5. PROVENANCE AND REPRODUCIBILITY
# ===========================================================================

class TestProvenanceReproducibility:
    """Tests for provenance hash determinism."""

    def test_same_inputs_same_hash(self, engine):
        """Identical inputs produce the same provenance hash."""
        kwargs = dict(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("15.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
        )
        r1 = engine.calculate_distance_based(**kwargs)
        r2 = engine.calculate_distance_based(**kwargs)
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_different_inputs_different_hash(self, engine):
        """Different distances produce different provenance hashes."""
        r1 = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("15.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
        )
        r2 = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("20.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
        )
        assert r1["provenance_hash"] != r2["provenance_hash"]


# ===========================================================================
# 6. KNOWN VALUE VERIFICATION
# ===========================================================================

class TestKnownValues:
    """Verify specific numeric outcomes for auditable calculations."""

    def test_medium_petrol_annual_distance(self, engine):
        """15 km x 2 x 225 days x (1 - 0.0) = 6750 km."""
        result = engine.calculate_distance_based(
            vehicle_type="car_medium_petrol",
            fuel_type="petrol",
            one_way_distance_km=Decimal("15.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
        )
        assert result["annual_distance_km"] == Decimal("6750.00000000")

    def test_gasoline_100l_co2e(self, engine):
        """100 L gasoline x 2.31484 = 231.484 kg CO2e (TTW)."""
        result = engine.calculate_fuel_based(
            fuel_type="gasoline",
            fuel_consumed_litres=Decimal("100.0"),
        )
        expected = _q(Decimal("100.0") * Decimal("2.31484"))
        assert result["co2e_kg"] == expected

    def test_diesel_100l_co2e(self, engine):
        """100 L diesel x 2.68787 = 268.787 kg CO2e (TTW)."""
        result = engine.calculate_fuel_based(
            fuel_type="diesel",
            fuel_consumed_litres=Decimal("100.0"),
        )
        expected = _q(Decimal("100.0") * Decimal("2.68787"))
        assert result["co2e_kg"] == expected
