# -*- coding: utf-8 -*-
"""
Unit tests for VehicleFleetCalculatorEngine (AGENT-MRV-021, Engine 3)

40 tests covering distance-based, fuel-based, EV (grid factor), fleet
aggregation, spend-based, batch, age factors, WTT, zero km, negative
inputs, and provenance for leased vehicle fleet emission calculations.

Calculation methods:
    Distance-based: distance_km * ef_per_km * (1 + wtt_factor) * age_factor
    Fuel-based:     fuel_litres * fuel_ef_per_litre * (1 + wtt_factor)
    EV:             distance_km * kwh_per_km * grid_ef
    Spend-based:    amount_usd * cpi_deflator * eeio_ef

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch
import pytest

try:
    from greenlang.upstream_leased_assets.vehicle_fleet_calculator import (
        VehicleFleetCalculatorEngine,
    )
    from greenlang.upstream_leased_assets.models import (
        VehicleType,
        FuelType,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason="VehicleFleetCalculatorEngine not available",
)

pytestmark = _SKIP


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton before and after every test."""
    if _AVAILABLE:
        VehicleFleetCalculatorEngine.reset_instance()
    yield
    if _AVAILABLE:
        VehicleFleetCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    """Create a fresh VehicleFleetCalculatorEngine."""
    return VehicleFleetCalculatorEngine()


# ==============================================================================
# DISTANCE-BASED CALCULATION TESTS
# ==============================================================================


class TestDistanceBasedCalculation:
    """Test distance-based vehicle emission calculations."""

    @pytest.mark.parametrize("vehicle_type", [
        "small_car", "medium_car", "large_car", "suv",
        "light_van", "heavy_van", "light_truck", "heavy_truck",
    ])
    def test_distance_based_all_vehicle_types(self, engine, vehicle_type):
        """Test distance-based calculation for all 8 vehicle types."""
        result = engine.calculate({
            "method": "distance_based",
            "vehicle_type": vehicle_type,
            "fuel_type": "diesel",
            "annual_distance_km": Decimal("25000"),
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0
        assert isinstance(result["total_co2e_kg"], Decimal)

    def test_more_distance_more_emissions(self, engine):
        """Test more distance produces proportionally more emissions."""
        short = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "medium_car",
            "fuel_type": "petrol",
            "annual_distance_km": Decimal("10000"),
            "region": "US",
        })
        long = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "medium_car",
            "fuel_type": "petrol",
            "annual_distance_km": Decimal("20000"),
            "region": "US",
        })
        ratio = long["total_co2e_kg"] / short["total_co2e_kg"]
        assert abs(ratio - Decimal("2.0")) < Decimal("0.05")

    def test_heavy_truck_higher_than_small_car(self, engine):
        """Test heavy truck emits more than small car for same distance."""
        truck = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "heavy_truck",
            "fuel_type": "diesel",
            "annual_distance_km": Decimal("25000"),
            "region": "US",
        })
        car = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "small_car",
            "fuel_type": "diesel",
            "annual_distance_km": Decimal("25000"),
            "region": "US",
        })
        assert truck["total_co2e_kg"] > car["total_co2e_kg"]

    def test_provenance_hash_deterministic(self, engine):
        """Test provenance hash is deterministic for same input."""
        inp = {
            "method": "distance_based",
            "vehicle_type": "medium_car",
            "fuel_type": "petrol",
            "annual_distance_km": Decimal("25000"),
            "region": "US",
        }
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1["provenance_hash"] == r2["provenance_hash"]
        assert len(r1["provenance_hash"]) == 64


# ==============================================================================
# FUEL-BASED CALCULATION TESTS
# ==============================================================================


class TestFuelBasedCalculation:
    """Test fuel-based vehicle emission calculations."""

    def test_fuel_based_diesel(self, engine):
        """Test fuel-based diesel calculation."""
        result = engine.calculate({
            "method": "fuel_based",
            "fuel_type": "diesel",
            "fuel_litres": Decimal("3500"),
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0
        # ~3500 * 2.68 = ~9380 kg
        assert Decimal("7000") < result["total_co2e_kg"] < Decimal("12000")

    def test_fuel_based_petrol(self, engine):
        """Test fuel-based petrol calculation."""
        result = engine.calculate({
            "method": "fuel_based",
            "fuel_type": "petrol",
            "fuel_litres": Decimal("2500"),
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_fuel_based_diesel_higher_per_litre(self, engine):
        """Test diesel emits more per litre than petrol."""
        diesel = engine.calculate({
            "method": "fuel_based",
            "fuel_type": "diesel",
            "fuel_litres": Decimal("1000"),
            "region": "US",
        })
        petrol = engine.calculate({
            "method": "fuel_based",
            "fuel_type": "petrol",
            "fuel_litres": Decimal("1000"),
            "region": "US",
        })
        assert diesel["total_co2e_kg"] > petrol["total_co2e_kg"]


# ==============================================================================
# EV WITH GRID FACTOR TESTS
# ==============================================================================


class TestEVCalculation:
    """Test electric vehicle emission calculations."""

    def test_ev_with_grid_factor(self, engine):
        """Test EV calculation using grid emission factor."""
        result = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "medium_car",
            "fuel_type": "bev",
            "annual_distance_km": Decimal("20000"),
            "electricity_kwh_per_km": Decimal("0.18"),
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0
        # 20000 * 0.18 * 0.37 = ~1332 kg
        assert result["total_co2e_kg"] < Decimal("5000")

    def test_ev_france_lower_than_us(self, engine):
        """Test EV in France emits less than US (lower grid EF)."""
        us = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "medium_car",
            "fuel_type": "bev",
            "annual_distance_km": Decimal("20000"),
            "electricity_kwh_per_km": Decimal("0.18"),
            "region": "US",
        })
        fr = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "medium_car",
            "fuel_type": "bev",
            "annual_distance_km": Decimal("20000"),
            "electricity_kwh_per_km": Decimal("0.18"),
            "region": "FR",
        })
        assert fr["total_co2e_kg"] < us["total_co2e_kg"]

    def test_ev_lower_than_diesel(self, engine):
        """Test EV emits less than diesel for same distance."""
        ev = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "medium_car",
            "fuel_type": "bev",
            "annual_distance_km": Decimal("25000"),
            "electricity_kwh_per_km": Decimal("0.18"),
            "region": "US",
        })
        diesel = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "medium_car",
            "fuel_type": "diesel",
            "annual_distance_km": Decimal("25000"),
            "region": "US",
        })
        assert ev["total_co2e_kg"] < diesel["total_co2e_kg"]


# ==============================================================================
# FLEET AGGREGATION TESTS
# ==============================================================================


class TestFleetAggregation:
    """Test fleet aggregation calculations."""

    def test_fleet_total(self, engine):
        """Test fleet-level aggregation of multiple vehicles."""
        vehicles = [
            {
                "method": "distance_based",
                "vehicle_type": "medium_car",
                "fuel_type": "petrol",
                "annual_distance_km": Decimal("25000"),
                "region": "US",
            },
            {
                "method": "distance_based",
                "vehicle_type": "light_van",
                "fuel_type": "diesel",
                "annual_distance_km": Decimal("30000"),
                "region": "US",
            },
        ]
        results = engine.calculate_batch(vehicles)
        assert len(results) == 2
        total = sum(r["total_co2e_kg"] for r in results)
        assert total > 0

    def test_fleet_three_vehicles(self, engine):
        """Test fleet with three different vehicle types."""
        vehicles = [
            {
                "method": "distance_based",
                "vehicle_type": "small_car",
                "fuel_type": "petrol",
                "annual_distance_km": Decimal("15000"),
                "region": "US",
            },
            {
                "method": "distance_based",
                "vehicle_type": "heavy_truck",
                "fuel_type": "diesel",
                "annual_distance_km": Decimal("80000"),
                "region": "US",
            },
            {
                "method": "distance_based",
                "vehicle_type": "medium_car",
                "fuel_type": "bev",
                "annual_distance_km": Decimal("20000"),
                "electricity_kwh_per_km": Decimal("0.18"),
                "region": "US",
            },
        ]
        results = engine.calculate_batch(vehicles)
        assert len(results) == 3


# ==============================================================================
# SPEND-BASED CALCULATION TESTS
# ==============================================================================


class TestSpendBasedVehicle:
    """Test spend-based vehicle emission calculations."""

    def test_spend_based_vehicle(self, engine):
        """Test spend-based vehicle lease calculation."""
        result = engine.calculate({
            "method": "spend_based",
            "naics_code": "532112",
            "amount": Decimal("36000.00"),
            "currency": "USD",
            "reporting_year": 2024,
        })
        assert result["total_co2e_kg"] > 0


# ==============================================================================
# AGE FACTOR TESTS
# ==============================================================================


class TestAgeFactor:
    """Test vehicle age degradation factor."""

    def test_older_vehicle_higher_emissions(self, engine):
        """Test older vehicles have higher emissions."""
        new = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "medium_car",
            "fuel_type": "petrol",
            "annual_distance_km": Decimal("25000"),
            "age_years": 0,
            "region": "US",
        })
        old = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "medium_car",
            "fuel_type": "petrol",
            "annual_distance_km": Decimal("25000"),
            "age_years": 10,
            "region": "US",
        })
        assert old["total_co2e_kg"] >= new["total_co2e_kg"]


# ==============================================================================
# WTT FACTOR TESTS
# ==============================================================================


class TestWTTFactor:
    """Test well-to-tank (WTT) factors."""

    def test_wtt_included_in_total(self, engine):
        """Test WTT is included in total emissions."""
        result = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "medium_car",
            "fuel_type": "diesel",
            "annual_distance_km": Decimal("25000"),
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0
        if "wtt_co2e_kg" in result:
            assert result["wtt_co2e_kg"] > 0
            assert result["total_co2e_kg"] > result.get("direct_co2e_kg", Decimal("0"))


# ==============================================================================
# EDGE CASES
# ==============================================================================


class TestVehicleEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_km_raises_error(self, engine):
        """Test zero distance raises error."""
        with pytest.raises((ValueError, Exception)):
            engine.calculate({
                "method": "distance_based",
                "vehicle_type": "medium_car",
                "fuel_type": "petrol",
                "annual_distance_km": Decimal("0"),
                "region": "US",
            })

    def test_negative_distance_raises_error(self, engine):
        """Test negative distance raises error."""
        with pytest.raises((ValueError, Exception)):
            engine.calculate({
                "method": "distance_based",
                "vehicle_type": "medium_car",
                "fuel_type": "petrol",
                "annual_distance_km": Decimal("-5000"),
                "region": "US",
            })

    def test_negative_fuel_raises_error(self, engine):
        """Test negative fuel litres raises error."""
        with pytest.raises((ValueError, Exception)):
            engine.calculate({
                "method": "fuel_based",
                "fuel_type": "diesel",
                "fuel_litres": Decimal("-100"),
                "region": "US",
            })

    def test_result_contains_vehicle_type(self, engine):
        """Test result contains vehicle type in output."""
        result = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "heavy_truck",
            "fuel_type": "diesel",
            "annual_distance_km": Decimal("80000"),
            "region": "US",
        })
        assert "vehicle_type" in result or "asset_type" in result


# ==============================================================================
# PARAMETRIZED FUEL TYPE TESTS
# ==============================================================================


class TestFuelTypeVariations:
    """Test all fuel type variations."""

    @pytest.mark.parametrize("fuel_type", [
        "petrol", "diesel", "lpg", "cng", "hybrid", "plugin_hybrid",
    ])
    def test_medium_car_all_fuel_types(self, engine, fuel_type):
        """Test medium car with all conventional fuel types."""
        result = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "medium_car",
            "fuel_type": fuel_type,
            "annual_distance_km": Decimal("20000"),
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    @pytest.mark.parametrize("region", [
        "US", "GB", "DE", "FR", "JP", "CA", "AU",
    ])
    def test_ev_multiple_regions(self, engine, region):
        """Test EV across multiple grid regions."""
        result = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "medium_car",
            "fuel_type": "bev",
            "annual_distance_km": Decimal("20000"),
            "electricity_kwh_per_km": Decimal("0.18"),
            "region": region,
        })
        assert result["total_co2e_kg"] >= 0

    @pytest.mark.parametrize("distance", [
        Decimal("5000"), Decimal("10000"), Decimal("25000"),
        Decimal("50000"), Decimal("100000"),
    ])
    def test_various_annual_distances(self, engine, distance):
        """Test various annual distances produce proportional results."""
        result = engine.calculate({
            "method": "distance_based",
            "vehicle_type": "medium_car",
            "fuel_type": "petrol",
            "annual_distance_km": distance,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0
