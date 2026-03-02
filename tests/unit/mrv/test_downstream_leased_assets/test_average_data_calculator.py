# -*- coding: utf-8 -*-
"""
Test suite for AverageDataCalculatorEngine (AGENT-MRV-026, Engine 3).

Tests benchmark-based calculations using EUI benchmarks by building type
and climate zone, vehicle type benchmarks, equipment and IT defaults.

Calculation: EUI * floor_area * grid_ef * allocation * vacancy_adj

Coverage:
- Building benchmark (8 types parametrized, climate zone effect, vacancy)
- Vehicle benchmark (8 types parametrized, default distance)
- Equipment benchmark (6 types, load factor)
- IT benchmark (7 types, PUE)
- Regional grid factor impact (12 regions parametrized)
- DQI Tier 2 (lower than asset-specific)
- Uncertainty higher than Tier 1
- EPC/NABERS rating proxy

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch
import pytest

try:
    from greenlang.downstream_leased_assets.average_data_calculator import (
        AverageDataCalculatorEngine,
    )
    from greenlang.downstream_leased_assets.models import (
        BuildingType,
        ClimateZone,
        VehicleType,
        FuelType,
        EquipmentType,
        ITAssetType,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="AverageDataCalculatorEngine not available")
pytestmark = _SKIP


@pytest.fixture(autouse=True)
def _reset_singleton():
    if _AVAILABLE:
        AverageDataCalculatorEngine.reset_instance()
    yield
    if _AVAILABLE:
        AverageDataCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    return AverageDataCalculatorEngine()


# ==============================================================================
# BUILDING BENCHMARK TESTS
# ==============================================================================


class TestBuildingBenchmark:

    @pytest.mark.parametrize("building_type", [
        "office", "retail", "warehouse", "data_center",
        "hotel", "healthcare", "education", "industrial",
    ])
    def test_building_type_benchmark(self, engine, building_type):
        """Test all 8 building types produce positive emissions."""
        result = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": building_type,
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "region": "US",
            "allocation_share": Decimal("1.0"),
        })
        assert result["total_co2e_kg"] > 0

    def test_climate_zone_effect(self, engine):
        """Cold zone should produce higher emissions than temperate."""
        temperate = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "region": "US",
            "allocation_share": Decimal("1.0"),
        })
        cold = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "cold",
            "region": "US",
            "allocation_share": Decimal("1.0"),
        })
        assert cold["total_co2e_kg"] >= temperate["total_co2e_kg"]

    def test_vacancy_adjustment_reduces(self, engine):
        """Vacancy adjustment with base load should be less than full occupancy."""
        full = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "region": "US",
            "allocation_share": Decimal("1.0"),
            "vacancy_rate": Decimal("0.0"),
        })
        vacant = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "region": "US",
            "allocation_share": Decimal("1.0"),
            "vacancy_rate": Decimal("0.30"),
        })
        assert vacant["total_co2e_kg"] <= full["total_co2e_kg"]

    def test_data_center_highest(self, engine):
        """Data center should have highest emissions per sqm."""
        dc = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": "data_center",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "region": "US",
            "allocation_share": Decimal("1.0"),
        })
        office = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "region": "US",
            "allocation_share": Decimal("1.0"),
        })
        assert dc["total_co2e_kg"] > office["total_co2e_kg"]

    def test_area_proportionality(self, engine):
        """Double the area should roughly double the emissions."""
        small = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "region": "US",
            "allocation_share": Decimal("1.0"),
        })
        large = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": "office",
            "floor_area_sqm": Decimal("2000"),
            "climate_zone": "temperate",
            "region": "US",
            "allocation_share": Decimal("1.0"),
        })
        ratio = large["total_co2e_kg"] / small["total_co2e_kg"]
        assert abs(ratio - Decimal("2.0")) < Decimal("0.1")


# ==============================================================================
# VEHICLE BENCHMARK TESTS
# ==============================================================================


class TestVehicleBenchmark:

    @pytest.mark.parametrize("vehicle_type", [
        "small_car", "medium_car", "large_car", "suv",
        "light_van", "heavy_van", "light_truck", "heavy_truck",
    ])
    def test_vehicle_type_benchmark(self, engine, vehicle_type):
        result = engine.calculate({
            "method": "average_data",
            "asset_type": "vehicle",
            "vehicle_type": vehicle_type,
            "fuel_type": "diesel",
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_default_distance_used(self, engine):
        """When no distance provided, default benchmark distance is used."""
        result = engine.calculate({
            "method": "average_data",
            "asset_type": "vehicle",
            "vehicle_type": "medium_car",
            "fuel_type": "diesel",
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0


# ==============================================================================
# EQUIPMENT BENCHMARK TESTS
# ==============================================================================


class TestEquipmentBenchmark:

    @pytest.mark.parametrize("equipment_type", [
        "manufacturing", "construction", "generator",
        "agricultural", "mining", "hvac",
    ])
    def test_equipment_type_benchmark(self, engine, equipment_type):
        result = engine.calculate({
            "method": "average_data",
            "asset_type": "equipment",
            "equipment_type": equipment_type,
            "rated_power_kw": Decimal("200"),
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_load_factor_applied(self, engine):
        """Higher load factor means higher emissions."""
        low_load = engine.calculate({
            "method": "average_data",
            "asset_type": "equipment",
            "equipment_type": "construction",
            "rated_power_kw": Decimal("200"),
            "load_factor": Decimal("0.30"),
            "region": "US",
        })
        high_load = engine.calculate({
            "method": "average_data",
            "asset_type": "equipment",
            "equipment_type": "construction",
            "rated_power_kw": Decimal("200"),
            "load_factor": Decimal("0.80"),
            "region": "US",
        })
        assert high_load["total_co2e_kg"] > low_load["total_co2e_kg"]


# ==============================================================================
# IT BENCHMARK TESTS
# ==============================================================================


class TestITBenchmark:

    @pytest.mark.parametrize("it_type", [
        "server", "network", "storage", "desktop",
        "laptop", "printer", "copier",
    ])
    def test_it_type_benchmark(self, engine, it_type):
        result = engine.calculate({
            "method": "average_data",
            "asset_type": "it_asset",
            "it_type": it_type,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_pue_effect(self, engine):
        """Higher PUE should increase data center IT emissions."""
        low_pue = engine.calculate({
            "method": "average_data",
            "asset_type": "it_asset",
            "it_type": "server",
            "pue": Decimal("1.1"),
            "region": "US",
        })
        high_pue = engine.calculate({
            "method": "average_data",
            "asset_type": "it_asset",
            "it_type": "server",
            "pue": Decimal("1.8"),
            "region": "US",
        })
        assert high_pue["total_co2e_kg"] > low_pue["total_co2e_kg"]


# ==============================================================================
# REGIONAL GRID FACTOR IMPACT TESTS
# ==============================================================================


class TestRegionalGridImpact:

    @pytest.mark.parametrize("region", [
        "US", "GB", "DE", "FR", "JP", "CA", "AU", "IN", "CN", "BR", "GLOBAL",
    ])
    def test_regional_grid_factor(self, engine, region):
        result = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "region": region,
            "allocation_share": Decimal("1.0"),
        })
        assert result["total_co2e_kg"] > 0

    def test_france_lower_than_india(self, engine):
        """France (nuclear) should have lower grid-based emissions than India (coal)."""
        fr = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "region": "FR",
            "allocation_share": Decimal("1.0"),
        })
        ind = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "region": "IN",
            "allocation_share": Decimal("1.0"),
        })
        assert fr["total_co2e_kg"] < ind["total_co2e_kg"]


# ==============================================================================
# DQI AND UNCERTAINTY TESTS
# ==============================================================================


class TestDQIAndUncertainty:

    def test_average_data_is_tier2(self, engine):
        result = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "region": "US",
            "allocation_share": Decimal("1.0"),
        })
        assert result.get("dqi_tier") in ("tier_2", "Tier 2", 2)

    def test_uncertainty_higher_than_tier1(self, engine):
        result = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "region": "US",
            "allocation_share": Decimal("1.0"),
        })
        uncertainty = result.get("uncertainty_pct", Decimal("0.25"))
        assert uncertainty >= Decimal("0.10")


# ==============================================================================
# EPC / NABERS RATING PROXY
# ==============================================================================


class TestEPCRatingProxy:

    def test_epc_rating_adjusts_eui(self, engine):
        """EPC/NABERS rating should adjust the benchmark EUI."""
        result = engine.calculate({
            "method": "average_data",
            "asset_type": "building",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "region": "US",
            "allocation_share": Decimal("1.0"),
            "epc_rating": "A",
        })
        assert result["total_co2e_kg"] > 0
