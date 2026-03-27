# -*- coding: utf-8 -*-
"""
Test suite for AssetSpecificCalculatorEngine (AGENT-MRV-026, Engine 2).

Tests metered energy calculations for downstream leased assets where the
reporter (lessor) has collected actual energy data from tenants.

Calculation: sum(energy_kwh * grid_ef) * allocation * (months/12) + vacancy_load

Coverage:
- Building: metered electricity, metered gas, multi-energy, common area
  allocation, vacancy adjustment, tenant allocation
- Vehicle: fuel-based, distance-based, BEV grid-based, fleet aggregation
- Equipment: operating hours with load factor
- IT: PUE adjustment, data center vs individual
- Lease share fraction (50% lease = 50% emissions)
- DQI Tier 1 validation
- Edge cases: zero energy, very large building

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch
import pytest

try:
    from greenlang.agents.mrv.downstream_leased_assets.asset_specific_calculator import (
        AssetSpecificCalculatorEngine,
    )
    from greenlang.agents.mrv.downstream_leased_assets.models import (
        BuildingType,
        ClimateZone,
        AllocationMethod,
        CalculationMethod,
        VehicleType,
        FuelType,
        EquipmentType,
        ITAssetType,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="AssetSpecificCalculatorEngine not available")
pytestmark = _SKIP


@pytest.fixture(autouse=True)
def _reset_singleton():
    if _AVAILABLE:
        AssetSpecificCalculatorEngine.reset_instance()
    yield
    if _AVAILABLE:
        AssetSpecificCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    return AssetSpecificCalculatorEngine()


# ==============================================================================
# BUILDING ASSET-SPECIFIC CALCULATION TESTS
# ==============================================================================


class TestBuildingAssetSpecific:

    def test_office_electricity_only(self, engine):
        result = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0
        assert isinstance(result["total_co2e_kg"], Decimal)
        assert len(result["provenance_hash"]) == 64

    def test_office_electricity_and_gas(self, engine):
        result = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "energy_sources": {
                "electricity_kwh": Decimal("450000"),
                "natural_gas_kwh": Decimal("120000"),
            },
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_multi_energy_higher_than_single(self, engine):
        single = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        multi = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "energy_sources": {
                "electricity_kwh": Decimal("450000"),
                "natural_gas_kwh": Decimal("120000"),
            },
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert multi["total_co2e_kg"] > single["total_co2e_kg"]

    def test_common_area_allocation(self, engine):
        """Test common area emissions allocated proportionally to tenants."""
        full = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        partial = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("0.35"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert partial["total_co2e_kg"] < full["total_co2e_kg"]

    def test_vacancy_adjustment(self, engine):
        """Test vacancy-adjusted emissions include base load."""
        result = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "vacancy_rate": Decimal("0.12"),
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_tenant_allocation(self, engine):
        """Test multi-tenant allocation produces lower per-tenant emissions."""
        full = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        tenant = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("0.25"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert tenant["total_co2e_kg"] < full["total_co2e_kg"]

    def test_partial_year_occupancy(self, engine):
        full = engine.calculate({
            "method": "asset_specific",
            "building_type": "warehouse",
            "floor_area_sqm": Decimal("5000"),
            "energy_sources": {"electricity_kwh": Decimal("180000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        partial = engine.calculate({
            "method": "asset_specific",
            "building_type": "warehouse",
            "floor_area_sqm": Decimal("5000"),
            "energy_sources": {"electricity_kwh": Decimal("180000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 9,
            "region": "US",
        })
        assert partial["total_co2e_kg"] <= full["total_co2e_kg"]


# ==============================================================================
# VEHICLE ASSET-SPECIFIC CALCULATION TESTS
# ==============================================================================


class TestVehicleAssetSpecific:

    def test_fuel_based_calculation(self, engine):
        result = engine.calculate({
            "method": "asset_specific",
            "asset_type": "vehicle",
            "vehicle_type": "medium_car",
            "fuel_type": "diesel",
            "annual_distance_km": Decimal("25000"),
            "fleet_size": 1,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_distance_based_calculation(self, engine):
        result = engine.calculate({
            "method": "asset_specific",
            "asset_type": "vehicle",
            "vehicle_type": "heavy_truck",
            "fuel_type": "diesel",
            "annual_distance_km": Decimal("80000"),
            "fleet_size": 1,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_bev_grid_based(self, engine):
        """Test BEV uses grid emission factor for electricity."""
        result = engine.calculate({
            "method": "asset_specific",
            "asset_type": "vehicle",
            "vehicle_type": "medium_car",
            "fuel_type": "bev",
            "annual_distance_km": Decimal("20000"),
            "electricity_kwh_per_km": Decimal("0.18"),
            "region": "US",
        })
        assert result["total_co2e_kg"] >= 0

    def test_fleet_aggregation(self, engine):
        """Test fleet of 10 vehicles has 10x emissions of single."""
        single = engine.calculate({
            "method": "asset_specific",
            "asset_type": "vehicle",
            "vehicle_type": "medium_car",
            "fuel_type": "diesel",
            "annual_distance_km": Decimal("25000"),
            "fleet_size": 1,
            "region": "US",
        })
        fleet = engine.calculate({
            "method": "asset_specific",
            "asset_type": "vehicle",
            "vehicle_type": "medium_car",
            "fuel_type": "diesel",
            "annual_distance_km": Decimal("25000"),
            "fleet_size": 10,
            "region": "US",
        })
        ratio = fleet["total_co2e_kg"] / single["total_co2e_kg"]
        assert abs(ratio - Decimal("10")) < Decimal("0.01")


# ==============================================================================
# EQUIPMENT ASSET-SPECIFIC CALCULATION TESTS
# ==============================================================================


class TestEquipmentAssetSpecific:

    def test_operating_hours_with_load_factor(self, engine):
        result = engine.calculate({
            "method": "asset_specific",
            "asset_type": "equipment",
            "equipment_type": "construction",
            "rated_power_kw": Decimal("200"),
            "annual_operating_hours": 2000,
            "load_factor": Decimal("0.60"),
            "energy_source": "diesel",
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0


# ==============================================================================
# IT ASSET-SPECIFIC CALCULATION TESTS
# ==============================================================================


class TestITAssetSpecific:

    def test_pue_adjustment(self, engine):
        """Test PUE multiplier increases IT emissions."""
        no_pue = engine.calculate({
            "method": "asset_specific",
            "asset_type": "it_asset",
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": Decimal("1.0"),
            "annual_hours": 8760,
            "region": "US",
        })
        with_pue = engine.calculate({
            "method": "asset_specific",
            "asset_type": "it_asset",
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "region": "US",
        })
        assert with_pue["total_co2e_kg"] > no_pue["total_co2e_kg"]

    def test_data_center_vs_individual(self, engine):
        """Test data center PUE applied to rack calculations."""
        individual = engine.calculate({
            "method": "asset_specific",
            "asset_type": "it_asset",
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": Decimal("1.0"),
            "annual_hours": 8760,
            "region": "US",
        })
        dc = engine.calculate({
            "method": "asset_specific",
            "asset_type": "it_asset",
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "region": "US",
        })
        assert dc["total_co2e_kg"] > individual["total_co2e_kg"]


# ==============================================================================
# LEASE SHARE FRACTION TESTS
# ==============================================================================


class TestLeaseShareFraction:

    def test_fifty_percent_lease(self, engine):
        """50% lease share = 50% of total emissions."""
        full = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        half = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("0.5"),
            "occupancy_months": 12,
            "region": "US",
        })
        ratio = half["total_co2e_kg"] / full["total_co2e_kg"]
        assert abs(ratio - Decimal("0.5")) < Decimal("0.01")


# ==============================================================================
# DQI TIER VALIDATION
# ==============================================================================


class TestDQITier:

    def test_asset_specific_is_tier1(self, engine):
        """Asset-specific method should produce Tier 1 DQI."""
        result = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert result.get("dqi_tier") in ("tier_1", "Tier 1", 1)


# ==============================================================================
# EDGE CASES
# ==============================================================================


class TestEdgeCases:

    def test_very_large_building(self, engine):
        result = engine.calculate({
            "method": "asset_specific",
            "building_type": "industrial",
            "floor_area_sqm": Decimal("100000"),
            "energy_sources": {"electricity_kwh": Decimal("50000000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_minimal_energy(self, engine):
        result = engine.calculate({
            "method": "asset_specific",
            "building_type": "warehouse",
            "floor_area_sqm": Decimal("100"),
            "energy_sources": {"electricity_kwh": Decimal("1")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 1,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_provenance_hash_deterministic(self, engine):
        inp = {
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        }
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1["provenance_hash"] == r2["provenance_hash"]
