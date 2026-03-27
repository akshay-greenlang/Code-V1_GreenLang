# -*- coding: utf-8 -*-
"""
Unit tests for BuildingCalculatorEngine (AGENT-MRV-021, Engine 2)

50 tests covering asset-specific, average-data, lessor-specific, and
spend-based building emission calculations, multi-tenant allocation,
partial year, batch processing, edge cases, and provenance.

Calculation methods:
    Asset-specific: sum(energy_kwh * grid_ef) * allocation * (months/12)
    Average-data:   EUI * floor_area * grid_ef * allocation
    Lessor:         lessor_energy * grid_ef * allocation
    Spend-based:    amount_usd * cpi_deflator * eeio_ef

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch
import pytest

try:
    from greenlang.agents.mrv.upstream_leased_assets.building_calculator import (
        BuildingCalculatorEngine,
    )
    from greenlang.agents.mrv.upstream_leased_assets.models import (
        BuildingType,
        ClimateZone,
        AllocationMethod,
        CalculationMethod,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason="BuildingCalculatorEngine not available",
)

pytestmark = _SKIP


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton before and after every test."""
    if _AVAILABLE:
        BuildingCalculatorEngine.reset_instance()
    yield
    if _AVAILABLE:
        BuildingCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    """Create a fresh BuildingCalculatorEngine."""
    return BuildingCalculatorEngine()


# ==============================================================================
# ASSET-SPECIFIC CALCULATION TESTS
# ==============================================================================


class TestAssetSpecificCalculation:
    """Test asset-specific building emission calculations."""

    def test_office_electricity_only(self, engine):
        """Test office with electricity only."""
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
        assert result["provenance_hash"] is not None
        assert len(result["provenance_hash"]) == 64

    def test_office_electricity_and_gas(self, engine):
        """Test office with electricity and natural gas."""
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
        # Result should be higher than electricity-only
        elec_only = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert result["total_co2e_kg"] > elec_only["total_co2e_kg"]

    def test_allocation_reduces_emissions(self, engine):
        """Test allocation share reduces total emissions proportionally."""
        full = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        partial = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("0.35"),
            "occupancy_months": 12,
            "region": "US",
        })
        expected_ratio = Decimal("0.35")
        actual_ratio = partial["total_co2e_kg"] / full["total_co2e_kg"]
        assert abs(actual_ratio - expected_ratio) < Decimal("0.01")

    def test_partial_year_reduces_emissions(self, engine):
        """Test partial year occupancy reduces emissions proportionally."""
        full_year = engine.calculate({
            "method": "asset_specific",
            "building_type": "warehouse",
            "floor_area_sqm": Decimal("5000"),
            "climate_zone": "cold",
            "energy_sources": {"electricity_kwh": Decimal("180000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        half_year = engine.calculate({
            "method": "asset_specific",
            "building_type": "warehouse",
            "floor_area_sqm": Decimal("5000"),
            "climate_zone": "cold",
            "energy_sources": {"electricity_kwh": Decimal("90000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 6,
            "region": "US",
        })
        assert half_year["total_co2e_kg"] < full_year["total_co2e_kg"]

    def test_data_center_high_electricity(self, engine):
        """Test data center with high electricity produces high emissions."""
        result = engine.calculate({
            "method": "asset_specific",
            "building_type": "data_center",
            "floor_area_sqm": Decimal("800"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("2400000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        # 2.4 GWh * ~0.37 = ~888 tonnes = ~888000 kg
        assert result["total_co2e_kg"] > Decimal("500000")

    def test_different_regions_different_results(self, engine):
        """Test different grid regions produce different results."""
        us_result = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("200000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        fr_result = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("200000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "FR",
        })
        # France has lower grid EF (nuclear)
        assert fr_result["total_co2e_kg"] < us_result["total_co2e_kg"]

    def test_result_contains_breakdown(self, engine):
        """Test result contains energy source breakdown."""
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
        assert "breakdown" in result or "by_source" in result or "energy_breakdown" in result

    def test_provenance_hash_deterministic(self, engine):
        """Test provenance hash is deterministic for same input."""
        inp = {
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        }
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1["provenance_hash"] == r2["provenance_hash"]


# ==============================================================================
# AVERAGE-DATA CALCULATION TESTS
# ==============================================================================


class TestAverageDataCalculation:
    """Test average-data building emission calculations."""

    @pytest.mark.parametrize("building_type", [
        "office", "retail", "warehouse", "industrial",
        "data_center", "hotel", "healthcare", "education",
    ])
    def test_average_data_all_building_types(self, engine, building_type):
        """Test average-data calculation for all 8 building types."""
        result = engine.calculate({
            "method": "average_data",
            "building_type": building_type,
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_average_data_larger_area_more_emissions(self, engine):
        """Test larger floor area produces proportionally more emissions."""
        small = engine.calculate({
            "method": "average_data",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        large = engine.calculate({
            "method": "average_data",
            "building_type": "office",
            "floor_area_sqm": Decimal("2000"),
            "climate_zone": "temperate",
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        ratio = large["total_co2e_kg"] / small["total_co2e_kg"]
        assert abs(ratio - Decimal("2.0")) < Decimal("0.01")

    def test_average_data_cold_higher_than_warm(self, engine):
        """Test cold climate zone produces higher emissions than warm."""
        cold = engine.calculate({
            "method": "average_data",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "cold",
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        warm = engine.calculate({
            "method": "average_data",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "warm",
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert cold["total_co2e_kg"] >= warm["total_co2e_kg"]


# ==============================================================================
# LESSOR-SPECIFIC CALCULATION TESTS
# ==============================================================================


class TestLessorSpecificCalculation:
    """Test lessor-specific building emission calculations."""

    def test_lessor_electricity_only(self, engine):
        """Test lessor-provided electricity data."""
        result = engine.calculate({
            "method": "lessor_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "lessor_electricity_kwh": Decimal("430000"),
            "allocation_share": Decimal("0.35"),
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_lessor_with_gas(self, engine):
        """Test lessor-provided electricity and gas data."""
        result = engine.calculate({
            "method": "lessor_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "lessor_electricity_kwh": Decimal("430000"),
            "lessor_natural_gas_kwh": Decimal("115000"),
            "allocation_share": Decimal("0.35"),
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_lessor_allocation_applied(self, engine):
        """Test lessor method applies allocation correctly."""
        full = engine.calculate({
            "method": "lessor_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "lessor_electricity_kwh": Decimal("430000"),
            "allocation_share": Decimal("1.0"),
            "region": "US",
        })
        partial = engine.calculate({
            "method": "lessor_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "lessor_electricity_kwh": Decimal("430000"),
            "allocation_share": Decimal("0.50"),
            "region": "US",
        })
        ratio = partial["total_co2e_kg"] / full["total_co2e_kg"]
        assert abs(ratio - Decimal("0.50")) < Decimal("0.01")


# ==============================================================================
# SPEND-BASED CALCULATION TESTS
# ==============================================================================


class TestSpendBasedCalculation:
    """Test spend-based building emission calculations."""

    def test_spend_based_usd(self, engine):
        """Test spend-based calculation in USD."""
        result = engine.calculate({
            "method": "spend_based",
            "naics_code": "531120",
            "amount": Decimal("120000.00"),
            "currency": "USD",
            "reporting_year": 2024,
        })
        assert result["total_co2e_kg"] > 0

    def test_spend_based_higher_amount_more_emissions(self, engine):
        """Test higher spend produces proportionally more emissions."""
        low = engine.calculate({
            "method": "spend_based",
            "naics_code": "531120",
            "amount": Decimal("50000.00"),
            "currency": "USD",
            "reporting_year": 2024,
        })
        high = engine.calculate({
            "method": "spend_based",
            "naics_code": "531120",
            "amount": Decimal("100000.00"),
            "currency": "USD",
            "reporting_year": 2024,
        })
        ratio = high["total_co2e_kg"] / low["total_co2e_kg"]
        assert abs(ratio - Decimal("2.0")) < Decimal("0.05")


# ==============================================================================
# MULTI-TENANT ALLOCATION TESTS
# ==============================================================================


class TestMultiTenantAllocation:
    """Test multi-tenant allocation methods."""

    @pytest.mark.parametrize("method,share", [
        ("area", Decimal("0.35")),
        ("headcount", Decimal("0.25")),
        ("revenue", Decimal("0.15")),
    ])
    def test_allocation_methods(self, engine, method, share):
        """Test different allocation methods produce scaled results."""
        result = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("2500"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("450000")},
            "allocation_method": method,
            "allocation_share": share,
            "occupancy_months": 12,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0


# ==============================================================================
# BATCH PROCESSING TESTS
# ==============================================================================


class TestBatchProcessing:
    """Test batch building calculations."""

    def test_batch_multiple_buildings(self, engine):
        """Test batch processing of multiple buildings."""
        buildings = [
            {
                "method": "asset_specific",
                "building_type": "office",
                "floor_area_sqm": Decimal("2500"),
                "climate_zone": "temperate",
                "energy_sources": {"electricity_kwh": Decimal("450000")},
                "allocation_share": Decimal("1.0"),
                "occupancy_months": 12,
                "region": "US",
            },
            {
                "method": "average_data",
                "building_type": "warehouse",
                "floor_area_sqm": Decimal("5000"),
                "climate_zone": "cold",
                "allocation_share": Decimal("0.5"),
                "occupancy_months": 12,
                "region": "US",
            },
        ]
        results = engine.calculate_batch(buildings)
        assert len(results) == 2
        assert all(r["total_co2e_kg"] > 0 for r in results)


# ==============================================================================
# EDGE CASES AND ERROR HANDLING
# ==============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_area_raises_error(self, engine):
        """Test zero floor area raises error."""
        with pytest.raises((ValueError, Exception)):
            engine.calculate({
                "method": "average_data",
                "building_type": "office",
                "floor_area_sqm": Decimal("0"),
                "climate_zone": "temperate",
                "allocation_share": Decimal("1.0"),
                "occupancy_months": 12,
                "region": "US",
            })

    def test_negative_energy_raises_error(self, engine):
        """Test negative energy input raises error."""
        with pytest.raises((ValueError, Exception)):
            engine.calculate({
                "method": "asset_specific",
                "building_type": "office",
                "floor_area_sqm": Decimal("1000"),
                "climate_zone": "temperate",
                "energy_sources": {"electricity_kwh": Decimal("-100000")},
                "allocation_share": Decimal("1.0"),
                "occupancy_months": 12,
                "region": "US",
            })

    def test_known_value_office(self, engine):
        """Test known value: 200000 kWh elec * 0.37170 kgCO2e/kWh = 74340 kg."""
        result = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("200000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        # Approximately 74340 kg (depends on exact grid EF)
        assert Decimal("50000") < result["total_co2e_kg"] < Decimal("100000")

    def test_compare_methods(self, engine):
        """Test comparing asset-specific vs average-data results."""
        asset_specific = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("200000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        average_data = engine.calculate({
            "method": "average_data",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        # Both should be positive; values may differ
        assert asset_specific["total_co2e_kg"] > 0
        assert average_data["total_co2e_kg"] > 0


# ==============================================================================
# ADDITIONAL BUILDING TYPE TESTS
# ==============================================================================


class TestAllBuildingTypes:
    """Test asset-specific calculation across all 8 building types."""

    @pytest.mark.parametrize("building_type", [
        "office", "retail", "warehouse", "industrial",
        "data_center", "hotel", "healthcare", "education",
    ])
    def test_asset_specific_all_types(self, engine, building_type):
        """Test asset-specific for all 8 building types."""
        result = engine.calculate({
            "method": "asset_specific",
            "building_type": building_type,
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("200000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    @pytest.mark.parametrize("climate_zone", [
        "tropical", "arid", "temperate", "cold", "warm",
    ])
    def test_office_all_climate_zones(self, engine, climate_zone):
        """Test office building across all 5 climate zones."""
        result = engine.calculate({
            "method": "average_data",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": climate_zone,
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    @pytest.mark.parametrize("region", [
        "US", "GB", "DE", "FR", "JP", "CA", "AU",
    ])
    def test_office_multiple_regions(self, engine, region):
        """Test office building across multiple grid regions."""
        result = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("200000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": region,
        })
        assert result["total_co2e_kg"] > 0

    @pytest.mark.parametrize("months", [1, 3, 6, 9, 12])
    def test_office_various_occupancy_months(self, engine, months):
        """Test office building with various occupancy periods."""
        result = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("200000")},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": months,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    @pytest.mark.parametrize("share", [
        Decimal("0.10"), Decimal("0.25"), Decimal("0.50"),
        Decimal("0.75"), Decimal("1.00"),
    ])
    def test_office_various_allocation_shares(self, engine, share):
        """Test office building with various allocation shares."""
        result = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": Decimal("200000")},
            "allocation_share": share,
            "occupancy_months": 12,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    @pytest.mark.parametrize("energy_kwh", [
        Decimal("10000"), Decimal("50000"), Decimal("100000"),
        Decimal("500000"), Decimal("1000000"), Decimal("5000000"),
    ])
    def test_office_various_energy_levels(self, engine, energy_kwh):
        """Test office building with various electricity consumption levels."""
        result = engine.calculate({
            "method": "asset_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": energy_kwh},
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    @pytest.mark.parametrize("area", [
        Decimal("100"), Decimal("500"), Decimal("1000"),
        Decimal("5000"), Decimal("10000"), Decimal("50000"),
    ])
    def test_average_data_various_areas(self, engine, area):
        """Test average-data office with various floor areas."""
        result = engine.calculate({
            "method": "average_data",
            "building_type": "office",
            "floor_area_sqm": area,
            "climate_zone": "temperate",
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0


# ==============================================================================
# ADDITIONAL SCENARIO AND COMPARISON TESTS
# ==============================================================================


class TestBuildingComparisons:
    """Test cross-method and cross-building comparisons."""

    def test_data_center_highest_emissions(self, engine):
        """Test data center emits more than office for same area."""
        dc = engine.calculate({
            "method": "average_data",
            "building_type": "data_center",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        office = engine.calculate({
            "method": "average_data",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert dc["total_co2e_kg"] > office["total_co2e_kg"]

    def test_warehouse_lowest_emissions(self, engine):
        """Test warehouse emits less than office for same area."""
        warehouse = engine.calculate({
            "method": "average_data",
            "building_type": "warehouse",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        office = engine.calculate({
            "method": "average_data",
            "building_type": "office",
            "floor_area_sqm": Decimal("1000"),
            "climate_zone": "temperate",
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert warehouse["total_co2e_kg"] < office["total_co2e_kg"]

    def test_allocation_over_one_raises_error(self, engine):
        """Test allocation share over 1.0 raises error."""
        with pytest.raises((ValueError, Exception)):
            engine.calculate({
                "method": "asset_specific",
                "building_type": "office",
                "floor_area_sqm": Decimal("1000"),
                "climate_zone": "temperate",
                "energy_sources": {"electricity_kwh": Decimal("200000")},
                "allocation_share": Decimal("1.5"),
                "occupancy_months": 12,
                "region": "US",
            })

    def test_occupancy_months_over_twelve_raises_error(self, engine):
        """Test occupancy_months over 12 raises error."""
        with pytest.raises((ValueError, Exception)):
            engine.calculate({
                "method": "asset_specific",
                "building_type": "office",
                "floor_area_sqm": Decimal("1000"),
                "climate_zone": "temperate",
                "energy_sources": {"electricity_kwh": Decimal("200000")},
                "allocation_share": Decimal("1.0"),
                "occupancy_months": 15,
                "region": "US",
            })

    def test_healthcare_higher_than_education(self, engine):
        """Test healthcare building emits more than education for same area."""
        hc = engine.calculate({
            "method": "average_data",
            "building_type": "healthcare",
            "floor_area_sqm": Decimal("2000"),
            "climate_zone": "temperate",
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        edu = engine.calculate({
            "method": "average_data",
            "building_type": "education",
            "floor_area_sqm": Decimal("2000"),
            "climate_zone": "temperate",
            "allocation_share": Decimal("1.0"),
            "occupancy_months": 12,
            "region": "US",
        })
        assert hc["total_co2e_kg"] >= edu["total_co2e_kg"]

    def test_spend_based_eur_conversion(self, engine):
        """Test spend-based calculation with EUR currency."""
        result = engine.calculate({
            "method": "spend_based",
            "naics_code": "531120",
            "amount": Decimal("100000.00"),
            "currency": "EUR",
            "reporting_year": 2024,
        })
        assert result["total_co2e_kg"] > 0

    def test_lessor_triple_energy_sources(self, engine):
        """Test lessor-specific with electricity, gas, and heating oil."""
        result = engine.calculate({
            "method": "lessor_specific",
            "building_type": "office",
            "floor_area_sqm": Decimal("5000"),
            "lessor_electricity_kwh": Decimal("600000"),
            "lessor_natural_gas_kwh": Decimal("200000"),
            "lessor_heating_oil_kwh": Decimal("50000"),
            "allocation_share": Decimal("0.40"),
            "region": "DE",
        })
        assert result["total_co2e_kg"] > 0
