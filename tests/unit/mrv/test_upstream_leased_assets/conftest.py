# -*- coding: utf-8 -*-
"""
Pytest fixtures for AGENT-MRV-021: Upstream Leased Assets Agent.

Provides comprehensive test fixtures for:
- Building inputs (office, retail, warehouse, data_center, hotel, healthcare,
  education, industrial) with climate zone variations
- Vehicle inputs (car, van, truck, SUV, EV) with fuel type variations
- Equipment inputs (manufacturing, construction, generator, agricultural,
  mining, HVAC) with load factor adjustments
- IT asset inputs (server, desktop, laptop, printer, copier, network,
  storage) with PUE adjustment
- Lessor-specific inputs (primary data from landlord/lessor)
- Spend-based inputs (EEIO factors with CPI deflation)
- Compliance inputs (7 frameworks)
- Allocation inputs (area, headcount, revenue based)
- Configuration objects (15 frozen dataclass configs)
- Mock engines (database, calculators, compliance, pipeline)

Usage:
    def test_something(sample_office_input, mock_database_engine):
        result = calculate(sample_office_input, mock_database_engine)
        assert result.total_co2e > 0

Author: GL-TestEngineer
Date: February 2026
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock
import pytest


# ============================================================================
# BUILDING INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_office_input() -> Dict[str, Any]:
    """Office building, 2500 sqm, temperate zone, all energy types."""
    return {
        "asset_type": "building",
        "building_type": "office",
        "floor_area_sqm": Decimal("2500.00"),
        "climate_zone": "temperate",
        "energy_sources": {
            "electricity_kwh": Decimal("450000"),
            "natural_gas_kwh": Decimal("120000"),
        },
        "occupancy_months": 12,
        "allocation_method": "area",
        "allocation_share": Decimal("0.35"),
        "region": "US",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_retail_input() -> Dict[str, Any]:
    """Retail building, 1200 sqm, warm zone."""
    return {
        "asset_type": "building",
        "building_type": "retail",
        "floor_area_sqm": Decimal("1200.00"),
        "climate_zone": "warm",
        "energy_sources": {
            "electricity_kwh": Decimal("280000"),
        },
        "occupancy_months": 12,
        "allocation_method": "area",
        "allocation_share": Decimal("1.0"),
        "region": "US",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_warehouse_input() -> Dict[str, Any]:
    """Warehouse, 5000 sqm, cold zone, partial year."""
    return {
        "asset_type": "building",
        "building_type": "warehouse",
        "floor_area_sqm": Decimal("5000.00"),
        "climate_zone": "cold",
        "energy_sources": {
            "electricity_kwh": Decimal("180000"),
            "natural_gas_kwh": Decimal("250000"),
        },
        "occupancy_months": 9,
        "allocation_method": "area",
        "allocation_share": Decimal("0.5"),
        "region": "CA",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_data_center_input() -> Dict[str, Any]:
    """Data center, 800 sqm, temperate zone, high electricity."""
    return {
        "asset_type": "building",
        "building_type": "data_center",
        "floor_area_sqm": Decimal("800.00"),
        "climate_zone": "temperate",
        "energy_sources": {
            "electricity_kwh": Decimal("2400000"),
        },
        "occupancy_months": 12,
        "pue": Decimal("1.40"),
        "allocation_method": "area",
        "allocation_share": Decimal("0.25"),
        "region": "US",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_hotel_building_input() -> Dict[str, Any]:
    """Hotel building, 3000 sqm, tropical zone."""
    return {
        "asset_type": "building",
        "building_type": "hotel",
        "floor_area_sqm": Decimal("3000.00"),
        "climate_zone": "tropical",
        "energy_sources": {
            "electricity_kwh": Decimal("650000"),
            "natural_gas_kwh": Decimal("100000"),
        },
        "occupancy_months": 12,
        "allocation_method": "revenue",
        "allocation_share": Decimal("0.15"),
        "region": "GB",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_healthcare_input() -> Dict[str, Any]:
    """Healthcare building, 4000 sqm, temperate zone."""
    return {
        "asset_type": "building",
        "building_type": "healthcare",
        "floor_area_sqm": Decimal("4000.00"),
        "climate_zone": "temperate",
        "energy_sources": {
            "electricity_kwh": Decimal("900000"),
            "natural_gas_kwh": Decimal("400000"),
        },
        "occupancy_months": 12,
        "allocation_method": "area",
        "allocation_share": Decimal("0.50"),
        "region": "DE",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_education_input() -> Dict[str, Any]:
    """Education building, 6000 sqm, cold zone."""
    return {
        "asset_type": "building",
        "building_type": "education",
        "floor_area_sqm": Decimal("6000.00"),
        "climate_zone": "cold",
        "energy_sources": {
            "electricity_kwh": Decimal("720000"),
            "natural_gas_kwh": Decimal("500000"),
        },
        "occupancy_months": 10,
        "allocation_method": "headcount",
        "allocation_share": Decimal("0.30"),
        "region": "CA",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_industrial_input() -> Dict[str, Any]:
    """Industrial building, 8000 sqm, arid zone."""
    return {
        "asset_type": "building",
        "building_type": "industrial",
        "floor_area_sqm": Decimal("8000.00"),
        "climate_zone": "arid",
        "energy_sources": {
            "electricity_kwh": Decimal("1600000"),
            "natural_gas_kwh": Decimal("800000"),
        },
        "occupancy_months": 12,
        "allocation_method": "area",
        "allocation_share": Decimal("0.60"),
        "region": "US",
        "lease_type": "operating",
    }


# ============================================================================
# VEHICLE INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_car_input() -> Dict[str, Any]:
    """Medium petrol car, 25000 km/year."""
    return {
        "asset_type": "vehicle",
        "vehicle_type": "medium_car",
        "fuel_type": "petrol",
        "annual_distance_km": Decimal("25000"),
        "region": "US",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_van_input() -> Dict[str, Any]:
    """Light van, diesel, 30000 km/year."""
    return {
        "asset_type": "vehicle",
        "vehicle_type": "light_van",
        "fuel_type": "diesel",
        "annual_distance_km": Decimal("30000"),
        "region": "GB",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_truck_input() -> Dict[str, Any]:
    """Heavy truck, diesel, 80000 km/year."""
    return {
        "asset_type": "vehicle",
        "vehicle_type": "heavy_truck",
        "fuel_type": "diesel",
        "annual_distance_km": Decimal("80000"),
        "region": "US",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_ev_input() -> Dict[str, Any]:
    """Battery electric vehicle, 20000 km/year, US grid."""
    return {
        "asset_type": "vehicle",
        "vehicle_type": "medium_car",
        "fuel_type": "bev",
        "annual_distance_km": Decimal("20000"),
        "electricity_kwh_per_km": Decimal("0.18"),
        "grid_ef_region": "US",
        "region": "US",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_suv_input() -> Dict[str, Any]:
    """SUV, petrol, 22000 km/year."""
    return {
        "asset_type": "vehicle",
        "vehicle_type": "suv",
        "fuel_type": "petrol",
        "annual_distance_km": Decimal("22000"),
        "region": "US",
        "lease_type": "operating",
    }


# ============================================================================
# EQUIPMENT INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_manufacturing_equipment() -> Dict[str, Any]:
    """Manufacturing equipment, 500 kW, 6000 hours/year."""
    return {
        "asset_type": "equipment",
        "equipment_type": "manufacturing",
        "rated_power_kw": Decimal("500"),
        "annual_operating_hours": 6000,
        "load_factor": Decimal("0.75"),
        "energy_source": "electricity",
        "region": "US",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_construction_equipment() -> Dict[str, Any]:
    """Construction equipment, 200 kW, diesel, 2000 hours/year."""
    return {
        "asset_type": "equipment",
        "equipment_type": "construction",
        "rated_power_kw": Decimal("200"),
        "annual_operating_hours": 2000,
        "load_factor": Decimal("0.60"),
        "energy_source": "diesel",
        "region": "US",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_generator_equipment() -> Dict[str, Any]:
    """Generator, 100 kW, diesel, 1500 hours/year."""
    return {
        "asset_type": "equipment",
        "equipment_type": "generator",
        "rated_power_kw": Decimal("100"),
        "annual_operating_hours": 1500,
        "load_factor": Decimal("0.80"),
        "energy_source": "diesel",
        "output_kwh": Decimal("120000"),
        "region": "US",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_hvac_equipment() -> Dict[str, Any]:
    """HVAC system, 50 kW, electricity, 4000 hours/year."""
    return {
        "asset_type": "equipment",
        "equipment_type": "hvac",
        "rated_power_kw": Decimal("50"),
        "annual_operating_hours": 4000,
        "load_factor": Decimal("0.65"),
        "energy_source": "electricity",
        "region": "DE",
        "lease_type": "operating",
    }


# ============================================================================
# IT ASSET INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_server_input() -> Dict[str, Any]:
    """Server, 500W rated, 90% utilization, PUE 1.4."""
    return {
        "asset_type": "it_asset",
        "it_type": "server",
        "rated_power_w": Decimal("500"),
        "utilization_pct": Decimal("0.90"),
        "pue": Decimal("1.40"),
        "annual_hours": 8760,
        "region": "US",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_desktop_input() -> Dict[str, Any]:
    """Desktop workstation, 200W rated, standard usage."""
    return {
        "asset_type": "it_asset",
        "it_type": "desktop",
        "rated_power_w": Decimal("200"),
        "utilization_pct": Decimal("0.50"),
        "pue": Decimal("1.00"),
        "annual_hours": 2080,
        "region": "US",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_laptop_input() -> Dict[str, Any]:
    """Laptop, 65W rated, standard usage."""
    return {
        "asset_type": "it_asset",
        "it_type": "laptop",
        "rated_power_w": Decimal("65"),
        "utilization_pct": Decimal("0.60"),
        "pue": Decimal("1.00"),
        "annual_hours": 2080,
        "region": "GB",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_network_switch_input() -> Dict[str, Any]:
    """Network switch, 350W, 24/7 operation, data center PUE."""
    return {
        "asset_type": "it_asset",
        "it_type": "network",
        "rated_power_w": Decimal("350"),
        "utilization_pct": Decimal("0.80"),
        "pue": Decimal("1.40"),
        "annual_hours": 8760,
        "region": "US",
        "lease_type": "operating",
    }


@pytest.fixture
def sample_storage_input() -> Dict[str, Any]:
    """Storage array, 800W, data center PUE 1.3."""
    return {
        "asset_type": "it_asset",
        "it_type": "storage",
        "rated_power_w": Decimal("800"),
        "utilization_pct": Decimal("0.70"),
        "pue": Decimal("1.30"),
        "annual_hours": 8760,
        "region": "DE",
        "lease_type": "operating",
    }


# ============================================================================
# LESSOR-SPECIFIC INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_lessor_input() -> Dict[str, Any]:
    """Lessor-provided energy data for office building."""
    return {
        "method": "lessor_specific",
        "asset_type": "building",
        "building_type": "office",
        "floor_area_sqm": Decimal("2500.00"),
        "lessor_electricity_kwh": Decimal("430000"),
        "lessor_natural_gas_kwh": Decimal("115000"),
        "lessor_district_heating_kwh": Decimal("0"),
        "lessor_data_year": 2024,
        "allocation_method": "area",
        "allocation_share": Decimal("0.35"),
        "region": "US",
    }


@pytest.fixture
def sample_lessor_vehicle_input() -> Dict[str, Any]:
    """Lessor-provided fuel data for fleet vehicle."""
    return {
        "method": "lessor_specific",
        "asset_type": "vehicle",
        "vehicle_type": "light_van",
        "lessor_fuel_litres": Decimal("3500"),
        "fuel_type": "diesel",
        "lessor_data_year": 2024,
        "region": "GB",
    }


# ============================================================================
# SPEND-BASED INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_spend_input() -> Dict[str, Any]:
    """Spend-based: office lease $120,000 USD, 2024."""
    return {
        "method": "spend_based",
        "naics_code": "531120",
        "amount": Decimal("120000.00"),
        "currency": "USD",
        "reporting_year": 2024,
        "description": "Office lease payments",
    }


@pytest.fixture
def sample_spend_input_eur() -> Dict[str, Any]:
    """Spend-based: warehouse lease EUR 85,000, 2024."""
    return {
        "method": "spend_based",
        "naics_code": "531130",
        "amount": Decimal("85000.00"),
        "currency": "EUR",
        "reporting_year": 2024,
        "description": "Warehouse lease payments",
    }


@pytest.fixture
def sample_spend_input_vehicle() -> Dict[str, Any]:
    """Spend-based: vehicle lease $36,000 USD, 2024."""
    return {
        "method": "spend_based",
        "naics_code": "532112",
        "amount": Decimal("36000.00"),
        "currency": "USD",
        "reporting_year": 2024,
        "description": "Fleet vehicle leases",
    }


# ============================================================================
# COMPLIANCE INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_compliance_input() -> Dict[str, Any]:
    """Compliance check for GHG Protocol + CSRD + CDP."""
    return {
        "frameworks": ["ghg_protocol", "iso_14064", "csrd_esrs", "cdp", "sbti", "sb_253", "gri"],
        "total_co2e": Decimal("85000.00"),
        "method_used": "asset_specific",
        "asset_breakdown_provided": True,
        "reporting_period": "2024",
        "lease_classification_disclosed": True,
    }


@pytest.fixture
def sample_compliance_input_minimal() -> Dict[str, Any]:
    """Minimal compliance check (GHG Protocol only)."""
    return {
        "frameworks": ["ghg_protocol"],
        "total_co2e": Decimal("50000.00"),
        "method_used": "average_data",
        "reporting_period": "2024",
    }


# ============================================================================
# ALLOCATION INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_area_allocation() -> Dict[str, Any]:
    """Area-based allocation for multi-tenant building."""
    return {
        "method": "area",
        "tenant_area_sqm": Decimal("875.00"),
        "total_area_sqm": Decimal("2500.00"),
    }


@pytest.fixture
def sample_headcount_allocation() -> Dict[str, Any]:
    """Headcount-based allocation for shared building."""
    return {
        "method": "headcount",
        "tenant_headcount": 50,
        "total_headcount": 200,
    }


@pytest.fixture
def sample_revenue_allocation() -> Dict[str, Any]:
    """Revenue-based allocation for shared building."""
    return {
        "method": "revenue",
        "tenant_revenue": Decimal("2500000.00"),
        "total_revenue": Decimal("10000000.00"),
    }


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def default_config():
    """Default UpstreamLeasedConfig with all 15 sections."""
    try:
        from greenlang.upstream_leased_assets.config import UpstreamLeasedConfig
        return UpstreamLeasedConfig()
    except ImportError:
        pytest.skip("UpstreamLeasedConfig not available")


@pytest.fixture
def default_general_config():
    """Default GeneralConfig."""
    try:
        from greenlang.upstream_leased_assets.config import GeneralConfig
        return GeneralConfig()
    except ImportError:
        pytest.skip("GeneralConfig not available")


@pytest.fixture
def default_database_config():
    """Default DatabaseConfig."""
    try:
        from greenlang.upstream_leased_assets.config import DatabaseConfig
        return DatabaseConfig()
    except ImportError:
        pytest.skip("DatabaseConfig not available")


@pytest.fixture
def default_building_config():
    """Default BuildingConfig."""
    try:
        from greenlang.upstream_leased_assets.config import BuildingConfig
        return BuildingConfig()
    except ImportError:
        pytest.skip("BuildingConfig not available")


@pytest.fixture
def default_vehicle_config():
    """Default VehicleConfig."""
    try:
        from greenlang.upstream_leased_assets.config import VehicleConfig
        return VehicleConfig()
    except ImportError:
        pytest.skip("VehicleConfig not available")


@pytest.fixture
def default_equipment_config():
    """Default EquipmentConfig."""
    try:
        from greenlang.upstream_leased_assets.config import EquipmentConfig
        return EquipmentConfig()
    except ImportError:
        pytest.skip("EquipmentConfig not available")


@pytest.fixture
def default_it_config():
    """Default ITAssetsConfig."""
    try:
        from greenlang.upstream_leased_assets.config import ITAssetsConfig
        return ITAssetsConfig()
    except ImportError:
        pytest.skip("ITAssetsConfig not available")


# ============================================================================
# BATCH INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_batch_buildings(
    sample_office_input,
    sample_retail_input,
    sample_warehouse_input,
) -> List[Dict[str, Any]]:
    """Batch of 3 building inputs."""
    return [sample_office_input, sample_retail_input, sample_warehouse_input]


@pytest.fixture
def sample_batch_vehicles(
    sample_car_input,
    sample_van_input,
    sample_truck_input,
) -> List[Dict[str, Any]]:
    """Batch of 3 vehicle inputs."""
    return [sample_car_input, sample_van_input, sample_truck_input]


@pytest.fixture
def sample_batch_mixed(
    sample_office_input,
    sample_car_input,
    sample_manufacturing_equipment,
    sample_server_input,
) -> List[Dict[str, Any]]:
    """Batch of 4 mixed asset type inputs."""
    return [
        sample_office_input,
        sample_car_input,
        sample_manufacturing_equipment,
        sample_server_input,
    ]
