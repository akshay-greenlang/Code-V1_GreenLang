# -*- coding: utf-8 -*-
"""
Pytest fixtures for AGENT-MRV-027: Franchises Agent.

Provides comprehensive test fixtures for:
- QSR franchise units (cooking, refrigeration, delivery)
- Hotel franchise units (rooms, amenities, occupancy)
- Convenience store units (24/7 operation)
- Retail, fitness, and automotive franchise units
- Network-level inputs with 5+ unit types
- Franchise-specific, average-data, and spend-based calculation inputs
- Hotel operations, cooking energy, refrigeration, and delivery fleet data
- Mocked FranchiseDatabaseEngine
- Test configuration and singleton resets

Usage:
    def test_something(sample_qsr_unit, mock_database_engine):
        result = calculate(sample_qsr_unit, mock_database_engine)
        assert result.total_co2e > 0

Author: GL-TestEngineer
Date: February 2026
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest


# ============================================================================
# QSR FRANCHISE UNIT FIXTURES
# ============================================================================


@pytest.fixture
def sample_qsr_unit() -> Dict[str, Any]:
    """QSR franchise unit with cooking, refrigeration, delivery."""
    return {
        "unit_id": "FRN-QSR-001",
        "franchise_type": "qsr_restaurant",
        "ownership_type": "franchised",
        "agreement_type": "single_unit",
        "unit_name": "Burger Palace #1234",
        "brand": "Burger Palace",
        "floor_area_m2": Decimal("250"),
        "country_code": "US",
        "region": "CAMX",
        "climate_zone": "temperate",
        "opening_date": "2020-03-15",
        "operating_months": 12,
        "status": "active",
        "annual_revenue_usd": Decimal("1200000"),
        "electricity_kwh": Decimal("180000"),
        "natural_gas_therms": Decimal("12000"),
        "cooking_energy": {
            "natural_gas_therms": Decimal("8500"),
            "propane_gallons": Decimal("200"),
            "electricity_kwh": Decimal("25000"),
        },
        "refrigeration": {
            "refrigerant_type": "R_404A",
            "charge_kg": Decimal("15.0"),
            "annual_leakage_rate": Decimal("0.15"),
        },
        "delivery_fleet": {
            "vehicle_count": 3,
            "vehicle_type": "light_commercial",
            "fuel_type": "gasoline",
            "annual_fuel_litres": Decimal("9000"),
            "annual_distance_km": Decimal("45000"),
        },
    }


# ============================================================================
# HOTEL FRANCHISE UNIT FIXTURES
# ============================================================================


@pytest.fixture
def sample_hotel_unit() -> Dict[str, Any]:
    """Hotel franchise unit with rooms, amenities, occupancy."""
    return {
        "unit_id": "FRN-HTL-001",
        "franchise_type": "hotel",
        "ownership_type": "franchised",
        "agreement_type": "single_unit",
        "unit_name": "GreenStay Hotel Downtown",
        "brand": "GreenStay Hotels",
        "floor_area_m2": Decimal("5000"),
        "country_code": "US",
        "region": "RFCE",
        "climate_zone": "temperate",
        "opening_date": "2018-06-01",
        "operating_months": 12,
        "status": "active",
        "room_count": 120,
        "hotel_class": "upscale",
        "annual_occupancy_rate": Decimal("0.72"),
        "annual_revenue_usd": Decimal("8500000"),
        "electricity_kwh": Decimal("950000"),
        "natural_gas_therms": Decimal("25000"),
        "hotel_operations": {
            "laundry_kg_per_year": Decimal("180000"),
            "pool_heated": True,
            "spa_facility": True,
            "restaurant_on_site": True,
            "conference_rooms": 5,
            "parking_spaces": 200,
        },
        "refrigeration": {
            "refrigerant_type": "R_410A",
            "charge_kg": Decimal("85.0"),
            "annual_leakage_rate": Decimal("0.08"),
        },
    }


# ============================================================================
# CONVENIENCE STORE FIXTURES
# ============================================================================


@pytest.fixture
def sample_convenience_unit() -> Dict[str, Any]:
    """Convenience store franchise unit with 24/7 operation."""
    return {
        "unit_id": "FRN-CVS-001",
        "franchise_type": "convenience_store",
        "ownership_type": "franchised",
        "agreement_type": "multi_unit",
        "unit_name": "QuickMart #0567",
        "brand": "QuickMart",
        "floor_area_m2": Decimal("150"),
        "country_code": "US",
        "region": "SRSO",
        "climate_zone": "arid",
        "opening_date": "2019-01-10",
        "operating_months": 12,
        "status": "active",
        "operating_hours_per_day": 24,
        "annual_revenue_usd": Decimal("2400000"),
        "electricity_kwh": Decimal("210000"),
        "natural_gas_therms": Decimal("800"),
        "refrigeration": {
            "refrigerant_type": "R_404A",
            "charge_kg": Decimal("30.0"),
            "annual_leakage_rate": Decimal("0.18"),
        },
    }


# ============================================================================
# RETAIL FRANCHISE FIXTURES
# ============================================================================


@pytest.fixture
def sample_retail_unit() -> Dict[str, Any]:
    """Retail franchise unit."""
    return {
        "unit_id": "FRN-RTL-001",
        "franchise_type": "retail_store",
        "ownership_type": "franchised",
        "agreement_type": "single_unit",
        "unit_name": "Fashion Forward #089",
        "brand": "Fashion Forward",
        "floor_area_m2": Decimal("400"),
        "country_code": "GB",
        "region": "GB",
        "climate_zone": "temperate",
        "opening_date": "2021-09-01",
        "operating_months": 12,
        "status": "active",
        "annual_revenue_usd": Decimal("950000"),
        "electricity_kwh": Decimal("85000"),
        "natural_gas_therms": Decimal("3500"),
    }


# ============================================================================
# FITNESS CENTER FIXTURES
# ============================================================================


@pytest.fixture
def sample_fitness_unit() -> Dict[str, Any]:
    """Fitness center franchise unit."""
    return {
        "unit_id": "FRN-FIT-001",
        "franchise_type": "fitness_center",
        "ownership_type": "franchised",
        "agreement_type": "area_development",
        "unit_name": "PowerGym #022",
        "brand": "PowerGym",
        "floor_area_m2": Decimal("1200"),
        "country_code": "US",
        "region": "NWPP",
        "climate_zone": "continental",
        "opening_date": "2022-01-15",
        "operating_months": 12,
        "status": "active",
        "annual_revenue_usd": Decimal("1800000"),
        "electricity_kwh": Decimal("350000"),
        "natural_gas_therms": Decimal("8000"),
        "pool_heated": True,
    }


# ============================================================================
# NETWORK INPUT FIXTURES
# ============================================================================


@pytest.fixture
def sample_network_input(
    sample_qsr_unit,
    sample_hotel_unit,
    sample_convenience_unit,
    sample_retail_unit,
    sample_fitness_unit,
) -> Dict[str, Any]:
    """Network input with 5+ unit types."""
    return {
        "network_id": "NET-001",
        "franchisor_name": "Global Brands Inc.",
        "reporting_year": 2025,
        "consolidation_approach": "financial_control",
        "brands": ["Burger Palace", "GreenStay Hotels", "QuickMart", "Fashion Forward", "PowerGym"],
        "units": [
            sample_qsr_unit,
            sample_hotel_unit,
            sample_convenience_unit,
            sample_retail_unit,
            sample_fitness_unit,
        ],
        "total_franchised_units": 5,
        "total_company_owned_units": 2,
        "currency": "USD",
        "tenant_id": "tenant-001",
    }


# ============================================================================
# CALCULATION METHOD INPUT FIXTURES
# ============================================================================


@pytest.fixture
def sample_franchise_specific_input(sample_qsr_unit) -> Dict[str, Any]:
    """Metered data input for franchise-specific calculation."""
    return {
        "method": "franchise_specific",
        "units": [sample_qsr_unit],
        "reporting_year": 2025,
        "gwp_source": "AR6",
        "include_wtt": True,
        "tenant_id": "tenant-001",
    }


@pytest.fixture
def sample_average_data_input() -> Dict[str, Any]:
    """Benchmark input for average-data calculation."""
    return {
        "method": "average_data",
        "franchise_type": "qsr_restaurant",
        "unit_count": 150,
        "avg_floor_area_m2": Decimal("220"),
        "climate_zone": "temperate",
        "country_code": "US",
        "region": "CAMX",
        "reporting_year": 2025,
        "currency": "USD",
        "tenant_id": "tenant-001",
    }


@pytest.fixture
def sample_spend_input() -> Dict[str, Any]:
    """Revenue/royalty input for spend-based calculation."""
    return {
        "method": "spend_based",
        "franchise_type": "qsr_restaurant",
        "naics_code": "722513",
        "total_revenue_usd": Decimal("450000000"),
        "royalty_income_usd": Decimal("27000000"),
        "total_units": 500,
        "reporting_year": 2025,
        "currency": "USD",
        "cpi_year": 2025,
        "tenant_id": "tenant-001",
    }


# ============================================================================
# HOTEL OPERATIONS FIXTURES
# ============================================================================


@pytest.fixture
def sample_hotel_operations() -> Dict[str, Any]:
    """Hotel-specific operations data."""
    return {
        "room_count": 120,
        "hotel_class": "upscale",
        "annual_occupancy_rate": Decimal("0.72"),
        "laundry_kg_per_year": Decimal("180000"),
        "pool_heated": True,
        "spa_facility": True,
        "restaurant_on_site": True,
        "conference_rooms": 5,
        "parking_spaces": 200,
        "water_heating_fuel": "natural_gas",
        "hvac_system": "central_chiller",
    }


# ============================================================================
# COOKING ENERGY FIXTURES
# ============================================================================


@pytest.fixture
def sample_cooking_energy() -> Dict[str, Any]:
    """QSR cooking energy data."""
    return {
        "natural_gas_therms": Decimal("8500"),
        "propane_gallons": Decimal("200"),
        "electricity_kwh": Decimal("25000"),
        "fryer_count": 4,
        "grill_count": 2,
        "oven_count": 2,
    }


# ============================================================================
# REFRIGERANT DATA FIXTURES
# ============================================================================


@pytest.fixture
def sample_refrigeration() -> Dict[str, Any]:
    """Refrigerant data for commercial systems."""
    return {
        "refrigerant_type": "R_404A",
        "charge_kg": Decimal("15.0"),
        "annual_leakage_rate": Decimal("0.15"),
        "equipment_type": "walk_in_cooler",
        "equipment_count": 2,
        "last_service_date": "2025-06-01",
    }


# ============================================================================
# DELIVERY FLEET FIXTURES
# ============================================================================


@pytest.fixture
def sample_delivery_fleet() -> Dict[str, Any]:
    """Vehicle fleet data for QSR delivery."""
    return {
        "vehicle_count": 3,
        "vehicle_type": "light_commercial",
        "fuel_type": "gasoline",
        "annual_fuel_litres": Decimal("9000"),
        "annual_distance_km": Decimal("45000"),
        "avg_fuel_efficiency_l_per_100km": Decimal("12.5"),
    }


# ============================================================================
# MOCK ENGINE FIXTURES
# ============================================================================


@pytest.fixture
def mock_database_engine() -> MagicMock:
    """Mocked FranchiseDatabaseEngine with deterministic responses."""
    engine = MagicMock()
    engine.get_eui_benchmark.return_value = Decimal("450.0")
    engine.get_revenue_intensity.return_value = Decimal("0.085")
    engine.get_grid_ef.return_value = Decimal("0.386")
    engine.get_fuel_ef.return_value = {
        "co2_per_unit": Decimal("5.302"),
        "ch4_per_unit": Decimal("0.001"),
        "n2o_per_unit": Decimal("0.0001"),
    }
    engine.get_refrigerant_gwp.return_value = Decimal("3922")
    engine.get_eeio_factor.return_value = Decimal("0.335")
    engine.get_hotel_benchmark.return_value = Decimal("380.0")
    engine.get_vehicle_ef.return_value = Decimal("2.31")
    engine.validate_franchise_type.return_value = True
    engine.get_dc_rule.return_value = {
        "rule_id": "DC-FRN-001",
        "description": "Company-owned units MUST be in Scope 1/2, NOT Cat 14",
        "action": "reject",
    }
    engine.search_benchmarks.return_value = [
        {"franchise_type": "qsr_restaurant", "climate_zone": "temperate", "eui_kwh_m2": Decimal("450.0")},
    ]
    return engine


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Test configuration dictionary."""
    return {
        "general": {
            "enabled": True,
            "debug": False,
            "log_level": "INFO",
            "agent_id": "GL-MRV-S3-014",
            "agent_component": "AGENT-MRV-027",
            "version": "1.0.0",
            "api_prefix": "/api/v1/franchises",
            "max_batch_size": 10000,
            "default_gwp": "AR6",
        },
        "franchise_specific": {
            "include_cooking_energy": True,
            "include_refrigerants": True,
            "include_delivery_fleet": True,
            "include_wtt": True,
        },
        "average_data": {
            "default_climate_zone": "temperate",
            "hotel_class_adjustment": True,
            "cooking_energy_adjustment": True,
        },
        "spend_based": {
            "default_currency": "USD",
            "cpi_base_year": 2021,
            "margin_removal_rate": Decimal("0.10"),
        },
        "compliance": {
            "default_frameworks": ["ghg_protocol"],
            "enforce_dc_rules": True,
        },
    }


# ============================================================================
# SINGLETON RESET FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all engine singletons before each test."""
    modules_to_reset = [
        "greenlang.franchises.franchise_database",
        "greenlang.franchises.franchise_specific_calculator",
        "greenlang.franchises.average_data_calculator",
        "greenlang.franchises.spend_based_calculator",
        "greenlang.franchises.hybrid_aggregator",
        "greenlang.franchises.compliance_checker",
        "greenlang.franchises.franchises_pipeline",
        "greenlang.franchises.config",
        "greenlang.franchises.provenance",
        "greenlang.franchises.setup",
    ]
    for module_name in modules_to_reset:
        try:
            import importlib
            mod = importlib.import_module(module_name)
            if hasattr(mod, "_instance"):
                mod._instance = None
            if hasattr(mod, "_config_instance"):
                mod._config_instance = None
            if hasattr(mod, "_service_instance"):
                mod._service_instance = None
            if hasattr(mod, "_reset_config"):
                mod._reset_config()
        except (ImportError, AttributeError):
            pass
    yield
