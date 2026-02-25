# -*- coding: utf-8 -*-
"""Shared fixtures for Steam/Heat Purchase Agent tests."""
import pytest
from decimal import Decimal


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all singletons before each test."""
    modules = [
        ("greenlang.steam_heat_purchase.config", "reset_config"),
        ("greenlang.steam_heat_purchase.metrics", "reset"),
        ("greenlang.steam_heat_purchase.provenance", "reset_provenance"),
        (
            "greenlang.steam_heat_purchase.steam_heat_database",
            "SteamHeatDatabaseEngine",
            "reset",
        ),
        (
            "greenlang.steam_heat_purchase.steam_emissions_calculator",
            "SteamEmissionsCalculatorEngine",
            "reset",
        ),
        (
            "greenlang.steam_heat_purchase.heat_cooling_calculator",
            "HeatCoolingCalculatorEngine",
            "reset",
        ),
        (
            "greenlang.steam_heat_purchase.chp_allocation",
            "CHPAllocationEngine",
            "reset",
        ),
        (
            "greenlang.steam_heat_purchase.uncertainty_quantifier",
            "UncertaintyQuantifierEngine",
            "reset",
        ),
        (
            "greenlang.steam_heat_purchase.compliance_checker",
            "ComplianceCheckerEngine",
            "reset",
        ),
        (
            "greenlang.steam_heat_purchase.steam_heat_pipeline",
            "SteamHeatPipelineEngine",
            "reset",
        ),
        ("greenlang.steam_heat_purchase.setup", "reset_service"),
    ]
    yield
    for entry in modules:
        try:
            import importlib

            mod = importlib.import_module(entry[0])
            if len(entry) == 2:
                getattr(mod, entry[1])()
            elif len(entry) == 3:
                cls = getattr(mod, entry[1])
                getattr(cls, entry[2])()
        except (ImportError, AttributeError):
            pass


# Common test fixtures
@pytest.fixture
def sample_steam_request():
    """Create a valid steam calculation request dictionary."""
    return {
        "facility_id": "test-facility-001",
        "consumption_gj": Decimal("1000"),
        "energy_type": "STEAM",
        "fuel_type": "natural_gas",
        "boiler_efficiency": Decimal("0.85"),
        "gwp_source": "AR6",
        "data_quality_tier": "TIER_1",
        "tenant_id": "test-tenant",
    }


@pytest.fixture
def sample_heating_request():
    """Create a valid district heating calculation request dictionary."""
    return {
        "facility_id": "test-facility-002",
        "consumption_gj": Decimal("500"),
        "region": "germany",
        "network_type": "MUNICIPAL",
        "gwp_source": "AR6",
        "tenant_id": "test-tenant",
    }


@pytest.fixture
def sample_cooling_request():
    """Create a valid district cooling calculation request dictionary."""
    return {
        "facility_id": "test-facility-003",
        "cooling_output_gj": Decimal("300"),
        "technology": "centrifugal_chiller",
        "cop": Decimal("6.0"),
        "grid_ef_kgco2e_per_kwh": Decimal("0.436"),
        "gwp_source": "AR6",
        "tenant_id": "test-tenant",
    }


@pytest.fixture
def sample_chp_request():
    """Create a valid CHP allocation request dictionary."""
    return {
        "facility_id": "test-facility-004",
        "total_fuel_gj": Decimal("2000"),
        "fuel_type": "natural_gas",
        "heat_output_gj": Decimal("900"),
        "power_output_gj": Decimal("700"),
        "method": "EFFICIENCY",
        "electrical_efficiency": Decimal("0.35"),
        "thermal_efficiency": Decimal("0.45"),
        "gwp_source": "AR6",
        "tenant_id": "test-tenant",
    }
