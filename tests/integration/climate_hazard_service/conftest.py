# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-DATA-020 Climate Hazard Connector integration tests.

Provides reusable test fixtures for:
- Override of parent conftest autouse fixtures (mock_agents, block_network)
- Configuration reset between tests (fresh_config)
- ClimateHazardService instance fixture
- Sample hazard data fixtures (sources, hazard data, assets)
- Sample multi-hazard and portfolio fixtures
- ProvenanceTracker fixture

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-020 Climate Hazard Connector (GL-DATA-GEO-002)
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from greenlang.climate_hazard.config import reset_config


# ---------------------------------------------------------------------------
# Override parent conftest autouse fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents fixture.

    The parent tests/integration/conftest.py defines an autouse fixture
    that patches greenlang.agents.registry.get_agent, which does not
    apply to Climate Hazard Connector integration tests.
    """
    return {}


@pytest.fixture(scope="session", autouse=True)
def block_network():
    """Override parent conftest block_network fixture.

    The parent tests/integration/conftest.py blocks all socket access,
    which can interfere with asyncio event loop creation. We disable it
    for Climate Hazard Connector integration tests since our tests are
    fully self-contained.
    """
    pass


# ---------------------------------------------------------------------------
# Configuration reset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_config():
    """Reset the singleton config before and after each test.

    Ensures each test starts with a clean default configuration
    and does not leak state to subsequent tests.
    """
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utcnow_iso() -> str:
    """Return current UTC datetime as ISO-8601 string."""
    return _utcnow().isoformat()


# ---------------------------------------------------------------------------
# Service fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Create a fresh ClimateHazardService instance for each test.

    Returns a fully initialised service with in-memory stores, ready
    for integration testing without external database dependencies.
    """
    from greenlang.climate_hazard.setup import ClimateHazardService

    svc = ClimateHazardService()
    return svc


@pytest.fixture
def started_service(service):
    """Create a ClimateHazardService that has been started via startup()."""
    service.startup()
    return service


# ---------------------------------------------------------------------------
# ProvenanceTracker fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def provenance_tracker():
    """Create a fresh ProvenanceTracker for provenance chain testing."""
    from greenlang.climate_hazard.provenance import ProvenanceTracker

    return ProvenanceTracker(genesis_hash="integration-test-genesis")


# ---------------------------------------------------------------------------
# HazardPipelineEngine fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_engine():
    """Create a fresh HazardPipelineEngine for pipeline testing."""
    from greenlang.climate_hazard.hazard_pipeline import HazardPipelineEngine

    return HazardPipelineEngine()


# ---------------------------------------------------------------------------
# Sample hazard source fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_source_noaa() -> Dict[str, Any]:
    """Sample NOAA climate hazard data source."""
    return {
        "name": "NOAA NCEI Climate Data",
        "source_type": "noaa",
        "hazard_types": ["flood", "storm", "tropical_cyclone"],
        "region": "North America",
        "description": "NOAA National Centers for Environmental Information",
        "metadata": {"agency": "NOAA", "data_format": "CSV", "update_frequency": "daily"},
    }


@pytest.fixture
def sample_source_copernicus() -> Dict[str, Any]:
    """Sample Copernicus climate hazard data source."""
    return {
        "name": "Copernicus Climate Data Store",
        "source_type": "copernicus",
        "hazard_types": ["heat_wave", "drought", "wildfire", "precipitation_change"],
        "region": "Europe",
        "description": "European Copernicus Climate Change Service (C3S)",
        "metadata": {"agency": "ECMWF", "data_format": "NetCDF"},
    }


@pytest.fixture
def sample_source_ipcc() -> Dict[str, Any]:
    """Sample IPCC climate scenario data source."""
    return {
        "name": "IPCC AR6 Climate Projections",
        "source_type": "ipcc",
        "hazard_types": ["temperature_change", "sea_level_rise", "precipitation_change"],
        "region": "Global",
        "description": "IPCC Sixth Assessment Report scenario projections",
        "metadata": {"report": "AR6", "working_group": "WG1"},
    }


@pytest.fixture
def sample_source_world_bank() -> Dict[str, Any]:
    """Sample World Bank climate data source."""
    return {
        "name": "World Bank Climate Knowledge Portal",
        "source_type": "world_bank",
        "hazard_types": ["drought", "water_stress", "heat_wave"],
        "region": "Global",
        "description": "World Bank Climate Change Knowledge Portal data",
        "metadata": {"portal": "CCKP", "data_format": "JSON"},
    }


# ---------------------------------------------------------------------------
# Sample hazard data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_hazard_data_flood() -> Dict[str, Any]:
    """Sample flood hazard data record."""
    now = _utcnow_iso()
    return {
        "hazard_type": "flood",
        "location_id": "loc_london_uk",
        "scenario": "SSP2-4.5",
        "value": 72.5,
        "unit": "mm/day",
        "parameters": {
            "return_period": 100,
            "depth_meters": 1.5,
            "duration_hours": 48,
        },
        "timestamp_start": "2025-01-01T00:00:00+00:00",
        "timestamp_end": "2025-12-31T23:59:59+00:00",
    }


@pytest.fixture
def sample_hazard_data_heat_wave() -> Dict[str, Any]:
    """Sample heat wave hazard data record."""
    return {
        "hazard_type": "heat_wave",
        "location_id": "loc_paris_fr",
        "scenario": "SSP5-8.5",
        "value": 42.3,
        "unit": "degrees_celsius",
        "parameters": {
            "consecutive_days": 14,
            "temperature_anomaly": 8.5,
            "humidity_percent": 35,
        },
        "timestamp_start": "2025-06-01T00:00:00+00:00",
        "timestamp_end": "2025-08-31T23:59:59+00:00",
    }


@pytest.fixture
def sample_hazard_data_drought() -> Dict[str, Any]:
    """Sample drought hazard data record."""
    return {
        "hazard_type": "drought",
        "location_id": "loc_madrid_es",
        "scenario": "SSP3-7.0",
        "value": 85.0,
        "unit": "spi_index",
        "parameters": {
            "drought_severity": "extreme",
            "duration_months": 6,
            "area_affected_km2": 50000,
        },
        "timestamp_start": "2025-03-01T00:00:00+00:00",
        "timestamp_end": "2025-09-30T23:59:59+00:00",
    }


@pytest.fixture
def sample_hazard_data_wildfire() -> Dict[str, Any]:
    """Sample wildfire hazard data record."""
    return {
        "hazard_type": "wildfire",
        "location_id": "loc_sydney_au",
        "scenario": "SSP5-8.5",
        "value": 91.2,
        "unit": "fwi",
        "parameters": {
            "fire_weather_index": 91.2,
            "area_burned_ha": 25000,
            "fire_danger_class": "extreme",
        },
        "timestamp_start": "2025-11-01T00:00:00+00:00",
        "timestamp_end": "2026-02-28T23:59:59+00:00",
    }


@pytest.fixture
def sample_hazard_data_sea_level_rise() -> Dict[str, Any]:
    """Sample sea level rise hazard data record."""
    return {
        "hazard_type": "sea_level_rise",
        "location_id": "loc_miami_us",
        "scenario": "SSP5-8.5",
        "value": 0.65,
        "unit": "meters",
        "parameters": {
            "thermal_expansion_cm": 30,
            "ice_sheet_contribution_cm": 25,
            "glacier_contribution_cm": 10,
        },
        "timestamp_start": "2050-01-01T00:00:00+00:00",
        "timestamp_end": "2050-12-31T23:59:59+00:00",
    }


# ---------------------------------------------------------------------------
# Sample asset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_asset_factory() -> Dict[str, Any]:
    """Sample factory asset for registration."""
    return {
        "name": "Munich Manufacturing Plant",
        "asset_type": "factory",
        "location_id": "loc_munich_de",
        "coordinates": {"lat": 48.1351, "lon": 11.5820},
        "value": 25_000_000.0,
        "currency": "EUR",
        "sector": "manufacturing",
        "metadata": {"employees": 500, "area_m2": 15000},
    }


@pytest.fixture
def sample_asset_warehouse() -> Dict[str, Any]:
    """Sample warehouse asset for registration."""
    return {
        "name": "Rotterdam Port Warehouse",
        "asset_type": "warehouse",
        "location_id": "loc_rotterdam_nl",
        "coordinates": {"lat": 51.9225, "lon": 4.4792},
        "value": 8_000_000.0,
        "currency": "EUR",
        "sector": "logistics",
        "metadata": {"capacity_m3": 50000},
    }


@pytest.fixture
def sample_asset_office() -> Dict[str, Any]:
    """Sample office asset for registration."""
    return {
        "name": "London HQ Office",
        "asset_type": "office",
        "location_id": "loc_london_uk",
        "coordinates": {"lat": 51.5074, "lon": -0.1278},
        "value": 15_000_000.0,
        "currency": "GBP",
        "sector": "financial_services",
        "metadata": {"floors": 12, "employees": 350},
    }


@pytest.fixture
def sample_asset_data_center() -> Dict[str, Any]:
    """Sample data center asset for registration."""
    return {
        "name": "Singapore Data Center",
        "asset_type": "data_center",
        "location_id": "loc_singapore_sg",
        "coordinates": {"lat": 1.3521, "lon": 103.8198},
        "value": 50_000_000.0,
        "currency": "USD",
        "sector": "technology",
        "metadata": {"tier": 4, "power_mw": 20},
    }


@pytest.fixture
def sample_asset_port() -> Dict[str, Any]:
    """Sample port asset for registration."""
    return {
        "name": "Miami Container Port",
        "asset_type": "port",
        "location_id": "loc_miami_us",
        "coordinates": {"lat": 25.7617, "lon": -80.1918},
        "value": 120_000_000.0,
        "currency": "USD",
        "sector": "transport",
        "metadata": {"teu_capacity": 1_200_000, "berths": 12},
    }


# ---------------------------------------------------------------------------
# Pipeline asset fixtures (for HazardPipelineEngine)
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_assets() -> List[Dict[str, Any]]:
    """List of asset dictionaries suitable for pipeline engine run_pipeline."""
    return [
        {
            "asset_id": "asset_hq_london",
            "name": "London HQ",
            "asset_type": "office",
            "location": {"lat": 51.5074, "lon": -0.1278},
        },
        {
            "asset_id": "asset_factory_munich",
            "name": "Munich Factory",
            "asset_type": "factory",
            "location": {"lat": 48.1351, "lon": 11.5820},
        },
        {
            "asset_id": "asset_warehouse_rotterdam",
            "name": "Rotterdam Warehouse",
            "asset_type": "warehouse",
            "location": {"lat": 51.9225, "lon": 4.4792},
        },
    ]


@pytest.fixture
def pipeline_single_asset() -> List[Dict[str, Any]]:
    """Single asset dictionary for basic pipeline testing."""
    return [
        {
            "asset_id": "asset_single_test",
            "name": "Test Office",
            "asset_type": "office",
            "location": {"lat": 40.7128, "lon": -74.0060},
        },
    ]


# ---------------------------------------------------------------------------
# Hazard type lists
# ---------------------------------------------------------------------------


@pytest.fixture
def single_hazard_type() -> List[str]:
    """Single hazard type for focused testing."""
    return ["flood"]


@pytest.fixture
def multi_hazard_types() -> List[str]:
    """Multiple hazard types for multi-hazard testing."""
    return ["flood", "heat_wave", "drought"]


@pytest.fixture
def all_hazard_types() -> List[str]:
    """All supported climate hazard types."""
    return [
        "flood",
        "drought",
        "wildfire",
        "heat_wave",
        "cold_wave",
        "storm",
        "sea_level_rise",
        "tropical_cyclone",
        "landslide",
        "water_stress",
        "precipitation_change",
        "temperature_change",
        "compound",
    ]


# ---------------------------------------------------------------------------
# Scenario and time horizon fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ssp_scenarios() -> List[str]:
    """SSP climate scenario pathways."""
    return ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]


@pytest.fixture
def time_horizons() -> List[str]:
    """Standard time horizons for scenario projection."""
    return ["SHORT_TERM", "MID_TERM", "LONG_TERM"]


@pytest.fixture
def report_frameworks() -> List[str]:
    """All supported compliance report frameworks."""
    return ["tcfd", "csrd", "eu_taxonomy"]


# ---------------------------------------------------------------------------
# Composite workflow helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def registered_source_id(service, sample_source_noaa) -> str:
    """Register a source and return its source_id for downstream tests."""
    result = service.register_source(**sample_source_noaa)
    return result["source_id"]


@pytest.fixture
def registered_asset_id(service, sample_asset_factory) -> str:
    """Register an asset and return its asset_id for downstream tests."""
    result = service.register_asset(**sample_asset_factory)
    return result["asset_id"]


@pytest.fixture
def ingested_record_id(service, registered_source_id, sample_hazard_data_flood) -> str:
    """Ingest hazard data and return the record_id for downstream tests."""
    data = dict(sample_hazard_data_flood)
    data["source_id"] = registered_source_id
    result = service.ingest_hazard_data(**data)
    return result["record_id"]
