"""Pytest configuration and shared fixtures for GL-009 tests.

This module provides shared pytest fixtures, test configuration,
and helper utilities used across all test modules.

Fixtures:
    - Mock energy meter data
    - Mock historian data
    - Sample thermal efficiency inputs
    - Test configuration
    - Database fixtures
    - Cache fixtures
    - API client fixtures

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import os
import tempfile
import json
from typing import Dict, List, Any
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from decimal import Decimal

# Test configuration constants
TEST_SEED = 42
TEST_PRECISION = 4
TEST_BALANCE_TOLERANCE = 2.0


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide test configuration settings."""
    return {
        "seed": TEST_SEED,
        "precision": TEST_PRECISION,
        "balance_tolerance_percent": TEST_BALANCE_TOLERANCE,
        "test_mode": True,
        "environment": "test",
        "log_level": "DEBUG"
    }


@pytest.fixture
def sample_reference_environment():
    """Create sample reference environment for exergy calculations."""
    from calculators.second_law_efficiency import ReferenceEnvironment

    return ReferenceEnvironment(
        temperature_k=298.15,  # 25C
        pressure_kpa=101.325,  # 1 atm
        relative_humidity_percent=60.0
    )


@pytest.fixture
def sample_surface_geometry():
    """Create sample surface geometry for heat loss calculations."""
    from calculators.heat_loss_calculator import SurfaceGeometry, SurfaceOrientation

    return SurfaceGeometry(
        surface_area_m2=50.0,
        length_m=5.0,
        orientation=SurfaceOrientation.VERTICAL,
        emissivity=0.85,
        view_factor=1.0
    )


@pytest.fixture
def sample_insulation_layers():
    """Create sample insulation layers for testing."""
    from calculators.heat_loss_calculator import InsulationLayer, InsulationMaterial

    return [
        InsulationLayer(
            material=InsulationMaterial.MINERAL_WOOL,
            thickness_m=0.1,
            thermal_conductivity_w_mk=0.045
        ),
        InsulationLayer(
            material=InsulationMaterial.CALCIUM_SILICATE,
            thickness_m=0.05,
            thermal_conductivity_w_mk=0.065
        )
    ]


@pytest.fixture
def sample_flue_gas_composition():
    """Create sample flue gas composition."""
    from calculators.heat_loss_calculator import FlueGasComposition

    return FlueGasComposition(
        co2_percent=12.0,
        o2_percent=6.0,
        n2_percent=75.0,
        h2o_percent=7.0,
        co_ppm=50.0,
        excess_air_percent=20.0
    )


@pytest.fixture
def mock_energy_meter_data() -> Dict[str, Any]:
    """Generate mock energy meter data for testing."""
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "meter_id": "EM-001",
        "fuel_flow_kg_s": 0.5,
        "fuel_type": "natural_gas",
        "fuel_hhv_kj_kg": 50000.0,
        "steam_flow_kg_s": 4.0,
        "steam_pressure_kpa": 1000.0,
        "steam_temperature_k": 473.15,
        "steam_enthalpy_kj_kg": 2776.0,
        "feedwater_temperature_k": 298.15,
        "feedwater_enthalpy_kj_kg": 104.89,
        "auxiliary_power_kw": 50.0
    }


@pytest.fixture
def mock_historian_data() -> List[Dict[str, Any]]:
    """Generate mock historian time-series data."""
    base_time = datetime.utcnow()
    data_points = []

    for i in range(24):  # 24 hours of hourly data
        timestamp = base_time - timedelta(hours=i)
        data_points.append({
            "timestamp": timestamp.isoformat() + "Z",
            "boiler_temperature_k": 473.15 + (i % 5) * 2.0,
            "surface_temperature_k": 323.15 + (i % 3) * 1.5,
            "ambient_temperature_k": 298.15,
            "flue_gas_temperature_k": 423.15 + (i % 4) * 3.0,
            "flue_gas_flow_kg_s": 5.0 + (i % 5) * 0.2,
            "fuel_flow_kg_s": 0.5 + (i % 3) * 0.05,
            "steam_production_kg_s": 4.0 + (i % 4) * 0.1,
            "efficiency_percent": 85.0 + (i % 3) * 0.5
        })

    return data_points


@pytest.fixture
def sample_thermal_efficiency_input() -> Dict[str, Any]:
    """Create sample input for thermal efficiency calculation."""
    return {
        "energy_inputs": {
            "natural_gas": 1000.0,
            "preheated_air": 50.0
        },
        "useful_outputs": {
            "steam": 850.0,
            "hot_water": 50.0
        },
        "losses": {
            "flue_gas": 70.0,
            "radiation": 15.0,
            "convection": 10.0,
            "other": 5.0
        }
    }


@pytest.fixture
def sample_boiler_parameters() -> Dict[str, Any]:
    """Create sample boiler operating parameters."""
    return {
        "fuel_input_kw": 10000.0,
        "steam_output_kw": 8500.0,
        "surface_area_m2": 100.0,
        "surface_temperature_k": 343.15,  # 70C
        "ambient_temperature_k": 298.15,  # 25C
        "flue_gas_temperature_k": 423.15,  # 150C
        "flue_gas_flow_kg_s": 5.0,
        "boiler_type": "fire_tube",
        "fuel_type": "natural_gas",
        "design_efficiency_percent": 85.0
    }


@pytest.fixture
def sample_exergy_streams():
    """Create sample exergy streams for Second Law calculations."""
    from calculators.second_law_efficiency import ExergyStream, StreamType

    input_streams = [
        ExergyStream(
            stream_type=StreamType.FUEL,
            stream_name="natural_gas",
            mass_flow_kg_s=0.5,
            temperature_k=298.15,
            pressure_kpa=500.0,
            specific_enthalpy_kj_kg=50000.0,
            specific_entropy_kj_kg_k=0.0,
            chemical_exergy_kj_kg=51000.0,
            is_input=True
        ),
        ExergyStream(
            stream_type=StreamType.COMBUSTION_AIR,
            stream_name="air",
            mass_flow_kg_s=5.0,
            temperature_k=298.15,
            pressure_kpa=101.325,
            specific_enthalpy_kj_kg=298.0,
            specific_entropy_kj_kg_k=6.86,
            is_input=True
        )
    ]

    output_streams = [
        ExergyStream(
            stream_type=StreamType.STEAM,
            stream_name="steam",
            mass_flow_kg_s=4.0,
            temperature_k=473.15,
            pressure_kpa=1000.0,
            specific_enthalpy_kj_kg=2776.0,
            specific_entropy_kj_kg_k=6.587,
            is_input=False
        ),
        ExergyStream(
            stream_type=StreamType.FLUE_GAS,
            stream_name="flue_gas",
            mass_flow_kg_s=5.5,
            temperature_k=423.15,
            pressure_kpa=101.325,
            specific_enthalpy_kj_kg=450.0,
            specific_entropy_kj_kg_k=7.2,
            is_input=False
        )
    ]

    return {"inputs": input_streams, "outputs": output_streams}


@pytest.fixture
def temp_database_path(tmp_path):
    """Create temporary database path for testing."""
    db_path = tmp_path / "test_thermal_efficiency.db"
    return str(db_path)


@pytest.fixture
def mock_database_connection(temp_database_path):
    """Create mock database connection."""
    mock_conn = Mock()
    mock_conn.connection_string = f"sqlite:///{temp_database_path}"
    mock_conn.is_connected = True

    # Mock cursor operations
    mock_cursor = Mock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_conn.cursor.return_value = mock_cursor

    return mock_conn


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory for testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def mock_redis_cache():
    """Create mock Redis cache for testing."""
    mock_cache = Mock()
    mock_cache.cache_data = {}

    def mock_get(key):
        return mock_cache.cache_data.get(key)

    def mock_set(key, value, ttl=None):
        mock_cache.cache_data[key] = value
        return True

    def mock_delete(key):
        if key in mock_cache.cache_data:
            del mock_cache.cache_data[key]
        return True

    def mock_exists(key):
        return key in mock_cache.cache_data

    mock_cache.get = mock_get
    mock_cache.set = mock_set
    mock_cache.delete = mock_delete
    mock_cache.exists = mock_exists

    return mock_cache


@pytest.fixture
async def mock_energy_meter_connector():
    """Create mock energy meter connector."""
    from integrations.energy_meter_connector import EnergyMeterConnector

    mock_connector = AsyncMock(spec=EnergyMeterConnector)
    mock_connector.connect = AsyncMock(return_value=True)
    mock_connector.disconnect = AsyncMock(return_value=True)
    mock_connector.read_current_values = AsyncMock(return_value={
        "fuel_flow_kg_s": 0.5,
        "steam_flow_kg_s": 4.0,
        "feedwater_temp_k": 298.15,
        "steam_temp_k": 473.15
    })
    mock_connector.is_connected = True

    return mock_connector


@pytest.fixture
async def mock_historian_connector():
    """Create mock historian connector."""
    from integrations.historian_connector import HistorianConnector

    mock_connector = AsyncMock(spec=HistorianConnector)
    mock_connector.connect = AsyncMock(return_value=True)
    mock_connector.disconnect = AsyncMock(return_value=True)
    mock_connector.query_time_series = AsyncMock(return_value=[
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "value": 85.0,
            "tag": "efficiency_percent"
        }
    ])
    mock_connector.is_connected = True

    return mock_connector


@pytest.fixture
def mock_api_client():
    """Create mock API client for testing."""
    from fastapi.testclient import TestClient

    # This would be imported from the actual main.py FastAPI app
    # For now, we'll create a mock
    mock_client = Mock(spec=TestClient)

    return mock_client


@pytest.fixture
def known_test_cases():
    """Load known test cases with expected results for validation."""
    return {
        "case_1": {
            "description": "Natural gas boiler - 85% efficiency",
            "inputs": {
                "fuel_input_kw": 1000.0,
                "steam_output_kw": 850.0,
                "auxiliary_kw": 0.0
            },
            "expected_efficiency": 85.0
        },
        "case_2": {
            "description": "Coal boiler with high losses",
            "inputs": {
                "fuel_input_kw": 1000.0,
                "useful_output_kw": 750.0,
                "flue_gas_loss_kw": 150.0,
                "radiation_loss_kw": 60.0,
                "other_losses_kw": 40.0
            },
            "expected_efficiency": 75.0
        },
        "case_3": {
            "description": "High efficiency condensing boiler",
            "inputs": {
                "fuel_input_kw": 1000.0,
                "steam_output_kw": 950.0,
                "auxiliary_kw": 20.0
            },
            "expected_efficiency": 93.0
        }
    }


@pytest.fixture
def benchmark_data():
    """Provide industry benchmark data for comparison tests."""
    return {
        "natural_gas_boilers": {
            "percentile_25": 80.0,
            "percentile_50": 85.0,
            "percentile_75": 88.0,
            "percentile_90": 90.0,
            "best_practice": 92.0
        },
        "coal_boilers": {
            "percentile_25": 75.0,
            "percentile_50": 80.0,
            "percentile_75": 83.0,
            "percentile_90": 85.0,
            "best_practice": 87.0
        },
        "biomass_boilers": {
            "percentile_25": 70.0,
            "percentile_50": 75.0,
            "percentile_75": 78.0,
            "percentile_90": 80.0,
            "best_practice": 82.0
        }
    }


# Pytest hooks and configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "determinism: marks tests for determinism verification"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add e2e marker to e2e tests
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.slow)

        # Add determinism marker to determinism tests
        if "determinism" in str(item.fspath):
            item.add_marker(pytest.mark.determinism)


# Helper functions for tests
def assert_close(actual: float, expected: float, rel_tol: float = 1e-6, abs_tol: float = 1e-9):
    """Assert that two floats are close within tolerance."""
    import math
    assert math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol), \
        f"Expected {expected}, got {actual} (difference: {abs(actual - expected)})"


def assert_provenance_valid(provenance_hash: str):
    """Assert that a provenance hash is valid SHA-256."""
    assert isinstance(provenance_hash, str), "Provenance hash must be string"
    assert len(provenance_hash) == 64, f"SHA-256 hash must be 64 chars, got {len(provenance_hash)}"
    assert all(c in '0123456789abcdef' for c in provenance_hash.lower()), \
        "Provenance hash must be valid hex string"


def assert_energy_balance(
    input_kw: float,
    output_kw: float,
    losses_kw: float,
    tolerance_percent: float = 2.0
):
    """Assert that energy balance holds within tolerance."""
    balance_error = abs(input_kw - output_kw - losses_kw)
    balance_error_percent = (balance_error / input_kw * 100) if input_kw > 0 else 0

    assert balance_error_percent <= tolerance_percent, \
        f"Energy balance error {balance_error_percent:.2f}% exceeds tolerance {tolerance_percent}%"
