"""
Pytest configuration and shared fixtures for GL-001 test suite.

Provides common fixtures, markers, and test configuration for all test modules.
"""

import pytest
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress debug logs from dependencies
logging.getLogger('asyncio').setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_output_dir(tmp_path):
    """Create temporary directory for test outputs."""
    return tmp_path / "test_outputs"


@pytest.fixture
def standard_process_data():
    """Standard process data for testing."""
    from agents.GL_001.process_heat_orchestrator import ProcessData

    return ProcessData(
        timestamp=datetime.utcnow(),
        temperature_c=250.0,
        pressure_bar=10.0,
        flow_rate_kg_s=5.0,
        energy_input_kw=1000.0,
        energy_output_kw=850.0,
        fuel_type="natural_gas",
        fuel_consumption_rate=10.0
    )


@pytest.fixture
def high_efficiency_process_data():
    """High-efficiency process data."""
    from agents.GL_001.process_heat_orchestrator import ProcessData

    return ProcessData(
        timestamp=datetime.utcnow(),
        temperature_c=400.0,
        pressure_bar=15.0,
        flow_rate_kg_s=8.0,
        energy_input_kw=1000.0,
        energy_output_kw=950.0,
        fuel_type="natural_gas",
        fuel_consumption_rate=10.0
    )


@pytest.fixture
def low_efficiency_process_data():
    """Low-efficiency process data."""
    from agents.GL_001.process_heat_orchestrator import ProcessData

    return ProcessData(
        timestamp=datetime.utcnow(),
        temperature_c=150.0,
        pressure_bar=5.0,
        flow_rate_kg_s=2.0,
        energy_input_kw=1000.0,
        energy_output_kw=600.0,
        fuel_type="coal",
        fuel_consumption_rate=20.0
    )


@pytest.fixture
def process_data_batch():
    """Batch of process data for statistical tests."""
    from agents.GL_001.process_heat_orchestrator import ProcessData

    return [
        ProcessData(
            timestamp=datetime.utcnow() - timedelta(hours=i),
            temperature_c=250.0 + (i * 5),
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0 - (i * 10),
            fuel_type="natural_gas",
            fuel_consumption_rate=10.0
        )
        for i in range(24)
    ]


@pytest.fixture
def mock_plant_config():
    """Mock plant configuration."""
    return {
        'plant_id': 'TEST-PLANT-001',
        'plant_name': 'Test Industrial Plant',
        'location': 'Test City, Test Country',
        'capacity_mw': 50.0,
        'fuel_type': 'natural_gas',
        'scada_host': 'localhost',
        'scada_port': 4840,
        'erp_host': 'localhost',
        'erp_port': 8000,
        'target_efficiency': 0.85,
        'regulatory_co2_limit_kg_mwh': 450.0,
        'emission_monitoring_enabled': True
    }


@pytest.fixture
def mock_scada_data():
    """Mock SCADA sensor data."""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'sensors': {
            'TEMP_001': {'value': 250.5, 'unit': 'C', 'status': 'OK'},
            'PRESSURE_001': {'value': 10.2, 'unit': 'bar', 'status': 'OK'},
            'FLOW_001': {'value': 5.1, 'unit': 'kg/s', 'status': 'OK'},
            'VALVE_001': {'value': 85.0, 'unit': '%', 'status': 'OK'},
            'PUMP_001': {'value': 3000.0, 'unit': 'RPM', 'status': 'OK'},
        }
    }


@pytest.fixture
def mock_erp_data():
    """Mock ERP system data."""
    return {
        'material_id': 'MAT-001',
        'material_name': 'Natural Gas',
        'unit_cost': 0.08,
        'availability': 1000.0,
        'supplier': 'GAS-SUPPLIER-001',
        'lead_time_days': 1
    }


@pytest.fixture
def sample_emission_data():
    """Sample emission measurement data."""
    return {
        'CO2': {'value': 350.0, 'unit': 'kg/MWh', 'limit': 450.0},
        'NOx': {'value': 45.0, 'unit': 'mg/Nm3', 'limit': 100.0},
        'SOx': {'value': 20.0, 'unit': 'mg/Nm3', 'limit': 50.0},
        'PM10': {'value': 15.0, 'unit': 'mg/Nm3', 'limit': 50.0},
    }


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )
    config.addinivalue_line(
        "markers", "compliance: mark test as compliance test"
    )
    config.addinivalue_line(
        "markers", "determinism: mark test as determinism test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )


# Collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Add asyncio marker to async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
