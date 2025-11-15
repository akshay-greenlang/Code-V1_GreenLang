"""
Pytest configuration and shared fixtures for GL-002 BoilerEfficiencyOptimizer tests.

This module provides:
- Configuration for pytest
- Shared fixtures for all test modules
- Test data generators
- Mock/stub implementations
- Async test support
"""

import pytest
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Generator
from unittest.mock import Mock, AsyncMock, MagicMock
import json

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tests.log')
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "determinism: mark test as a determinism test"
    )
    config.addinivalue_line(
        "markers", "compliance: mark test as a compliance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "boundary: mark test as a boundary test"
    )
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def boiler_config_data() -> Dict[str, Any]:
    """Create valid boiler configuration data."""
    return {
        'boiler_id': 'BOILER-001',
        'manufacturer': 'Cleaver-Brooks',
        'model': 'CB-8000',
        'type': 'water-tube',
        'max_steam_capacity_kg_hr': 50000,
        'min_steam_capacity_kg_hr': 10000,
        'design_pressure_bar': 40,
        'design_temperature_c': 450,
        'primary_fuel_type': 'natural_gas',
        'fuel_heating_value_mj_kg': 50,
        'design_efficiency_percent': 85,
        'actual_efficiency_percent': 80,
        'heating_surface_area_m2': 500,
        'furnace_volume_m3': 100,
        'commissioning_date': datetime(2020, 1, 1),
        'operating_hours': 20000
    }


@pytest.fixture
def operational_constraints_data() -> Dict[str, Any]:
    """Create operational constraints data."""
    return {
        'max_pressure_bar': 42,
        'min_pressure_bar': 5,
        'max_temperature_c': 480,
        'min_temperature_c': 150,
        'min_excess_air_percent': 5.0,
        'max_excess_air_percent': 25.0,
        'min_o2_percent': 2.0,
        'max_co_ppm': 100.0,
        'min_load_percent': 20.0,
        'max_load_percent': 100.0,
        'max_load_change_rate_percent_min': 5.0,
        'min_steam_quality': 0.95,
        'max_tds_ppm': 3500,
        'max_moisture_percent': 0.5
    }


@pytest.fixture
def emission_limits_data() -> Dict[str, Any]:
    """Create emission limits data."""
    return {
        'nox_limit_ppm': 30,
        'nox_limit_mg_nm3': 65,
        'co_limit_ppm': 50,
        'co_limit_mg_nm3': 100,
        'regulation_standard': 'EPA-NSPS'
    }


@pytest.fixture
def optimization_parameters_data() -> Dict[str, Any]:
    """Create optimization parameters data."""
    return {
        'primary_objective': 'efficiency',
        'secondary_objectives': ['emissions', 'cost'],
        'optimization_interval_seconds': 60,
        'prediction_horizon_minutes': 30,
        'control_horizon_minutes': 10,
        'convergence_tolerance': 0.001,
        'max_iterations': 100,
        'efficiency_weight': 0.4,
        'emissions_weight': 0.3,
        'cost_weight': 0.3
    }


@pytest.fixture
def integration_settings_data() -> Dict[str, Any]:
    """Create integration settings data."""
    return {
        'scada_enabled': True,
        'scada_polling_interval_seconds': 5,
        'dcs_enabled': True,
        'dcs_write_enabled': False,
        'historian_enabled': True,
        'historian_retention_days': 365,
        'alert_enabled': True,
        'alert_channels': ['email', 'sms']
    }


# ============================================================================
# OPERATIONAL DATA FIXTURES
# ============================================================================

@pytest.fixture
def boiler_operational_data() -> Dict[str, Any]:
    """Create realistic boiler operational data."""
    return {
        'timestamp': datetime.now(),
        'fuel_flow_rate_kg_hr': 1500.0,
        'steam_flow_rate_kg_hr': 20000.0,
        'combustion_temperature_c': 1200.0,
        'excess_air_percent': 15.0,
        'flue_gas_temperature_c': 180.0,
        'steam_pressure_bar': 35.0,
        'steam_temperature_c': 400.0,
        'feedwater_temperature_c': 100.0,
        'drum_level_percent': 50.0,
        'o2_flue_gas_percent': 4.5,
        'co2_flue_gas_percent': 10.0,
        'co_ppm': 15.0,
        'boiler_load_percent': 75.0,
        'efficiency_percent': 82.5,
        'co2_emissions_kg_hr': 3900.0,
        'nox_emissions_ppm': 22.0,
        'ambient_temperature_c': 25.0,
        'ambient_humidity_percent': 60.0
    }


@pytest.fixture
def sensor_data_with_quality() -> Dict[str, Any]:
    """Create sensor data with quality indicators."""
    return {
        'fuel_flow_rate': {
            'value': 1500.0,
            'quality': 'good',
            'timestamp': datetime.now()
        },
        'steam_flow_rate': {
            'value': 20000.0,
            'quality': 'good',
            'timestamp': datetime.now()
        },
        'combustion_temperature': {
            'value': 1200.0,
            'quality': 'good',
            'timestamp': datetime.now()
        },
        'flue_gas_temperature': {
            'value': 180.0,
            'quality': 'good',
            'timestamp': datetime.now()
        },
        'o2_percent': {
            'value': 4.5,
            'quality': 'good',
            'timestamp': datetime.now()
        },
        'co2_percent': {
            'value': 10.0,
            'quality': 'good',
            'timestamp': datetime.now()
        }
    }


@pytest.fixture
def boundary_test_cases() -> List[Dict[str, Any]]:
    """Create test cases for boundary testing."""
    return [
        {
            'name': 'min_fuel_flow',
            'fuel_flow_kg_hr': 100.0,
            'expected_status': 'valid'
        },
        {
            'name': 'max_fuel_flow',
            'fuel_flow_kg_hr': 3000.0,
            'expected_status': 'valid'
        },
        {
            'name': 'below_min_load',
            'load_percent': 10.0,
            'expected_status': 'invalid'
        },
        {
            'name': 'above_max_load',
            'load_percent': 110.0,
            'expected_status': 'invalid'
        },
        {
            'name': 'zero_fuel_flow',
            'fuel_flow_kg_hr': 0.0,
            'expected_status': 'invalid'
        },
        {
            'name': 'negative_temperature',
            'temperature_c': -50.0,
            'expected_status': 'invalid'
        },
        {
            'name': 'excessive_excess_air',
            'excess_air_percent': 50.0,
            'expected_status': 'invalid'
        }
    ]


# ============================================================================
# MOCK/STUB FIXTURES
# ============================================================================

@pytest.fixture
def mock_scada_connector() -> AsyncMock:
    """Create mock SCADA connector."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.read_tags = AsyncMock(return_value={
        'fuel_flow': 1500.0,
        'steam_flow': 20000.0,
        'temperature': 1200.0
    })
    mock.write_setpoint = AsyncMock(return_value=True)
    mock.get_alarms = AsyncMock(return_value=[])
    mock.disconnect = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_dcs_connector() -> AsyncMock:
    """Create mock DCS connector."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.read_process_data = AsyncMock(return_value={
        'pressure': 35.0,
        'temperature': 400.0,
        'flow': 20000.0
    })
    mock.send_command = AsyncMock(return_value=True)
    mock.disconnect = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_historian() -> AsyncMock:
    """Create mock historian connector."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.write_data = AsyncMock(return_value=True)
    mock.query_historical = AsyncMock(return_value=[])
    mock.get_statistics = AsyncMock(return_value={
        'count': 1000,
        'min': 0.0,
        'max': 100.0,
        'avg': 50.0
    })
    mock.disconnect = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_agent_intelligence() -> AsyncMock:
    """Create mock agent intelligence."""
    mock = AsyncMock()
    mock.classify_operation_mode = AsyncMock(
        return_value={'mode': 'normal', 'confidence': 0.95}
    )
    mock.classify_anomaly = AsyncMock(
        return_value={'anomaly': False, 'confidence': 0.99}
    )
    mock.generate_recommendations = AsyncMock(
        return_value=[
            'Increase combustion air flow',
            'Reduce fuel flow rate'
        ]
    )
    return mock


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

class TestDataGenerator:
    """Generate test data for various scenarios."""

    @staticmethod
    def generate_efficiency_test_cases() -> List[Dict[str, Any]]:
        """Generate efficiency calculation test cases."""
        return [
            {
                'name': 'high_efficiency',
                'fuel_flow': 1000.0,
                'steam_flow': 850.0,
                'expected_efficiency': 0.85,
                'tolerance': 0.01
            },
            {
                'name': 'medium_efficiency',
                'fuel_flow': 1000.0,
                'steam_flow': 800.0,
                'expected_efficiency': 0.80,
                'tolerance': 0.01
            },
            {
                'name': 'low_efficiency',
                'fuel_flow': 1000.0,
                'steam_flow': 700.0,
                'expected_efficiency': 0.70,
                'tolerance': 0.01
            }
        ]

    @staticmethod
    def generate_combustion_test_cases() -> List[Dict[str, Any]]:
        """Generate combustion calculation test cases."""
        return [
            {
                'fuel_type': 'natural_gas',
                'fuel_flow': 1500.0,
                'o2_percent': 4.5,
                'expected_excess_air': 15.0,
                'tolerance': 2.0
            },
            {
                'fuel_type': 'fuel_oil',
                'fuel_flow': 1200.0,
                'o2_percent': 3.5,
                'expected_excess_air': 10.0,
                'tolerance': 2.0
            },
            {
                'fuel_type': 'coal',
                'fuel_flow': 2000.0,
                'o2_percent': 5.5,
                'expected_excess_air': 20.0,
                'tolerance': 2.0
            }
        ]

    @staticmethod
    def generate_emissions_test_cases() -> List[Dict[str, Any]]:
        """Generate emissions calculation test cases."""
        return [
            {
                'fuel_type': 'natural_gas',
                'fuel_flow': 1500.0,
                'expected_co2': 3000.0,
                'tolerance': 100.0
            },
            {
                'fuel_type': 'fuel_oil',
                'fuel_flow': 1200.0,
                'expected_co2': 3200.0,
                'tolerance': 150.0
            },
            {
                'fuel_type': 'coal',
                'fuel_flow': 2000.0,
                'expected_co2': 5200.0,
                'tolerance': 300.0
            }
        ]

    @staticmethod
    def generate_steam_quality_test_cases() -> List[Dict[str, Any]]:
        """Generate steam quality test cases."""
        return [
            {
                'pressure_bar': 10.0,
                'temperature_c': 179.9,
                'expected_quality': 0.95,
                'tolerance': 0.05
            },
            {
                'pressure_bar': 20.0,
                'temperature_c': 212.0,
                'expected_quality': 0.98,
                'tolerance': 0.02
            },
            {
                'pressure_bar': 40.0,
                'temperature_c': 250.0,
                'expected_quality': 0.99,
                'tolerance': 0.01
            }
        ]


@pytest.fixture
def test_data_generator() -> TestDataGenerator:
    """Provide test data generator."""
    return TestDataGenerator()


# ============================================================================
# TIMING AND PERFORMANCE FIXTURES
# ============================================================================

@pytest.fixture
def performance_timer():
    """Context manager for timing test execution."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = datetime.now()
            return self

        def __exit__(self, *args):
            self.end_time = datetime.now()

        @property
        def elapsed_ms(self) -> float:
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds() * 1000
            return 0.0

    return Timer


@pytest.fixture
def benchmark_targets() -> Dict[str, float]:
    """Provide performance benchmark targets."""
    return {
        'orchestrator_process_ms': 3000.0,
        'calculator_efficiency_ms': 100.0,
        'calculator_combustion_ms': 100.0,
        'calculator_emissions_ms': 80.0,
        'scada_read_ms': 500.0,
        'scada_write_ms': 500.0,
        'memory_usage_mb': 500.0,
        'throughput_rps': 100.0
    }


# ============================================================================
# CLEANUP FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up resources after each test."""
    yield
    # Cleanup code here if needed
    logger.debug("Test cleanup completed")


# ============================================================================
# LOGGING FIXTURES
# ============================================================================

@pytest.fixture
def caplog_with_level(caplog):
    """Fixture for capturing logs at DEBUG level."""
    caplog.set_level(logging.DEBUG)
    return caplog


# ============================================================================
# CREDENTIALS MANAGEMENT (Environment-based, no hardcoding)
# ============================================================================

def get_test_credentials(credential_type: str) -> Dict[str, str]:
    """
    Get test credentials from environment variables.

    This ensures credentials are never hardcoded in test files.
    For CI/CD or local testing, set environment variables:
    - TEST_SCADA_USERNAME, TEST_SCADA_PASSWORD
    - TEST_DCS_USERNAME, TEST_DCS_PASSWORD
    - TEST_ERP_API_KEY
    - TEST_HISTORIAN_USERNAME, TEST_HISTORIAN_PASSWORD
    - TEST_CLOUD_API_KEY

    Args:
        credential_type: Type of credentials to retrieve

    Returns:
        Dictionary with credentials or mock credentials for testing
    """
    if credential_type == "scada_dcs":
        return {
            "username": os.getenv("TEST_SCADA_USERNAME", "test_user"),
            "password": os.getenv("TEST_SCADA_PASSWORD", "test_pass")
        }
    elif credential_type == "erp":
        return {
            "api_key": os.getenv("TEST_ERP_API_KEY", "test-api-key")
        }
    elif credential_type == "historian":
        return {
            "username": os.getenv("TEST_HISTORIAN_USERNAME", "reader"),
            "password": os.getenv("TEST_HISTORIAN_PASSWORD", "readonly")
        }
    elif credential_type == "cloud":
        return {
            "api_key": os.getenv("TEST_CLOUD_API_KEY", "test-cloud-api-key")
        }
    else:
        raise ValueError(f"Unknown credential type: {credential_type}")


@pytest.fixture
def scada_dcs_credentials() -> Dict[str, str]:
    """Provide SCADA/DCS credentials from environment."""
    return get_test_credentials("scada_dcs")


@pytest.fixture
def erp_credentials() -> Dict[str, str]:
    """Provide ERP credentials from environment."""
    return get_test_credentials("erp")


@pytest.fixture
def historian_credentials() -> Dict[str, str]:
    """Provide historian credentials from environment."""
    return get_test_credentials("historian")


@pytest.fixture
def cloud_credentials() -> Dict[str, str]:
    """Provide cloud credentials from environment."""
    return get_test_credentials("cloud")
