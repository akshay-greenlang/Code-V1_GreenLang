# -*- coding: utf-8 -*-
"""
Pytest configuration and shared fixtures for GL-003 SteamSystemAnalyzer tests.

This module provides:
- Configuration for pytest
- Shared fixtures for all test modules
- Test data generators for steam system components
- Mock/stub implementations for external systems
- Async test support

GL-003 focuses on steam generation, distribution, and condensate recovery optimization.
"""

import pytest
import asyncio
import logging
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Any, List, Generator
from unittest.mock import Mock, AsyncMock, MagicMock
import json
from greenlang.determinism import DeterministicClock

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
# BOILER CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def boiler_config_data() -> Dict[str, Any]:
    """Create valid boiler configuration data."""
    return {
        'boiler_id': 'BOILER-STEAM-001',
        'boiler_type': 'firetube',
        'manufacturer': 'Cleaver-Brooks',
        'model': 'CB-700',
        'rated_capacity_lb_hr': 10000,
        'steam_pressure_psig': 150,
        'design_temperature_f': 366,  # Saturation temp at 150 psig
        'fuel_type': 'natural_gas',
        'fuel_heating_value_btu_lb': 21500,
        'design_efficiency_percent': 85.0,
        'actual_efficiency_percent': 80.0,
        'feedwater_temperature_f': 180,
        'stack_temperature_f': 350,
        'excess_air_percent': 15.0,
        'blowdown_percent': 5.0,
        'commissioning_date': datetime(2020, 1, 1),
        'operating_hours': 50000
    }


@pytest.fixture
def steam_system_config() -> Dict[str, Any]:
    """Create steam system configuration."""
    return {
        'system_id': 'STEAM-SYS-001',
        'total_steam_production_lb_hr': 10000,
        'distribution_pressure_psig': 150,
        'condensate_return_percent': 40,
        'trap_count': 500,
        'operating_hours_per_year': 8400,
        'ambient_temperature_f': 70
    }


@pytest.fixture
def steam_trap_config() -> Dict[str, Any]:
    """Create steam trap configuration data."""
    return {
        'total_trap_count': 500,
        'trap_types': [
            {
                'trap_type': 'thermostatic',
                'count': 200,
                'steam_pressure_psig': 100,
                'orifice_size_inch': 0.125
            },
            {
                'trap_type': 'inverted_bucket',
                'count': 200,
                'steam_pressure_psig': 150,
                'orifice_size_inch': 0.25
            },
            {
                'trap_type': 'thermodynamic',
                'count': 100,
                'steam_pressure_psig': 100,
                'orifice_size_inch': 0.125
            }
        ],
        'last_inspection_months': 24,
        'failure_rate_percent': 20,
        'steam_cost_per_1000lb': 8.50,
        'operating_hours_per_year': 8400
    }


@pytest.fixture
def condensate_recovery_config() -> Dict[str, Any]:
    """Create condensate recovery configuration."""
    return {
        'steam_production_lb_hr': 10000,
        'current_condensate_return_percent': 40,
        'condensate_temperature_f': 180,
        'makeup_water_temperature_f': 60,
        'makeup_water_cost_per_1000gal': 3.50,
        'water_treatment_cost_per_1000gal': 2.50,
        'fuel_cost_per_mmbtu': 5.00,
        'boiler_efficiency': 0.80,
        'operating_hours_per_year': 8400,
        'target_condensate_return_percent': 80
    }


@pytest.fixture
def pressure_optimization_config() -> Dict[str, Any]:
    """Create pressure optimization configuration."""
    return {
        'current_pressure_psig': 150,
        'minimum_process_pressure_psig': 80,
        'pressure_drop_distribution_psi': 10,
        'safety_margin_psi': 15,
        'steam_production_lb_hr': 10000,
        'boiler_efficiency': 0.80,
        'fuel_cost_per_mmbtu': 5.00,
        'operating_hours_per_year': 8400
    }


@pytest.fixture
def insulation_config() -> Dict[str, Any]:
    """Create insulation assessment configuration."""
    return {
        'components': [
            {
                'component_type': 'pipe',
                'diameter_inches': 4,
                'length_feet': 500,
                'steam_temperature_f': 350,
                'current_insulation': 'poor_1inch'
            },
            {
                'component_type': 'valve',
                'diameter_inches': 4,
                'length_feet': 5,
                'steam_temperature_f': 350,
                'current_insulation': 'none'
            },
            {
                'component_type': 'flange',
                'diameter_inches': 4,
                'length_feet': 2,
                'steam_temperature_f': 350,
                'current_insulation': 'none'
            }
        ],
        'ambient_temperature_f': 70,
        'fuel_cost_per_mmbtu': 5.00,
        'operating_hours_per_year': 8760
    }


# ============================================================================
# OPERATIONAL DATA FIXTURES
# ============================================================================

@pytest.fixture
def boiler_operational_data() -> Dict[str, Any]:
    """Create realistic boiler operational data."""
    return {
        'timestamp': datetime.now(timezone.utc),
        'steam_flow_rate_lb_hr': 8000.0,
        'steam_pressure_psig': 148.0,
        'steam_temperature_f': 365.0,
        'feedwater_flow_rate_lb_hr': 8400.0,
        'feedwater_temperature_f': 180.0,
        'fuel_flow_rate_cuft_hr': 10000.0,
        'stack_temperature_f': 350.0,
        'o2_percent': 4.5,
        'co2_percent': 10.0,
        'co_ppm': 15.0,
        'excess_air_percent': 15.0,
        'drum_level_percent': 50.0,
        'blowdown_flow_rate_lb_hr': 400.0,
        'combustion_efficiency_percent': 82.5,
        'thermal_efficiency_percent': 78.3,
        'ambient_temperature_f': 70.0,
        'ambient_humidity_percent': 60.0
    }


@pytest.fixture
def steam_meter_data() -> Dict[str, Any]:
    """Create steam meter reading data."""
    return {
        'meter_id': 'STEAM-METER-001',
        'timestamp': datetime.now(timezone.utc),
        'flow_rate_lb_hr': 8000.0,
        'total_flow_lb': 67200000.0,
        'pressure_psig': 148.0,
        'temperature_f': 365.0,
        'quality_percent': 98.5,
        'density_lb_cuft': 0.395,
        'quality': 'good',
        'alarm_status': 'normal'
    }


@pytest.fixture
def pressure_sensor_data() -> Dict[str, Any]:
    """Create pressure sensor data."""
    return {
        'sensor_id': 'PRESS-SENSOR-001',
        'timestamp': datetime.now(timezone.utc),
        'pressure_psig': 148.0,
        'temperature_f': 365.0,
        'quality': 'good',
        'calibration_date': datetime(2024, 1, 1),
        'alarm_high': 160.0,
        'alarm_low': 140.0,
        'status': 'normal'
    }


@pytest.fixture
def steam_trap_inspection_data() -> Dict[str, Any]:
    """Create steam trap inspection data."""
    return {
        'trap_id': 'TRAP-001',
        'trap_type': 'inverted_bucket',
        'location': 'Building-A-Line-1',
        'inspection_date': datetime.now(timezone.utc),
        'inspector': 'John Doe',
        'status': 'failed',
        'failure_mode': 'blowing_through',
        'steam_pressure_psig': 150,
        'orifice_size_inch': 0.25,
        'estimated_loss_lb_hr': 150.0,
        'estimated_cost_loss_usd_year': 10710.0,
        'recommendation': 'Replace immediately',
        'priority': 'high'
    }


# ============================================================================
# SENSOR DATA WITH QUALITY INDICATORS
# ============================================================================

@pytest.fixture
def sensor_data_with_quality() -> Dict[str, Any]:
    """Create sensor data with quality indicators."""
    return {
        'steam_flow': {
            'value': 8000.0,
            'quality': 'good',
            'timestamp': datetime.now(timezone.utc),
            'unit': 'lb/hr'
        },
        'steam_pressure': {
            'value': 148.0,
            'quality': 'good',
            'timestamp': datetime.now(timezone.utc),
            'unit': 'psig'
        },
        'steam_temperature': {
            'value': 365.0,
            'quality': 'good',
            'timestamp': datetime.now(timezone.utc),
            'unit': 'degF'
        },
        'feedwater_temperature': {
            'value': 180.0,
            'quality': 'good',
            'timestamp': datetime.now(timezone.utc),
            'unit': 'degF'
        },
        'stack_temperature': {
            'value': 350.0,
            'quality': 'good',
            'timestamp': datetime.now(timezone.utc),
            'unit': 'degF'
        },
        'o2_percent': {
            'value': 4.5,
            'quality': 'good',
            'timestamp': datetime.now(timezone.utc),
            'unit': '%'
        }
    }


# ============================================================================
# BOUNDARY TEST CASES
# ============================================================================

@pytest.fixture
def boundary_test_cases() -> List[Dict[str, Any]]:
    """Create test cases for boundary testing."""
    return [
        {
            'name': 'min_steam_capacity',
            'rated_capacity_lb_hr': 1000.0,
            'expected_status': 'valid'
        },
        {
            'name': 'max_steam_capacity',
            'rated_capacity_lb_hr': 100000.0,
            'expected_status': 'valid'
        },
        {
            'name': 'zero_steam_production',
            'steam_production_lb_hr': 0.0,
            'expected_status': 'invalid'
        },
        {
            'name': 'negative_pressure',
            'steam_pressure_psig': -10.0,
            'expected_status': 'invalid'
        },
        {
            'name': 'excessive_pressure',
            'steam_pressure_psig': 2000.0,
            'expected_status': 'invalid'
        },
        {
            'name': 'stack_temp_too_low',
            'stack_temperature_f': 150.0,
            'expected_status': 'invalid'
        },
        {
            'name': 'stack_temp_too_high',
            'stack_temperature_f': 900.0,
            'expected_status': 'invalid'
        },
        {
            'name': 'excessive_excess_air',
            'excess_air_percent': 120.0,
            'expected_status': 'invalid'
        },
        {
            'name': '100_percent_condensate_return',
            'condensate_return_percent': 100.0,
            'expected_status': 'valid'
        },
        {
            'name': 'negative_trap_count',
            'trap_count': -10,
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
        'BOILER.STEAM.FLOW': 8000.0,
        'BOILER.STEAM.PRESSURE': 148.0,
        'BOILER.STEAM.TEMPERATURE': 365.0,
        'BOILER.FEEDWATER.FLOW': 8400.0,
        'BOILER.FEEDWATER.TEMP': 180.0,
        'BOILER.FUEL.FLOW': 10000.0,
        'BOILER.STACK.TEMP': 350.0,
        'BOILER.O2.PERCENT': 4.5
    })
    mock.write_setpoint = AsyncMock(return_value=True)
    mock.get_alarms = AsyncMock(return_value=[])
    mock.disconnect = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_steam_meter_connector() -> AsyncMock:
    """Create mock steam meter connector."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.read_flow_data = AsyncMock(return_value={
        'flow_rate_lb_hr': 8000.0,
        'total_flow_lb': 67200000.0,
        'pressure_psig': 148.0,
        'temperature_f': 365.0,
        'quality_percent': 98.5
    })
    mock.disconnect = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_pressure_sensor() -> AsyncMock:
    """Create mock pressure sensor."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.read_pressure = AsyncMock(return_value={
        'pressure_psig': 148.0,
        'temperature_f': 365.0,
        'quality': 'good'
    })
    mock.disconnect = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_mqtt_broker() -> AsyncMock:
    """Create mock MQTT broker for IoT sensors."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.publish = AsyncMock(return_value=True)
    mock.subscribe = AsyncMock(return_value=True)
    mock.on_message = Mock()
    mock.disconnect = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_agent_intelligence() -> AsyncMock:
    """Create mock agent intelligence."""
    mock = AsyncMock()
    mock.classify_operation_mode = AsyncMock(
        return_value={'mode': 'normal', 'confidence': 0.95}
    )
    mock.classify_trap_failure = AsyncMock(
        return_value={'failed': False, 'confidence': 0.92}
    )
    mock.generate_recommendations = AsyncMock(
        return_value=[
            'Improve condensate recovery system',
            'Upgrade insulation on distribution lines',
            'Implement trap monitoring program'
        ]
    )
    return mock


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

class TestDataGenerator:
    """Generate test data for various scenarios."""

    @staticmethod
    def generate_boiler_efficiency_test_cases() -> List[Dict[str, Any]]:
        """Generate boiler efficiency calculation test cases."""
        return [
            {
                'name': 'high_efficiency_natural_gas',
                'boiler_type': 'firetube',
                'fuel_type': 'natural_gas',
                'rated_capacity_lb_hr': 10000,
                'stack_temperature_f': 300,
                'excess_air_percent': 10,
                'expected_efficiency_min': 82.0,
                'expected_efficiency_max': 88.0
            },
            {
                'name': 'medium_efficiency_fuel_oil',
                'boiler_type': 'firetube',
                'fuel_type': 'fuel_oil',
                'rated_capacity_lb_hr': 10000,
                'stack_temperature_f': 400,
                'excess_air_percent': 20,
                'expected_efficiency_min': 75.0,
                'expected_efficiency_max': 82.0
            },
            {
                'name': 'low_efficiency_old_system',
                'boiler_type': 'firetube',
                'fuel_type': 'coal',
                'rated_capacity_lb_hr': 10000,
                'stack_temperature_f': 550,
                'excess_air_percent': 35,
                'expected_efficiency_min': 65.0,
                'expected_efficiency_max': 75.0
            }
        ]

    @staticmethod
    def generate_steam_trap_audit_test_cases() -> List[Dict[str, Any]]:
        """Generate steam trap audit test cases."""
        return [
            {
                'total_trap_count': 100,
                'failure_rate_percent': 10,
                'expected_failed_traps': 10,
                'tolerance': 2
            },
            {
                'total_trap_count': 500,
                'failure_rate_percent': 25,
                'expected_failed_traps': 125,
                'tolerance': 10
            },
            {
                'total_trap_count': 1000,
                'failure_rate_percent': 30,
                'expected_failed_traps': 300,
                'tolerance': 20
            }
        ]

    @staticmethod
    def generate_condensate_recovery_test_cases() -> List[Dict[str, Any]]:
        """Generate condensate recovery test cases."""
        return [
            {
                'steam_production_lb_hr': 10000,
                'current_return_percent': 40,
                'target_return_percent': 80,
                'expected_savings_min_mmbtu': 3500,
                'expected_savings_max_mmbtu': 4500
            },
            {
                'steam_production_lb_hr': 5000,
                'current_return_percent': 20,
                'target_return_percent': 70,
                'expected_savings_min_mmbtu': 2000,
                'expected_savings_max_mmbtu': 3000
            }
        ]

    @staticmethod
    def generate_pressure_optimization_test_cases() -> List[Dict[str, Any]]:
        """Generate pressure optimization test cases."""
        return [
            {
                'current_pressure_psig': 150,
                'minimum_process_pressure_psig': 80,
                'expected_reduction_psi_min': 35,
                'expected_reduction_psi_max': 55,
                'expected_savings_percent': 4.5
            },
            {
                'current_pressure_psig': 200,
                'minimum_process_pressure_psig': 100,
                'expected_reduction_psi_min': 65,
                'expected_reduction_psi_max': 85,
                'expected_savings_percent': 8.0
            }
        ]

    @staticmethod
    def generate_insulation_loss_test_cases() -> List[Dict[str, Any]]:
        """Generate insulation loss test cases."""
        return [
            {
                'component_type': 'pipe',
                'diameter_inches': 4,
                'length_feet': 100,
                'insulation': 'none',
                'expected_loss_btu_hr_min': 13000,
                'expected_loss_btu_hr_max': 17000
            },
            {
                'component_type': 'valve',
                'diameter_inches': 6,
                'length_feet': 8,
                'insulation': 'none',
                'expected_loss_btu_hr_min': 16000,
                'expected_loss_btu_hr_max': 20000
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
            self.start_time = DeterministicClock.now()
            return self

        def __exit__(self, *args):
            self.end_time = DeterministicClock.now()

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
        'calculator_boiler_efficiency_ms': 100.0,
        'calculator_steam_trap_ms': 150.0,
        'calculator_condensate_recovery_ms': 100.0,
        'calculator_pressure_optimization_ms': 80.0,
        'calculator_insulation_loss_ms': 120.0,
        'scada_read_ms': 500.0,
        'steam_meter_read_ms': 300.0,
        'memory_usage_mb': 512.0,
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
    - TEST_MQTT_USERNAME, TEST_MQTT_PASSWORD
    - TEST_STEAM_METER_API_KEY
    - TEST_PRESSURE_SENSOR_API_KEY
    - TEST_CLOUD_API_KEY

    Args:
        credential_type: Type of credentials to retrieve

    Returns:
        Dictionary with credentials or mock credentials for testing
    """
    if credential_type == "scada":
        return {
            "username": os.getenv("TEST_SCADA_USERNAME", "test_user"),
            "password": os.getenv("TEST_SCADA_PASSWORD", "test_pass")
        }
    elif credential_type == "mqtt":
        return {
            "username": os.getenv("TEST_MQTT_USERNAME", "test_mqtt_user"),
            "password": os.getenv("TEST_MQTT_PASSWORD", "test_mqtt_pass"),
            "client_id": os.getenv("TEST_MQTT_CLIENT_ID", "test-client-001")
        }
    elif credential_type == "steam_meter":
        return {
            "api_key": os.getenv("TEST_STEAM_METER_API_KEY", "test-meter-api-key")
        }
    elif credential_type == "pressure_sensor":
        return {
            "api_key": os.getenv("TEST_PRESSURE_SENSOR_API_KEY", "test-sensor-api-key")
        }
    elif credential_type == "cloud":
        return {
            "api_key": os.getenv("TEST_CLOUD_API_KEY", "test-cloud-api-key")
        }
    else:
        raise ValueError(f"Unknown credential type: {credential_type}")


@pytest.fixture
def scada_credentials() -> Dict[str, str]:
    """Provide SCADA credentials from environment."""
    return get_test_credentials("scada")


@pytest.fixture
def mqtt_credentials() -> Dict[str, str]:
    """Provide MQTT credentials from environment."""
    return get_test_credentials("mqtt")


@pytest.fixture
def steam_meter_credentials() -> Dict[str, str]:
    """Provide steam meter credentials from environment."""
    return get_test_credentials("steam_meter")


@pytest.fixture
def pressure_sensor_credentials() -> Dict[str, str]:
    """Provide pressure sensor credentials from environment."""
    return get_test_credentials("pressure_sensor")


@pytest.fixture
def cloud_credentials() -> Dict[str, str]:
    """Provide cloud credentials from environment."""
    return get_test_credentials("cloud")


# ============================================================================
# EDGE CASE TESTING FIXTURES
# ============================================================================

@pytest.fixture
def extreme_values() -> Dict[str, Any]:
    """Provide extreme values for boundary testing."""
    return {
        'max_float': sys.float_info.max,
        'min_float': sys.float_info.min,
        'epsilon': sys.float_info.epsilon,
        'max_int': sys.maxsize,
        'positive_inf': float('inf'),
        'negative_inf': float('-inf'),
        'nan': float('nan'),
        'very_small': 1e-300,
        'very_large': 1e300,
        'positive_zero': 0.0,
        'negative_zero': -0.0
    }


@pytest.fixture
def invalid_data_samples() -> List[Dict[str, Any]]:
    """Provide invalid data samples for error testing."""
    return [
        {'type': 'none_value', 'data': None},
        {'type': 'empty_dict', 'data': {}},
        {'type': 'empty_string', 'data': ''},
        {'type': 'negative_value', 'data': -100},
        {'type': 'zero_value', 'data': 0},
        {'type': 'infinity', 'data': float('inf')},
        {'type': 'nan', 'data': float('nan')},
        {'type': 'invalid_type', 'data': 'not_a_number'},
        {'type': 'missing_required', 'data': {'incomplete': True}},
    ]


@pytest.fixture
def unicode_test_strings() -> List[str]:
    """Provide Unicode test strings for edge case testing."""
    return [
        "蒸汽系统-001",  # Chinese
        "паровая-001",  # Russian
        "भाप-001",  # Hindi
        "بخار-001",  # Arabic
        "スチーム-001",  # Japanese
        "🔥-STEAM-001",  # Emoji
        "STEAM\u0000NULL",  # Null character
        "STEAM\n\t\r",  # Control characters
        "'; DROP TABLE steam_systems; --",  # SQL injection
        "A" * 10000,  # Very long string
    ]


@pytest.fixture
def malformed_sensor_data() -> List[Dict[str, Any]]:
    """Provide malformed sensor data for testing."""
    return [
        {'steam_flow_lb_hr': 'not_a_number'},
        {'steam_flow_lb_hr': None},
        {'steam_flow_lb_hr': -1000},
        {'steam_flow_lb_hr': float('inf')},
        {'steam_flow_lb_hr': float('nan')},
        {'steam_pressure_psig': -50},
        {'steam_pressure_psig': 5000},
        {'condensate_return_percent': -10},
        {'condensate_return_percent': 150},
        {},  # Missing all fields
    ]


@pytest.fixture
def performance_test_data() -> Dict[str, Any]:
    """Provide data for performance testing."""
    return {
        'small_dataset': [{'id': i, 'value': i * 2} for i in range(10)],
        'medium_dataset': [{'id': i, 'value': i * 2} for i in range(1000)],
        'large_dataset': [{'id': i, 'value': i * 2} for i in range(10000)],
        'max_concurrent_requests': 100,
        'burst_sizes': [100, 500, 1000, 2000, 5000],
        'load_test_duration_seconds': 5,
        'throughput_target_rps': 1000,
    }


@pytest.fixture
def timeout_scenarios() -> List[Dict[str, Any]]:
    """Provide timeout scenarios for testing."""
    return [
        {'name': 'immediate', 'delay_seconds': 0, 'timeout_seconds': 0.1},
        {'name': 'fast', 'delay_seconds': 0.05, 'timeout_seconds': 0.1},
        {'name': 'boundary', 'delay_seconds': 0.1, 'timeout_seconds': 0.1},
        {'name': 'slow', 'delay_seconds': 0.5, 'timeout_seconds': 0.1},
        {'name': 'very_slow', 'delay_seconds': 5.0, 'timeout_seconds': 0.1},
    ]


@pytest.fixture
def integration_failure_scenarios() -> List[Dict[str, Any]]:
    """Provide integration failure scenarios."""
    return [
        {
            'name': 'connection_refused',
            'error_type': ConnectionRefusedError,
            'message': 'Connection refused'
        },
        {
            'name': 'timeout',
            'error_type': TimeoutError,
            'message': 'Connection timeout'
        },
        {
            'name': 'connection_reset',
            'error_type': ConnectionResetError,
            'message': 'Connection reset by peer'
        },
        {
            'name': 'broken_pipe',
            'error_type': BrokenPipeError,
            'message': 'Broken pipe'
        },
        {
            'name': 'permission_denied',
            'error_type': PermissionError,
            'message': 'Permission denied'
        },
        {
            'name': 'network_unreachable',
            'error_type': OSError,
            'message': 'Network is unreachable'
        },
    ]


@pytest.fixture
def mock_failing_scada() -> AsyncMock:
    """Create SCADA connector that fails intermittently."""
    mock = AsyncMock()
    call_count = [0]

    async def failing_connect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] % 3 == 0:
            return True
        raise ConnectionError("SCADA connection failed")

    mock.connect = AsyncMock(side_effect=failing_connect)
    mock.read_tags = AsyncMock(side_effect=TimeoutError("Read timeout"))
    mock.disconnect = AsyncMock(return_value=True)

    return mock


@pytest.fixture
def mock_rate_limited_api() -> AsyncMock:
    """Create API mock that implements rate limiting."""
    mock = AsyncMock()
    request_times = []

    async def rate_limited_call(*args, **kwargs):
        current_time = time.time()
        request_times.append(current_time)

        # Check rate (max 10 requests per second)
        recent_requests = [t for t in request_times if current_time - t < 1.0]

        if len(recent_requests) > 10:
            raise Exception("Rate limit exceeded: 429 Too Many Requests")

        return {'status': 'success', 'data': {}}

    mock.call = AsyncMock(side_effect=rate_limited_call)

    return mock


@pytest.fixture
def stress_test_config() -> Dict[str, Any]:
    """Provide configuration for stress testing."""
    return {
        'max_concurrent_operations': 100,
        'sustained_load_duration_seconds': 5,
        'burst_sizes': [100, 500, 1000, 2000],
        'memory_limit_mb': 512,
        'cpu_intensive_iterations': 100000,
    }


@pytest.fixture
async def async_test_helper():
    """Helper for async test operations."""

    class AsyncTestHelper:
        """Helper class for async testing utilities."""

        @staticmethod
        async def wait_with_timeout(coro, timeout_seconds: float):
            """Wait for coroutine with timeout."""
            try:
                return await asyncio.wait_for(coro, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Operation timed out after {timeout_seconds}s")

        @staticmethod
        async def run_concurrent(operations: List, max_concurrent: int = 10):
            """Run operations with concurrency limit."""
            semaphore = asyncio.Semaphore(max_concurrent)

            async def bounded_operation(op):
                async with semaphore:
                    return await op

            return await asyncio.gather(*[bounded_operation(op) for op in operations])

        @staticmethod
        def measure_execution_time(func):
            """Decorator to measure execution time."""
            async def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - start
                return result, duration

            return wrapper

    return AsyncTestHelper()


# ============================================================================
# BENCHMARK AND METRICS FIXTURES
# ============================================================================

@pytest.fixture
def benchmark_thresholds() -> Dict[str, float]:
    """Provide performance benchmark thresholds."""
    return {
        'cache_operation_ms': 1.0,
        'boiler_efficiency_calculation_ms': 100.0,
        'steam_trap_audit_ms': 150.0,
        'condensate_recovery_ms': 100.0,
        'pressure_optimization_ms': 80.0,
        'insulation_assessment_ms': 120.0,
        'orchestrator_execute_ms': 3000.0,
        'throughput_min_rps': 100.0,
        'memory_increase_max_mb': 100.0,
        'p99_latency_ms': 10.0,
        'cache_hit_rate_min': 0.7,
    }


@pytest.fixture
def metrics_collector():
    """Provide metrics collector for test measurements."""

    class MetricsCollector:
        """Collect and analyze test metrics."""

        def __init__(self):
            self.metrics = {
                'execution_times': [],
                'memory_samples': [],
                'throughput_samples': [],
                'error_counts': {},
                'cache_stats': {'hits': 0, 'misses': 0},
            }

        def record_execution_time(self, operation: str, duration_ms: float):
            """Record execution time for an operation."""
            self.metrics['execution_times'].append({
                'operation': operation,
                'duration_ms': duration_ms,
                'timestamp': datetime.now(timezone.utc)
            })

        def record_error(self, error_type: str):
            """Record error occurrence."""
            self.metrics['error_counts'][error_type] = \
                self.metrics['error_counts'].get(error_type, 0) + 1

        def record_cache_hit(self):
            """Record cache hit."""
            self.metrics['cache_stats']['hits'] += 1

        def record_cache_miss(self):
            """Record cache miss."""
            self.metrics['cache_stats']['misses'] += 1

        def get_summary(self) -> Dict[str, Any]:
            """Get metrics summary."""
            exec_times = [m['duration_ms'] for m in self.metrics['execution_times']]

            summary = {
                'total_operations': len(exec_times),
                'avg_execution_time_ms': sum(exec_times) / len(exec_times) if exec_times else 0,
                'min_execution_time_ms': min(exec_times) if exec_times else 0,
                'max_execution_time_ms': max(exec_times) if exec_times else 0,
                'total_errors': sum(self.metrics['error_counts'].values()),
                'cache_hit_rate': (
                    self.metrics['cache_stats']['hits'] /
                    (self.metrics['cache_stats']['hits'] + self.metrics['cache_stats']['misses'])
                    if (self.metrics['cache_stats']['hits'] + self.metrics['cache_stats']['misses']) > 0
                    else 0
                ),
            }

            return summary

    return MetricsCollector()


logger.info("Conftest.py loaded for GL-003 SteamSystemAnalyzer with comprehensive fixtures")
