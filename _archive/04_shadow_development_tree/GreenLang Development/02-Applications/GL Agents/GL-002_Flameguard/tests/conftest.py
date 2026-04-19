"""
GL-002 FLAMEGUARD - Test Configuration

Pytest fixtures and configuration for comprehensive testing.
"""

import pytest
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, AsyncMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "performance: performance tests")
    config.addinivalue_line("markers", "compliance: compliance tests")
    config.addinivalue_line("markers", "slow: slow tests")


# =============================================================================
# PROCESS DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_process_data() -> Dict[str, float]:
    """Sample boiler process data."""
    return {
        "drum_pressure_psig": 125.5,
        "drum_level_inches": 0.5,
        "steam_flow_klb_hr": 150.0,
        "steam_temperature_f": 450.0,
        "feedwater_temperature_f": 220.0,
        "flue_gas_temperature_f": 350.0,
        "o2_percent": 3.5,
        "co_ppm": 25.0,
        "fuel_flow_scfh": 25000.0,
        "fuel_pressure_psig": 15.0,
        "air_flow_scfm": 8500.0,
        "ambient_temperature_f": 70.0,
        "load_percent": 75.0,
    }


@pytest.fixture
def high_load_process_data() -> Dict[str, float]:
    """Process data at high load (95%)."""
    return {
        "drum_pressure_psig": 145.0,
        "drum_level_inches": 0.3,
        "steam_flow_klb_hr": 190.0,
        "steam_temperature_f": 480.0,
        "feedwater_temperature_f": 225.0,
        "flue_gas_temperature_f": 380.0,
        "o2_percent": 2.5,
        "co_ppm": 50.0,
        "fuel_flow_scfh": 32000.0,
        "fuel_pressure_psig": 18.0,
        "air_flow_scfm": 11000.0,
        "ambient_temperature_f": 75.0,
        "load_percent": 95.0,
    }


@pytest.fixture
def low_load_process_data() -> Dict[str, float]:
    """Process data at low load (30%)."""
    return {
        "drum_pressure_psig": 100.0,
        "drum_level_inches": 0.8,
        "steam_flow_klb_hr": 60.0,
        "steam_temperature_f": 400.0,
        "feedwater_temperature_f": 210.0,
        "flue_gas_temperature_f": 300.0,
        "o2_percent": 5.0,
        "co_ppm": 15.0,
        "fuel_flow_scfh": 9500.0,
        "fuel_pressure_psig": 12.0,
        "air_flow_scfm": 4000.0,
        "ambient_temperature_f": 70.0,
        "load_percent": 30.0,
    }


# =============================================================================
# FUEL PROPERTIES FIXTURES
# =============================================================================


@pytest.fixture
def sample_fuel_properties() -> Dict[str, float]:
    """Sample natural gas properties."""
    return {
        "hhv_btu_scf": 1050.0,
        "lhv_btu_scf": 950.0,
        "specific_gravity": 0.6,
        "h2_percent": 0.0,
        "moisture_percent": 0.0,
        "carbon_content": 0.75,
    }


@pytest.fixture
def fuel_oil_properties() -> Dict[str, float]:
    """Fuel oil #2 properties."""
    return {
        "hhv_btu_lb": 19500.0,
        "lhv_btu_lb": 18300.0,
        "specific_gravity": 0.87,
        "sulfur_percent": 0.3,
        "carbon_percent": 87.0,
        "hydrogen_percent": 12.0,
        "ash_percent": 0.01,
    }


@pytest.fixture
def coal_properties() -> Dict[str, float]:
    """Bituminous coal properties."""
    return {
        "hhv_btu_lb": 12500.0,
        "lhv_btu_lb": 11800.0,
        "carbon_percent": 75.0,
        "hydrogen_percent": 5.0,
        "sulfur_percent": 1.5,
        "nitrogen_percent": 1.5,
        "oxygen_percent": 7.0,
        "moisture_percent": 5.0,
        "ash_percent": 9.0,
    }


# =============================================================================
# CALCULATION INPUT FIXTURES
# =============================================================================


@pytest.fixture
def sample_efficiency_input() -> Dict[str, float]:
    """Sample efficiency calculation input."""
    return {
        "fuel_input_mmbtu_hr": 185.0,
        "steam_output_mmbtu_hr": 152.0,
        "blowdown_mmbtu_hr": 3.7,
        "flue_gas_temp_f": 350.0,
        "ambient_temp_f": 70.0,
        "o2_percent": 3.5,
        "fuel_moisture_percent": 0.0,
        "air_humidity_percent": 50.0,
    }


@pytest.fixture
def sample_emissions_input() -> Dict[str, float]:
    """Sample emissions calculation input."""
    return {
        "fuel_flow_scfh": 25000.0,
        "fuel_hhv_btu_scf": 1050.0,
        "o2_percent": 3.5,
        "nox_ppm": 45.0,
        "co_ppm": 25.0,
        "flue_gas_flow_scfm": 10000.0,
    }


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def boiler_config() -> Dict:
    """Sample boiler configuration."""
    return {
        "boiler_id": "BOILER-001",
        "name": "Main Boiler #1",
        "capacity_klb_hr": 200.0,
        "design_pressure_psig": 150.0,
        "fuel_type": "natural_gas",
        "o2_setpoint_min": 1.5,
        "o2_setpoint_max": 6.0,
        "co_limit_ppm": 400.0,
    }


@pytest.fixture
def multi_boiler_config() -> List[Dict]:
    """Configuration for multiple boilers."""
    return [
        {
            "boiler_id": "BOILER-001",
            "name": "Main Boiler #1",
            "capacity_klb_hr": 200.0,
            "design_efficiency": 82.0,
            "fuel_cost_per_mmbtu": 5.0,
        },
        {
            "boiler_id": "BOILER-002",
            "name": "Main Boiler #2",
            "capacity_klb_hr": 150.0,
            "design_efficiency": 80.0,
            "fuel_cost_per_mmbtu": 5.5,
        },
        {
            "boiler_id": "BOILER-003",
            "name": "Auxiliary Boiler",
            "capacity_klb_hr": 100.0,
            "design_efficiency": 78.0,
            "fuel_cost_per_mmbtu": 6.0,
        },
    ]


# =============================================================================
# SAFETY CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def safety_interlocks() -> Dict:
    """Sample safety interlock configuration."""
    return {
        "STEAM_PRESSURE": {
            "trip_high": 150.0,
            "alarm_high": 140.0,
            "sil_level": 2,
        },
        "DRUM_LEVEL": {
            "trip_high": 8.0,
            "trip_low": -4.0,
            "alarm_high": 6.0,
            "alarm_low": -2.0,
            "sil_level": 3,
        },
        "FUEL_PRESSURE": {
            "trip_high": 25.0,
            "trip_low": 2.0,
            "sil_level": 2,
        },
    }


@pytest.fixture
def flame_scanner_config() -> Dict:
    """Flame scanner configuration."""
    return {
        "voting_logic": "2oo3",
        "scanners": [
            {"id": "UV-1", "type": "UV", "sightline": "main"},
            {"id": "UV-2", "type": "UV", "sightline": "main"},
            {"id": "UV-3", "type": "UV", "sightline": "pilot"},
        ],
        "min_signal_percent": 10.0,
        "unstable_threshold_percent": 20.0,
        "failure_time_s": 4.0,
    }


# =============================================================================
# O2 TRIM CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def o2_trim_config() -> Dict:
    """O2 trim controller configuration."""
    return {
        "o2_setpoint_curve": {
            0.25: 5.0,
            0.50: 3.5,
            0.75: 3.0,
            1.00: 2.5,
        },
        "pid_kp": 2.0,
        "pid_ki": 0.008,
        "pid_kd": 0.0,
        "output_limit_percent": 10.0,
        "co_limit_ppm": 400.0,
        "co_response_gain": 0.5,
    }


# =============================================================================
# SCADA CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def modbus_config() -> Dict:
    """Modbus TCP configuration."""
    return {
        "protocol": "modbus_tcp",
        "host": "192.168.1.100",
        "port": 502,
        "unit_id": 1,
        "timeout_ms": 5000,
        "retry_count": 3,
        "poll_interval_ms": 1000,
    }


@pytest.fixture
def opcua_config() -> Dict:
    """OPC-UA configuration."""
    return {
        "protocol": "opc_ua",
        "endpoint_url": "opc.tcp://192.168.1.100:4840",
        "security_policy": "Basic256Sha256",
        "security_mode": "SignAndEncrypt",
        "timeout_ms": 5000,
    }


# =============================================================================
# MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_scada_handler():
    """Mock SCADA protocol handler."""
    handler = MagicMock()
    handler.connect = AsyncMock(return_value=True)
    handler.disconnect = AsyncMock()
    handler.read_tags = AsyncMock(return_value={})
    handler.write_tag = AsyncMock(return_value=True)
    handler.is_connected = Mock(return_value=True)
    return handler


@pytest.fixture
def mock_event_handler():
    """Mock event handler."""
    handler = MagicMock()
    handler.handle = AsyncMock()
    handler.name = "mock_handler"
    return handler


@pytest.fixture
def mock_trip_callback():
    """Mock trip callback function."""
    return Mock()


@pytest.fixture
def mock_alarm_callback():
    """Mock alarm callback function."""
    return Mock()
