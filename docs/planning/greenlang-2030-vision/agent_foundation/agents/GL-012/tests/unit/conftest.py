# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL Test Suite - Shared Pytest Fixtures.

This module provides comprehensive test fixtures for the SteamQualityController
test suite, including sample steam conditions, pressure data, mock sensors,
desuperheater configurations, and test utilities.

Coverage Target: 95%+
Standards Compliance:
- IAPWS-IF97: Industrial Formulation for Water and Steam Properties
- ASME PTC 19.11: Steam and Water Sampling
- ASME PTC 6: Steam Turbines
- ISO 3046: Reciprocating internal combustion engines

Author: GL-TestEngineer
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from dataclasses import dataclass, field

import pytest

# Add parent directories to path for imports
TEST_DIR = Path(__file__).parent
AGENT_DIR = TEST_DIR.parent.parent
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(AGENT_DIR.parent.parent))


# =============================================================================
# ENUMS AND DATACLASSES FOR TESTING
# =============================================================================

class SteamState(Enum):
    """Steam thermodynamic state enumeration."""
    SUBCOOLED = "subcooled"
    SATURATED_LIQUID = "saturated_liquid"
    WET_STEAM = "wet_steam"
    SATURATED_VAPOR = "saturated_vapor"
    SUPERHEATED = "superheated"
    SUPERCRITICAL = "supercritical"


class RiskLevel(Enum):
    """Risk level for moisture/condensation."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValveCharacteristic(Enum):
    """Valve characteristic curve types."""
    LINEAR = "linear"
    EQUAL_PERCENTAGE = "equal_percentage"
    QUICK_OPENING = "quick_opening"


@dataclass
class SteamProperties:
    """Steam thermodynamic properties."""
    pressure_bar: float
    temperature_c: float
    dryness_fraction: float
    specific_enthalpy_kj_kg: float
    specific_entropy_kj_kg_k: float
    specific_volume_m3_kg: float
    state: SteamState
    quality_index: float = 0.0


@dataclass
class DesuperheaterConfig:
    """Desuperheater configuration parameters."""
    max_injection_rate_kg_s: float = 10.0
    min_injection_rate_kg_s: float = 0.1
    spray_water_temp_c: float = 105.0
    spray_water_pressure_bar: float = 20.0
    target_superheat_c: float = 15.0
    pid_kp: float = 2.0
    pid_ki: float = 0.5
    pid_kd: float = 0.1
    nozzle_count: int = 4
    nozzle_cv: float = 2.5


@dataclass
class PressureControlConfig:
    """Pressure control configuration parameters."""
    target_pressure_bar: float = 10.0
    pressure_tolerance_bar: float = 0.2
    valve_cv: float = 100.0
    valve_characteristic: ValveCharacteristic = ValveCharacteristic.EQUAL_PERCENTAGE
    pid_kp: float = 1.5
    pid_ki: float = 0.3
    pid_kd: float = 0.05
    max_valve_position: float = 100.0
    min_valve_position: float = 0.0
    valve_stroke_time_s: float = 30.0


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "determinism: Determinism tests")
    config.addinivalue_line("markers", "compliance: Regulatory compliance tests")
    config.addinivalue_line("markers", "golden: Golden test cases")
    config.addinivalue_line("markers", "concurrency: Thread safety tests")
    config.addinivalue_line("markers", "edge_case: Edge case tests")
    config.addinivalue_line("markers", "error_path: Error handling tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "asyncio: Async tests")
    config.addinivalue_line("markers", "iapws: IAPWS-IF97 standard tests")
    config.addinivalue_line("markers", "asme: ASME standard tests")


# =============================================================================
# EVENT LOOP FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# STEAM PROPERTY FIXTURES - IAPWS-IF97 Reference Data
# =============================================================================

@pytest.fixture
def saturated_steam_100c() -> SteamProperties:
    """Saturated steam at 100C (1.01325 bar) - IAPWS-IF97 reference."""
    return SteamProperties(
        pressure_bar=1.01325,
        temperature_c=100.0,
        dryness_fraction=1.0,
        specific_enthalpy_kj_kg=2675.5,  # hg at 100C
        specific_entropy_kj_kg_k=7.3549,  # sg at 100C
        specific_volume_m3_kg=1.6729,  # vg at 100C
        state=SteamState.SATURATED_VAPOR,
        quality_index=100.0
    )


@pytest.fixture
def saturated_steam_180c() -> SteamProperties:
    """Saturated steam at 180C (10.027 bar) - IAPWS-IF97 reference."""
    return SteamProperties(
        pressure_bar=10.027,
        temperature_c=180.0,
        dryness_fraction=1.0,
        specific_enthalpy_kj_kg=2777.1,  # hg at 180C
        specific_entropy_kj_kg_k=6.5857,  # sg at 180C
        specific_volume_m3_kg=0.1943,  # vg at 180C
        state=SteamState.SATURATED_VAPOR,
        quality_index=100.0
    )


@pytest.fixture
def wet_steam_10bar_90pct() -> SteamProperties:
    """Wet steam at 10 bar with 90% dryness fraction."""
    # Saturation temperature at 10 bar = 179.88C
    # hf = 762.6 kJ/kg, hfg = 2013.6 kJ/kg
    # h = hf + x*hfg = 762.6 + 0.9*2013.6 = 2574.84 kJ/kg
    return SteamProperties(
        pressure_bar=10.0,
        temperature_c=179.88,
        dryness_fraction=0.90,
        specific_enthalpy_kj_kg=2574.84,
        specific_entropy_kj_kg_k=6.273,  # Interpolated
        specific_volume_m3_kg=0.1753,  # vf + x*vfg
        state=SteamState.WET_STEAM,
        quality_index=90.0
    )


@pytest.fixture
def superheated_steam_10bar_250c() -> SteamProperties:
    """Superheated steam at 10 bar, 250C - IAPWS-IF97 reference."""
    return SteamProperties(
        pressure_bar=10.0,
        temperature_c=250.0,
        dryness_fraction=1.0,  # Always 1.0 for superheated
        specific_enthalpy_kj_kg=2942.6,
        specific_entropy_kj_kg_k=6.9247,
        specific_volume_m3_kg=0.2328,
        state=SteamState.SUPERHEATED,
        quality_index=100.0
    )


@pytest.fixture
def subcooled_water_10bar() -> SteamProperties:
    """Subcooled water at 10 bar, 150C."""
    return SteamProperties(
        pressure_bar=10.0,
        temperature_c=150.0,
        dryness_fraction=0.0,
        specific_enthalpy_kj_kg=632.2,
        specific_entropy_kj_kg_k=1.8418,
        specific_volume_m3_kg=0.001091,
        state=SteamState.SUBCOOLED,
        quality_index=0.0
    )


@pytest.fixture
def supercritical_steam() -> SteamProperties:
    """Supercritical steam at 250 bar, 400C."""
    return SteamProperties(
        pressure_bar=250.0,
        temperature_c=400.0,
        dryness_fraction=1.0,
        specific_enthalpy_kj_kg=2952.0,
        specific_entropy_kj_kg_k=5.946,
        specific_volume_m3_kg=0.00892,
        state=SteamState.SUPERCRITICAL,
        quality_index=100.0
    )


# =============================================================================
# DRYNESS FRACTION TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def dryness_fraction_test_cases() -> List[Dict[str, Any]]:
    """
    Test cases for dryness fraction calculations.

    Values based on IAPWS-IF97 steam tables.
    """
    return [
        {
            'name': 'saturated_liquid_x0',
            'pressure_bar': 10.0,
            'enthalpy_kj_kg': 762.6,  # hf at 10 bar
            'expected_dryness': 0.0,
            'expected_state': SteamState.SATURATED_LIQUID,
            'tolerance': 0.001
        },
        {
            'name': 'wet_steam_x0.5',
            'pressure_bar': 10.0,
            'enthalpy_kj_kg': 1769.4,  # hf + 0.5*hfg
            'expected_dryness': 0.5,
            'expected_state': SteamState.WET_STEAM,
            'tolerance': 0.01
        },
        {
            'name': 'wet_steam_x0.9',
            'pressure_bar': 10.0,
            'enthalpy_kj_kg': 2574.84,  # hf + 0.9*hfg
            'expected_dryness': 0.9,
            'expected_state': SteamState.WET_STEAM,
            'tolerance': 0.01
        },
        {
            'name': 'wet_steam_x0.95',
            'pressure_bar': 10.0,
            'enthalpy_kj_kg': 2675.52,  # hf + 0.95*hfg
            'expected_dryness': 0.95,
            'expected_state': SteamState.WET_STEAM,
            'tolerance': 0.01
        },
        {
            'name': 'saturated_vapor_x1.0',
            'pressure_bar': 10.0,
            'enthalpy_kj_kg': 2776.2,  # hg at 10 bar
            'expected_dryness': 1.0,
            'expected_state': SteamState.SATURATED_VAPOR,
            'tolerance': 0.001
        },
    ]


@pytest.fixture
def superheat_test_cases() -> List[Dict[str, Any]]:
    """Test cases for superheat degree calculations."""
    return [
        {
            'name': 'subcooled_negative_superheat',
            'pressure_bar': 10.0,
            'temperature_c': 150.0,  # Tsat = 179.88C
            'expected_superheat_c': -29.88,  # Subcooling
            'expected_state': SteamState.SUBCOOLED,
            'tolerance': 0.1
        },
        {
            'name': 'saturated_zero_superheat',
            'pressure_bar': 10.0,
            'temperature_c': 179.88,  # Tsat at 10 bar
            'expected_superheat_c': 0.0,
            'expected_state': SteamState.SATURATED_VAPOR,
            'tolerance': 0.1
        },
        {
            'name': 'superheated_50K',
            'pressure_bar': 10.0,
            'temperature_c': 229.88,  # Tsat + 50
            'expected_superheat_c': 50.0,
            'expected_state': SteamState.SUPERHEATED,
            'tolerance': 0.1
        },
        {
            'name': 'superheated_100K',
            'pressure_bar': 10.0,
            'temperature_c': 279.88,  # Tsat + 100
            'expected_superheat_c': 100.0,
            'expected_state': SteamState.SUPERHEATED,
            'tolerance': 0.1
        },
    ]


# =============================================================================
# PRESSURE TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def pressure_test_conditions() -> List[Dict[str, Any]]:
    """Test conditions for various pressure levels."""
    return [
        {
            'name': 'atmospheric',
            'pressure_bar': 1.01325,
            'saturation_temp_c': 100.0,
            'hf_kj_kg': 419.1,
            'hg_kj_kg': 2675.5,
            'hfg_kj_kg': 2256.4,
        },
        {
            'name': 'low_pressure',
            'pressure_bar': 5.0,
            'saturation_temp_c': 151.83,
            'hf_kj_kg': 640.1,
            'hg_kj_kg': 2747.5,
            'hfg_kj_kg': 2107.4,
        },
        {
            'name': 'medium_pressure',
            'pressure_bar': 10.0,
            'saturation_temp_c': 179.88,
            'hf_kj_kg': 762.6,
            'hg_kj_kg': 2776.2,
            'hfg_kj_kg': 2013.6,
        },
        {
            'name': 'high_pressure',
            'pressure_bar': 40.0,
            'saturation_temp_c': 250.33,
            'hf_kj_kg': 1087.4,
            'hg_kj_kg': 2799.4,
            'hfg_kj_kg': 1712.0,
        },
        {
            'name': 'very_high_pressure',
            'pressure_bar': 100.0,
            'saturation_temp_c': 311.0,
            'hf_kj_kg': 1408.0,
            'hg_kj_kg': 2725.5,
            'hfg_kj_kg': 1317.5,
        },
        {
            'name': 'near_critical',
            'pressure_bar': 200.0,
            'saturation_temp_c': 365.75,
            'hf_kj_kg': 1826.5,
            'hg_kj_kg': 2410.4,
            'hfg_kj_kg': 583.9,
        },
    ]


@pytest.fixture
def extreme_pressure_cases() -> List[Dict[str, Any]]:
    """Edge cases for extreme pressure conditions."""
    return [
        {
            'name': 'very_low_pressure',
            'pressure_bar': 0.01,  # 10 mbar vacuum
            'expected_saturation_temp_c': 7.0,  # Approximate
            'is_valid': True,
        },
        {
            'name': 'zero_pressure',
            'pressure_bar': 0.0,
            'expected_error': 'ValueError',
            'is_valid': False,
        },
        {
            'name': 'negative_pressure',
            'pressure_bar': -1.0,
            'expected_error': 'ValueError',
            'is_valid': False,
        },
        {
            'name': 'above_critical',
            'pressure_bar': 250.0,  # Above critical point (220.64 bar)
            'expected_state': SteamState.SUPERCRITICAL,
            'is_valid': True,
        },
        {
            'name': 'extremely_high',
            'pressure_bar': 1000.0,  # Very high pressure
            'is_valid': True,  # Should handle gracefully
        },
    ]


# =============================================================================
# DESUPERHEATER CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def desuperheater_config() -> DesuperheaterConfig:
    """Standard desuperheater configuration."""
    return DesuperheaterConfig(
        max_injection_rate_kg_s=10.0,
        min_injection_rate_kg_s=0.1,
        spray_water_temp_c=105.0,
        spray_water_pressure_bar=20.0,
        target_superheat_c=15.0,
        pid_kp=2.0,
        pid_ki=0.5,
        pid_kd=0.1,
        nozzle_count=4,
        nozzle_cv=2.5
    )


@pytest.fixture
def desuperheater_test_cases() -> List[Dict[str, Any]]:
    """Test cases for desuperheater calculations."""
    return [
        {
            'name': 'normal_operation',
            'inlet_temp_c': 350.0,
            'inlet_pressure_bar': 40.0,
            'inlet_flow_kg_s': 50.0,
            'target_temp_c': 300.0,
            'spray_water_temp_c': 105.0,
            'expected_injection_rate_kg_s': 3.5,  # Approximate
            'tolerance': 0.5,
        },
        {
            'name': 'high_superheat_reduction',
            'inlet_temp_c': 450.0,
            'inlet_pressure_bar': 40.0,
            'inlet_flow_kg_s': 50.0,
            'target_temp_c': 280.0,
            'spray_water_temp_c': 105.0,
            'expected_injection_rate_kg_s': 8.0,  # Higher injection
            'tolerance': 1.0,
        },
        {
            'name': 'minimal_superheat_reduction',
            'inlet_temp_c': 290.0,
            'inlet_pressure_bar': 40.0,
            'inlet_flow_kg_s': 50.0,
            'target_temp_c': 280.0,
            'spray_water_temp_c': 105.0,
            'expected_injection_rate_kg_s': 0.5,  # Minimal injection
            'tolerance': 0.2,
        },
    ]


# =============================================================================
# PRESSURE CONTROL FIXTURES
# =============================================================================

@pytest.fixture
def pressure_control_config() -> PressureControlConfig:
    """Standard pressure control configuration."""
    return PressureControlConfig(
        target_pressure_bar=10.0,
        pressure_tolerance_bar=0.2,
        valve_cv=100.0,
        valve_characteristic=ValveCharacteristic.EQUAL_PERCENTAGE,
        pid_kp=1.5,
        pid_ki=0.3,
        pid_kd=0.05,
        max_valve_position=100.0,
        min_valve_position=0.0,
        valve_stroke_time_s=30.0
    )


@pytest.fixture
def valve_test_cases() -> List[Dict[str, Any]]:
    """Test cases for valve position calculations."""
    return [
        {
            'name': 'linear_50pct',
            'characteristic': ValveCharacteristic.LINEAR,
            'position_percent': 50.0,
            'expected_cv_ratio': 0.50,
            'tolerance': 0.01,
        },
        {
            'name': 'linear_25pct',
            'characteristic': ValveCharacteristic.LINEAR,
            'position_percent': 25.0,
            'expected_cv_ratio': 0.25,
            'tolerance': 0.01,
        },
        {
            'name': 'equal_percentage_50pct',
            'characteristic': ValveCharacteristic.EQUAL_PERCENTAGE,
            'position_percent': 50.0,
            'expected_cv_ratio': 0.316,  # R^(x-1) where R=50, x=0.5
            'tolerance': 0.02,
        },
        {
            'name': 'equal_percentage_100pct',
            'characteristic': ValveCharacteristic.EQUAL_PERCENTAGE,
            'position_percent': 100.0,
            'expected_cv_ratio': 1.0,
            'tolerance': 0.01,
        },
        {
            'name': 'quick_opening_50pct',
            'characteristic': ValveCharacteristic.QUICK_OPENING,
            'position_percent': 50.0,
            'expected_cv_ratio': 0.707,  # sqrt(x)
            'tolerance': 0.02,
        },
    ]


@pytest.fixture
def pid_test_cases() -> List[Dict[str, Any]]:
    """Test cases for PID controller response."""
    return [
        {
            'name': 'proportional_only',
            'setpoint': 10.0,
            'process_value': 9.0,
            'kp': 2.0, 'ki': 0.0, 'kd': 0.0,
            'expected_output': 2.0,  # Kp * error
            'tolerance': 0.01,
        },
        {
            'name': 'steady_state_no_error',
            'setpoint': 10.0,
            'process_value': 10.0,
            'kp': 2.0, 'ki': 0.5, 'kd': 0.1,
            'expected_output': 0.0,  # No error, no output change
            'tolerance': 0.01,
        },
        {
            'name': 'negative_error',
            'setpoint': 10.0,
            'process_value': 12.0,
            'kp': 2.0, 'ki': 0.0, 'kd': 0.0,
            'expected_output': -4.0,  # Kp * (-2)
            'tolerance': 0.01,
        },
    ]


# =============================================================================
# MOISTURE ANALYZER FIXTURES
# =============================================================================

@pytest.fixture
def moisture_analysis_test_cases() -> List[Dict[str, Any]]:
    """Test cases for moisture content analysis."""
    return [
        {
            'name': 'dry_steam',
            'dryness_fraction': 1.0,
            'wetness_percent': 0.0,
            'expected_risk': RiskLevel.NONE,
            'condensation_possible': False,
        },
        {
            'name': 'slight_moisture',
            'dryness_fraction': 0.98,
            'wetness_percent': 2.0,
            'expected_risk': RiskLevel.LOW,
            'condensation_possible': True,
        },
        {
            'name': 'moderate_moisture',
            'dryness_fraction': 0.95,
            'wetness_percent': 5.0,
            'expected_risk': RiskLevel.MEDIUM,
            'condensation_possible': True,
        },
        {
            'name': 'high_moisture',
            'dryness_fraction': 0.88,
            'wetness_percent': 12.0,
            'expected_risk': RiskLevel.HIGH,
            'condensation_possible': True,
        },
        {
            'name': 'critical_moisture',
            'dryness_fraction': 0.80,
            'wetness_percent': 20.0,
            'expected_risk': RiskLevel.CRITICAL,
            'condensation_possible': True,
        },
    ]


@pytest.fixture
def condensation_risk_factors() -> Dict[str, float]:
    """Risk factor weights for condensation analysis."""
    return {
        'wetness_weight': 0.35,
        'temperature_weight': 0.25,
        'pressure_drop_weight': 0.20,
        'insulation_weight': 0.10,
        'ambient_temp_weight': 0.10,
    }


# =============================================================================
# GOLDEN TEST CASES - KNOWN GOOD VALUES
# =============================================================================

@pytest.fixture
def golden_test_cases() -> List[Dict[str, Any]]:
    """
    Golden test cases with known input-output pairs.

    These values must remain constant across all test runs.
    Based on IAPWS-IF97 standard steam tables.
    """
    return [
        {
            'name': 'steam_quality_10bar_saturated',
            'input': {
                'pressure_bar': 10.0,
                'temperature_c': 179.88,
                'enthalpy_kj_kg': 2776.2,
            },
            'expected': {
                'dryness_fraction': 1.0,
                'state': SteamState.SATURATED_VAPOR,
                'quality_index': 100.0,
            },
            'tolerance': {
                'dryness_fraction': 0.001,
                'quality_index': 0.1,
            },
        },
        {
            'name': 'steam_quality_10bar_wet_90pct',
            'input': {
                'pressure_bar': 10.0,
                'temperature_c': 179.88,
                'enthalpy_kj_kg': 2574.84,
            },
            'expected': {
                'dryness_fraction': 0.9,
                'state': SteamState.WET_STEAM,
                'quality_index': 90.0,
            },
            'tolerance': {
                'dryness_fraction': 0.01,
                'quality_index': 1.0,
            },
        },
        {
            'name': 'desuperheater_normal',
            'input': {
                'inlet_temp_c': 350.0,
                'inlet_pressure_bar': 40.0,
                'inlet_flow_kg_s': 50.0,
                'target_temp_c': 300.0,
                'spray_temp_c': 105.0,
            },
            'expected': {
                'injection_rate_range': (2.5, 4.5),  # kg/s range
                'outlet_temp_c': 300.0,
            },
            'tolerance': {
                'outlet_temp_c': 2.0,
            },
        },
    ]


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_steam_sensor():
    """Create mock steam quality sensor."""
    mock = MagicMock()
    mock.connect.return_value = True
    mock.disconnect.return_value = True
    mock.is_connected.return_value = True
    mock.read_pressure.return_value = 10.0  # bar
    mock.read_temperature.return_value = 250.0  # C
    mock.read_flow_rate.return_value = 50.0  # kg/s
    mock.read_dryness.return_value = 0.98
    mock.get_sensor_status.return_value = {
        'status': 'healthy',
        'last_calibration': datetime.now(timezone.utc).isoformat(),
        'accuracy_percent': 0.5,
    }
    return mock


@pytest.fixture
def mock_desuperheater_controller():
    """Create mock desuperheater controller."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.disconnect = AsyncMock(return_value=True)
    mock.get_injection_rate = AsyncMock(return_value=3.5)
    mock.set_injection_rate = AsyncMock(return_value=True)
    mock.get_spray_water_pressure = AsyncMock(return_value=20.0)
    mock.get_inlet_temperature = AsyncMock(return_value=350.0)
    mock.get_outlet_temperature = AsyncMock(return_value=300.0)
    mock.is_operating = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_pressure_controller():
    """Create mock pressure control valve."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.disconnect = AsyncMock(return_value=True)
    mock.get_valve_position = AsyncMock(return_value=50.0)
    mock.set_valve_position = AsyncMock(return_value=True)
    mock.get_upstream_pressure = AsyncMock(return_value=15.0)
    mock.get_downstream_pressure = AsyncMock(return_value=10.0)
    mock.get_flow_rate = AsyncMock(return_value=50.0)
    mock.is_actuator_healthy = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_scada_connector():
    """Create mock SCADA connector."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.disconnect = AsyncMock(return_value=True)
    mock.read_tags = AsyncMock(return_value={
        'steam_pressure': 10.0,
        'steam_temperature': 250.0,
        'steam_flow': 50.0,
        'dryness_fraction': 0.98,
    })
    mock.write_setpoint = AsyncMock(return_value=True)
    mock.get_alarms = AsyncMock(return_value=[])
    return mock


# =============================================================================
# CONCURRENCY FIXTURES
# =============================================================================

@pytest.fixture
def thread_pool():
    """Create thread pool for concurrency tests."""
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=50)
    yield executor
    executor.shutdown(wait=True)


@pytest.fixture
def concurrency_barrier():
    """Create barrier for synchronized concurrent access."""
    return threading.Barrier(10)


# =============================================================================
# PERFORMANCE FIXTURES
# =============================================================================

@pytest.fixture
def performance_timer():
    """Context manager for timing test execution."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()

        @property
        def elapsed_ms(self) -> float:
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time) * 1000
            return 0.0

    return Timer


@pytest.fixture
def benchmark_targets() -> Dict[str, float]:
    """Performance benchmark targets in milliseconds."""
    return {
        'steam_quality_calculation_ms': 5.0,
        'desuperheater_calculation_ms': 10.0,
        'pressure_control_calculation_ms': 5.0,
        'moisture_analysis_ms': 3.0,
        'pid_update_ms': 1.0,
        'provenance_hash_ms': 2.0,
        'batch_100_calculations_ms': 100.0,
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_provenance_hash(data: Dict[str, Any]) -> str:
    """Generate SHA-256 hash for provenance verification."""
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()


def assert_within_tolerance(actual: float, expected: float, tolerance: float, message: str = ""):
    """Assert value is within tolerance of expected."""
    assert abs(actual - expected) <= tolerance, (
        f"{message} Expected {expected} +/- {tolerance}, got {actual}"
    )


def assert_deterministic(results: List[Any], message: str = ""):
    """Assert all results are identical (deterministic)."""
    if len(results) < 2:
        return
    first = results[0]
    for i, result in enumerate(results[1:], 2):
        assert result == first, f"{message} Result {i} differs from result 1"


def calculate_saturation_temperature(pressure_bar: float) -> float:
    """
    Approximate saturation temperature from pressure.

    Uses simplified correlation for testing purposes.
    For production, use full IAPWS-IF97 implementation.
    """
    if pressure_bar <= 0:
        raise ValueError("Pressure must be positive")

    # Simplified correlation (not exact, for testing)
    # Tsat = 100 * (pressure_bar)^0.25 for approximate values
    import math
    return 100.0 + 58.0 * math.log10(pressure_bar)


# =============================================================================
# TEMPORARY DIRECTORY FIXTURES
# =============================================================================

@pytest.fixture
def temp_directory():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_file(temp_directory) -> Path:
    """Create temporary configuration file."""
    config = {
        'agent_id': 'GL-012',
        'agent_name': 'STEAMQUAL',
        'version': '1.0.0',
        'steam_quality': {
            'min_dryness_fraction': 0.85,
            'target_dryness_fraction': 0.98,
            'max_moisture_percent': 15.0,
        },
        'desuperheater': {
            'target_superheat_c': 15.0,
            'max_injection_rate_kg_s': 10.0,
        },
        'pressure_control': {
            'target_pressure_bar': 10.0,
            'tolerance_bar': 0.2,
        },
    }
    config_file = temp_directory / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    return config_file


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Cleanup logic if needed


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_session():
    """Clean up after test session."""
    yield
    # Session cleanup logic if needed
