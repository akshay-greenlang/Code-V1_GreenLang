"""
Pytest configuration and shared fixtures for GL-005 CombustionControlAgent tests.

This module provides:
- Configuration for pytest
- Shared fixtures for all test modules
- Test data generators
- Mock implementations for hardware integrations
- Async test support
- Performance measurement utilities
"""

import pytest
import asyncio
import logging
import sys
import os
import time
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Any, List, Generator, Tuple
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass, field

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tests/logs/gl005_tests.log')
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "e2e: mark test as an end-to-end test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "determinism: mark test as a determinism test")
    config.addinivalue_line("markers", "compliance: mark test as a compliance test")
    config.addinivalue_line("markers", "security: mark test as a security test")
    config.addinivalue_line("markers", "boundary: mark test as a boundary test")
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "realtime: mark test as real-time control test")
    config.addinivalue_line("markers", "safety: mark test as safety interlock test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "stress: mark test as stress test")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# COMBUSTION STATE DATACLASSES
# ============================================================================

@dataclass
class CombustionState:
    """Represents current combustion state."""
    timestamp: datetime
    fuel_flow_rate_kg_hr: float
    air_flow_rate_kg_hr: float
    combustion_temperature_c: float
    furnace_pressure_mbar: float
    o2_percent: float
    co_ppm: float
    nox_ppm: float
    flame_intensity_percent: float
    flame_stability_index: float
    fuel_air_ratio: float
    excess_air_percent: float
    heat_output_mw: float


@dataclass
class SafetyLimits:
    """Safety limits for combustion control."""
    max_temperature_c: float = 1400.0
    min_temperature_c: float = 800.0
    max_pressure_mbar: float = 150.0
    min_pressure_mbar: float = 50.0
    max_fuel_flow_kg_hr: float = 1000.0
    min_fuel_flow_kg_hr: float = 50.0
    max_air_flow_kg_hr: float = 10000.0
    min_air_flow_kg_hr: float = 500.0
    max_o2_percent: float = 10.0
    min_o2_percent: float = 2.0
    max_co_ppm: float = 100.0
    max_nox_ppm: float = 50.0
    min_flame_intensity: float = 30.0


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def combustion_config() -> Dict[str, Any]:
    """Create combustion control configuration."""
    return {
        'controller_id': 'CC-001',
        'control_loop_interval_ms': 100,
        'safety_check_interval_ms': 50,
        'optimization_enabled': True,
        'deterministic_mode': True,
        'enable_logging': True,
        'enable_monitoring': True
    }


@pytest.fixture
def safety_limits() -> SafetyLimits:
    """Create safety limits fixture."""
    return SafetyLimits()


@pytest.fixture
def control_parameters() -> Dict[str, Any]:
    """Create control parameters."""
    return {
        'pid_kp': 1.5,
        'pid_ki': 0.3,
        'pid_kd': 0.1,
        'setpoint_temperature_c': 1200.0,
        'setpoint_o2_percent': 3.5,
        'fuel_air_ratio_target': 0.95,
        'control_deadband': 2.0,
        'ramp_rate_limit_c_per_min': 50.0
    }


@pytest.fixture
def optimization_config() -> Dict[str, Any]:
    """Create optimization configuration."""
    return {
        'objective': 'fuel_efficiency',
        'constraints': ['emissions', 'temperature', 'stability'],
        'convergence_tolerance': 0.01,
        'max_iterations': 100,
        'optimization_interval_sec': 300
    }


# ============================================================================
# COMBUSTION STATE FIXTURES
# ============================================================================

@pytest.fixture
def normal_combustion_state() -> CombustionState:
    """Create normal operating combustion state."""
    return CombustionState(
        timestamp=datetime.now(timezone.utc),
        fuel_flow_rate_kg_hr=500.0,
        air_flow_rate_kg_hr=5000.0,
        combustion_temperature_c=1200.0,
        furnace_pressure_mbar=100.0,
        o2_percent=3.5,
        co_ppm=25.0,
        nox_ppm=30.0,
        flame_intensity_percent=85.0,
        flame_stability_index=0.95,
        fuel_air_ratio=0.1,
        excess_air_percent=15.0,
        heat_output_mw=12.5
    )


@pytest.fixture
def high_temp_combustion_state() -> CombustionState:
    """Create high temperature combustion state."""
    return CombustionState(
        timestamp=datetime.now(timezone.utc),
        fuel_flow_rate_kg_hr=800.0,
        air_flow_rate_kg_hr=7500.0,
        combustion_temperature_c=1380.0,
        furnace_pressure_mbar=135.0,
        o2_percent=2.8,
        co_ppm=45.0,
        nox_ppm=48.0,
        flame_intensity_percent=95.0,
        flame_stability_index=0.88,
        fuel_air_ratio=0.107,
        excess_air_percent=8.0,
        heat_output_mw=18.5
    )


@pytest.fixture
def low_load_combustion_state() -> CombustionState:
    """Create low load combustion state."""
    return CombustionState(
        timestamp=datetime.now(timezone.utc),
        fuel_flow_rate_kg_hr=150.0,
        air_flow_rate_kg_hr=2000.0,
        combustion_temperature_c=950.0,
        furnace_pressure_mbar=65.0,
        o2_percent=6.5,
        co_ppm=15.0,
        nox_ppm=18.0,
        flame_intensity_percent=45.0,
        flame_stability_index=0.75,
        fuel_air_ratio=0.075,
        excess_air_percent=35.0,
        heat_output_mw=3.8
    )


@pytest.fixture
def unstable_combustion_state() -> CombustionState:
    """Create unstable combustion state for testing."""
    return CombustionState(
        timestamp=datetime.now(timezone.utc),
        fuel_flow_rate_kg_hr=450.0,
        air_flow_rate_kg_hr=4800.0,
        combustion_temperature_c=1100.0,
        furnace_pressure_mbar=90.0,
        o2_percent=5.2,
        co_ppm=85.0,
        nox_ppm=25.0,
        flame_intensity_percent=55.0,
        flame_stability_index=0.45,
        fuel_air_ratio=0.094,
        excess_air_percent=25.0,
        heat_output_mw=9.5
    )


# ============================================================================
# MOCK DCS/PLC FIXTURES
# ============================================================================

@pytest.fixture
def mock_dcs_connector() -> AsyncMock:
    """Create mock DCS (Distributed Control System) connector."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.disconnect = AsyncMock(return_value=True)
    mock.read_process_variables = AsyncMock(return_value={
        'fuel_flow': 500.0,
        'air_flow': 5000.0,
        'temperature': 1200.0,
        'pressure': 100.0,
        'o2': 3.5
    })
    mock.write_setpoint = AsyncMock(return_value=True)
    mock.subscribe_alarms = AsyncMock(return_value=True)
    mock.get_alarms = AsyncMock(return_value=[])
    mock.is_connected = True
    return mock


@pytest.fixture
def mock_plc_connector() -> AsyncMock:
    """Create mock PLC (Programmable Logic Controller) connector."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.disconnect = AsyncMock(return_value=True)
    mock.read_coils = AsyncMock(return_value=[True, False, True])
    mock.read_registers = AsyncMock(return_value=[1200, 500, 5000])
    mock.write_coil = AsyncMock(return_value=True)
    mock.write_register = AsyncMock(return_value=True)
    mock.is_connected = True
    return mock


@pytest.fixture
def mock_combustion_analyzer() -> AsyncMock:
    """Create mock combustion analyzer."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.disconnect = AsyncMock(return_value=True)
    mock.read_o2 = AsyncMock(return_value=3.5)
    mock.read_co = AsyncMock(return_value=25.0)
    mock.read_co2 = AsyncMock(return_value=12.0)
    mock.read_nox = AsyncMock(return_value=30.0)
    mock.read_all = AsyncMock(return_value={
        'o2_percent': 3.5,
        'co_ppm': 25.0,
        'co2_percent': 12.0,
        'nox_ppm': 30.0,
        'timestamp': datetime.now(timezone.utc)
    })
    mock.is_connected = True
    return mock


@pytest.fixture
def mock_flame_scanner() -> AsyncMock:
    """Create mock flame scanner."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.disconnect = AsyncMock(return_value=True)
    mock.detect_flame = AsyncMock(return_value=True)
    mock.get_intensity = AsyncMock(return_value=85.0)
    mock.get_stability_index = AsyncMock(return_value=0.95)
    mock.get_flame_data = AsyncMock(return_value={
        'flame_detected': True,
        'intensity_percent': 85.0,
        'stability_index': 0.95,
        'timestamp': datetime.now(timezone.utc)
    })
    mock.is_connected = True
    return mock


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

class CombustionTestDataGenerator:
    """Generate test data for combustion control scenarios."""

    @staticmethod
    def generate_control_cycle_data(num_cycles: int = 10) -> List[Dict[str, Any]]:
        """Generate test data for multiple control cycles."""
        cycles = []
        base_temp = 1200.0
        base_fuel = 500.0

        for i in range(num_cycles):
            cycle = {
                'cycle_number': i,
                'timestamp': datetime.now(timezone.utc) + timedelta(seconds=i),
                'fuel_flow': base_fuel + (i * 10),
                'air_flow': (base_fuel + (i * 10)) * 10,
                'temperature': base_temp + (i * 5),
                'pressure': 100.0 + (i * 2),
                'o2': 3.5 - (i * 0.1)
            }
            cycles.append(cycle)

        return cycles

    @staticmethod
    def generate_stability_test_cases() -> List[Dict[str, Any]]:
        """Generate stability test cases."""
        return [
            {
                'name': 'high_stability',
                'flame_intensity': 85.0,
                'intensity_variance': 2.0,
                'expected_index': 0.95
            },
            {
                'name': 'medium_stability',
                'flame_intensity': 70.0,
                'intensity_variance': 8.0,
                'expected_index': 0.75
            },
            {
                'name': 'low_stability',
                'flame_intensity': 50.0,
                'intensity_variance': 15.0,
                'expected_index': 0.45
            }
        ]

    @staticmethod
    def generate_fuel_air_ratio_test_cases() -> List[Dict[str, Any]]:
        """Generate fuel-air ratio test cases."""
        return [
            {
                'fuel_flow': 500.0,
                'air_flow': 5000.0,
                'expected_ratio': 0.1,
                'expected_excess_air': 15.0
            },
            {
                'fuel_flow': 600.0,
                'air_flow': 5500.0,
                'expected_ratio': 0.109,
                'expected_excess_air': 10.0
            },
            {
                'fuel_flow': 400.0,
                'air_flow': 5200.0,
                'expected_ratio': 0.077,
                'expected_excess_air': 25.0
            }
        ]

    @staticmethod
    def generate_safety_violation_scenarios() -> List[Dict[str, Any]]:
        """Generate safety violation scenarios."""
        return [
            {
                'name': 'high_temperature',
                'temperature': 1450.0,
                'violation': 'MAX_TEMPERATURE',
                'action': 'emergency_shutdown'
            },
            {
                'name': 'low_temperature',
                'temperature': 750.0,
                'violation': 'MIN_TEMPERATURE',
                'action': 'increase_fuel_flow'
            },
            {
                'name': 'high_pressure',
                'pressure': 160.0,
                'violation': 'MAX_PRESSURE',
                'action': 'emergency_shutdown'
            },
            {
                'name': 'high_co',
                'co_ppm': 120.0,
                'violation': 'MAX_CO',
                'action': 'increase_air_flow'
            },
            {
                'name': 'flame_loss',
                'flame_detected': False,
                'violation': 'FLAME_LOSS',
                'action': 'emergency_shutdown'
            }
        ]


@pytest.fixture
def test_data_generator() -> CombustionTestDataGenerator:
    """Provide test data generator."""
    return CombustionTestDataGenerator()


# ============================================================================
# PERFORMANCE MEASUREMENT FIXTURES
# ============================================================================

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
def benchmark_thresholds() -> Dict[str, float]:
    """Provide performance benchmark thresholds."""
    return {
        'control_loop_max_latency_ms': 100.0,
        'cycle_execution_max_ms': 500.0,
        'stability_calculation_max_ms': 50.0,
        'pid_calculation_max_ms': 10.0,
        'safety_check_max_ms': 20.0,
        'dcs_read_max_ms': 100.0,
        'plc_write_max_ms': 50.0,
        'min_throughput_cps': 10.0
    }


# ============================================================================
# DETERMINISM VALIDATION FIXTURES
# ============================================================================

@pytest.fixture
def determinism_validator():
    """Validator for deterministic calculations."""
    class DeterminismValidator:
        @staticmethod
        def calculate_hash(data: Dict[str, Any]) -> str:
            """Calculate deterministic hash of data."""
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()

        @staticmethod
        def validate_reproducibility(func, input_data: Dict, num_runs: int = 10) -> bool:
            """Validate function produces identical results."""
            hashes = set()
            for _ in range(num_runs):
                result = func(input_data)
                result_hash = hashlib.sha256(
                    json.dumps(result, sort_keys=True, default=str).encode()
                ).hexdigest()
                hashes.add(result_hash)

            return len(hashes) == 1

    return DeterminismValidator()


# ============================================================================
# CLEANUP FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up resources after each test."""
    yield
    logger.debug("Test cleanup completed")


# ============================================================================
# ASYNC TEST HELPERS
# ============================================================================

@pytest.fixture
async def async_test_helper():
    """Helper for async test operations."""
    class AsyncTestHelper:
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

    return AsyncTestHelper()


logger.info("GL-005 conftest.py loaded with comprehensive fixtures")
