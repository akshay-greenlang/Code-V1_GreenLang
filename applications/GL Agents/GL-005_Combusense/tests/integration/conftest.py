# -*- coding: utf-8 -*-
"""
Integration test fixtures for GL-005 CombustionControlAgent.

Provides mock servers, test data, and integration test helpers.
"""

import pytest
import asyncio
import time
from typing import Generator, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from .mock_servers import MockServerManager, MockOPCUAServer, MockModbusServer, MockMQTTBroker, MockFlameScannerServer


# ============================================================================
# SAFETY LIMITS DATA CLASS
# ============================================================================

@dataclass
class SafetyLimits:
    """Safety limits for combustion control."""
    max_temperature_c: float = 1400.0
    min_temperature_c: float = 800.0
    max_pressure_mbar: float = 150.0
    min_pressure_mbar: float = 50.0
    max_fuel_flow_kg_hr: float = 1000.0
    min_fuel_flow_kg_hr: float = 50.0
    max_co_ppm: float = 100.0
    max_nox_ppm: float = 50.0


# ============================================================================
# CONTROL PARAMETERS DATA CLASS
# ============================================================================

@dataclass
class ControlParameters:
    """Control parameters for combustion optimization."""
    setpoint_temperature_c: float = 1200.0
    setpoint_o2_percent: float = 3.0
    fuel_air_ratio_target: float = 0.1
    pid_kp: float = 1.5
    pid_ki: float = 0.3
    pid_kd: float = 0.1


# ============================================================================
# BENCHMARK THRESHOLDS
# ============================================================================

@dataclass
class BenchmarkThresholds:
    """Performance benchmark thresholds."""
    cycle_execution_max_ms: float = 100.0
    control_loop_max_latency_ms: float = 50.0
    min_throughput_cps: float = 10.0  # cycles per second


# ============================================================================
# PERFORMANCE TIMER
# ============================================================================

class PerformanceTimer:
    """Timer for measuring performance."""

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000.0


# ============================================================================
# MOCK SERVER FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def mock_server_manager() -> Generator[MockServerManager, None, None]:
    """Provide mock server manager for integration tests."""
    manager = MockServerManager()
    await manager.start_all()
    yield manager
    await manager.stop_all()


@pytest.fixture
async def opcua_server(mock_server_manager) -> MockOPCUAServer:
    """Provide mock OPC UA server."""
    # Reset to default state
    mock_server_manager.opcua_server._init_default_nodes()
    if not mock_server_manager.opcua_server.running:
        await mock_server_manager.opcua_server.start()
    return mock_server_manager.opcua_server


@pytest.fixture
async def modbus_server(mock_server_manager) -> MockModbusServer:
    """Provide mock Modbus server."""
    # Reset to default state
    mock_server_manager.modbus_server._init_default_values()
    if not mock_server_manager.modbus_server.running:
        await mock_server_manager.modbus_server.start()
    return mock_server_manager.modbus_server


@pytest.fixture
async def mqtt_broker(mock_server_manager) -> MockMQTTBroker:
    """Provide mock MQTT broker."""
    # Clear topics
    mock_server_manager.mqtt_broker.topics = {}
    if not mock_server_manager.mqtt_broker.running:
        await mock_server_manager.mqtt_broker.start()
    return mock_server_manager.mqtt_broker


@pytest.fixture
async def flame_scanner(mock_server_manager) -> MockFlameScannerServer:
    """Provide mock flame scanner server."""
    # Reset to default state
    mock_server_manager.flame_scanner.restore_flame()
    if not mock_server_manager.flame_scanner.running:
        await mock_server_manager.flame_scanner.start()
    return mock_server_manager.flame_scanner


# ============================================================================
# SAFETY AND CONTROL FIXTURES
# ============================================================================

@pytest.fixture
def safety_limits() -> SafetyLimits:
    """Provide safety limits for tests."""
    return SafetyLimits()


@pytest.fixture
def control_parameters() -> Dict[str, float]:
    """Provide control parameters for tests."""
    return {
        'setpoint_temperature_c': 1200.0,
        'setpoint_o2_percent': 3.0,
        'fuel_air_ratio_target': 0.1,
        'pid_kp': 1.5,
        'pid_ki': 0.3,
        'pid_kd': 0.1
    }


@pytest.fixture
def benchmark_thresholds() -> Dict[str, float]:
    """Provide benchmark thresholds for performance tests."""
    return {
        'cycle_execution_max_ms': 100.0,
        'control_loop_max_latency_ms': 50.0,
        'min_throughput_cps': 10.0
    }


# ============================================================================
# PERFORMANCE FIXTURES
# ============================================================================

@pytest.fixture
def performance_timer():
    """Provide performance timer as context manager factory."""
    def create_timer():
        return PerformanceTimer()
    return create_timer


# ============================================================================
# MOCK DATA ENHANCEMENTS
# ============================================================================

# Add additional methods to MockMQTTBroker for testing
def _get_all_messages(self, topic: str):
    """Get all messages from a topic."""
    if topic in self.topics:
        return [msg['message'] for msg in self.topics[topic]]
    return []


def _clear_messages(self, topic: str):
    """Clear all messages from a topic."""
    if topic in self.topics:
        self.topics[topic] = []


def _get_last_update_time(self, topic: str):
    """Get timestamp of last message."""
    if topic in self.topics and len(self.topics[topic]) > 0:
        return self.topics[topic][-1]['timestamp'].timestamp() * 1000
    return None


# Patch MockMQTTBroker with additional methods
MockMQTTBroker.get_all_messages = _get_all_messages
MockMQTTBroker.clear_messages = _clear_messages
MockMQTTBroker.get_last_update_time = _get_last_update_time


# Add is_running property to servers
@property
def _opcua_is_running(self):
    return self.running

@property
def _modbus_is_running(self):
    return self.running

@property
def _mqtt_is_running(self):
    return self.running

@property
def _flame_is_running(self):
    return self.running


MockOPCUAServer.is_running = _opcua_is_running
MockModbusServer.is_running = _modbus_is_running
MockMQTTBroker.is_running = _mqtt_is_running
MockFlameScannerServer.is_running = _flame_is_running


# Add additional methods to MockModbusServer
async def _write_holding_register(self, address: int, value: int) -> bool:
    """Alias for write_register."""
    return await self.write_register(address, value)

MockModbusServer.write_holding_register = _write_holding_register


# Add additional methods to MockFlameScannerServer
def _simulate_communication_failure(self):
    """Simulate communication failure."""
    self.running = False

def _restore_communication(self):
    """Restore communication after failure."""
    self.running = True

def _set_signal_strength(self, strength: float):
    """Set signal strength (0.0 to 1.0)."""
    self._signal_strength = strength

def _set_primary_failed(self, failed: bool):
    """Set primary scanner failure state."""
    self._primary_failed = failed


async def _get_flame_status_enhanced(self) -> Dict[str, Any]:
    """Enhanced get_flame_status with additional fields."""
    if not self.running:
        raise ConnectionError("Server not running")

    await asyncio.sleep(0.01)

    return {
        'flame_detected': self.flame_detected,
        'intensity': self.flame_intensity,
        'intensity_percent': self.flame_intensity,
        'stability_index': self.flame_stability,
        'signal_strength': getattr(self, '_signal_strength', 1.0),
        'primary_failed': getattr(self, '_primary_failed', False),
        'backup_active': getattr(self, '_primary_failed', False),
        'timestamp': time.time()
    }


MockFlameScannerServer.simulate_communication_failure = _simulate_communication_failure
MockFlameScannerServer.restore_communication = _restore_communication
MockFlameScannerServer.set_signal_strength = _set_signal_strength
MockFlameScannerServer.set_primary_failed = _set_primary_failed
MockFlameScannerServer.get_flame_status = _get_flame_status_enhanced
