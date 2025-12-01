# -*- coding: utf-8 -*-
"""
Integration Test Fixtures and Configuration for GL-012 SteamQualityController

Comprehensive fixtures for all integration tests including:
- Mock servers (SCADA, Modbus, OPC-UA)
- Mock steam quality equipment (meters, valves, desuperheaters)
- Mock agent communication infrastructure
- Test data generators
- Performance monitoring utilities
- Assertion helpers

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
import asyncio
import os
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
import json
import hashlib


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    markers = [
        "integration: Integration tests requiring external services",
        "steam_meter: Steam quality meter tests",
        "control_valve: Control valve tests",
        "desuperheater: Desuperheater control tests",
        "scada: SCADA integration tests",
        "orchestrator: Orchestrator workflow tests",
        "multi_agent: Multi-agent coordination tests",
        "performance: Performance and benchmark tests",
        "safety: Safety interlock tests",
        "slow: Slow-running tests (>30s)",
        "docker: Tests requiring Docker infrastructure",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--docker",
        action="store_true",
        default=False,
        help="Run tests with Docker infrastructure"
    )
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="Run performance tests"
    )


# =============================================================================
# EVENT LOOP FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# =============================================================================
# MOCK STEAM QUALITY CONTROLLER
# =============================================================================

@pytest.fixture
async def steam_quality_controller():
    """Create mock SteamQualityController for testing."""
    controller = MockSteamQualityController()
    await controller.initialize()
    yield controller
    await controller.shutdown()


@pytest.fixture
async def steam_quality_orchestrator():
    """Create mock SteamQualityOrchestrator for testing."""
    orchestrator = MockSteamQualityOrchestrator()
    await orchestrator.initialize()
    yield orchestrator
    await orchestrator.shutdown()


# =============================================================================
# MOCK STEAM EQUIPMENT
# =============================================================================

@pytest.fixture
async def mock_steam_meter():
    """Mock steam quality meter."""
    meter = MockSteamQualityMeter()
    await meter.start()
    yield meter
    await meter.stop()


@pytest.fixture
async def mock_modbus_steam_meter():
    """Mock Modbus steam quality meter."""
    meter = MockModbusSteamMeter(host="localhost", port=5020)
    await meter.start()
    yield meter
    await meter.stop()


@pytest.fixture
async def mock_opcua_steam_meter():
    """Mock OPC-UA steam quality meter."""
    meter = MockOPCUASteamMeter(host="localhost", port=4850)
    await meter.start()
    yield meter
    await meter.stop()


@pytest.fixture
async def mock_slow_meter():
    """Mock meter with slow responses for timeout testing."""
    meter = MockSlowMeter(host="localhost", port=5021)
    await meter.start()
    yield meter
    await meter.stop()


@pytest.fixture
async def mock_unstable_meter():
    """Mock meter with intermittent failures."""
    meter = MockUnstableMeter(host="localhost", port=5022)
    await meter.start()
    yield meter
    await meter.stop()


@pytest.fixture
async def mock_control_valve():
    """Mock control valve for testing."""
    valve = MockControlValve(valve_id="CV-STEAM-001")
    await valve.start()
    yield valve
    await valve.stop()


@pytest.fixture
async def mock_desuperheater():
    """Mock desuperheater for testing."""
    dsh = MockDesuperheater(desuperheater_id="DSH-001")
    await dsh.start()
    yield dsh
    await dsh.stop()


@pytest.fixture
async def mock_water_supply():
    """Mock spray water supply system."""
    supply = MockWaterSupply()
    await supply.start()
    yield supply
    await supply.stop()


# =============================================================================
# MOCK SCADA SERVERS
# =============================================================================

@pytest.fixture
async def mock_scada_server():
    """Mock SCADA OPC-UA server."""
    server = MockSCADAServer(host="localhost", port=4860)
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def mock_modbus_scada_server():
    """Mock SCADA Modbus server."""
    server = MockModbusSCADAServer(host="localhost", port=5030)
    await server.start()
    yield server
    await server.stop()


# =============================================================================
# MOCK AGENT INFRASTRUCTURE
# =============================================================================

@pytest.fixture
async def mock_gl001_agent():
    """Mock GL-001 ProcessHeatOrchestrator agent."""
    agent = MockAgent(
        agent_id="GL-001",
        agent_type="process_heat_orchestrator"
    )
    await agent.start()
    yield agent
    await agent.stop()


@pytest.fixture
async def mock_gl003_agent():
    """Mock GL-003 SteamSystemAnalyzer agent."""
    agent = MockAgent(
        agent_id="GL-003",
        agent_type="steam_system_analyzer"
    )
    await agent.start()
    yield agent
    await agent.stop()


@pytest.fixture
async def mock_message_bus():
    """Mock message bus for agent communication."""
    bus = MockMessageBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
def mock_event_handler():
    """Mock event handler for testing events."""
    return MockEventHandler()


@pytest.fixture
def mock_alarm_receiver():
    """Mock alarm receiver for testing alarm forwarding."""
    return MockAlarmReceiver()


# =============================================================================
# MOCK SENSORS
# =============================================================================

@pytest.fixture
def mock_pressure_sensor():
    """Mock pressure sensor."""
    return MockSensor(sensor_type="pressure", initial_value=100.0)


@pytest.fixture
def mock_temperature_sensor():
    """Mock temperature sensor."""
    return MockSensor(sensor_type="temperature", initial_value=400.0)


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture."""
    return PerformanceMonitor()


# =============================================================================
# MOCK IMPLEMENTATIONS
# =============================================================================

class MockSteamQualityController:
    """Mock implementation of SteamQualityController."""

    def __init__(self):
        self.connection_state = "disconnected"
        self.consecutive_failures = 0
        self.reconnect_attempts = 0
        self._config = {
            'temperature_setpoint_c': 400.0,
            'quality_target': 0.95
        }
        self._meters = {}
        self._valves = {}

    async def initialize(self):
        """Initialize controller."""
        pass

    async def shutdown(self):
        """Shutdown controller."""
        pass

    async def connect_meter(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to steam meter."""
        try:
            self.connection_state = "connected"
            return {
                'status': 'connected',
                'protocol': config.get('protocol', 'modbus_tcp'),
                'meter_id': f"METER-{len(self._meters) + 1:03d}",
                'session_id': f"SESSION-{random.randint(1000, 9999)}",
                'authenticated': True
            }
        except Exception as e:
            self.consecutive_failures += 1
            return {'status': 'error', 'error_message': str(e)}

    async def read_steam_quality(self) -> Dict[str, Any]:
        """Read steam quality parameters."""
        if self.connection_state != "connected":
            return {'status': 'disconnected', 'quality': 'STALE'}

        return {
            'dryness_fraction': random.uniform(0.92, 0.98),
            'moisture_percent': random.uniform(2.0, 8.0),
            'quality': 'GOOD',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'validation': {
                'dryness_valid': True,
                'rate_of_change_valid': True
            }
        }

    async def read_steam_parameters(self) -> Dict[str, Any]:
        """Read all steam parameters."""
        return {
            'pressure_bar': random.uniform(95, 105),
            'temperature_c': random.uniform(395, 405),
            'flow_rate_kg_s': random.uniform(45, 55),
            'superheat_c': random.uniform(20, 30),
            'enthalpy_kj_kg': random.uniform(2800, 3000),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'validation': {
                'pressure_valid': True,
                'thermodynamic_consistent': True
            }
        }

    async def read_all_parameters(self) -> Dict[str, Any]:
        """Read all parameters including quality and process."""
        quality = await self.read_steam_quality()
        params = await self.read_steam_parameters()
        return {**quality, **params}

    async def set_valve_position(
        self,
        valve_id: str,
        position_percent: float,
        wait_for_position: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Set valve position."""
        if position_percent < 0 or position_percent > 100:
            return {'status': 'error', 'error_code': 'OUT_OF_RANGE'}

        if wait_for_position:
            await asyncio.sleep(0.5)

        return {
            'status': 'success',
            'commanded_position': position_percent,
            'actual_position': position_percent + random.uniform(-0.5, 0.5),
            'feedback_correction_applied': False
        }

    async def read_valve_status(self, valve_id: str) -> Dict[str, Any]:
        """Read valve status."""
        return {
            'position_percent': random.uniform(45, 55),
            'operating_mode': 'AUTO',
            'actuator_state': 'IDLE',
            'limit_switches': {'closed': False, 'open': False},
            'actuator_torque_percent': random.uniform(20, 40),
            'stroke_time_seconds': 10.0,
            'cycle_count': random.randint(1000, 10000),
            'reset_required': False,
            'locked_out': False
        }

    async def read_valve_diagnostics(self, valve_id: str) -> Dict[str, Any]:
        """Read valve diagnostics."""
        return {
            'health_status': 'GOOD',
            'actuator_temperature_c': random.uniform(40, 60),
            'friction_signature': {
                'static_friction': random.uniform(5, 10),
                'dynamic_friction': random.uniform(3, 7)
            },
            'deadband_percent': random.uniform(0.5, 2.0),
            'positioner': {
                'calibration_status': 'GOOD',
                'supply_pressure_bar': random.uniform(4.5, 5.5)
            },
            'active_alerts': [],
            'maintenance_prediction': {
                'remaining_life_percent': random.uniform(70, 95),
                'next_service_date': (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()
            }
        }

    async def emergency_close(self, valve_id: str, reason: str) -> Dict[str, Any]:
        """Emergency close valve."""
        return {
            'status': 'success',
            'emergency_close_executed': True,
            'final_position': 0.0,
            'response_time_ms': random.uniform(100, 500)
        }

    async def acknowledge_emergency(
        self,
        valve_id: str,
        operator_id: str,
        acknowledgment_code: str
    ) -> Dict[str, Any]:
        """Acknowledge emergency and reset lockout."""
        return {
            'status': 'success',
            'lockout_cleared': True
        }

    async def check_safety_interlocks(self) -> Dict[str, Any]:
        """Check all safety interlocks."""
        return {
            'interlock_triggered': False,
            'active_interlocks': [],
            'highest_priority_interlock': None
        }

    async def connect_scada(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to SCADA system."""
        return {
            'status': 'connected',
            'protocol': config.get('protocol', 'opcua'),
            'server_state': 'Running',
            'security_active': config.get('security_mode', 'None') != 'None'
        }

    async def read_tag(self, tag_id: str) -> Dict[str, Any]:
        """Read SCADA tag."""
        return {
            'value': random.uniform(100, 500),
            'quality': 'Good',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    async def write_tag(self, tag_id: str, value: Any) -> Dict[str, Any]:
        """Write SCADA tag."""
        return {
            'status': 'success',
            'written_value': value
        }

    async def get_configuration(self) -> Dict[str, Any]:
        """Get controller configuration."""
        return self._config

    def configure_reconnection(self, config: Dict[str, Any]):
        """Configure reconnection parameters."""
        pass

    def configure_valve(self, **kwargs):
        """Configure valve parameters."""
        pass

    def register_event_handler(self, handler):
        """Register event handler."""
        pass

    async def attempt_reconnect(self):
        """Attempt reconnection."""
        self.reconnect_attempts += 1


class MockSteamQualityOrchestrator:
    """Mock implementation of SteamQualityOrchestrator."""

    def __init__(self):
        self._cycle_count = 0
        self._config = {
            'quality_target': 0.95,
            'temperature_setpoint_c': 400.0
        }

    async def initialize(self):
        """Initialize orchestrator."""
        pass

    async def shutdown(self):
        """Shutdown orchestrator."""
        pass

    async def configure(self, config: Dict[str, Any]):
        """Configure orchestrator."""
        self._config.update(config)

    async def execute_cycle(self) -> Dict[str, Any]:
        """Execute single orchestration cycle."""
        self._cycle_count += 1
        return {
            'status': 'success',
            'cycle_id': f"CYCLE-{self._cycle_count:06d}",
            'execution_time_ms': random.uniform(50, 200),
            'actions_taken': ['READ_QUALITY', 'ANALYZE', 'ADJUST'],
            'current_quality': random.uniform(0.93, 0.97),
            'quality_deviation_detected': False,
            'corrective_actions': [],
            'provenance_hash': hashlib.sha256(str(self._cycle_count).encode()).hexdigest()
        }

    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute workflow."""
        return {
            'status': 'completed',
            'workflow_id': workflow_id,
            'steps_executed': [
                {'step_type': 'DATA_ACQUISITION', 'status': 'success'},
                {'step_type': 'QUALITY_ANALYSIS', 'status': 'success'},
                {'step_type': 'CONTROL_ACTION', 'status': 'success'}
            ]
        }

    async def analyze_quality(self) -> Dict[str, Any]:
        """Analyze steam quality."""
        return {
            'quality_metrics': {
                'dryness_fraction': 0.95,
                'superheat_c': 25.0,
                'enthalpy_kj_kg': 2900.0
            },
            'deviation_detected': False,
            'recommended_actions': []
        }

    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute control action."""
        return {
            'status': 'success',
            'action_executed': True
        }

    async def get_state(self) -> Dict[str, Any]:
        """Get orchestrator state."""
        return {
            'cycle_count': self._cycle_count,
            'last_cycle_time': datetime.now(timezone.utc).isoformat(),
            'quality_target': self._config['quality_target']
        }

    async def get_configuration(self) -> Dict[str, Any]:
        """Get orchestrator configuration."""
        return self._config

    async def request_agent_data(
        self,
        agent_id: str,
        request_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Request data from another agent."""
        return {
            'status': 'success',
            'data': {'headers_status': 'OK'}
        }

    async def coordinate_with_agent(
        self,
        agent_id: str,
        coordination_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate with another agent."""
        return {
            'status': 'success',
            'recommendations': []
        }

    async def publish_status_event(self, event: Dict[str, Any]):
        """Publish status event."""
        pass

    async def publish_message(
        self,
        topic: str,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Publish message to message bus."""
        return {'status': 'success'}

    async def subscribe(self, topic: str, handler: Callable):
        """Subscribe to topic."""
        pass

    async def request_from_agent(
        self,
        agent_id: str,
        request_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Request from agent."""
        return {'status': 'success', 'data': {}}


class MockSteamQualityMeter:
    """Mock steam quality meter."""

    def __init__(self):
        self.running = False
        self._dryness_fraction = 0.95
        self._historical_data = []

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

    def set_dryness_fraction(self, value: float):
        self._dryness_fraction = value

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, f'_{key}', value)

    def add_historical_reading(self, timestamp: datetime, dryness_fraction: float):
        self._historical_data.append({
            'timestamp': timestamp,
            'dryness_fraction': dryness_fraction
        })


class MockModbusSteamMeter:
    """Mock Modbus steam quality meter."""

    def __init__(self, host: str = "localhost", port: int = 502):
        self.host = host
        self.port = port
        self.running = False
        self.connection_count = 0
        self._registers = {}
        self._exceptions = {}

    async def start(self):
        self.running = True
        self.connection_count += 1

    async def stop(self):
        self.running = False

    def set_register_value(self, name: str, value: float):
        self._registers[name] = value

    def set_float32_register(self, address: int, value: float):
        self._registers[address] = value

    def set_exception_response(self, address: int, exception_code: int):
        self._exceptions[address] = exception_code


class MockOPCUASteamMeter:
    """Mock OPC-UA steam quality meter."""

    def __init__(self, host: str = "localhost", port: int = 4840):
        self.host = host
        self.port = port
        self.running = False
        self._nodes = {}
        self._statuses = {}

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

    def set_node_status(self, node_id: str, status_code: str):
        self._statuses[node_id] = status_code


class MockSlowMeter:
    """Mock meter with slow responses."""

    def __init__(self, host: str = "localhost", port: int = 502):
        self.host = host
        self.port = port
        self.running = False
        self._response_delay = 0.0

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

    def set_response_delay(self, delay_seconds: float):
        self._response_delay = delay_seconds


class MockUnstableMeter:
    """Mock meter with intermittent failures."""

    def __init__(self, host: str = "localhost", port: int = 502):
        self.host = host
        self.port = port
        self.running = False

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False


class MockControlValve:
    """Mock control valve."""

    def __init__(self, valve_id: str):
        self.valve_id = valve_id
        self.running = False
        self._position = 50.0
        self._position_offset = 0.0

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

    async def get_status(self) -> Dict[str, Any]:
        return {
            'position_percent': self._position + self._position_offset
        }

    def set_position_offset(self, offset: float):
        self._position_offset = offset

    def set_dead_time(self, dead_time: float):
        pass

    def simulate_stall(self):
        pass

    def simulate_overcurrent(self):
        pass

    def simulate_communication_loss(self):
        pass


class MockDesuperheater:
    """Mock desuperheater."""

    def __init__(self, desuperheater_id: str):
        self.desuperheater_id = desuperheater_id
        self.running = False
        self._injection_rate = 10.0
        self._inlet_temperature = 500.0
        self._outlet_temperature = 400.0

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

    async def get_status(self) -> Dict[str, Any]:
        return {
            'injection_rate_kg_s': self._injection_rate,
            'inlet_temperature_c': self._inlet_temperature,
            'outlet_temperature_c': self._outlet_temperature
        }

    def set_inlet_temperature(self, temp: float):
        self._inlet_temperature = temp

    def set_injection_rate(self, rate: float):
        self._injection_rate = rate

    def simulate_stuck_valve(self):
        pass

    def simulate_sensor_failure(self, sensor: str):
        pass

    def simulate_failure(self):
        pass

    def clear_faults(self):
        pass

    def simulate_nozzle_blockage(self, nozzle_id: str):
        pass

    def simulate_failure_during_action(self):
        pass

    def apply_disturbance(self, parameter: str, magnitude: float):
        pass


class MockWaterSupply:
    """Mock water supply system."""

    def __init__(self):
        self.running = False
        self._pressure = 20.0

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

    def set_pressure(self, pressure: float):
        self._pressure = pressure

    def simulate_failure(self):
        pass


class MockSCADAServer:
    """Mock SCADA server."""

    def __init__(self, host: str = "localhost", port: int = 4840):
        self.host = host
        self.port = port
        self.running = False
        self._tags = {}
        self._alarms = []
        self._connection_failures = 0

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

    def set_tag_value(self, tag_id: str, value: Any):
        self._tags[tag_id] = value

    def set_tag_quality(self, tag_id: str, quality: str):
        pass

    def set_tag_readonly(self, tag_id: str):
        pass

    def trigger_alarm(self, alarm_id: str, severity: str, message: str):
        self._alarms.append({
            'alarm_id': alarm_id,
            'severity': severity,
            'message': message
        })

    def set_connection_failure_count(self, count: int):
        self._connection_failures = count

    def set_server_state(self, state: str):
        pass

    def expire_session(self):
        pass

    def send_shutdown_notification(self, reason: str, delay_seconds: int):
        pass

    async def disconnect_client(self):
        pass


class MockModbusSCADAServer:
    """Mock Modbus SCADA server."""

    def __init__(self, host: str = "localhost", port: int = 502):
        self.host = host
        self.port = port
        self.running = False

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False


class MockAgent:
    """Mock agent for multi-agent testing."""

    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.running = False
        self._received_messages = []
        self._received_statuses = []
        self._received_emergencies = []
        self._received_metrics = []
        self._task_completions = []
        self._last_recommendation = None
        self._available = True

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

    def send_recommendation(self, recommendation: Dict[str, Any]):
        self._last_recommendation = recommendation

    def send_command(self, command: Dict[str, Any]):
        pass

    def send_directive(self, directive: Dict[str, Any]):
        pass

    def send_message(self, recipient_id: str, message_type: str, content: Dict[str, Any]):
        pass

    def assign_task(self, agent_id: str, task: Dict[str, Any]):
        pass

    def approve_proposal(self, proposal_id: str):
        pass

    def initiate_optimization(self, config: Dict[str, Any]):
        pass

    def get_received_messages(self) -> List[Dict[str, Any]]:
        return self._received_messages

    def get_received_statuses(self) -> List[Dict[str, Any]]:
        return self._received_statuses

    def get_received_emergencies(self) -> List[Dict[str, Any]]:
        return self._received_emergencies

    def get_received_metrics(self) -> List[Dict[str, Any]]:
        return self._received_metrics

    def get_received_notifications(self) -> List[Dict[str, Any]]:
        return []

    def get_task_completions(self) -> List[Dict[str, Any]]:
        return self._task_completions

    def get_last_recommendation(self) -> Optional[Dict[str, Any]]:
        return self._last_recommendation

    def set_availability(self, available: bool):
        self._available = available


class MockMessageBus:
    """Mock message bus."""

    def __init__(self):
        self.running = False
        self._messages = {}
        self._published_events = {}

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

    async def publish(self, topic: str, message: Dict[str, Any]):
        if topic not in self._messages:
            self._messages[topic] = []
        self._messages[topic].append(message)

    async def disconnect(self):
        self.running = False

    def get_messages(self, topic: str) -> List[Dict[str, Any]]:
        return self._messages.get(topic, [])

    def get_published_events(self, event_type: str) -> List[Dict[str, Any]]:
        return self._published_events.get(event_type, [])

    def simulate_failure(self):
        pass


class MockEventHandler:
    """Mock event handler."""

    def __init__(self):
        self.connection_lost_called = False
        self.last_event = None

    def on_connection_lost(self, event: Dict[str, Any]):
        self.connection_lost_called = True
        self.last_event = event


class MockAlarmReceiver:
    """Mock alarm receiver."""

    def __init__(self):
        self.received_alarms_count = 0

    def receive_alarm(self, alarm: Dict[str, Any]):
        self.received_alarms_count += 1


class MockSensor:
    """Mock sensor."""

    def __init__(self, sensor_type: str, initial_value: float):
        self.sensor_type = sensor_type
        self._value = initial_value

    def set_value(self, value: float):
        self._value = value

    def get_value(self) -> float:
        return self._value


class PerformanceMonitor:
    """Performance monitoring utility."""

    def __init__(self):
        self.start_time = None
        self.metrics = {}
        self.events = []

    def start(self):
        self.start_time = datetime.now(timezone.utc)
        self.metrics = {}
        self.events = []

    def record_metric(self, metric_name: str, value: float):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def record_event(self, event_name: str, metadata: Optional[Dict[str, Any]] = None):
        self.events.append({
            'name': event_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metadata': metadata or {}
        })

    def elapsed_ms(self) -> float:
        if self.start_time is None:
            return 0
        elapsed = datetime.now(timezone.utc) - self.start_time
        return elapsed.total_seconds() * 1000

    def get_report(self) -> Dict[str, Any]:
        metric_stats = {}
        for metric_name, values in self.metrics.items():
            if values:
                metric_stats[metric_name] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count': len(values)
                }

        return {
            'total_duration_ms': self.elapsed_ms(),
            'event_count': len(self.events),
            'metrics': metric_stats
        }


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

class SteamQualityTestDataGenerator:
    """Generate test data for steam quality tests."""

    @staticmethod
    def generate_quality_reading(
        dryness_fraction: float = None,
        pressure_bar: float = None,
        temperature_c: float = None
    ) -> Dict[str, Any]:
        """Generate steam quality reading."""
        return {
            'dryness_fraction': dryness_fraction or random.uniform(0.90, 0.98),
            'moisture_percent': (1 - (dryness_fraction or 0.95)) * 100,
            'pressure_bar': pressure_bar or random.uniform(95, 105),
            'temperature_c': temperature_c or random.uniform(395, 410),
            'flow_rate_kg_s': random.uniform(45, 55),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'quality': 'GOOD'
        }

    @staticmethod
    def generate_historical_data(
        hours: int = 24,
        interval_minutes: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate historical steam quality data."""
        data = []
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        current_time = start_time
        while current_time <= end_time:
            data.append({
                'timestamp': current_time.isoformat(),
                'dryness_fraction': random.uniform(0.92, 0.97),
                'pressure_bar': random.uniform(95, 105),
                'temperature_c': random.uniform(395, 410)
            })
            current_time += timedelta(minutes=interval_minutes)

        return data


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return SteamQualityTestDataGenerator()


# =============================================================================
# ASSERTION HELPERS
# =============================================================================

class IntegrationTestAssertions:
    """Custom assertions for integration tests."""

    @staticmethod
    def assert_quality_in_range(quality: float, min_val: float = 0.90, max_val: float = 1.0):
        """Assert quality is within valid range."""
        assert min_val <= quality <= max_val, \
            f"Quality {quality} not in range [{min_val}, {max_val}]"

    @staticmethod
    def assert_response_time_ok(response_time_ms: float, target_ms: float):
        """Assert response time meets target."""
        assert response_time_ms <= target_ms, \
            f"Response time {response_time_ms}ms exceeds target {target_ms}ms"

    @staticmethod
    def assert_connection_healthy(status: Dict[str, Any]):
        """Assert connection is healthy."""
        assert status.get('status') == 'connected', "Connection not healthy"


@pytest.fixture
def integration_assertions():
    """Provide integration test assertions."""
    return IntegrationTestAssertions()
