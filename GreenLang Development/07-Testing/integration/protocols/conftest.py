# -*- coding: utf-8 -*-
"""
Protocol Integration Test Fixtures
===================================

Pytest fixtures for mocking industrial protocol servers and brokers.
Enables integration testing without real infrastructure dependencies.

Fixtures Provided:
- mock_opcua_server: Mock OPC-UA server for client testing
- mock_modbus_server: Mock Modbus TCP/RTU server
- mock_mqtt_broker: Mock MQTT broker
- mock_kafka_cluster: Mock Kafka cluster with producer/consumer

Additional Fixtures:
- Protocol client wrappers for testing actual implementations
- Performance measurement utilities
- Error injection helpers
- Reconnection scenario simulators

Author: GreenLang Test Engineering Team
Date: December 2025
"""

import asyncio
import hashlib
import json
import logging
import struct
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Mock Data Classes
# =============================================================================


@dataclass
class MockOPCUANode:
    """Mock OPC-UA node for testing."""
    node_id: str
    browse_name: str
    display_name: str
    value: Any = None
    data_type: str = "Double"
    historizing: bool = True
    history: List[Dict[str, Any]] = field(default_factory=list)
    subscriptions: List[Callable] = field(default_factory=list)
    source_timestamp: datetime = field(default_factory=datetime.utcnow)
    server_timestamp: datetime = field(default_factory=datetime.utcnow)
    status_code: int = 0
    access_level: int = 3  # Read/Write


@dataclass
class MockModbusRegister:
    """Mock Modbus register for testing."""
    address: int
    value: int = 0
    register_type: str = "holding"  # holding, input, coil, discrete


@dataclass
class MockMQTTMessage:
    """Mock MQTT message for testing."""
    topic: str
    payload: bytes
    qos: int = 1
    retain: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class MockKafkaRecord:
    """Mock Kafka record for testing."""
    topic: str
    partition: int
    offset: int
    key: Optional[bytes]
    value: bytes
    headers: List[tuple] = field(default_factory=list)
    timestamp: int = field(default_factory=lambda: int(datetime.utcnow().timestamp() * 1000))


class SecurityMode(str, Enum):
    """OPC-UA security modes."""
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class QoSLevel(IntEnum):
    """MQTT QoS levels."""
    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1
    EXACTLY_ONCE = 2


# =============================================================================
# Mock OPC-UA Server Fixture
# =============================================================================


class MockOPCUAServer:
    """
    Mock OPC-UA server for integration testing.

    Simulates OPC-UA server behavior including:
    - Node browsing and reading
    - Value writing with provenance
    - Subscription management
    - Security mode simulation
    - Reconnection behavior
    - Historical data queries
    """

    def __init__(self, endpoint: str = "opc.tcp://localhost:4840/greenlang/"):
        self.endpoint = endpoint
        self.nodes: Dict[str, MockOPCUANode] = {}
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self.security_mode: SecurityMode = SecurityMode.NONE
        self.running: bool = False
        self._error_mode: bool = False
        self._latency_ms: int = 0
        self._disconnect_after_ops: int = 0
        self._ops_count: int = 0
        self._setup_default_nodes()

    def _setup_default_nodes(self) -> None:
        """Setup default test nodes for process heat scenarios."""
        default_nodes = [
            # Server status
            MockOPCUANode(
                node_id="ns=0;i=2256",
                browse_name="ServerStatus",
                display_name="Server Status",
                value={"state": "Running"},
                data_type="Object"
            ),
            # Process heat nodes
            MockOPCUANode(
                node_id="ns=2;s=Temperature",
                browse_name="Temperature",
                display_name="Process Temperature",
                value=85.5,
                data_type="Double"
            ),
            MockOPCUANode(
                node_id="ns=2;s=Pressure",
                browse_name="Pressure",
                display_name="Process Pressure",
                value=2.5,
                data_type="Double"
            ),
            MockOPCUANode(
                node_id="ns=2;s=FlowRate",
                browse_name="FlowRate",
                display_name="Flow Rate",
                value=150.0,
                data_type="Double"
            ),
            MockOPCUANode(
                node_id="ns=2;s=FuelConsumption",
                browse_name="FuelConsumption",
                display_name="Fuel Consumption",
                value=45.2,
                data_type="Double"
            ),
            MockOPCUANode(
                node_id="ns=2;s=EmissionsFactor",
                browse_name="EmissionsFactor",
                display_name="Emissions Factor",
                value=2.68,
                data_type="Double"
            ),
            MockOPCUANode(
                node_id="ns=2;s=TotalEmissions",
                browse_name="TotalEmissions",
                display_name="Total CO2 Emissions",
                value=121.14,
                data_type="Double"
            ),
            MockOPCUANode(
                node_id="ns=2;s=Status",
                browse_name="Status",
                display_name="System Status",
                value="RUNNING",
                data_type="String"
            ),
            MockOPCUANode(
                node_id="ns=2;s=Efficiency",
                browse_name="Efficiency",
                display_name="Thermal Efficiency",
                value=0.92,
                data_type="Double"
            ),
        ]
        for node in default_nodes:
            self.nodes[node.node_id] = node

    async def start(self) -> None:
        """Start mock server."""
        self.running = True
        logger.info(f"MockOPCUAServer started at {self.endpoint}")

    async def stop(self) -> None:
        """Stop mock server."""
        self.running = False
        self.connected_clients.clear()
        self.subscriptions.clear()
        logger.info("MockOPCUAServer stopped")

    def add_node(self, node: MockOPCUANode) -> None:
        """Add a node to the server."""
        self.nodes[node.node_id] = node

    async def connect_client(
        self,
        client_id: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        certificate: Optional[bytes] = None
    ) -> bool:
        """Connect a client to the server."""
        if not self.running:
            raise ConnectionError("Server not running")

        if self._error_mode:
            raise ConnectionError("Simulated connection error")

        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000)

        # Security validation
        if self.security_mode != SecurityMode.NONE:
            if not certificate and not (username and password):
                raise PermissionError("Authentication required")

        self.connected_clients[client_id] = {
            "username": username,
            "connected_at": datetime.utcnow(),
            "security_mode": self.security_mode.value
        }
        return True

    async def disconnect_client(self, client_id: str) -> None:
        """Disconnect a client from the server."""
        if client_id in self.connected_clients:
            del self.connected_clients[client_id]
            # Remove client subscriptions
            subs_to_remove = [
                sub_id for sub_id, sub in self.subscriptions.items()
                if sub.get("client_id") == client_id
            ]
            for sub_id in subs_to_remove:
                del self.subscriptions[sub_id]

    async def read_value(self, node_id: str) -> Any:
        """Read value from node."""
        self._check_running()
        self._maybe_disconnect()

        if self._error_mode:
            raise Exception("Simulated read error")

        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000)

        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        return self.nodes[node_id].value

    async def read_values(self, node_ids: List[str]) -> Dict[str, Any]:
        """Read multiple values in batch."""
        self._check_running()

        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000)

        results = {}
        for node_id in node_ids:
            if node_id in self.nodes:
                results[node_id] = self.nodes[node_id].value
            else:
                results[node_id] = None
        return results

    async def write_value(
        self,
        node_id: str,
        value: Any,
        source_timestamp: Optional[datetime] = None
    ) -> str:
        """Write value to node and return provenance hash."""
        self._check_running()
        self._maybe_disconnect()

        if self._error_mode:
            raise Exception("Simulated write error")

        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        node = self.nodes[node_id]

        # Check write access
        if not (node.access_level & 0x02):
            raise PermissionError(f"Node {node_id} is read-only")

        old_value = node.value
        node.value = value
        node.source_timestamp = source_timestamp or datetime.utcnow()
        node.server_timestamp = datetime.utcnow()

        # Record history
        node.history.append({
            "timestamp": node.server_timestamp,
            "value": value,
            "old_value": old_value
        })

        # Notify subscribers
        await self._notify_subscribers(node_id, value, node)

        # Calculate provenance
        provenance_str = f"{node_id}:{value}:{node.server_timestamp.isoformat()}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    async def browse(self, parent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Browse child nodes."""
        self._check_running()

        if parent_id is None:
            return [
                {
                    "node_id": node.node_id,
                    "browse_name": node.browse_name,
                    "display_name": node.display_name,
                    "data_type": node.data_type
                }
                for node in self.nodes.values()
            ]
        return []

    async def subscribe(
        self,
        client_id: str,
        node_id: str,
        callback: Callable,
        publishing_interval_ms: int = 1000
    ) -> str:
        """Subscribe to node changes."""
        self._check_running()

        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        sub_id = f"sub_{uuid4().hex[:8]}"
        self.nodes[node_id].subscriptions.append(callback)
        self.subscriptions[sub_id] = {
            "client_id": client_id,
            "node_id": node_id,
            "callback": callback,
            "publishing_interval_ms": publishing_interval_ms,
            "created_at": datetime.utcnow()
        }
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from a subscription."""
        if subscription_id in self.subscriptions:
            sub = self.subscriptions[subscription_id]
            node_id = sub["node_id"]
            callback = sub["callback"]
            if node_id in self.nodes:
                try:
                    self.nodes[node_id].subscriptions.remove(callback)
                except ValueError:
                    pass
            del self.subscriptions[subscription_id]

    async def read_history(
        self,
        node_id: str,
        start_time: datetime,
        end_time: datetime,
        num_values: int = 1000
    ) -> List[Dict[str, Any]]:
        """Read historical data from a node."""
        self._check_running()

        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        node = self.nodes[node_id]
        if not node.historizing:
            raise ValueError(f"Node {node_id} does not have historizing enabled")

        # Filter history by time range
        filtered = [
            dp for dp in node.history
            if start_time <= dp["timestamp"] <= end_time
        ]
        return filtered[:num_values]

    async def _notify_subscribers(
        self,
        node_id: str,
        value: Any,
        node: MockOPCUANode
    ) -> None:
        """Notify all subscribers of a node value change."""
        for callback in node.subscriptions:
            try:
                notification = {
                    "node_id": node_id,
                    "value": value,
                    "source_timestamp": node.source_timestamp,
                    "server_timestamp": node.server_timestamp,
                    "status_code": node.status_code
                }
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification)
                else:
                    callback(notification)
            except Exception as e:
                logger.error(f"Subscription callback error: {e}")

    def _check_running(self) -> None:
        """Check if server is running."""
        if not self.running:
            raise ConnectionError("Server not running")

    def _maybe_disconnect(self) -> None:
        """Maybe disconnect after N operations for testing."""
        if self._disconnect_after_ops > 0:
            self._ops_count += 1
            if self._ops_count >= self._disconnect_after_ops:
                self.running = False
                self._ops_count = 0

    # Test helper methods
    def simulate_disconnect(self) -> None:
        """Simulate server disconnect for testing reconnection."""
        self.running = False
        self.connected_clients.clear()

    def simulate_reconnect(self) -> None:
        """Simulate server coming back online."""
        self.running = True

    def enable_error_mode(self) -> None:
        """Enable error simulation mode."""
        self._error_mode = True

    def disable_error_mode(self) -> None:
        """Disable error simulation mode."""
        self._error_mode = False

    def set_latency(self, latency_ms: int) -> None:
        """Set simulated network latency."""
        self._latency_ms = latency_ms

    def set_disconnect_after(self, ops: int) -> None:
        """Set server to disconnect after N operations."""
        self._disconnect_after_ops = ops
        self._ops_count = 0

    def set_security_mode(self, mode: SecurityMode) -> None:
        """Set security mode."""
        self.security_mode = mode

    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "endpoint": self.endpoint,
            "running": self.running,
            "security_mode": self.security_mode.value,
            "node_count": len(self.nodes),
            "connected_clients": len(self.connected_clients),
            "active_subscriptions": len(self.subscriptions),
            "error_mode": self._error_mode,
            "latency_ms": self._latency_ms
        }


@pytest.fixture
def mock_opcua_server():
    """Create a mock OPC-UA server for testing."""
    server = MockOPCUAServer()
    yield server


@pytest.fixture
async def running_mock_opcua_server(mock_opcua_server):
    """Create and start a mock OPC-UA server."""
    await mock_opcua_server.start()
    yield mock_opcua_server
    await mock_opcua_server.stop()


# =============================================================================
# Mock Modbus Server Fixture
# =============================================================================


class MockModbusServer:
    """
    Mock Modbus server for integration testing.

    Simulates Modbus TCP/RTU behavior including:
    - Holding registers (read/write)
    - Input registers (read-only)
    - Coils (read/write)
    - Discrete inputs (read-only)
    - Error simulation
    - Latency simulation
    """

    def __init__(self, host: str = "localhost", port: int = 502):
        self.host = host
        self.port = port
        self.holding_registers: Dict[int, int] = {}
        self.input_registers: Dict[int, int] = {}
        self.coils: Dict[int, bool] = {}
        self.discrete_inputs: Dict[int, bool] = {}
        self.connected: bool = False
        self.error_mode: bool = False
        self.latency_ms: int = 0
        self._timeout_mode: bool = False
        self._invalid_address_mode: bool = False
        self._connection_pool: Dict[str, Dict[str, Any]] = {}
        self._setup_default_registers()

    def _setup_default_registers(self) -> None:
        """Setup default test registers for process heat scenarios."""
        # Holding registers (address: value)
        self.holding_registers = {
            0: 855,    # Temperature (scale 0.1 = 85.5 C)
            1: 25,     # Pressure (scale 0.1 = 2.5 bar)
            2: 1500,   # Flow rate high word
            3: 0,      # Flow rate low word
            10: 268,   # Emission factor (scale 0.01 = 2.68)
            11: 92,    # Efficiency percentage
            20: 452,   # Fuel consumption (scale 0.1 = 45.2)
            21: 12114, # Total emissions (scale 0.01 = 121.14)
            100: 1,    # Status register (1 = running)
        }

        # Input registers (read-only sensor values)
        self.input_registers = {
            0: 857,    # Actual temperature (85.7 C)
            1: 24,     # Actual pressure (2.4 bar)
            2: 1495,   # Actual flow high
            3: 0,      # Actual flow low
            10: 269,   # Current emission factor
        }

        # Coils (boolean outputs)
        self.coils = {
            0: True,   # System running
            1: False,  # Alarm active
            2: True,   # Pump 1 on
            3: False,  # Pump 2 on
            4: True,   # Heater on
            5: False,  # Emergency stop
        }

        # Discrete inputs (boolean inputs)
        self.discrete_inputs = {
            0: True,   # Emergency stop released
            1: True,   # Safety circuit OK
            2: False,  # High temperature alarm
            3: False,  # Low pressure alarm
        }

    async def connect(self) -> bool:
        """Connect to mock server."""
        if self.error_mode:
            raise ConnectionError("Simulated connection error")

        if self._timeout_mode:
            await asyncio.sleep(30)  # Simulate timeout
            raise TimeoutError("Connection timeout")

        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)

        self.connected = True
        return True

    async def disconnect(self) -> None:
        """Disconnect from mock server."""
        self.connected = False

    async def read_holding_registers(
        self,
        address: int,
        count: int = 1,
        unit: int = 1
    ) -> List[int]:
        """Read holding registers (function code 03)."""
        self._check_connected()
        self._check_address(address, count)

        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)

        return [
            self.holding_registers.get(address + i, 0)
            for i in range(count)
        ]

    async def read_input_registers(
        self,
        address: int,
        count: int = 1,
        unit: int = 1
    ) -> List[int]:
        """Read input registers (function code 04)."""
        self._check_connected()
        self._check_address(address, count)

        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)

        return [
            self.input_registers.get(address + i, 0)
            for i in range(count)
        ]

    async def read_coils(
        self,
        address: int,
        count: int = 1,
        unit: int = 1
    ) -> List[bool]:
        """Read coils (function code 01)."""
        self._check_connected()
        return [
            self.coils.get(address + i, False)
            for i in range(count)
        ]

    async def read_discrete_inputs(
        self,
        address: int,
        count: int = 1,
        unit: int = 1
    ) -> List[bool]:
        """Read discrete inputs (function code 02)."""
        self._check_connected()
        return [
            self.discrete_inputs.get(address + i, False)
            for i in range(count)
        ]

    async def write_register(
        self,
        address: int,
        value: int,
        unit: int = 1
    ) -> str:
        """Write single holding register (function code 06)."""
        self._check_connected()

        if address < 0 or value < 0 or value > 65535:
            raise ValueError("Invalid register address or value")

        self.holding_registers[address] = value

        # Return provenance hash
        provenance_str = f"register:{address}:{value}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    async def write_registers(
        self,
        address: int,
        values: List[int],
        unit: int = 1
    ) -> str:
        """Write multiple holding registers (function code 16)."""
        self._check_connected()

        for i, value in enumerate(values):
            if value < 0 or value > 65535:
                raise ValueError(f"Invalid value at index {i}: {value}")
            self.holding_registers[address + i] = value

        provenance_str = f"registers:{address}:{values}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    async def write_coil(
        self,
        address: int,
        value: bool,
        unit: int = 1
    ) -> str:
        """Write single coil (function code 05)."""
        self._check_connected()
        self.coils[address] = value

        provenance_str = f"coil:{address}:{value}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    async def write_coils(
        self,
        address: int,
        values: List[bool],
        unit: int = 1
    ) -> str:
        """Write multiple coils (function code 15)."""
        self._check_connected()

        for i, value in enumerate(values):
            self.coils[address + i] = value

        provenance_str = f"coils:{address}:{values}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _check_connected(self) -> None:
        """Check if connected."""
        if not self.connected:
            raise ConnectionError("Not connected to Modbus server")
        if self.error_mode:
            raise Exception("Simulated Modbus error")

    def _check_address(self, address: int, count: int) -> None:
        """Check if address is valid."""
        if self._invalid_address_mode:
            raise ValueError(f"Invalid address: {address}")
        if address < 0 or address + count > 65535:
            raise ValueError(f"Address out of range: {address}")

    # Test helper methods
    def simulate_error(self) -> None:
        """Enable error mode for testing error handling."""
        self.error_mode = True

    def clear_error(self) -> None:
        """Disable error mode."""
        self.error_mode = False

    def set_latency(self, latency_ms: int) -> None:
        """Set simulated latency."""
        self.latency_ms = latency_ms

    def enable_timeout_mode(self) -> None:
        """Enable timeout simulation."""
        self._timeout_mode = True

    def disable_timeout_mode(self) -> None:
        """Disable timeout simulation."""
        self._timeout_mode = False

    def enable_invalid_address_mode(self) -> None:
        """Enable invalid address error simulation."""
        self._invalid_address_mode = True

    def disable_invalid_address_mode(self) -> None:
        """Disable invalid address error simulation."""
        self._invalid_address_mode = False

    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "host": self.host,
            "port": self.port,
            "connected": self.connected,
            "holding_register_count": len(self.holding_registers),
            "input_register_count": len(self.input_registers),
            "coil_count": len(self.coils),
            "discrete_input_count": len(self.discrete_inputs),
            "error_mode": self.error_mode,
            "latency_ms": self.latency_ms
        }


@pytest.fixture
def mock_modbus_server():
    """Create a mock Modbus server for testing."""
    return MockModbusServer()


@pytest.fixture
async def connected_mock_modbus_server(mock_modbus_server):
    """Create and connect a mock Modbus server."""
    await mock_modbus_server.connect()
    yield mock_modbus_server
    await mock_modbus_server.disconnect()


# =============================================================================
# Mock MQTT Broker Fixture
# =============================================================================


class MockMQTTBroker:
    """
    Mock MQTT broker for integration testing.

    Simulates MQTT broker behavior including:
    - Connect/disconnect with authentication
    - Publish with QoS 0, 1, 2
    - Subscribe with wildcards (+ and #)
    - Retained messages
    - Last Will and Testament (LWT)
    - Clean session handling
    """

    def __init__(self, host: str = "localhost", port: int = 1883):
        self.host = host
        self.port = port
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.subscriptions: Dict[str, List[Dict[str, Any]]] = {}
        self.retained_messages: Dict[str, MockMQTTMessage] = {}
        self.message_history: List[MockMQTTMessage] = []
        self.will_messages: Dict[str, MockMQTTMessage] = {}
        self.running: bool = True
        self._qos_delivery_tracking: Dict[str, Dict[str, Any]] = {}
        self._message_id_counter: int = 0

    async def connect(
        self,
        client_id: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        clean_session: bool = True,
        will_topic: Optional[str] = None,
        will_message: Optional[bytes] = None,
        will_qos: int = 1,
        will_retain: bool = False
    ) -> bool:
        """Connect a client to the broker."""
        if not self.running:
            raise ConnectionError("Broker not running")

        # Store client info
        self.connected_clients[client_id] = {
            "username": username,
            "clean_session": clean_session,
            "connected_at": datetime.utcnow(),
            "subscriptions": []
        }

        # Store will message (LWT)
        if will_topic and will_message:
            self.will_messages[client_id] = MockMQTTMessage(
                topic=will_topic,
                payload=will_message,
                qos=will_qos,
                retain=will_retain
            )

        return True

    async def disconnect(self, client_id: str, graceful: bool = True) -> None:
        """Disconnect a client."""
        if client_id not in self.connected_clients:
            return

        # Publish will message if not graceful disconnect
        if not graceful and client_id in self.will_messages:
            will = self.will_messages[client_id]
            await self._deliver_message(will)
            del self.will_messages[client_id]

        # Remove will message on graceful disconnect
        if graceful and client_id in self.will_messages:
            del self.will_messages[client_id]

        # Remove from connected clients
        del self.connected_clients[client_id]

    async def publish(
        self,
        client_id: str,
        topic: str,
        payload: bytes,
        qos: int = 1,
        retain: bool = False
    ) -> str:
        """Publish a message and return message ID."""
        if client_id not in self.connected_clients:
            raise ConnectionError("Client not connected")

        self._message_id_counter += 1
        message_id = str(self._message_id_counter)

        message = MockMQTTMessage(
            topic=topic,
            payload=payload,
            qos=qos,
            retain=retain,
            message_id=message_id
        )

        self.message_history.append(message)

        # Handle retained messages
        if retain:
            if payload:  # Non-empty payload
                self.retained_messages[topic] = message
            elif topic in self.retained_messages:
                # Empty payload clears retained message
                del self.retained_messages[topic]

        # Deliver to subscribers
        await self._deliver_message(message)

        # Calculate provenance hash
        provenance_str = f"{topic}:{payload.decode() if payload else ''}:{message.timestamp.isoformat()}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    async def subscribe(
        self,
        client_id: str,
        topic: str,
        qos: int = 1,
        callback: Optional[Callable] = None
    ) -> None:
        """Subscribe to a topic."""
        if client_id not in self.connected_clients:
            raise ConnectionError("Client not connected")

        if topic not in self.subscriptions:
            self.subscriptions[topic] = []

        subscription = {
            "client_id": client_id,
            "qos": qos,
            "callback": callback,
            "subscribed_at": datetime.utcnow()
        }

        self.subscriptions[topic].append(subscription)
        self.connected_clients[client_id]["subscriptions"].append(topic)

        # Deliver retained messages for matching topics
        for retained_topic, message in self.retained_messages.items():
            if self._topic_matches(topic, retained_topic):
                if callback:
                    effective_qos = min(qos, message.qos)
                    await callback(MockMQTTMessage(
                        topic=retained_topic,
                        payload=message.payload,
                        qos=effective_qos,
                        retain=True
                    ))

    async def unsubscribe(self, client_id: str, topic: str) -> None:
        """Unsubscribe from a topic."""
        if topic in self.subscriptions:
            self.subscriptions[topic] = [
                sub for sub in self.subscriptions[topic]
                if sub["client_id"] != client_id
            ]

        if client_id in self.connected_clients:
            subs = self.connected_clients[client_id]["subscriptions"]
            if topic in subs:
                subs.remove(topic)

    async def _deliver_message(self, message: MockMQTTMessage) -> None:
        """Deliver message to matching subscribers."""
        for pattern, subscribers in self.subscriptions.items():
            if self._topic_matches(pattern, message.topic):
                for sub in subscribers:
                    if sub["callback"]:
                        # Apply QoS downgrade if necessary
                        effective_qos = min(sub["qos"], message.qos)

                        # Track QoS 1 and 2 delivery
                        if effective_qos >= 1:
                            self._track_qos_delivery(message.message_id, sub["client_id"])

                        try:
                            if asyncio.iscoroutinefunction(sub["callback"]):
                                await sub["callback"](MockMQTTMessage(
                                    topic=message.topic,
                                    payload=message.payload,
                                    qos=effective_qos,
                                    retain=message.retain,
                                    message_id=message.message_id
                                ))
                            else:
                                sub["callback"](MockMQTTMessage(
                                    topic=message.topic,
                                    payload=message.payload,
                                    qos=effective_qos,
                                    retain=message.retain,
                                    message_id=message.message_id
                                ))
                        except Exception as e:
                            logger.error(f"MQTT callback error: {e}")

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Check if topic matches subscription pattern with wildcards."""
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")

        for i, part in enumerate(pattern_parts):
            if part == "#":
                return True  # Multi-level wildcard matches rest
            if i >= len(topic_parts):
                return False
            if part != "+" and part != topic_parts[i]:
                return False  # Single-level wildcard or exact match

        return len(pattern_parts) == len(topic_parts)

    def _track_qos_delivery(self, message_id: str, client_id: str) -> None:
        """Track QoS delivery for reliability testing."""
        if message_id not in self._qos_delivery_tracking:
            self._qos_delivery_tracking[message_id] = {}
        self._qos_delivery_tracking[message_id][client_id] = {
            "delivered": True,
            "acknowledged": False,
            "timestamp": datetime.utcnow()
        }

    def acknowledge_message(self, message_id: str, client_id: str) -> None:
        """Acknowledge message delivery (for QoS 1/2 testing)."""
        if message_id in self._qos_delivery_tracking:
            if client_id in self._qos_delivery_tracking[message_id]:
                self._qos_delivery_tracking[message_id][client_id]["acknowledged"] = True

    # Test helper methods
    def simulate_disconnect(self) -> None:
        """Simulate broker disconnect."""
        self.running = False
        # Trigger will messages for all connected clients
        for client_id in list(self.connected_clients.keys()):
            asyncio.create_task(self.disconnect(client_id, graceful=False))

    def simulate_reconnect(self) -> None:
        """Simulate broker coming back online."""
        self.running = True

    def get_message_count(self, topic: Optional[str] = None) -> int:
        """Get count of published messages."""
        if topic:
            return len([m for m in self.message_history if self._topic_matches(topic, m.topic)])
        return len(self.message_history)

    def get_retained_message(self, topic: str) -> Optional[MockMQTTMessage]:
        """Get retained message for a topic."""
        return self.retained_messages.get(topic)

    def get_statistics(self) -> Dict[str, Any]:
        """Get broker statistics."""
        return {
            "host": self.host,
            "port": self.port,
            "running": self.running,
            "connected_clients": len(self.connected_clients),
            "active_subscriptions": sum(len(subs) for subs in self.subscriptions.values()),
            "retained_messages": len(self.retained_messages),
            "total_messages": len(self.message_history)
        }


@pytest.fixture
def mock_mqtt_broker():
    """Create a mock MQTT broker for testing."""
    return MockMQTTBroker()


@pytest.fixture
async def connected_mock_mqtt_broker(mock_mqtt_broker):
    """Create a mock MQTT broker with a test client connected."""
    await mock_mqtt_broker.connect("test-client")
    yield mock_mqtt_broker
    await mock_mqtt_broker.disconnect("test-client")


# =============================================================================
# Mock Kafka Cluster Fixture
# =============================================================================


class MockKafkaCluster:
    """
    Mock Kafka cluster for integration testing.

    Simulates Kafka behavior including:
    - Producer send (sync/async)
    - Consumer group management
    - Offset management (commit/seek)
    - Partition assignment
    - Schema registry (Avro)
    - Transaction support
    - Error injection
    """

    def __init__(self, bootstrap_servers: List[str] = None):
        self.bootstrap_servers = bootstrap_servers or ["localhost:9092"]
        self.topics: Dict[str, Dict[int, List[MockKafkaRecord]]] = {}
        self.consumer_groups: Dict[str, Dict[str, Any]] = {}
        self.committed_offsets: Dict[str, Dict[str, int]] = {}
        self.schemas: Dict[str, Dict] = {}
        self.running: bool = True
        self._offset_counter: int = 0
        self._transaction_active: Dict[str, bool] = {}
        self._transaction_records: Dict[str, List[MockKafkaRecord]] = {}

    def create_topic(
        self,
        topic: str,
        num_partitions: int = 3,
        replication_factor: int = 1
    ) -> None:
        """Create a topic."""
        self.topics[topic] = {
            p: [] for p in range(num_partitions)
        }

    async def produce(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        partition: Optional[int] = None,
        headers: Optional[List[tuple]] = None,
        producer_id: Optional[str] = None
    ) -> MockKafkaRecord:
        """Produce a message."""
        if not self.running:
            raise ConnectionError("Kafka cluster not running")

        if topic not in self.topics:
            self.create_topic(topic)

        # Determine partition
        if partition is None:
            if key:
                partition = hash(key) % len(self.topics[topic])
            else:
                partition = self._offset_counter % len(self.topics[topic])

        self._offset_counter += 1
        offset = len(self.topics[topic][partition])

        # Add provenance header
        all_headers = list(headers or [])
        provenance_str = f"{topic}:{key}:{offset}:{datetime.utcnow().isoformat()}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()
        all_headers.append(("provenance_hash", provenance_hash.encode()))

        record = MockKafkaRecord(
            topic=topic,
            partition=partition,
            offset=offset,
            key=key,
            value=value,
            headers=all_headers
        )

        # Handle transactions
        if producer_id and self._transaction_active.get(producer_id):
            if producer_id not in self._transaction_records:
                self._transaction_records[producer_id] = []
            self._transaction_records[producer_id].append(record)
        else:
            self.topics[topic][partition].append(record)

        return record

    async def produce_batch(
        self,
        records: List[Dict[str, Any]],
        producer_id: Optional[str] = None
    ) -> List[MockKafkaRecord]:
        """Produce multiple messages in a batch."""
        results = []
        for record_data in records:
            result = await self.produce(
                topic=record_data["topic"],
                value=record_data["value"],
                key=record_data.get("key"),
                partition=record_data.get("partition"),
                headers=record_data.get("headers"),
                producer_id=producer_id
            )
            results.append(result)
        return results

    async def consume(
        self,
        group_id: str,
        topics: List[str],
        timeout_ms: int = 1000,
        max_records: int = 100
    ) -> List[MockKafkaRecord]:
        """Consume messages from topics."""
        if not self.running:
            raise ConnectionError("Kafka cluster not running")

        if group_id not in self.consumer_groups:
            self.consumer_groups[group_id] = {
                "topics": topics,
                "members": [],
                "created_at": datetime.utcnow()
            }

        records = []

        for topic in topics:
            if topic not in self.topics:
                continue

            for partition, partition_records in self.topics[topic].items():
                key = f"{topic}:{partition}"

                if group_id not in self.committed_offsets:
                    self.committed_offsets[group_id] = {}

                start_offset = self.committed_offsets[group_id].get(key, 0)

                for record in partition_records[start_offset:]:
                    records.append(record)
                    if len(records) >= max_records:
                        break

                if len(records) >= max_records:
                    break

        return records

    async def consume_one(
        self,
        group_id: str,
        topics: List[str],
        timeout_ms: int = 1000
    ) -> Optional[MockKafkaRecord]:
        """Consume a single message."""
        records = await self.consume(group_id, topics, timeout_ms, max_records=1)
        return records[0] if records else None

    async def commit(
        self,
        group_id: str,
        offsets: Dict[str, int]
    ) -> None:
        """Commit offsets for consumer group."""
        if group_id not in self.committed_offsets:
            self.committed_offsets[group_id] = {}

        self.committed_offsets[group_id].update(offsets)

    async def seek(
        self,
        group_id: str,
        topic: str,
        partition: int,
        offset: int
    ) -> None:
        """Seek to specific offset."""
        key = f"{topic}:{partition}"

        if group_id not in self.committed_offsets:
            self.committed_offsets[group_id] = {}

        self.committed_offsets[group_id][key] = offset

    async def seek_to_beginning(
        self,
        group_id: str,
        topic: str,
        partition: int
    ) -> None:
        """Seek to beginning of partition."""
        await self.seek(group_id, topic, partition, 0)

    async def seek_to_end(
        self,
        group_id: str,
        topic: str,
        partition: int
    ) -> None:
        """Seek to end of partition."""
        if topic in self.topics and partition in self.topics[topic]:
            end_offset = len(self.topics[topic][partition])
            await self.seek(group_id, topic, partition, end_offset)

    # Transaction support
    async def begin_transaction(self, producer_id: str) -> None:
        """Begin a transaction."""
        self._transaction_active[producer_id] = True
        self._transaction_records[producer_id] = []

    async def commit_transaction(self, producer_id: str) -> None:
        """Commit a transaction."""
        if not self._transaction_active.get(producer_id):
            raise RuntimeError("No active transaction")

        # Move transaction records to topics
        for record in self._transaction_records.get(producer_id, []):
            self.topics[record.topic][record.partition].append(record)

        self._transaction_active[producer_id] = False
        self._transaction_records[producer_id] = []

    async def abort_transaction(self, producer_id: str) -> None:
        """Abort a transaction."""
        self._transaction_active[producer_id] = False
        self._transaction_records[producer_id] = []

    # Schema registry
    def register_schema(self, subject: str, schema: Dict) -> int:
        """Register an Avro schema."""
        self.schemas[subject] = schema
        return len(self.schemas)

    def get_schema(self, subject: str) -> Optional[Dict]:
        """Get schema by subject."""
        return self.schemas.get(subject)

    # Utility methods
    def get_topic_partitions(self, topic: str) -> List[int]:
        """Get partition IDs for a topic."""
        if topic not in self.topics:
            return []
        return list(self.topics[topic].keys())

    def get_consumer_group_info(self, group_id: str) -> Optional[Dict]:
        """Get consumer group information."""
        return self.consumer_groups.get(group_id)

    def get_topic_message_count(self, topic: str) -> int:
        """Get total message count for a topic."""
        if topic not in self.topics:
            return 0
        return sum(len(records) for records in self.topics[topic].values())

    # Test helper methods
    def simulate_cluster_down(self) -> None:
        """Simulate cluster going down."""
        self.running = False

    def simulate_cluster_up(self) -> None:
        """Simulate cluster coming back up."""
        self.running = True

    def simulate_leader_election(self, topic: str, partition: int) -> None:
        """Simulate leader election (for testing failover)."""
        pass  # No-op in mock, but could add delays

    def clear_all(self) -> None:
        """Clear all data (for test isolation)."""
        self.topics.clear()
        self.consumer_groups.clear()
        self.committed_offsets.clear()
        self.schemas.clear()
        self._offset_counter = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        return {
            "bootstrap_servers": self.bootstrap_servers,
            "running": self.running,
            "topic_count": len(self.topics),
            "consumer_group_count": len(self.consumer_groups),
            "schema_count": len(self.schemas),
            "total_messages": sum(
                sum(len(records) for records in partitions.values())
                for partitions in self.topics.values()
            )
        }


@pytest.fixture
def mock_kafka_cluster():
    """Create a mock Kafka cluster for testing."""
    cluster = MockKafkaCluster()
    cluster.create_topic("test-topic", num_partitions=3)
    cluster.create_topic("emissions-events", num_partitions=3)
    cluster.create_topic("process-heat-data", num_partitions=3)
    cluster.create_topic("dlq-topic", num_partitions=1)
    return cluster


@pytest.fixture
async def mock_kafka_with_data(mock_kafka_cluster):
    """Create a mock Kafka cluster with pre-populated test data."""
    # Add test messages
    test_messages = [
        {"type": "emission", "value": 2.68, "unit": "kg_co2"},
        {"type": "temperature", "value": 85.5, "unit": "celsius"},
        {"type": "pressure", "value": 2.5, "unit": "bar"},
        {"type": "flow_rate", "value": 150.0, "unit": "kg_h"},
        {"type": "efficiency", "value": 0.92, "unit": "percent"},
    ]

    for msg in test_messages:
        await mock_kafka_cluster.produce(
            "test-topic",
            json.dumps(msg).encode(),
            key=msg["type"].encode()
        )

    yield mock_kafka_cluster
    mock_kafka_cluster.clear_all()


# =============================================================================
# Protocol Client Patches (for testing without real libraries)
# =============================================================================


@pytest.fixture
def patch_asyncua():
    """Patch asyncua imports for testing without the library."""
    mock_ua = MagicMock()
    mock_ua.ObjectIds.Server_ServerStatus = "ns=0;i=2256"
    mock_ua.SecurityPolicyType.NoSecurity = "NoSecurity"
    mock_ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt = "Basic256Sha256"
    mock_ua.DataValue = MagicMock
    mock_ua.Variant = MagicMock

    mock_node = AsyncMock()
    mock_node.read_value = AsyncMock(return_value=85.5)
    mock_node.write_value = AsyncMock()
    mock_node.read_browse_name = AsyncMock(return_value=MagicMock(to_string=lambda: "Temperature"))
    mock_node.read_display_name = AsyncMock(return_value=MagicMock(Text="Process Temperature"))
    mock_node.read_node_class = AsyncMock(return_value="Variable")
    mock_node.nodeid = MagicMock(to_string=lambda: "ns=2;s=Temperature")

    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client.get_node = MagicMock(return_value=mock_node)
    mock_client.read_values = AsyncMock(return_value=[85.5, 2.5, 150.0])
    mock_client.nodes = MagicMock(root=mock_node)
    mock_client.create_subscription = AsyncMock()
    mock_client.set_user = MagicMock()
    mock_client.set_password = MagicMock()
    mock_client.set_security_string = MagicMock()

    mock_subscription = AsyncMock()
    mock_subscription.subscribe_data_change = AsyncMock()
    mock_subscription.delete = AsyncMock()
    mock_client.create_subscription.return_value = mock_subscription

    with patch.dict("sys.modules", {
        "asyncua": MagicMock(Client=MagicMock(return_value=mock_client), ua=mock_ua),
        "asyncua.common.subscription": MagicMock(Subscription=MagicMock())
    }):
        yield mock_client, mock_ua


@pytest.fixture
def patch_pymodbus():
    """Patch pymodbus imports for testing without the library."""
    mock_result = MagicMock()
    mock_result.isError = MagicMock(return_value=False)
    mock_result.registers = [855, 25]
    mock_result.bits = [True, False, True, False]

    mock_client = AsyncMock()
    mock_client.connect = AsyncMock(return_value=True)
    mock_client.close = MagicMock()
    mock_client.read_holding_registers = AsyncMock(return_value=mock_result)
    mock_client.read_input_registers = AsyncMock(return_value=mock_result)
    mock_client.read_coils = AsyncMock(return_value=mock_result)
    mock_client.read_discrete_inputs = AsyncMock(return_value=mock_result)
    mock_client.write_register = AsyncMock(return_value=mock_result)
    mock_client.write_registers = AsyncMock(return_value=mock_result)
    mock_client.write_coil = AsyncMock(return_value=mock_result)

    mock_decoder = MagicMock()
    mock_decoder.decode_16bit_uint = MagicMock(return_value=855)
    mock_decoder.decode_32bit_float = MagicMock(return_value=85.5)

    with patch.dict("sys.modules", {
        "pymodbus": MagicMock(),
        "pymodbus.client": MagicMock(
            AsyncModbusTcpClient=MagicMock(return_value=mock_client),
            AsyncModbusSerialClient=MagicMock(return_value=mock_client)
        ),
        "pymodbus.exceptions": MagicMock(ModbusException=Exception),
        "pymodbus.payload": MagicMock(
            BinaryPayloadDecoder=MagicMock(fromRegisters=MagicMock(return_value=mock_decoder)),
            BinaryPayloadBuilder=MagicMock()
        ),
        "pymodbus.constants": MagicMock(Endian=MagicMock(BIG=">", LITTLE="<"))
    }):
        yield mock_client


@pytest.fixture
def patch_aiomqtt():
    """Patch aiomqtt imports for testing without the library."""
    mock_message = MagicMock()
    mock_message.topic = "test/topic"
    mock_message.payload = b'{"value": 85.5}'
    mock_message.qos = 1
    mock_message.retain = False

    mock_messages = AsyncMock()
    mock_messages.__aiter__ = MagicMock(return_value=iter([mock_message]))

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()
    mock_client.publish = AsyncMock()
    mock_client.subscribe = AsyncMock()
    mock_client.unsubscribe = AsyncMock()
    mock_client.messages = MagicMock(return_value=mock_messages)

    with patch.dict("sys.modules", {
        "aiomqtt": MagicMock(
            Client=MagicMock(return_value=mock_client),
            MqttError=Exception
        )
    }):
        yield mock_client


@pytest.fixture
def patch_aiokafka():
    """Patch aiokafka imports for testing without the library."""
    mock_record_metadata = MagicMock()
    mock_record_metadata.topic = "test-topic"
    mock_record_metadata.partition = 0
    mock_record_metadata.offset = 0

    mock_producer = AsyncMock()
    mock_producer.start = AsyncMock()
    mock_producer.stop = AsyncMock()
    mock_producer.flush = AsyncMock()
    mock_producer.send_and_wait = AsyncMock(return_value=mock_record_metadata)

    mock_message = MagicMock()
    mock_message.topic = "test-topic"
    mock_message.partition = 0
    mock_message.offset = 0
    mock_message.key = b"test-key"
    mock_message.value = b'{"value": 85.5}'
    mock_message.headers = []
    mock_message.timestamp = int(datetime.utcnow().timestamp() * 1000)

    mock_consumer = AsyncMock()
    mock_consumer.start = AsyncMock()
    mock_consumer.stop = AsyncMock()
    mock_consumer.subscribe = MagicMock()
    mock_consumer.unsubscribe = MagicMock()
    mock_consumer.commit = AsyncMock()
    mock_consumer.seek = MagicMock()
    mock_consumer.pause = MagicMock()
    mock_consumer.resume = MagicMock()
    mock_consumer.paused = MagicMock(return_value=[])
    mock_consumer.assignment = MagicMock(return_value=[])
    mock_consumer.partitions_for_topic = MagicMock(return_value=[0, 1, 2])
    mock_consumer.getone = AsyncMock(return_value=mock_message)
    mock_consumer.__aiter__ = MagicMock(return_value=iter([mock_message]))

    mock_topic_partition = MagicMock()

    with patch.dict("sys.modules", {
        "aiokafka": MagicMock(
            AIOKafkaProducer=MagicMock(return_value=mock_producer),
            AIOKafkaConsumer=MagicMock(return_value=mock_consumer),
            TopicPartition=mock_topic_partition
        ),
        "aiokafka.errors": MagicMock(
            KafkaError=Exception,
            OffsetOutOfRangeError=Exception
        )
    }):
        yield mock_producer, mock_consumer


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_process_heat_data():
    """Sample process heat data for testing."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "temperature_celsius": 85.5,
        "pressure_bar": 2.5,
        "flow_rate_kg_h": 150.0,
        "fuel_type": "natural_gas",
        "fuel_consumption_kg_h": 45.2,
        "emission_factor_kg_co2_per_kg": 2.68,
        "heat_output_kw": 500.0,
        "efficiency": 0.92,
        "total_emissions_kg_co2": 121.14
    }


@pytest.fixture
def sample_emission_event():
    """Sample emission event for Kafka testing."""
    return {
        "event_id": f"evt_{uuid4().hex[:12]}",
        "timestamp": datetime.utcnow().isoformat(),
        "source": "boiler_1",
        "emission_type": "CO2",
        "value_kg": 121.14,
        "calculation_method": "mass_balance",
        "fuel_type": "natural_gas",
        "provenance_hash": hashlib.sha256(b"test").hexdigest()
    }


@pytest.fixture
def sample_modbus_register_map():
    """Sample Modbus register mapping for process heat."""
    return [
        {"name": "temperature", "address": 0, "data_type": "uint16", "scale": 0.1, "unit": "C"},
        {"name": "pressure", "address": 1, "data_type": "uint16", "scale": 0.1, "unit": "bar"},
        {"name": "flow_rate", "address": 2, "data_type": "uint32", "scale": 1.0, "unit": "kg/h"},
        {"name": "emission_factor", "address": 10, "data_type": "uint16", "scale": 0.01, "unit": "kg_co2/kg"},
        {"name": "efficiency", "address": 11, "data_type": "uint16", "scale": 0.01, "unit": "%"},
        {"name": "status", "address": 100, "data_type": "uint16", "scale": 1, "unit": ""},
    ]


@pytest.fixture
def sample_mqtt_topics():
    """Sample MQTT topics for process heat."""
    return {
        "temperature": "process-heat/sensors/temperature",
        "pressure": "process-heat/sensors/pressure",
        "flow_rate": "process-heat/sensors/flow-rate",
        "emissions": "process-heat/calculated/emissions",
        "status": "process-heat/status",
        "alarms": "process-heat/alarms/#",
        "commands": "process-heat/commands/+",
    }


@pytest.fixture
def sample_kafka_topics():
    """Sample Kafka topics for process heat."""
    return {
        "raw_sensor_data": "process-heat-raw",
        "calculated_emissions": "emissions-events",
        "aggregated_data": "process-heat-aggregated",
        "alerts": "process-heat-alerts",
        "dlq": "process-heat-dlq",
    }


# =============================================================================
# Performance Measurement Utilities
# =============================================================================


@pytest.fixture
def performance_timer():
    """Timer utility for performance testing."""
    class PerformanceTimer:
        def __init__(self):
            self.start_time: Optional[float] = None
            self.end_time: Optional[float] = None
            self.measurements: List[float] = []

        def start(self) -> None:
            self.start_time = time.perf_counter()

        def stop(self) -> float:
            self.end_time = time.perf_counter()
            duration = self.end_time - self.start_time
            self.measurements.append(duration)
            return duration

        def reset(self) -> None:
            self.start_time = None
            self.end_time = None

        @property
        def elapsed_ms(self) -> float:
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time) * 1000
            return 0.0

        @property
        def average_ms(self) -> float:
            if not self.measurements:
                return 0.0
            return (sum(self.measurements) / len(self.measurements)) * 1000

        @property
        def min_ms(self) -> float:
            if not self.measurements:
                return 0.0
            return min(self.measurements) * 1000

        @property
        def max_ms(self) -> float:
            if not self.measurements:
                return 0.0
            return max(self.measurements) * 1000

    return PerformanceTimer()


@pytest.fixture
def throughput_calculator():
    """Throughput calculator for performance testing."""
    class ThroughputCalculator:
        def __init__(self):
            self.start_time: Optional[float] = None
            self.message_count: int = 0
            self.byte_count: int = 0

        def start(self) -> None:
            self.start_time = time.perf_counter()
            self.message_count = 0
            self.byte_count = 0

        def record_message(self, size_bytes: int = 0) -> None:
            self.message_count += 1
            self.byte_count += size_bytes

        def get_throughput(self) -> Dict[str, float]:
            if not self.start_time:
                return {"messages_per_sec": 0.0, "bytes_per_sec": 0.0}

            elapsed = time.perf_counter() - self.start_time
            if elapsed == 0:
                return {"messages_per_sec": 0.0, "bytes_per_sec": 0.0}

            return {
                "messages_per_sec": self.message_count / elapsed,
                "bytes_per_sec": self.byte_count / elapsed,
                "total_messages": self.message_count,
                "total_bytes": self.byte_count,
                "elapsed_seconds": elapsed
            }

    return ThroughputCalculator()


# =============================================================================
# Event Loop Fixture
# =============================================================================


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
