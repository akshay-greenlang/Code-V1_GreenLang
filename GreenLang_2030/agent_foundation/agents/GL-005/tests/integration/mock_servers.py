"""
Mock servers for GL-005 CombustionControlAgent integration testing.

Provides mock implementations of:
- OPC UA server (DCS simulation)
- Modbus server (PLC simulation)
- MQTT broker (Combustion analyzer simulation)
- HTTP server (Flame scanner simulation)

These mocks simulate real hardware behavior for testing without
requiring actual industrial equipment.
"""

import asyncio
import threading
import time
import random
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# MOCK OPC UA SERVER (DCS Simulation)
# ============================================================================

class MockOPCUAServer:
    """Mock OPC UA server for DCS integration testing."""

    def __init__(self, host: str = 'localhost', port: int = 4840):
        self.host = host
        self.port = port
        self.running = False
        self.nodes: Dict[str, Any] = {}
        self.subscriptions: List[Dict] = []
        self.alarms: List[Dict] = []
        self._init_default_nodes()

    def _init_default_nodes(self):
        """Initialize default OPC UA nodes."""
        self.nodes = {
            'fuel_flow': 500.0,
            'air_flow': 5000.0,
            'combustion_temperature': 1200.0,
            'furnace_pressure': 100.0,
            'o2_percent': 3.5,
            'co2_percent': 12.0,
            'steam_flow': 10000.0,
            'feedwater_temp': 105.0
        }

    async def start(self):
        """Start the mock OPC UA server."""
        self.running = True
        logger.info(f"Mock OPC UA server started on {self.host}:{self.port}")

    async def stop(self):
        """Stop the mock OPC UA server."""
        self.running = False
        logger.info("Mock OPC UA server stopped")

    async def read_node(self, node_id: str) -> Optional[Any]:
        """Read value from OPC UA node."""
        if not self.running:
            raise ConnectionError("Server not running")

        # Simulate network delay
        await asyncio.sleep(random.uniform(0.01, 0.05))

        if node_id in self.nodes:
            # Add small random variation to simulate sensor noise
            value = self.nodes[node_id]
            if isinstance(value, (int, float)):
                variation = value * 0.02  # 2% variation
                value += random.uniform(-variation, variation)
            return value
        else:
            raise ValueError(f"Node {node_id} not found")

    async def write_node(self, node_id: str, value: Any) -> bool:
        """Write value to OPC UA node."""
        if not self.running:
            raise ConnectionError("Server not running")

        # Simulate network delay
        await asyncio.sleep(random.uniform(0.01, 0.05))

        self.nodes[node_id] = value
        logger.debug(f"Node {node_id} set to {value}")
        return True

    async def read_multiple_nodes(self, node_ids: List[str]) -> Dict[str, Any]:
        """Read multiple nodes at once."""
        results = {}
        for node_id in node_ids:
            try:
                results[node_id] = await self.read_node(node_id)
            except ValueError:
                results[node_id] = None
        return results

    def subscribe_to_alarms(self, callback):
        """Subscribe to alarm notifications."""
        self.subscriptions.append({'callback': callback})

    def trigger_alarm(self, alarm_type: str, severity: str, message: str):
        """Trigger an alarm."""
        alarm = {
            'type': alarm_type,
            'severity': severity,
            'message': message,
            'timestamp': datetime.now(timezone.utc)
        }
        self.alarms.append(alarm)

        # Notify subscribers
        for subscription in self.subscriptions:
            subscription['callback'](alarm)


# ============================================================================
# MOCK MODBUS SERVER (PLC Simulation)
# ============================================================================

class MockModbusServer:
    """Mock Modbus TCP server for PLC integration testing."""

    def __init__(self, host: str = 'localhost', port: int = 502):
        self.host = host
        self.port = port
        self.running = False
        self.coils: Dict[int, bool] = {}
        self.discrete_inputs: Dict[int, bool] = {}
        self.holding_registers: Dict[int, int] = {}
        self.input_registers: Dict[int, int] = {}
        self._init_default_values()

    def _init_default_values(self):
        """Initialize default Modbus values."""
        # Coils (digital outputs)
        self.coils[0] = True  # System enabled
        self.coils[1] = False  # Emergency stop
        self.coils[2] = True  # Flame detected

        # Holding registers (analog outputs/setpoints)
        self.holding_registers[0] = 1200  # Temperature setpoint (scaled)
        self.holding_registers[1] = 500   # Fuel flow setpoint (scaled)
        self.holding_registers[2] = 5000  # Air flow setpoint (scaled)

        # Input registers (analog inputs/measurements)
        self.input_registers[0] = 1205  # Temperature actual (scaled)
        self.input_registers[1] = 505   # Fuel flow actual (scaled)
        self.input_registers[2] = 5050  # Air flow actual (scaled)

    async def start(self):
        """Start the mock Modbus server."""
        self.running = True
        logger.info(f"Mock Modbus server started on {self.host}:{self.port}")

    async def stop(self):
        """Stop the mock Modbus server."""
        self.running = False
        logger.info("Mock Modbus server stopped")

    async def read_coils(self, address: int, count: int = 1) -> List[bool]:
        """Read coil values."""
        if not self.running:
            raise ConnectionError("Server not running")

        await asyncio.sleep(random.uniform(0.005, 0.02))

        coils = []
        for i in range(count):
            coils.append(self.coils.get(address + i, False))
        return coils

    async def write_coil(self, address: int, value: bool) -> bool:
        """Write coil value."""
        if not self.running:
            raise ConnectionError("Server not running")

        await asyncio.sleep(random.uniform(0.005, 0.02))

        self.coils[address] = value
        logger.debug(f"Coil {address} set to {value}")
        return True

    async def read_holding_registers(self, address: int, count: int = 1) -> List[int]:
        """Read holding register values."""
        if not self.running:
            raise ConnectionError("Server not running")

        await asyncio.sleep(random.uniform(0.005, 0.02))

        registers = []
        for i in range(count):
            registers.append(self.holding_registers.get(address + i, 0))
        return registers

    async def write_register(self, address: int, value: int) -> bool:
        """Write holding register value."""
        if not self.running:
            raise ConnectionError("Server not running")

        await asyncio.sleep(random.uniform(0.005, 0.02))

        self.holding_registers[address] = value
        logger.debug(f"Register {address} set to {value}")
        return True

    async def read_input_registers(self, address: int, count: int = 1) -> List[int]:
        """Read input register values."""
        if not self.running:
            raise ConnectionError("Server not running")

        await asyncio.sleep(random.uniform(0.005, 0.02))

        registers = []
        for i in range(count):
            # Simulate sensor variation
            base_value = self.input_registers.get(address + i, 0)
            variation = int(base_value * 0.02)
            registers.append(base_value + random.randint(-variation, variation))
        return registers


# ============================================================================
# MOCK MQTT BROKER (Combustion Analyzer Simulation)
# ============================================================================

class MockMQTTBroker:
    """Mock MQTT broker for combustion analyzer integration testing."""

    def __init__(self, host: str = 'localhost', port: int = 1883):
        self.host = host
        self.port = port
        self.running = False
        self.topics: Dict[str, List[Dict]] = {}
        self.subscribers: Dict[str, List] = {}
        self.message_queue: List[Dict] = []

    async def start(self):
        """Start the mock MQTT broker."""
        self.running = True
        logger.info(f"Mock MQTT broker started on {self.host}:{self.port}")

        # Start publishing simulated analyzer data
        asyncio.create_task(self._publish_analyzer_data())

    async def stop(self):
        """Stop the mock MQTT broker."""
        self.running = False
        logger.info("Mock MQTT broker stopped")

    async def _publish_analyzer_data(self):
        """Continuously publish simulated analyzer data."""
        while self.running:
            # Publish O2, CO, CO2, NOx readings
            analyzer_data = {
                'o2_percent': 3.5 + random.uniform(-0.3, 0.3),
                'co_ppm': 25.0 + random.uniform(-5.0, 5.0),
                'co2_percent': 12.0 + random.uniform(-0.5, 0.5),
                'nox_ppm': 30.0 + random.uniform(-3.0, 3.0),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            await self.publish('combustion/analyzer/data', analyzer_data)
            await asyncio.sleep(1.0)  # Publish every second

    async def publish(self, topic: str, message: Dict):
        """Publish message to topic."""
        if not self.running:
            return

        # Store message
        if topic not in self.topics:
            self.topics[topic] = []

        self.topics[topic].append({
            'message': message,
            'timestamp': datetime.now(timezone.utc)
        })

        # Notify subscribers
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                callback(topic, message)

    def subscribe(self, topic: str, callback):
        """Subscribe to topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []

        self.subscribers[topic].append(callback)
        logger.debug(f"Subscribed to topic: {topic}")

    def get_latest_message(self, topic: str) -> Optional[Dict]:
        """Get latest message from topic."""
        if topic in self.topics and len(self.topics[topic]) > 0:
            return self.topics[topic][-1]['message']
        return None


# ============================================================================
# MOCK FLAME SCANNER HTTP SERVER
# ============================================================================

class MockFlameScannerServer:
    """Mock HTTP server for flame scanner integration testing."""

    def __init__(self, host: str = 'localhost', port: int = 8080):
        self.host = host
        self.port = port
        self.running = False
        self.flame_detected = True
        self.flame_intensity = 85.0
        self.flame_stability = 0.95

    async def start(self):
        """Start the mock flame scanner server."""
        self.running = True
        logger.info(f"Mock flame scanner server started on {self.host}:{self.port}")

        # Start simulating flame variations
        asyncio.create_task(self._simulate_flame_variations())

    async def stop(self):
        """Stop the mock flame scanner server."""
        self.running = False
        logger.info("Mock flame scanner server stopped")

    async def _simulate_flame_variations(self):
        """Simulate flame intensity variations."""
        while self.running:
            # Add random variations to simulate real flame behavior
            if self.flame_detected:
                self.flame_intensity += random.uniform(-2.0, 2.0)
                self.flame_intensity = max(30.0, min(100.0, self.flame_intensity))

                # Calculate stability based on recent variations
                self.flame_stability = 0.95 - (abs(85.0 - self.flame_intensity) / 100)

            await asyncio.sleep(0.1)

    async def get_flame_status(self) -> Dict[str, Any]:
        """Get current flame status."""
        if not self.running:
            raise ConnectionError("Server not running")

        await asyncio.sleep(random.uniform(0.01, 0.03))

        return {
            'flame_detected': self.flame_detected,
            'intensity_percent': self.flame_intensity,
            'stability_index': self.flame_stability,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def set_flame_loss(self):
        """Simulate flame loss."""
        self.flame_detected = False
        self.flame_intensity = 0.0
        self.flame_stability = 0.0

    def restore_flame(self):
        """Restore flame."""
        self.flame_detected = True
        self.flame_intensity = 85.0
        self.flame_stability = 0.95


# ============================================================================
# MOCK SERVER MANAGER
# ============================================================================

class MockServerManager:
    """Manages all mock servers for integration testing."""

    def __init__(self):
        self.opcua_server = MockOPCUAServer()
        self.modbus_server = MockModbusServer()
        self.mqtt_broker = MockMQTTBroker()
        self.flame_scanner = MockFlameScannerServer()

    async def start_all(self):
        """Start all mock servers."""
        await self.opcua_server.start()
        await self.modbus_server.start()
        await self.mqtt_broker.start()
        await self.flame_scanner.start()
        logger.info("All mock servers started")

    async def stop_all(self):
        """Stop all mock servers."""
        await self.opcua_server.stop()
        await self.modbus_server.stop()
        await self.mqtt_broker.stop()
        await self.flame_scanner.stop()
        logger.info("All mock servers stopped")


# ============================================================================
# PYTEST FIXTURES FOR MOCK SERVERS
# ============================================================================

async def create_mock_server_manager():
    """Create and start mock server manager."""
    manager = MockServerManager()
    await manager.start_all()
    return manager


async def cleanup_mock_server_manager(manager: MockServerManager):
    """Stop and cleanup mock server manager."""
    await manager.stop_all()
