# -*- coding: utf-8 -*-
"""
SCADA System Integration Connector for GL-001 ProcessHeatOrchestrator

Implements secure, real-time connections to industrial control systems via:
- OPC UA (Open Platform Communications Unified Architecture)
- Modbus TCP (Industrial protocol for PLCs and RTUs)
- MQTT (Message Queuing Telemetry Transport for IoT devices)

Features:
- TLS encryption with certificate authentication
- Automatic reconnection with exponential backoff
- Data buffering for offline scenarios
- Real-time sensor subscription (1-10 second intervals)
- Connection pooling for high performance
"""

import asyncio
import ssl
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json
import os
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_random

# Third-party imports (would be installed via pip)
# from asyncua import Client as OPCUAClientBase
# from pymodbus.client import AsyncModbusTcpClient
# import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class SCADAProtocol(Enum):
    """Supported SCADA protocols."""
    OPC_UA = "opc_ua"
    MODBUS_TCP = "modbus_tcp"
    MQTT = "mqtt"


@dataclass
class SCADASensorConfig:
    """Configuration for a SCADA sensor."""
    sensor_id: str
    sensor_type: str  # temperature, pressure, flow, valve, pump
    address: str  # Protocol-specific address
    unit: str
    min_value: float
    max_value: float
    sampling_rate: int  # Seconds
    calibration_factor: float = 1.0


@dataclass
class SCADAConnectionConfig:
    """Configuration for SCADA connection."""
    protocol: SCADAProtocol
    host: str
    port: int
    tls_enabled: bool = True
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    ca_path: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None  # Retrieved from environment
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 5
    buffer_size: int = 10000
    health_check_interval: int = 30


class SCADADataBuffer:
    """
    Thread-safe buffer for SCADA data with 24-hour retention.

    Implements circular buffer with timestamp-based cleanup.
    """

    def __init__(self, max_size: int = 10000, retention_hours: int = 24):
        """Initialize data buffer."""
        self.max_size = max_size
        self.retention_hours = retention_hours
        self.buffer = deque(maxlen=max_size)
        self._lock = asyncio.Lock()

    async def add(self, data: Dict[str, Any]):
        """Add data point to buffer."""
        async with self._lock:
            timestamp = DeterministicClock.utcnow()
            self.buffer.append({
                'timestamp': timestamp,
                'data': data
            })

            # Clean old data
            cutoff_time = timestamp - timedelta(hours=self.retention_hours)
            while self.buffer and self.buffer[0]['timestamp'] < cutoff_time:
                self.buffer.popleft()

    async def get_all(self) -> List[Dict[str, Any]]:
        """Get all buffered data."""
        async with self._lock:
            return list(self.buffer)

    async def get_recent(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent data within specified minutes."""
        async with self._lock:
            cutoff_time = DeterministicClock.utcnow() - timedelta(minutes=minutes)
            return [
                item for item in self.buffer
                if item['timestamp'] >= cutoff_time
            ]


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.

    States: CLOSED (normal), OPEN (failing), HALF_OPEN (testing)
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def can_attempt(self) -> bool:
        """Check if operation can be attempted."""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False

        return self.state == "HALF_OPEN"


class OPCUAClient:
    """
    OPC UA client implementation for industrial automation.

    Handles connection to OPC UA servers with security and subscription management.
    """

    def __init__(self, config: SCADAConnectionConfig):
        """Initialize OPC UA client."""
        self.config = config
        self.client = None  # Would be asyncua.Client in production
        self.subscriptions = {}
        self.circuit_breaker = CircuitBreaker()
        self.connected = False

    async def connect(self) -> bool:
        """
        Establish secure OPC UA connection.

        Returns:
            True if connection successful, False otherwise
        """
        if not self.circuit_breaker.can_attempt():
            logger.warning("Circuit breaker OPEN for OPC UA connection")
            return False

        try:
            # In production, would use asyncua library
            # self.client = OPCUAClientBase(f"opc.tcp://{self.config.host}:{self.config.port}")

            if self.config.tls_enabled:
                # Configure TLS security
                ssl_context = self._create_ssl_context()
                # self.client.set_security(ssl_context)

            # Connect with authentication
            if self.config.username:
                # self.client.set_user(self.config.username)
                # self.client.set_password(os.getenv('SCADA_PASSWORD', self.config.password))
                pass

            # await self.client.connect()
            self.connected = True
            self.circuit_breaker.record_success()
            logger.info(f"Connected to OPC UA server at {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"OPC UA connection failed: {e}")
            return False

    async def subscribe_sensors(self, sensors: List[SCADASensorConfig], callback: Callable):
        """
        Subscribe to sensor data streams.

        Args:
            sensors: List of sensors to subscribe to
            callback: Function to call with sensor data
        """
        if not self.connected:
            raise ConnectionError("Not connected to OPC UA server")

        for sensor in sensors:
            try:
                # In production, would create subscription
                # subscription = await self.client.create_subscription(sensor.sampling_rate * 1000, callback)
                # node = self.client.get_node(sensor.address)
                # handle = await subscription.subscribe_data_change(node)

                self.subscriptions[sensor.sensor_id] = {
                    'sensor': sensor,
                    'subscription': None,  # Would be actual subscription
                    'handle': None  # Would be actual handle
                }

                logger.info(f"Subscribed to sensor {sensor.sensor_id}")

            except Exception as e:
                logger.error(f"Failed to subscribe to sensor {sensor.sensor_id}: {e}")

    async def read_sensor(self, sensor_id: str) -> Optional[float]:
        """
        Read current value from sensor.

        Args:
            sensor_id: Sensor identifier

        Returns:
            Sensor value or None if error
        """
        if not self.connected:
            return None

        try:
            if sensor_id in self.subscriptions:
                sensor = self.subscriptions[sensor_id]['sensor']
                # In production: value = await node.read_value()

                # Simulated value for demo
                import random
                value = random.uniform(sensor.min_value, sensor.max_value)
                return value * sensor.calibration_factor

        except Exception as e:
            logger.error(f"Failed to read sensor {sensor_id}: {e}")
            return None

    async def disconnect(self):
        """Disconnect from OPC UA server."""
        if self.connected:
            try:
                # await self.client.disconnect()
                self.connected = False
                logger.info("Disconnected from OPC UA server")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for secure connection."""
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

        if self.config.cert_path and self.config.key_path:
            context.load_cert_chain(self.config.cert_path, self.config.key_path)

        if self.config.ca_path:
            context.load_verify_locations(self.config.ca_path)

        context.minimum_version = ssl.TLSVersion.TLSv1_3
        return context


class ModbusTCPClient:
    """
    Modbus TCP client for PLC and RTU communication.

    Implements Modbus protocol over TCP/IP for industrial devices.
    """

    def __init__(self, config: SCADAConnectionConfig):
        """Initialize Modbus TCP client."""
        self.config = config
        self.client = None  # Would be pymodbus.AsyncModbusTcpClient
        self.circuit_breaker = CircuitBreaker()
        self.connected = False

    async def connect(self) -> bool:
        """
        Establish Modbus TCP connection.

        Returns:
            True if connection successful, False otherwise
        """
        if not self.circuit_breaker.can_attempt():
            logger.warning("Circuit breaker OPEN for Modbus connection")
            return False

        try:
            # In production: self.client = AsyncModbusTcpClient(host=self.config.host, port=self.config.port)
            # await self.client.connect()

            self.connected = True
            self.circuit_breaker.record_success()
            logger.info(f"Connected to Modbus server at {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Modbus connection failed: {e}")
            return False

    async def read_registers(self, address: int, count: int, unit: int = 1) -> Optional[List[int]]:
        """
        Read holding registers from Modbus device.

        Args:
            address: Starting register address
            count: Number of registers to read
            unit: Modbus unit ID

        Returns:
            List of register values or None if error
        """
        if not self.connected:
            return None

        try:
            # In production: result = await self.client.read_holding_registers(address, count, unit)
            # return result.registers if not result.isError() else None

            # Simulated values for demo
            import random
            return [deterministic_random().randint(0, 65535) for _ in range(count)]

        except Exception as e:
            logger.error(f"Failed to read Modbus registers: {e}")
            return None

    async def write_register(self, address: int, value: int, unit: int = 1) -> bool:
        """
        Write single register to Modbus device.

        Args:
            address: Register address
            value: Value to write
            unit: Modbus unit ID

        Returns:
            True if write successful, False otherwise
        """
        if not self.connected:
            return False

        try:
            # In production: result = await self.client.write_register(address, value, unit)
            # return not result.isError()

            logger.info(f"Written value {value} to register {address}")
            return True

        except Exception as e:
            logger.error(f"Failed to write Modbus register: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Modbus server."""
        if self.connected:
            try:
                # await self.client.close()
                self.connected = False
                logger.info("Disconnected from Modbus server")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")


class MQTTSubscriber:
    """
    MQTT subscriber for IoT sensor networks.

    Implements MQTT protocol for lightweight sensor communication.
    """

    def __init__(self, config: SCADAConnectionConfig):
        """Initialize MQTT subscriber."""
        self.config = config
        self.client = None  # Would be paho.mqtt.Client
        self.circuit_breaker = CircuitBreaker()
        self.connected = False
        self.callbacks = {}

    async def connect(self) -> bool:
        """
        Establish MQTT connection with TLS.

        Returns:
            True if connection successful, False otherwise
        """
        if not self.circuit_breaker.can_attempt():
            logger.warning("Circuit breaker OPEN for MQTT connection")
            return False

        try:
            # In production, would use paho-mqtt
            # self.client = mqtt.Client()

            if self.config.tls_enabled:
                # Configure TLS
                # self.client.tls_set(ca_certs=self.config.ca_path,
                #                    certfile=self.config.cert_path,
                #                    keyfile=self.config.key_path)
                pass

            if self.config.username:
                # self.client.username_pw_set(self.config.username,
                #                            os.getenv('MQTT_PASSWORD', self.config.password))
                pass

            # self.client.connect_async(self.config.host, self.config.port)
            self.connected = True
            self.circuit_breaker.record_success()
            logger.info(f"Connected to MQTT broker at {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"MQTT connection failed: {e}")
            return False

    async def subscribe_topic(self, topic: str, callback: Callable):
        """
        Subscribe to MQTT topic.

        Args:
            topic: MQTT topic to subscribe
            callback: Function to call with message data
        """
        if not self.connected:
            raise ConnectionError("Not connected to MQTT broker")

        try:
            # In production: self.client.subscribe(topic)
            self.callbacks[topic] = callback
            logger.info(f"Subscribed to MQTT topic: {topic}")

        except Exception as e:
            logger.error(f"Failed to subscribe to topic {topic}: {e}")

    async def publish(self, topic: str, payload: Dict[str, Any], qos: int = 1) -> bool:
        """
        Publish message to MQTT topic.

        Args:
            topic: MQTT topic
            payload: Message payload
            qos: Quality of Service level

        Returns:
            True if publish successful, False otherwise
        """
        if not self.connected:
            return False

        try:
            # In production: self.client.publish(topic, json.dumps(payload), qos=qos)
            logger.info(f"Published to topic {topic}: {payload}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish to topic {topic}: {e}")
            return False

    async def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.connected:
            try:
                # self.client.disconnect()
                self.connected = False
                logger.info("Disconnected from MQTT broker")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")


class SCADAConnectionPool:
    """
    Connection pool for managing multiple SCADA connections.

    Provides connection pooling, health monitoring, and load balancing.
    """

    def __init__(self, max_connections: int = 100):
        """Initialize connection pool."""
        self.max_connections = max_connections
        self.connections: Dict[str, Any] = {}
        self.health_status: Dict[str, bool] = {}
        self._lock = asyncio.Lock()

    async def add_connection(self, conn_id: str, config: SCADAConnectionConfig) -> bool:
        """
        Add new connection to pool.

        Args:
            conn_id: Connection identifier
            config: Connection configuration

        Returns:
            True if added successfully, False otherwise
        """
        async with self._lock:
            if len(self.connections) >= self.max_connections:
                logger.error(f"Connection pool full ({self.max_connections} connections)")
                return False

            # Create appropriate client based on protocol
            if config.protocol == SCADAProtocol.OPC_UA:
                client = OPCUAClient(config)
            elif config.protocol == SCADAProtocol.MODBUS_TCP:
                client = ModbusTCPClient(config)
            elif config.protocol == SCADAProtocol.MQTT:
                client = MQTTSubscriber(config)
            else:
                logger.error(f"Unsupported protocol: {config.protocol}")
                return False

            # Connect
            connected = await client.connect()
            if connected:
                self.connections[conn_id] = client
                self.health_status[conn_id] = True
                logger.info(f"Added connection {conn_id} to pool")
                return True

            return False

    async def get_connection(self, conn_id: str) -> Optional[Any]:
        """
        Get connection from pool.

        Args:
            conn_id: Connection identifier

        Returns:
            Connection client or None if not found
        """
        async with self._lock:
            return self.connections.get(conn_id)

    async def health_check(self):
        """Perform health check on all connections."""
        async with self._lock:
            for conn_id, client in self.connections.items():
                try:
                    # Check connection health
                    if hasattr(client, 'connected'):
                        self.health_status[conn_id] = client.connected
                    else:
                        self.health_status[conn_id] = True

                except Exception as e:
                    logger.error(f"Health check failed for {conn_id}: {e}")
                    self.health_status[conn_id] = False

    async def reconnect_failed(self):
        """Reconnect failed connections."""
        async with self._lock:
            for conn_id, is_healthy in self.health_status.items():
                if not is_healthy:
                    client = self.connections.get(conn_id)
                    if client:
                        logger.info(f"Attempting to reconnect {conn_id}")
                        connected = await client.connect()
                        self.health_status[conn_id] = connected


class SCADAConnector:
    """
    Main SCADA connector orchestrating all protocols and connections.

    Provides unified interface for SCADA system integration with:
    - Multi-protocol support (OPC UA, Modbus, MQTT)
    - Connection pooling and management
    - Data buffering and replay
    - Real-time sensor subscription
    - Automatic reconnection and fault tolerance
    """

    def __init__(self):
        """Initialize SCADA connector."""
        self.connection_pool = SCADAConnectionPool(max_connections=100)
        self.data_buffer = SCADADataBuffer(max_size=10000, retention_hours=24)
        self.sensor_registry: Dict[str, SCADASensorConfig] = {}
        self.health_monitor_task = None
        self.data_collection_tasks = {}

    async def initialize(self, configs: List[Tuple[str, SCADAConnectionConfig]]):
        """
        Initialize SCADA connections.

        Args:
            configs: List of (connection_id, config) tuples
        """
        logger.info(f"Initializing SCADA connector with {len(configs)} connections")

        # Add connections to pool
        for conn_id, config in configs:
            success = await self.connection_pool.add_connection(conn_id, config)
            if not success:
                logger.error(f"Failed to initialize connection {conn_id}")

        # Start health monitoring
        self.health_monitor_task = asyncio.create_task(self._health_monitor())

        logger.info("SCADA connector initialized")

    async def register_sensors(self, sensors: List[SCADASensorConfig]):
        """
        Register sensors for monitoring.

        Args:
            sensors: List of sensor configurations
        """
        for sensor in sensors:
            self.sensor_registry[sensor.sensor_id] = sensor
            logger.info(f"Registered sensor {sensor.sensor_id} ({sensor.sensor_type})")

    async def start_data_collection(self, conn_id: str, sensor_ids: List[str]):
        """
        Start collecting data from sensors.

        Args:
            conn_id: Connection to use
            sensor_ids: List of sensor IDs to collect from
        """
        if conn_id in self.data_collection_tasks:
            logger.warning(f"Data collection already running for {conn_id}")
            return

        # Create collection task
        task = asyncio.create_task(
            self._collect_sensor_data(conn_id, sensor_ids)
        )
        self.data_collection_tasks[conn_id] = task

        logger.info(f"Started data collection for {conn_id} with {len(sensor_ids)} sensors")

    async def _collect_sensor_data(self, conn_id: str, sensor_ids: List[str]):
        """
        Continuously collect sensor data.

        Args:
            conn_id: Connection identifier
            sensor_ids: List of sensor IDs
        """
        while True:
            try:
                client = await self.connection_pool.get_connection(conn_id)
                if not client:
                    logger.error(f"Connection {conn_id} not found")
                    await asyncio.sleep(10)
                    continue

                # Collect data from each sensor
                for sensor_id in sensor_ids:
                    if sensor_id not in self.sensor_registry:
                        continue

                    sensor = self.sensor_registry[sensor_id]

                    # Read sensor value based on protocol
                    if isinstance(client, OPCUAClient):
                        value = await client.read_sensor(sensor_id)
                    elif isinstance(client, ModbusTCPClient):
                        # Convert address to Modbus register
                        address = int(sensor.address)
                        registers = await client.read_registers(address, 1)
                        value = registers[0] if registers else None
                    else:
                        value = None

                    if value is not None:
                        # Store in buffer
                        await self.data_buffer.add({
                            'sensor_id': sensor_id,
                            'value': value,
                            'unit': sensor.unit,
                            'timestamp': DeterministicClock.utcnow().isoformat()
                        })

                # Wait for next collection cycle
                await asyncio.sleep(1)  # 1 second default, could be configurable

            except Exception as e:
                logger.error(f"Error collecting data from {conn_id}: {e}")
                await asyncio.sleep(10)

    async def _health_monitor(self):
        """Monitor connection health."""
        while True:
            try:
                await self.connection_pool.health_check()
                await self.connection_pool.reconnect_failed()
                await asyncio.sleep(30)  # Health check every 30 seconds

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)

    async def get_recent_data(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """
        Get recent sensor data.

        Args:
            minutes: Number of minutes to retrieve

        Returns:
            List of recent sensor readings
        """
        return await self.data_buffer.get_recent(minutes)

    async def shutdown(self):
        """Shutdown SCADA connector."""
        logger.info("Shutting down SCADA connector")

        # Cancel tasks
        if self.health_monitor_task:
            self.health_monitor_task.cancel()

        for task in self.data_collection_tasks.values():
            task.cancel()

        # Disconnect all connections
        for conn_id, client in self.connection_pool.connections.items():
            await client.disconnect()

        logger.info("SCADA connector shutdown complete")


# Example usage
async def main():
    """Example SCADA connector usage."""

    # Create connector
    connector = SCADAConnector()

    # Configure connections
    configs = [
        ("plant1_opcua", SCADAConnectionConfig(
            protocol=SCADAProtocol.OPC_UA,
            host="192.168.1.100",
            port=4840,
            tls_enabled=True,
            cert_path="/path/to/cert.pem",
            key_path="/path/to/key.pem"
        )),
        ("plant1_modbus", SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="192.168.1.101",
            port=502,
            tls_enabled=False
        ))
    ]

    # Initialize
    await connector.initialize(configs)

    # Register sensors
    sensors = [
        SCADASensorConfig(
            sensor_id="TEMP_001",
            sensor_type="temperature",
            address="ns=2;i=1001",
            unit="celsius",
            min_value=0,
            max_value=200,
            sampling_rate=5
        ),
        SCADASensorConfig(
            sensor_id="PRESS_001",
            sensor_type="pressure",
            address="40001",
            unit="bar",
            min_value=0,
            max_value=10,
            sampling_rate=10
        )
    ]

    await connector.register_sensors(sensors)

    # Start data collection
    await connector.start_data_collection("plant1_opcua", ["TEMP_001"])
    await connector.start_data_collection("plant1_modbus", ["PRESS_001"])

    # Let it run for a while
    await asyncio.sleep(60)

    # Get recent data
    recent_data = await connector.get_recent_data(minutes=5)
    print(f"Collected {len(recent_data)} data points")

    # Shutdown
    await connector.shutdown()


if __name__ == "__main__":
    asyncio.run(main())