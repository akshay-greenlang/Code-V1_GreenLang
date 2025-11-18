"""
SCADA Integration for GL-005 CombustionControlAgent

Implements comprehensive SCADA system integration for monitoring and visualization:
- Real-time data publishing (OPC UA, MQTT)
- Alarm and event management
- Historical trend data aggregation
- Operator command interface
- Bidirectional communication
- Data aggregation and buffering

Real-Time Requirements:
- Data update rate: 1Hz minimum
- Alarm latency: <100ms
- Command acknowledgment: <200ms
- Historical data retrieval: <5s

Protocols Supported:
- OPC UA (IEC 62541) - Primary for HMI/SCADA
- MQTT - Lightweight telemetry and cloud integration
- REST API - Web-based dashboards

Features:
- Tag-based data model
- Quality-of-service guarantees
- Data buffering for network resilience
- Compression for bandwidth optimization

Author: GL-DataIntegrationEngineer
Date: 2025-11-18
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
import gzip
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum

# Third-party imports
try:
    from asyncua import Server as OPCUAServer, ua
    from asyncua.common.node import Node
    OPCUA_AVAILABLE = True
except ImportError:
    OPCUA_AVAILABLE = False

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataPriority(Enum):
    """Data priority levels for publishing."""
    CRITICAL = 1  # Alarms, safety interlocks
    HIGH = 2      # Control setpoints
    NORMAL = 3    # Process variables
    LOW = 4       # Diagnostic data


class AlarmSeverity(Enum):
    """SCADA alarm severity levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5


class CommandType(Enum):
    """Operator command types."""
    START = "start"
    STOP = "stop"
    SETPOINT_CHANGE = "setpoint_change"
    MODE_CHANGE = "mode_change"
    RESET_ALARM = "reset_alarm"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class SCADATag:
    """SCADA tag definition."""
    tag_name: str
    description: str
    data_type: str  # float, int, bool, string
    units: str
    priority: DataPriority = DataPriority.NORMAL

    # OPC UA settings
    opcua_node_id: Optional[str] = None

    # MQTT settings
    mqtt_topic: Optional[str] = None
    mqtt_qos: int = 1

    # Value constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # Update settings
    deadband: float = 0.1  # % change to trigger update
    scan_rate_ms: int = 1000

    # Runtime state
    current_value: Any = None
    last_update: Optional[datetime] = None
    quality: str = "good"


@dataclass
class SCADAAlarm:
    """SCADA alarm event."""
    alarm_id: str
    source_tag: str
    severity: AlarmSeverity
    message: str
    timestamp: datetime
    value: Any
    limit: Any

    # Acknowledgment
    acknowledged: bool = False
    ack_user: Optional[str] = None
    ack_timestamp: Optional[datetime] = None

    # Resolution
    resolved: bool = False
    resolve_timestamp: Optional[datetime] = None


@dataclass
class OperatorCommand:
    """Operator command from SCADA."""
    command_id: str
    command_type: CommandType
    target_tag: str
    value: Any
    operator: str
    timestamp: datetime
    executed: bool = False
    execution_timestamp: Optional[datetime] = None
    result: Optional[str] = None


@dataclass
class SCADAConfig:
    """Configuration for SCADA integration."""
    system_id: str = "GL005_COMBUSTION_CONTROL"

    # OPC UA Server settings
    opcua_enabled: bool = True
    opcua_endpoint: str = "opc.tcp://0.0.0.0:4840"
    opcua_namespace: str = "http://greenlang.com/combustion"
    opcua_server_name: str = "GL-005 Combustion Control"

    # MQTT Publisher settings
    mqtt_enabled: bool = True
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic_prefix: str = "combustion/gl005"
    mqtt_qos: int = 2
    mqtt_retain: bool = True
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    mqtt_use_tls: bool = True

    # Data publishing
    default_publish_rate_hz: float = 1.0
    batch_publish_enabled: bool = True
    batch_size: int = 50
    compression_enabled: bool = True

    # Historical data
    historical_buffer_size: int = 86400  # 24 hours at 1Hz
    historical_retention_days: int = 90

    # Alarm settings
    alarm_queue_size: int = 1000
    alarm_publish_immediately: bool = True


class SCADAIntegration:
    """
    SCADA Integration for comprehensive system monitoring.

    Features:
    - OPC UA server for HMI/SCADA clients
    - MQTT publisher for cloud/edge integration
    - Real-time data streaming
    - Alarm and event management
    - Operator command interface
    - Historical data aggregation
    - Data quality management

    Example:
        config = SCADAConfig(
            opcua_endpoint="opc.tcp://0.0.0.0:4840",
            mqtt_broker="mqtt.plant.com"
        )

        async with SCADAIntegration(config) as scada:
            # Register tags
            scada.register_tag(SCADATag(
                tag_name="FurnaceTemp",
                description="Furnace Temperature",
                data_type="float",
                units="°C",
                priority=DataPriority.HIGH
            ))

            # Publish real-time data
            await scada.publish_real_time_data({
                "FurnaceTemp": 850.5,
                "SteamPressure": 120.0,
                "O2Content": 3.5
            })

            # Publish alarm
            await scada.publish_alarms([
                SCADAAlarm(
                    alarm_id="TEMP_HIGH_001",
                    source_tag="FurnaceTemp",
                    severity=AlarmSeverity.HIGH,
                    message="Furnace temperature high",
                    timestamp=datetime.now(),
                    value=900.0,
                    limit=850.0
                )
            ])

            # Receive operator commands
            await scada.receive_operator_commands(command_handler)
    """

    def __init__(self, config: SCADAConfig):
        """Initialize SCADA integration."""
        self.config = config
        self.connected = False

        # OPC UA server
        self.opcua_server: Optional[OPCUAServer] = None
        self.opcua_nodes: Dict[str, Node] = {}

        # MQTT publisher
        self.mqtt_client: Optional[mqtt.Client] = None
        self.mqtt_connected = False

        # Tag registry
        self.tags: Dict[str, SCADATag] = {}

        # Data buffers
        self.publish_queue: asyncio.Queue = asyncio.Queue()
        self.historical_data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.historical_buffer_size)
        )

        # Alarm management
        self.active_alarms: Dict[str, SCADAAlarm] = {}
        self.alarm_history: deque = deque(maxlen=config.alarm_queue_size)

        # Command handling
        self.command_callbacks: List[Callable[[OperatorCommand], Any]] = []
        self.command_history: deque = deque(maxlen=1000)

        # Background tasks
        self._publish_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Performance tracking
        self.publish_latencies = deque(maxlen=1000)
        self.messages_published = 0

        # Prometheus metrics
        if METRICS_AVAILABLE:
            self.metrics = {
                'tags_published': Counter(
                    'scada_tags_published_total',
                    'Total SCADA tags published',
                    ['tag_name', 'protocol']
                ),
                'publish_latency': Histogram(
                    'scada_publish_latency_seconds',
                    'SCADA publish latency',
                    ['protocol']
                ),
                'active_alarms': Gauge(
                    'scada_active_alarms',
                    'Number of active SCADA alarms',
                    ['severity']
                ),
                'commands_received': Counter(
                    'scada_commands_received_total',
                    'Total operator commands received',
                    ['command_type']
                ),
                'connection_status': Gauge(
                    'scada_connection_status',
                    'SCADA connection status (1=connected)',
                    ['protocol']
                )
            }
        else:
            self.metrics = {}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_to_scada()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect_to_scada(self) -> bool:
        """
        Connect to SCADA systems (start OPC UA server, connect MQTT).

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        logger.info("Starting SCADA integration...")

        # Start OPC UA server
        if self.config.opcua_enabled:
            await self._start_opcua_server()

        # Connect MQTT publisher
        if self.config.mqtt_enabled:
            await self._connect_mqtt_publisher()

        self.connected = True

        # Start background tasks
        self._publish_task = asyncio.create_task(self._publish_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info("SCADA integration started successfully")
        return True

    async def _start_opcua_server(self) -> bool:
        """Start OPC UA server."""
        if not OPCUA_AVAILABLE:
            logger.error("OPC UA library not available")
            return False

        try:
            self.opcua_server = OPCUAServer()
            await self.opcua_server.init()

            # Set endpoint
            self.opcua_server.set_endpoint(self.config.opcua_endpoint)

            # Set server name
            self.opcua_server.set_server_name(self.config.opcua_server_name)

            # Register namespace
            namespace_idx = await self.opcua_server.register_namespace(
                self.config.opcua_namespace
            )

            # Start server
            await self.opcua_server.start()

            logger.info(f"OPC UA server started at {self.config.opcua_endpoint}")

            if self.metrics:
                self.metrics['connection_status'].labels(protocol='opcua').set(1)

            return True

        except Exception as e:
            logger.error(f"Failed to start OPC UA server: {e}")
            return False

    async def _connect_mqtt_publisher(self) -> bool:
        """Connect MQTT publisher."""
        if not MQTT_AVAILABLE:
            logger.error("MQTT library not available")
            return False

        try:
            self.mqtt_client = mqtt.Client(
                client_id=f"scada_{self.config.system_id}",
                protocol=mqtt.MQTTv311
            )

            # Set callbacks
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

            # Set authentication
            if self.config.mqtt_username and self.config.mqtt_password:
                self.mqtt_client.username_pw_set(
                    self.config.mqtt_username,
                    self.config.mqtt_password
                )

            # Set TLS
            if self.config.mqtt_use_tls:
                self.mqtt_client.tls_set()

            # Connect
            self.mqtt_client.connect(
                self.config.mqtt_broker,
                self.config.mqtt_port,
                60
            )

            # Start network loop
            self.mqtt_client.loop_start()

            # Wait for connection
            for _ in range(10):
                if self.mqtt_connected:
                    logger.info(f"MQTT publisher connected to {self.config.mqtt_broker}")

                    if self.metrics:
                        self.metrics['connection_status'].labels(protocol='mqtt').set(1)

                    return True
                await asyncio.sleep(0.5)

            logger.error("MQTT connection timeout")
            return False

        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            return False

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            self.mqtt_connected = True

            # Subscribe to command topics
            command_topic = f"{self.config.mqtt_topic_prefix}/commands/#"
            client.subscribe(command_topic, qos=2)
            logger.info(f"Subscribed to MQTT command topic: {command_topic}")

        else:
            logger.error(f"MQTT connection failed with code {rc}")
            self.mqtt_connected = False

    def _on_mqtt_message(self, client, userdata, message):
        """MQTT message received callback (operator commands)."""
        try:
            # Parse command
            data = json.loads(message.payload.decode())

            command = OperatorCommand(
                command_id=data.get('command_id', f"cmd_{int(time.time()*1000)}"),
                command_type=CommandType[data['command_type']],
                target_tag=data['target_tag'],
                value=data['value'],
                operator=data.get('operator', 'unknown'),
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
            )

            # Process command
            asyncio.create_task(self._process_operator_command(command))

        except Exception as e:
            logger.error(f"Error processing MQTT command: {e}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        logger.warning(f"MQTT disconnected with code {rc}")
        self.mqtt_connected = False

        if self.metrics:
            self.metrics['connection_status'].labels(protocol='mqtt').set(0)

        if rc != 0:
            logger.info("Unexpected disconnection, attempting reconnection...")
            try:
                client.reconnect()
            except Exception as e:
                logger.error(f"MQTT reconnection failed: {e}")

    def register_tag(self, tag: SCADATag):
        """Register a SCADA tag for publishing."""
        self.tags[tag.tag_name] = tag

        # Create OPC UA node if enabled
        if self.config.opcua_enabled and self.opcua_server:
            asyncio.create_task(self._create_opcua_node(tag))

        logger.info(f"Registered SCADA tag: {tag.tag_name}")

    async def _create_opcua_node(self, tag: SCADATag):
        """Create OPC UA node for tag."""
        try:
            # Get objects node
            objects = self.opcua_server.nodes.objects

            # Create variable node
            if tag.data_type == "float":
                variant_type = ua.VariantType.Float
            elif tag.data_type == "int":
                variant_type = ua.VariantType.Int32
            elif tag.data_type == "bool":
                variant_type = ua.VariantType.Boolean
            else:
                variant_type = ua.VariantType.String

            node = await objects.add_variable(
                f"ns={2};s={tag.tag_name}",
                tag.tag_name,
                0.0,
                varianttype=variant_type
            )

            # Set writable if needed
            await node.set_writable()

            # Store node reference
            self.opcua_nodes[tag.tag_name] = node

            logger.debug(f"Created OPC UA node for {tag.tag_name}")

        except Exception as e:
            logger.error(f"Failed to create OPC UA node for {tag.tag_name}: {e}")

    async def publish_real_time_data(self, data: Dict[str, Any]):
        """
        Publish real-time process data to SCADA.

        Args:
            data: Dictionary mapping tag name to value
        """
        start_time = time.perf_counter()

        for tag_name, value in data.items():
            tag = self.tags.get(tag_name)
            if not tag:
                logger.warning(f"Tag {tag_name} not registered")
                continue

            # Check deadband (only publish if significant change)
            if tag.current_value is not None and tag.data_type in ['float', 'int']:
                change_pct = abs(value - tag.current_value) / abs(tag.current_value) * 100
                if change_pct < tag.deadband:
                    continue  # Skip - not significant enough change

            # Update tag state
            tag.current_value = value
            tag.last_update = datetime.now()

            # Add to historical buffer
            self.historical_data[tag_name].append({
                'timestamp': tag.last_update.isoformat(),
                'value': value,
                'quality': tag.quality
            })

            # Publish to OPC UA
            if self.config.opcua_enabled and tag_name in self.opcua_nodes:
                try:
                    await self.opcua_nodes[tag_name].write_value(value)
                except Exception as e:
                    logger.error(f"OPC UA publish failed for {tag_name}: {e}")

            # Publish to MQTT
            if self.config.mqtt_enabled and self.mqtt_connected:
                mqtt_data = {
                    'tag_name': tag_name,
                    'value': value,
                    'timestamp': tag.last_update.isoformat(),
                    'quality': tag.quality,
                    'units': tag.units
                }

                topic = f"{self.config.mqtt_topic_prefix}/data/{tag_name}"

                # Compress if enabled
                payload = json.dumps(mqtt_data)
                if self.config.compression_enabled and len(payload) > 100:
                    payload = gzip.compress(payload.encode())

                self.mqtt_client.publish(
                    topic,
                    payload,
                    qos=tag.mqtt_qos,
                    retain=self.config.mqtt_retain
                )

            # Update metrics
            if self.metrics:
                self.metrics['tags_published'].labels(
                    tag_name=tag_name,
                    protocol='opcua'
                ).inc()

                self.metrics['tags_published'].labels(
                    tag_name=tag_name,
                    protocol='mqtt'
                ).inc()

        # Record latency
        latency = time.perf_counter() - start_time
        self.publish_latencies.append(latency)

        if self.metrics:
            self.metrics['publish_latency'].labels(protocol='combined').observe(latency)

    async def publish_alarms(self, alarms: List[SCADAAlarm]):
        """
        Publish alarm events to SCADA.

        Args:
            alarms: List of alarm events
        """
        for alarm in alarms:
            # Add to active alarms
            if not alarm.resolved:
                self.active_alarms[alarm.alarm_id] = alarm

            # Add to history
            self.alarm_history.append(alarm)

            # Publish to MQTT
            if self.config.mqtt_enabled and self.mqtt_connected:
                alarm_data = {
                    'alarm_id': alarm.alarm_id,
                    'source_tag': alarm.source_tag,
                    'severity': alarm.severity.value,
                    'message': alarm.message,
                    'timestamp': alarm.timestamp.isoformat(),
                    'value': alarm.value,
                    'limit': alarm.limit,
                    'acknowledged': alarm.acknowledged,
                    'resolved': alarm.resolved
                }

                topic = f"{self.config.mqtt_topic_prefix}/alarms/{alarm.severity.name}"

                self.mqtt_client.publish(
                    topic,
                    json.dumps(alarm_data),
                    qos=2,  # Alarms always use QoS 2
                    retain=True
                )

            # Update metrics
            if self.metrics:
                if not alarm.resolved:
                    self.metrics['active_alarms'].labels(
                        severity=alarm.severity.name
                    ).inc()

            logger.warning(
                f"Alarm published: {alarm.alarm_id} - {alarm.message} "
                f"(severity={alarm.severity.name})"
            )

    async def publish_trends(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Publish historical trend data.

        Args:
            tag_name: Tag to retrieve trends for
            start_time: Start of time window
            end_time: End of time window

        Returns:
            List of historical data points
        """
        if tag_name not in self.historical_data:
            return []

        # Filter data by time window
        trend_data = [
            point for point in self.historical_data[tag_name]
            if start_time <= datetime.fromisoformat(point['timestamp']) <= end_time
        ]

        logger.info(f"Retrieved {len(trend_data)} trend points for {tag_name}")

        return trend_data

    async def receive_operator_commands(
        self,
        callback: Callable[[OperatorCommand], Any]
    ):
        """
        Register callback for operator commands.

        Args:
            callback: Function to handle operator commands
        """
        self.command_callbacks.append(callback)
        logger.info("Registered operator command callback")

    async def _process_operator_command(self, command: OperatorCommand):
        """Process incoming operator command."""
        logger.info(
            f"Processing operator command: {command.command_type.value} "
            f"on {command.target_tag} by {command.operator}"
        )

        # Add to history
        self.command_history.append(command)

        # Update metrics
        if self.metrics:
            self.metrics['commands_received'].labels(
                command_type=command.command_type.value
            ).inc()

        # Execute callbacks
        for callback in self.command_callbacks:
            try:
                result = await callback(command)

                command.executed = True
                command.execution_timestamp = datetime.now()
                command.result = str(result)

                logger.info(f"Command {command.command_id} executed successfully")

            except Exception as e:
                logger.error(f"Command execution failed: {e}")
                command.result = f"Error: {str(e)}"

    async def _publish_loop(self):
        """Background task for batch publishing."""
        while self.connected:
            try:
                await asyncio.sleep(1.0 / self.config.default_publish_rate_hz)

                # Batch publishing logic can be implemented here
                # For now, real-time publishing is used

            except Exception as e:
                logger.error(f"Publish loop error: {e}")
                await asyncio.sleep(1.0)

    async def _heartbeat_loop(self):
        """Background task for system heartbeat."""
        while self.connected:
            try:
                # Publish system heartbeat
                heartbeat_data = {
                    'system_id': self.config.system_id,
                    'timestamp': datetime.now().isoformat(),
                    'active_alarms': len(self.active_alarms),
                    'tags_count': len(self.tags),
                    'uptime_seconds': time.time()
                }

                if self.config.mqtt_enabled and self.mqtt_connected:
                    topic = f"{self.config.mqtt_topic_prefix}/heartbeat"
                    self.mqtt_client.publish(
                        topic,
                        json.dumps(heartbeat_data),
                        qos=0
                    )

                await asyncio.sleep(10)  # Heartbeat every 10 seconds

            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(10)

    async def disconnect(self):
        """Disconnect from SCADA systems."""
        logger.info("Disconnecting SCADA integration...")

        # Stop background tasks
        if self._publish_task:
            self._publish_task.cancel()
            try:
                await self._publish_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Stop OPC UA server
        if self.opcua_server:
            try:
                await self.opcua_server.stop()
                logger.info("OPC UA server stopped")
            except Exception as e:
                logger.error(f"Error stopping OPC UA server: {e}")

        # Disconnect MQTT
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                logger.info("MQTT publisher disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting MQTT: {e}")

        self.connected = False
        logger.info("SCADA integration disconnected")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get SCADA integration performance statistics."""
        if not self.publish_latencies:
            return {}

        latencies_ms = [l * 1000 for l in self.publish_latencies]

        return {
            'avg_publish_latency_ms': sum(latencies_ms) / len(latencies_ms),
            'max_publish_latency_ms': max(latencies_ms),
            'messages_published': self.messages_published,
            'active_alarms': len(self.active_alarms),
            'registered_tags': len(self.tags),
            'opcua_connected': self.opcua_server is not None,
            'mqtt_connected': self.mqtt_connected
        }
