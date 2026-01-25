"""
IoT Sensor Connector Module for GL-013 PREDICTMAINT (Predictive Maintenance Agent).

Provides integration with IoT sensors and gateways for predictive maintenance data
collection. Supports MQTT broker connections, REST API for sensor gateways,
and various sensor types including vibration, temperature, pressure, current, and flow.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import asyncio
import json
import logging
import struct
import time
import uuid
from collections import deque

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    CircuitState,
    ConnectionState,
    ConnectorType,
    DataQualityLevel,
    DataQualityResult,
    HealthCheckResult,
    HealthStatus,
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    ConnectorError,
    ProtocolError,
    RateLimitError,
    TimeoutError,
    ValidationError,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class SensorType(str, Enum):
    """Types of IoT sensors supported."""

    VIBRATION_ACCELEROMETER = "vibration_accelerometer"
    VIBRATION_VELOCITY = "vibration_velocity"
    TEMPERATURE_RTD = "temperature_rtd"
    TEMPERATURE_THERMOCOUPLE = "temperature_thermocouple"
    TEMPERATURE_IR = "temperature_ir"
    PRESSURE_TRANSMITTER = "pressure_transmitter"
    PRESSURE_GAUGE = "pressure_gauge"
    CURRENT_CT = "current_ct"
    CURRENT_SHUNT = "current_shunt"
    FLOW_MAGNETIC = "flow_magnetic"
    FLOW_ULTRASONIC = "flow_ultrasonic"
    FLOW_VORTEX = "flow_vortex"
    LEVEL_ULTRASONIC = "level_ultrasonic"
    LEVEL_RADAR = "level_radar"
    HUMIDITY = "humidity"
    ULTRASONIC_ACOUSTIC = "ultrasonic_acoustic"
    OIL_PARTICLE_COUNTER = "oil_particle_counter"
    OIL_MOISTURE = "oil_moisture"
    VOLTAGE = "voltage"
    POWER = "power"


class SensorProtocol(str, Enum):
    """Communication protocols for IoT sensors."""

    MQTT = "mqtt"
    REST_API = "rest_api"
    COAP = "coap"
    LORAWAN = "lorawan"
    ZIGBEE = "zigbee"
    MODBUS_TCP = "modbus_tcp"
    OPC_UA = "opc_ua"


class MQTTQoS(int, Enum):
    """MQTT Quality of Service levels."""

    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1
    EXACTLY_ONCE = 2


class SensorStatus(str, Enum):
    """Sensor operational status."""

    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    LOW_BATTERY = "low_battery"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class DataFormat(str, Enum):
    """Sensor data payload formats."""

    JSON = "json"
    BINARY = "binary"
    CSV = "csv"
    PROTOBUF = "protobuf"
    MSGPACK = "msgpack"


class TimeSyncMode(str, Enum):
    """Timestamp synchronization modes."""

    SENSOR_TIMESTAMP = "sensor_timestamp"
    GATEWAY_TIMESTAMP = "gateway_timestamp"
    SERVER_TIMESTAMP = "server_timestamp"
    NTP_SYNC = "ntp_sync"


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class MQTTConfig(BaseModel):
    """MQTT connection configuration."""

    model_config = ConfigDict(extra="forbid")

    broker_host: str = Field(..., description="MQTT broker hostname")
    broker_port: int = Field(default=1883, ge=1, le=65535, description="MQTT broker port")
    use_tls: bool = Field(default=False, description="Use TLS encryption")
    use_websockets: bool = Field(default=False, description="Use WebSocket transport")

    # Authentication
    username: Optional[str] = Field(default=None, description="MQTT username")
    password: Optional[str] = Field(default=None, description="MQTT password")
    client_id: str = Field(
        default_factory=lambda: f"gl-013-{uuid.uuid4().hex[:8]}",
        description="MQTT client ID"
    )

    # TLS settings
    ca_cert_path: Optional[str] = Field(default=None, description="CA certificate path")
    client_cert_path: Optional[str] = Field(default=None, description="Client certificate path")
    client_key_path: Optional[str] = Field(default=None, description="Client private key path")

    # Connection settings
    keepalive_seconds: int = Field(default=60, ge=10, le=600, description="Keepalive interval")
    clean_session: bool = Field(default=True, description="Clean session on connect")
    reconnect_on_failure: bool = Field(default=True, description="Auto-reconnect on failure")
    reconnect_delay_seconds: float = Field(default=5.0, ge=1.0, le=60.0, description="Reconnect delay")

    # QoS settings
    default_qos: MQTTQoS = Field(default=MQTTQoS.AT_LEAST_ONCE, description="Default QoS level")
    retain_messages: bool = Field(default=False, description="Retain messages")

    # Topic configuration
    topic_prefix: str = Field(default="sensors/", description="Topic prefix for all subscriptions")
    will_topic: Optional[str] = Field(default=None, description="Last will topic")
    will_message: Optional[str] = Field(default=None, description="Last will message")


class SensorGatewayConfig(BaseModel):
    """Sensor gateway REST API configuration."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(..., description="Gateway API base URL")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    username: Optional[str] = Field(default=None, description="Username for auth")
    password: Optional[str] = Field(default=None, description="Password for auth")

    # Request settings
    timeout_seconds: float = Field(default=30.0, ge=5.0, le=120.0, description="Request timeout")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")

    # Polling settings
    poll_interval_seconds: float = Field(default=60.0, ge=1.0, le=3600.0, description="Poll interval")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Batch size for requests")


class IoTSensorConnectorConfig(BaseConnectorConfig):
    """Configuration for IoT sensor connector."""

    model_config = ConfigDict(extra="forbid")

    # Protocol settings
    protocol: SensorProtocol = Field(default=SensorProtocol.MQTT, description="Communication protocol")

    # Protocol-specific configuration
    mqtt_config: Optional[MQTTConfig] = Field(default=None, description="MQTT configuration")
    gateway_config: Optional[SensorGatewayConfig] = Field(default=None, description="Gateway configuration")

    # Sensor settings
    supported_sensor_types: List[SensorType] = Field(
        default_factory=lambda: list(SensorType),
        description="Supported sensor types"
    )
    sensor_timeout_seconds: float = Field(
        default=300.0,
        ge=60.0,
        le=3600.0,
        description="Sensor offline threshold"
    )

    # Data handling
    data_format: DataFormat = Field(default=DataFormat.JSON, description="Expected data format")
    time_sync_mode: TimeSyncMode = Field(
        default=TimeSyncMode.SERVER_TIMESTAMP,
        description="Timestamp synchronization mode"
    )
    max_timestamp_drift_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description="Maximum allowed timestamp drift"
    )

    # Buffer settings
    buffer_size: int = Field(default=10000, ge=100, le=1000000, description="Message buffer size")
    buffer_flush_interval_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Buffer flush interval"
    )

    # Data quality
    validate_readings: bool = Field(default=True, description="Validate sensor readings")
    reject_outliers: bool = Field(default=True, description="Reject outlier readings")
    outlier_threshold_std: float = Field(
        default=4.0,
        ge=2.0,
        le=10.0,
        description="Outlier threshold (std dev)"
    )
    min_quality_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum data quality score"
    )

    # Batch retrieval
    enable_batch_retrieval: bool = Field(default=True, description="Enable batch data retrieval")
    batch_retrieval_limit: int = Field(default=1000, ge=100, le=100000, description="Batch retrieval limit")

    @field_validator('connector_type', mode='before')
    @classmethod
    def set_connector_type(cls, v):
        return ConnectorType.IOT_SENSOR


# =============================================================================
# Pydantic Models - Data Objects
# =============================================================================


class IoTSensor(BaseModel):
    """IoT sensor registration model."""

    model_config = ConfigDict(extra="allow")

    sensor_id: str = Field(..., description="Unique sensor identifier")
    sensor_name: str = Field(..., description="Human-readable sensor name")
    sensor_type: SensorType = Field(..., description="Type of sensor")
    description: Optional[str] = Field(default=None, description="Sensor description")

    # Equipment association
    equipment_id: str = Field(..., description="Associated equipment ID")
    equipment_name: Optional[str] = Field(default=None, description="Equipment name")
    location: Optional[str] = Field(default=None, description="Physical location")
    installation_point: Optional[str] = Field(default=None, description="Installation point description")

    # Technical specifications
    manufacturer: Optional[str] = Field(default=None, description="Sensor manufacturer")
    model: Optional[str] = Field(default=None, description="Sensor model")
    serial_number: Optional[str] = Field(default=None, description="Serial number")
    firmware_version: Optional[str] = Field(default=None, description="Firmware version")

    # Measurement configuration
    measurement_unit: str = Field(..., description="Measurement unit")
    measurement_range_min: Optional[float] = Field(default=None, description="Minimum measurement range")
    measurement_range_max: Optional[float] = Field(default=None, description="Maximum measurement range")
    resolution: Optional[float] = Field(default=None, description="Measurement resolution")
    accuracy: Optional[float] = Field(default=None, description="Measurement accuracy")
    sampling_rate_hz: float = Field(default=1.0, ge=0.001, le=100000, description="Sampling rate (Hz)")

    # Alarm thresholds
    low_low_threshold: Optional[float] = Field(default=None, description="Low-low alarm threshold")
    low_threshold: Optional[float] = Field(default=None, description="Low alarm threshold")
    high_threshold: Optional[float] = Field(default=None, description="High alarm threshold")
    high_high_threshold: Optional[float] = Field(default=None, description="High-high alarm threshold")

    # Communication
    protocol: SensorProtocol = Field(default=SensorProtocol.MQTT, description="Communication protocol")
    mqtt_topic: Optional[str] = Field(default=None, description="MQTT topic for this sensor")
    gateway_id: Optional[str] = Field(default=None, description="Gateway ID if applicable")

    # Status
    status: SensorStatus = Field(default=SensorStatus.UNKNOWN, description="Current status")
    last_seen: Optional[datetime] = Field(default=None, description="Last data received timestamp")
    battery_level: Optional[float] = Field(default=None, ge=0, le=100, description="Battery level %")
    signal_strength: Optional[int] = Field(default=None, description="Signal strength (dBm)")

    # Calibration
    calibration_date: Optional[datetime] = Field(default=None, description="Last calibration date")
    calibration_due_date: Optional[datetime] = Field(default=None, description="Next calibration due")
    calibration_offset: float = Field(default=0.0, description="Calibration offset")
    calibration_scale: float = Field(default=1.0, description="Calibration scale factor")

    # Metadata
    tags: List[str] = Field(default_factory=list, description="Sensor tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SensorReading(BaseModel):
    """Single sensor reading with full context."""

    model_config = ConfigDict(extra="allow")

    reading_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique reading ID"
    )
    sensor_id: str = Field(..., description="Sensor ID")
    equipment_id: str = Field(..., description="Equipment ID")
    sensor_type: SensorType = Field(..., description="Sensor type")

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Reading timestamp")
    received_at: datetime = Field(default_factory=datetime.utcnow, description="Server receive time")
    gateway_timestamp: Optional[datetime] = Field(default=None, description="Gateway timestamp")
    sensor_timestamp: Optional[datetime] = Field(default=None, description="Sensor timestamp")

    # Value
    value: float = Field(..., description="Measured value")
    raw_value: Optional[float] = Field(default=None, description="Raw sensor value before calibration")
    unit: str = Field(..., description="Measurement unit")

    # Quality
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Data quality score")
    is_valid: bool = Field(default=True, description="Reading validity")
    validation_status: str = Field(default="valid", description="Validation status message")
    quality_flags: List[str] = Field(default_factory=list, description="Quality issue flags")

    # Alarm status
    alarm_state: Optional[str] = Field(default=None, description="Alarm state if triggered")
    threshold_exceeded: Optional[str] = Field(default=None, description="Threshold that was exceeded")

    # Sensor context
    battery_level: Optional[float] = Field(default=None, description="Battery level at reading time")
    signal_strength: Optional[int] = Field(default=None, description="Signal strength at reading time")

    # Source information
    source_topic: Optional[str] = Field(default=None, description="MQTT topic if applicable")
    gateway_id: Optional[str] = Field(default=None, description="Gateway ID if applicable")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SensorBatch(BaseModel):
    """Batch of sensor readings."""

    model_config = ConfigDict(extra="allow")

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Batch ID"
    )
    sensor_id: str = Field(..., description="Sensor ID")
    equipment_id: str = Field(..., description="Equipment ID")

    # Time range
    start_time: datetime = Field(..., description="Batch start time")
    end_time: datetime = Field(..., description="Batch end time")

    # Readings
    readings: List[SensorReading] = Field(..., description="Readings in batch")
    reading_count: int = Field(..., ge=0, description="Number of readings")

    # Statistics
    min_value: Optional[float] = Field(default=None, description="Minimum value")
    max_value: Optional[float] = Field(default=None, description="Maximum value")
    avg_value: Optional[float] = Field(default=None, description="Average value")
    std_dev: Optional[float] = Field(default=None, description="Standard deviation")

    # Quality summary
    valid_readings: int = Field(default=0, ge=0, description="Number of valid readings")
    invalid_readings: int = Field(default=0, ge=0, description="Number of invalid readings")
    avg_quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Average quality score")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GatewayStatus(BaseModel):
    """IoT gateway status."""

    model_config = ConfigDict(extra="allow")

    gateway_id: str = Field(..., description="Gateway ID")
    gateway_name: str = Field(..., description="Gateway name")
    status: str = Field(..., description="Gateway status")

    # Connection info
    ip_address: Optional[str] = Field(default=None, description="IP address")
    mac_address: Optional[str] = Field(default=None, description="MAC address")
    firmware_version: Optional[str] = Field(default=None, description="Firmware version")

    # Statistics
    connected_sensors: int = Field(default=0, ge=0, description="Number of connected sensors")
    messages_per_minute: float = Field(default=0.0, ge=0, description="Messages per minute")
    uptime_seconds: Optional[int] = Field(default=None, ge=0, description="Uptime in seconds")

    # Health
    cpu_usage_percent: Optional[float] = Field(default=None, ge=0, le=100, description="CPU usage %")
    memory_usage_percent: Optional[float] = Field(default=None, ge=0, le=100, description="Memory usage %")
    disk_usage_percent: Optional[float] = Field(default=None, ge=0, le=100, description="Disk usage %")

    last_heartbeat: Optional[datetime] = Field(default=None, description="Last heartbeat timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# MQTT Client Wrapper
# =============================================================================


class MQTTClientWrapper:
    """Wrapper for MQTT client operations."""

    def __init__(self, config: MQTTConfig) -> None:
        """Initialize MQTT client wrapper."""
        self._config = config
        self._client = None
        self._connected = False
        self._lock = asyncio.Lock()

        # Message handling
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._message_handlers: Dict[str, List[Callable]] = {}
        self._subscriptions: Set[str] = set()

        # Reconnection
        self._reconnect_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Establish MQTT connection."""
        try:
            import asyncio_mqtt as aiomqtt

            # Build connection parameters
            connect_kwargs = {
                "hostname": self._config.broker_host,
                "port": self._config.broker_port,
                "keepalive": self._config.keepalive_seconds,
                "clean_session": self._config.clean_session,
            }

            if self._config.username:
                connect_kwargs["username"] = self._config.username
            if self._config.password:
                connect_kwargs["password"] = self._config.password
            if self._config.client_id:
                connect_kwargs["client_id"] = self._config.client_id

            # TLS configuration
            if self._config.use_tls:
                import ssl
                tls_context = ssl.create_default_context()
                if self._config.ca_cert_path:
                    tls_context.load_verify_locations(self._config.ca_cert_path)
                if self._config.client_cert_path and self._config.client_key_path:
                    tls_context.load_cert_chain(
                        self._config.client_cert_path,
                        self._config.client_key_path
                    )
                connect_kwargs["tls_context"] = tls_context

            self._client = aiomqtt.Client(**connect_kwargs)
            await self._client.__aenter__()
            self._connected = True

            logger.info(f"MQTT connected to {self._config.broker_host}:{self._config.broker_port}")

        except ImportError:
            raise ConfigurationError("asyncio-mqtt package not installed. Install with: pip install asyncio-mqtt")
        except Exception as e:
            raise ConnectionError(f"MQTT connection failed: {str(e)}")

    async def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        if self._client and self._connected:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error during MQTT disconnect: {e}")
            finally:
                self._connected = False

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[str, bytes], None],
        qos: MQTTQoS = MQTTQoS.AT_LEAST_ONCE,
    ) -> None:
        """
        Subscribe to MQTT topic.

        Args:
            topic: Topic pattern to subscribe to
            handler: Callback function for messages
            qos: Quality of service level
        """
        if not self._connected:
            raise ConnectionError("MQTT not connected")

        await self._client.subscribe(topic, qos=qos.value)
        self._subscriptions.add(topic)

        if topic not in self._message_handlers:
            self._message_handlers[topic] = []
        self._message_handlers[topic].append(handler)

        logger.info(f"Subscribed to MQTT topic: {topic}")

    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from topic."""
        if not self._connected:
            return

        await self._client.unsubscribe(topic)
        self._subscriptions.discard(topic)

        if topic in self._message_handlers:
            del self._message_handlers[topic]

    async def publish(
        self,
        topic: str,
        payload: Union[str, bytes, Dict],
        qos: MQTTQoS = MQTTQoS.AT_LEAST_ONCE,
        retain: bool = False,
    ) -> None:
        """
        Publish message to topic.

        Args:
            topic: Target topic
            payload: Message payload
            qos: Quality of service
            retain: Retain message on broker
        """
        if not self._connected:
            raise ConnectionError("MQTT not connected")

        if isinstance(payload, dict):
            payload = json.dumps(payload).encode()
        elif isinstance(payload, str):
            payload = payload.encode()

        await self._client.publish(topic, payload, qos=qos.value, retain=retain)

    async def listen(self) -> AsyncGenerator[Tuple[str, bytes], None]:
        """
        Listen for messages on subscribed topics.

        Yields:
            Tuple of (topic, payload)
        """
        if not self._connected:
            raise ConnectionError("MQTT not connected")

        async with self._client.messages() as messages:
            async for message in messages:
                yield (str(message.topic), message.payload)

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# =============================================================================
# Sensor Data Validator
# =============================================================================


class SensorDataValidator:
    """Validates sensor readings for data quality."""

    def __init__(
        self,
        reject_outliers: bool = True,
        outlier_threshold_std: float = 4.0,
        max_timestamp_drift_seconds: float = 60.0,
    ) -> None:
        """Initialize validator."""
        self._reject_outliers = reject_outliers
        self._outlier_threshold = outlier_threshold_std
        self._max_timestamp_drift = max_timestamp_drift_seconds

        # Historical data for outlier detection
        self._history: Dict[str, deque] = {}
        self._history_size = 100

    def validate(
        self,
        reading: SensorReading,
        sensor: Optional[IoTSensor] = None,
    ) -> Tuple[bool, str, float, List[str]]:
        """
        Validate a sensor reading.

        Returns:
            Tuple of (is_valid, status_message, quality_score, quality_flags)
        """
        quality_score = 1.0
        quality_flags = []

        # Check for null/nan values
        if reading.value is None:
            return False, "null_value", 0.0, ["null_value"]

        import math
        if math.isnan(reading.value) or math.isinf(reading.value):
            return False, "invalid_number", 0.0, ["invalid_number"]

        # Check timestamp drift
        if reading.sensor_timestamp:
            drift = abs((reading.received_at - reading.sensor_timestamp).total_seconds())
            if drift > self._max_timestamp_drift:
                quality_score *= 0.8
                quality_flags.append("timestamp_drift")

        # Check range limits if sensor config available
        if sensor:
            if sensor.measurement_range_min is not None:
                if reading.value < sensor.measurement_range_min:
                    quality_score *= 0.5
                    quality_flags.append("below_range")

            if sensor.measurement_range_max is not None:
                if reading.value > sensor.measurement_range_max:
                    quality_score *= 0.5
                    quality_flags.append("above_range")

        # Check for outliers
        if self._reject_outliers:
            sensor_id = reading.sensor_id
            if sensor_id not in self._history:
                self._history[sensor_id] = deque(maxlen=self._history_size)

            history = self._history[sensor_id]
            if len(history) >= 10:
                import statistics
                mean = statistics.mean(history)
                std_dev = statistics.stdev(history) if len(history) > 1 else 0

                if std_dev > 0:
                    z_score = abs(reading.value - mean) / std_dev
                    if z_score > self._outlier_threshold:
                        quality_score *= 0.3
                        quality_flags.append("statistical_outlier")

            history.append(reading.value)

        # Check sensor health indicators
        if reading.battery_level is not None and reading.battery_level < 20:
            quality_score *= 0.9
            quality_flags.append("low_battery")

        if reading.signal_strength is not None and reading.signal_strength < -90:
            quality_score *= 0.9
            quality_flags.append("weak_signal")

        # Determine status
        if quality_score >= 0.9:
            status = "valid"
        elif quality_score >= 0.5:
            status = f"degraded:{','.join(quality_flags)}"
        else:
            status = f"poor:{','.join(quality_flags)}"

        return quality_score >= 0.5, status, quality_score, quality_flags


# =============================================================================
# Message Parser
# =============================================================================


class SensorMessageParser:
    """Parses sensor messages from various formats."""

    def __init__(self, data_format: DataFormat = DataFormat.JSON) -> None:
        """Initialize parser."""
        self._format = data_format

    def parse(
        self,
        payload: bytes,
        topic: Optional[str] = None,
        sensor: Optional[IoTSensor] = None,
    ) -> Optional[SensorReading]:
        """
        Parse sensor message payload.

        Args:
            payload: Raw message payload
            topic: MQTT topic if applicable
            sensor: Sensor configuration if known

        Returns:
            Parsed sensor reading or None if parsing fails
        """
        try:
            if self._format == DataFormat.JSON:
                return self._parse_json(payload, topic, sensor)
            elif self._format == DataFormat.BINARY:
                return self._parse_binary(payload, topic, sensor)
            elif self._format == DataFormat.MSGPACK:
                return self._parse_msgpack(payload, topic, sensor)
            else:
                logger.warning(f"Unsupported data format: {self._format}")
                return None
        except Exception as e:
            logger.warning(f"Failed to parse sensor message: {e}")
            return None

    def _parse_json(
        self,
        payload: bytes,
        topic: Optional[str],
        sensor: Optional[IoTSensor],
    ) -> Optional[SensorReading]:
        """Parse JSON format message."""
        data = json.loads(payload.decode('utf-8'))

        # Handle different JSON structures
        # Common fields to look for
        value = data.get('value') or data.get('v') or data.get('reading')
        sensor_id = data.get('sensor_id') or data.get('id') or data.get('sensorId')
        timestamp = data.get('timestamp') or data.get('ts') or data.get('time')

        if value is None and sensor_id is None:
            return None

        # Use sensor config if available
        if sensor:
            sensor_id = sensor_id or sensor.sensor_id
            equipment_id = sensor.equipment_id
            sensor_type = sensor.sensor_type
            unit = sensor.measurement_unit
        else:
            equipment_id = data.get('equipment_id', 'unknown')
            sensor_type = SensorType(data.get('type', 'vibration_accelerometer'))
            unit = data.get('unit', 'unknown')

        # Parse timestamp
        if timestamp:
            if isinstance(timestamp, (int, float)):
                sensor_timestamp = datetime.utcfromtimestamp(timestamp)
            else:
                sensor_timestamp = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
        else:
            sensor_timestamp = None

        return SensorReading(
            sensor_id=sensor_id,
            equipment_id=equipment_id,
            sensor_type=sensor_type,
            value=float(value),
            unit=unit,
            sensor_timestamp=sensor_timestamp,
            source_topic=topic,
            battery_level=data.get('battery') or data.get('battery_level'),
            signal_strength=data.get('rssi') or data.get('signal_strength'),
            metadata=data.get('metadata', {}),
        )

    def _parse_binary(
        self,
        payload: bytes,
        topic: Optional[str],
        sensor: Optional[IoTSensor],
    ) -> Optional[SensorReading]:
        """Parse binary format message."""
        if len(payload) < 8:
            return None

        # Assume simple binary format: sensor_id (4 bytes), value (4 bytes float)
        sensor_id_bytes = payload[:4]
        value_bytes = payload[4:8]

        sensor_id = sensor_id_bytes.hex()
        value = struct.unpack('>f', value_bytes)[0]

        if sensor:
            return SensorReading(
                sensor_id=sensor.sensor_id,
                equipment_id=sensor.equipment_id,
                sensor_type=sensor.sensor_type,
                value=value,
                unit=sensor.measurement_unit,
                raw_value=value,
                source_topic=topic,
            )

        return SensorReading(
            sensor_id=sensor_id,
            equipment_id='unknown',
            sensor_type=SensorType.VIBRATION_ACCELEROMETER,
            value=value,
            unit='unknown',
            source_topic=topic,
        )

    def _parse_msgpack(
        self,
        payload: bytes,
        topic: Optional[str],
        sensor: Optional[IoTSensor],
    ) -> Optional[SensorReading]:
        """Parse MessagePack format message."""
        try:
            import msgpack
            data = msgpack.unpackb(payload, raw=False)
            # Treat as dictionary and use JSON parser logic
            return self._parse_json(json.dumps(data).encode(), topic, sensor)
        except ImportError:
            raise ConfigurationError("msgpack package not installed")


# =============================================================================
# IoT Sensor Connector Implementation
# =============================================================================


class IoTSensorConnector(BaseConnector):
    """
    IoT Sensor Connector for predictive maintenance.

    Provides integration with IoT sensors and gateways:
    - MQTT broker connections
    - REST API for sensor gateways
    - Vibration sensors (accelerometers)
    - Temperature sensors (RTD, thermocouple)
    - Pressure sensors (transmitters)
    - Current sensors (CT)
    - Flow sensors

    Features:
    - Real-time data streaming via MQTT
    - Batch data retrieval via REST API
    - Data quality validation
    - Timestamp synchronization
    - Automatic sensor discovery
    """

    def __init__(self, config: IoTSensorConnectorConfig) -> None:
        """
        Initialize IoT sensor connector.

        Args:
            config: Connector configuration
        """
        super().__init__(config)
        self._iot_config = config

        # Protocol clients
        self._mqtt_client: Optional[MQTTClientWrapper] = None
        self._http_client = None

        # Data handling
        self._parser = SensorMessageParser(config.data_format)
        self._validator = SensorDataValidator(
            reject_outliers=config.reject_outliers,
            outlier_threshold_std=config.outlier_threshold_std,
            max_timestamp_drift_seconds=config.max_timestamp_drift_seconds,
        )

        # Sensor registry
        self._sensors: Dict[str, IoTSensor] = {}
        self._sensor_last_seen: Dict[str, datetime] = {}

        # Message buffer
        self._message_buffer: asyncio.Queue = asyncio.Queue(maxsize=config.buffer_size)

        # Streaming state
        self._streaming_task: Optional[asyncio.Task] = None
        self._streaming_callbacks: List[Callable[[SensorReading], None]] = []

        # Status check task
        self._status_check_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Establish connection to IoT sensor system."""
        self._logger.info(f"Connecting to IoT sensor system via {self._iot_config.protocol.value}...")

        if self._iot_config.protocol == SensorProtocol.MQTT:
            if not self._iot_config.mqtt_config:
                raise ConfigurationError("MQTT configuration required")

            self._mqtt_client = MQTTClientWrapper(self._iot_config.mqtt_config)
            await self._mqtt_client.connect()

            # Subscribe to sensor topics
            topic_prefix = self._iot_config.mqtt_config.topic_prefix
            await self._mqtt_client.subscribe(
                f"{topic_prefix}#",
                self._handle_mqtt_message,
                self._iot_config.mqtt_config.default_qos,
            )

            # Start message processing
            self._streaming_task = asyncio.create_task(self._mqtt_message_loop())

        elif self._iot_config.protocol == SensorProtocol.REST_API:
            if not self._iot_config.gateway_config:
                raise ConfigurationError("Gateway configuration required")

            import httpx
            self._http_client = httpx.AsyncClient(
                base_url=self._iot_config.gateway_config.base_url,
                timeout=httpx.Timeout(self._iot_config.gateway_config.timeout_seconds),
                verify=self._iot_config.gateway_config.verify_ssl,
            )

            # Test connection
            await self._test_gateway_connection()

        # Start sensor status monitoring
        self._status_check_task = asyncio.create_task(self._sensor_status_loop())

        # Discover sensors
        await self._discover_sensors()

        self._logger.info(
            f"Connected to IoT sensor system. {len(self._sensors)} sensors registered."
        )

    async def disconnect(self) -> None:
        """Disconnect from IoT sensor system."""
        # Stop background tasks
        for task in [self._streaming_task, self._status_check_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Disconnect clients
        if self._mqtt_client:
            await self._mqtt_client.disconnect()
            self._mqtt_client = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._logger.info("Disconnected from IoT sensor system")

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on connection."""
        start_time = time.time()

        try:
            if self._iot_config.protocol == SensorProtocol.MQTT:
                if not self._mqtt_client or not self._mqtt_client.is_connected:
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        message="MQTT client not connected",
                        latency_ms=(time.time() - start_time) * 1000,
                    )

            elif self._iot_config.protocol == SensorProtocol.REST_API:
                if not self._http_client:
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        message="HTTP client not initialized",
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                await self._test_gateway_connection()

            # Check sensor status
            online_sensors = sum(
                1 for s in self._sensors.values()
                if s.status == SensorStatus.ONLINE
            )
            total_sensors = len(self._sensors)

            latency_ms = (time.time() - start_time) * 1000

            if total_sensors == 0:
                status = HealthStatus.DEGRADED
                message = "No sensors registered"
            elif online_sensors == 0:
                status = HealthStatus.UNHEALTHY
                message = "All sensors offline"
            elif online_sensors < total_sensors:
                status = HealthStatus.DEGRADED
                message = f"{online_sensors}/{total_sensors} sensors online"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {total_sensors} sensors online"

            return HealthCheckResult(
                status=status,
                message=message,
                latency_ms=latency_ms,
                details={
                    "protocol": self._iot_config.protocol.value,
                    "total_sensors": total_sensors,
                    "online_sensors": online_sensors,
                    "buffer_size": self._message_buffer.qsize(),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def validate_configuration(self) -> bool:
        """Validate connector configuration."""
        if self._iot_config.protocol == SensorProtocol.MQTT:
            if not self._iot_config.mqtt_config:
                raise ConfigurationError("MQTT configuration required")
            if not self._iot_config.mqtt_config.broker_host:
                raise ConfigurationError("MQTT broker host required")

        elif self._iot_config.protocol == SensorProtocol.REST_API:
            if not self._iot_config.gateway_config:
                raise ConfigurationError("Gateway configuration required")
            if not self._iot_config.gateway_config.base_url:
                raise ConfigurationError("Gateway base URL required")

        return True

    async def _test_gateway_connection(self) -> None:
        """Test gateway API connection."""
        headers = {}
        if self._iot_config.gateway_config.api_key:
            headers["X-API-Key"] = self._iot_config.gateway_config.api_key

        response = await self._http_client.get("/health", headers=headers)
        response.raise_for_status()

    async def _discover_sensors(self) -> None:
        """Discover and register sensors."""
        if self._iot_config.protocol == SensorProtocol.REST_API:
            try:
                headers = {}
                if self._iot_config.gateway_config.api_key:
                    headers["X-API-Key"] = self._iot_config.gateway_config.api_key

                response = await self._http_client.get("/sensors", headers=headers)
                response.raise_for_status()
                data = response.json()

                for sensor_data in data.get("sensors", []):
                    sensor = IoTSensor(**sensor_data)
                    self._sensors[sensor.sensor_id] = sensor
                    self._logger.info(f"Discovered sensor: {sensor.sensor_id} ({sensor.sensor_type.value})")

            except Exception as e:
                self._logger.warning(f"Sensor discovery failed: {e}")

    # =========================================================================
    # MQTT Message Handling
    # =========================================================================

    async def _handle_mqtt_message(self, topic: str, payload: bytes) -> None:
        """Handle incoming MQTT message."""
        try:
            # Parse message
            reading = self._parser.parse(payload, topic)
            if reading is None:
                return

            # Get sensor config if available
            sensor = self._sensors.get(reading.sensor_id)

            # Validate reading
            if self._iot_config.validate_readings:
                is_valid, status, quality, flags = self._validator.validate(reading, sensor)
                reading.is_valid = is_valid
                reading.validation_status = status
                reading.quality_score = quality
                reading.quality_flags = flags

                if not is_valid and quality < self._iot_config.min_quality_score:
                    return  # Drop low quality readings

            # Apply calibration if sensor config available
            if sensor:
                reading.raw_value = reading.value
                reading.value = (reading.value + sensor.calibration_offset) * sensor.calibration_scale

                # Check alarm thresholds
                reading.alarm_state = self._check_alarm_thresholds(sensor, reading.value)

            # Update sensor last seen
            self._sensor_last_seen[reading.sensor_id] = datetime.utcnow()

            # Add to buffer
            try:
                self._message_buffer.put_nowait(reading)
            except asyncio.QueueFull:
                # Drop oldest message
                try:
                    self._message_buffer.get_nowait()
                    self._message_buffer.put_nowait(reading)
                except asyncio.QueueEmpty:
                    pass

            # Call streaming callbacks
            for callback in self._streaming_callbacks:
                try:
                    callback(reading)
                except Exception as e:
                    self._logger.warning(f"Streaming callback error: {e}")

        except Exception as e:
            self._logger.warning(f"Error handling MQTT message: {e}")

    async def _mqtt_message_loop(self) -> None:
        """Background loop for processing MQTT messages."""
        try:
            async for topic, payload in self._mqtt_client.listen():
                await self._handle_mqtt_message(topic, payload)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.error(f"MQTT message loop error: {e}")

    def _check_alarm_thresholds(self, sensor: IoTSensor, value: float) -> Optional[str]:
        """Check value against sensor alarm thresholds."""
        if sensor.high_high_threshold and value >= sensor.high_high_threshold:
            return "high_high"
        elif sensor.high_threshold and value >= sensor.high_threshold:
            return "high"
        elif sensor.low_low_threshold and value <= sensor.low_low_threshold:
            return "low_low"
        elif sensor.low_threshold and value <= sensor.low_threshold:
            return "low"
        return None

    # =========================================================================
    # Sensor Status Monitoring
    # =========================================================================

    async def _sensor_status_loop(self) -> None:
        """Background task for monitoring sensor status."""
        check_interval = 60.0  # Check every minute

        while True:
            try:
                await asyncio.sleep(check_interval)
                await self._update_sensor_status()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Sensor status check error: {e}")

    async def _update_sensor_status(self) -> None:
        """Update status of all sensors."""
        now = datetime.utcnow()
        timeout = timedelta(seconds=self._iot_config.sensor_timeout_seconds)

        for sensor_id, sensor in self._sensors.items():
            last_seen = self._sensor_last_seen.get(sensor_id)

            if last_seen is None:
                sensor.status = SensorStatus.UNKNOWN
            elif now - last_seen > timeout:
                if sensor.status != SensorStatus.OFFLINE:
                    self._logger.warning(f"Sensor {sensor_id} went offline")
                sensor.status = SensorStatus.OFFLINE
            else:
                sensor.status = SensorStatus.ONLINE

            sensor.last_seen = last_seen

    # =========================================================================
    # Sensor Management
    # =========================================================================

    async def register_sensor(self, sensor: IoTSensor) -> None:
        """
        Register a new sensor.

        Args:
            sensor: Sensor configuration
        """
        self._sensors[sensor.sensor_id] = sensor
        self._logger.info(f"Registered sensor: {sensor.sensor_id}")

        # Subscribe to sensor-specific topic if MQTT
        if self._iot_config.protocol == SensorProtocol.MQTT and sensor.mqtt_topic:
            await self._mqtt_client.subscribe(
                sensor.mqtt_topic,
                self._handle_mqtt_message,
                self._iot_config.mqtt_config.default_qos,
            )

    async def unregister_sensor(self, sensor_id: str) -> None:
        """
        Unregister a sensor.

        Args:
            sensor_id: Sensor ID
        """
        if sensor_id in self._sensors:
            sensor = self._sensors[sensor_id]

            # Unsubscribe from MQTT topic
            if self._iot_config.protocol == SensorProtocol.MQTT and sensor.mqtt_topic:
                await self._mqtt_client.unsubscribe(sensor.mqtt_topic)

            del self._sensors[sensor_id]
            self._logger.info(f"Unregistered sensor: {sensor_id}")

    async def get_sensor(self, sensor_id: str) -> Optional[IoTSensor]:
        """Get sensor by ID."""
        return self._sensors.get(sensor_id)

    async def get_sensors(
        self,
        equipment_id: Optional[str] = None,
        sensor_type: Optional[SensorType] = None,
        status: Optional[SensorStatus] = None,
    ) -> List[IoTSensor]:
        """
        Get sensors with optional filters.

        Args:
            equipment_id: Filter by equipment
            sensor_type: Filter by sensor type
            status: Filter by status

        Returns:
            List of sensors
        """
        sensors = list(self._sensors.values())

        if equipment_id:
            sensors = [s for s in sensors if s.equipment_id == equipment_id]
        if sensor_type:
            sensors = [s for s in sensors if s.sensor_type == sensor_type]
        if status:
            sensors = [s for s in sensors if s.status == status]

        return sensors

    async def update_sensor_config(
        self,
        sensor_id: str,
        updates: Dict[str, Any],
    ) -> IoTSensor:
        """
        Update sensor configuration.

        Args:
            sensor_id: Sensor ID
            updates: Configuration updates

        Returns:
            Updated sensor
        """
        if sensor_id not in self._sensors:
            raise ValidationError(f"Unknown sensor: {sensor_id}")

        sensor = self._sensors[sensor_id]
        updated_data = {**sensor.model_dump(), **updates}
        updated_sensor = IoTSensor(**updated_data)
        self._sensors[sensor_id] = updated_sensor

        return updated_sensor

    # =========================================================================
    # Real-time Data Access
    # =========================================================================

    async def start_streaming(
        self,
        callback: Optional[Callable[[SensorReading], None]] = None,
    ) -> None:
        """
        Start real-time data streaming.

        Args:
            callback: Optional callback for each reading
        """
        if callback:
            self._streaming_callbacks.append(callback)
        self._logger.info("Started sensor data streaming")

    async def stop_streaming(self) -> None:
        """Stop real-time data streaming."""
        self._streaming_callbacks.clear()
        self._logger.info("Stopped sensor data streaming")

    async def get_buffered_readings(
        self,
        max_items: int = 100,
        timeout: float = 1.0,
    ) -> List[SensorReading]:
        """
        Get buffered sensor readings.

        Args:
            max_items: Maximum items to retrieve
            timeout: Timeout for waiting

        Returns:
            List of buffered readings
        """
        readings = []

        try:
            for _ in range(max_items):
                reading = await asyncio.wait_for(
                    self._message_buffer.get(),
                    timeout=timeout,
                )
                readings.append(reading)
        except asyncio.TimeoutError:
            pass

        return readings

    async def get_current_reading(self, sensor_id: str) -> Optional[SensorReading]:
        """
        Get current reading for a sensor.

        Args:
            sensor_id: Sensor ID

        Returns:
            Current reading or None
        """
        if sensor_id not in self._sensors:
            raise ValidationError(f"Unknown sensor: {sensor_id}")

        if self._iot_config.protocol == SensorProtocol.REST_API:
            headers = {}
            if self._iot_config.gateway_config.api_key:
                headers["X-API-Key"] = self._iot_config.gateway_config.api_key

            response = await self._http_client.get(
                f"/sensors/{sensor_id}/current",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            sensor = self._sensors[sensor_id]
            return SensorReading(
                sensor_id=sensor_id,
                equipment_id=sensor.equipment_id,
                sensor_type=sensor.sensor_type,
                value=data.get("value"),
                unit=sensor.measurement_unit,
                timestamp=datetime.fromisoformat(data.get("timestamp")),
            )

        # For MQTT, return most recent from buffer if available
        # (would need additional tracking for per-sensor current value)
        return None

    async def stream_readings(
        self,
        sensor_ids: Optional[List[str]] = None,
        equipment_id: Optional[str] = None,
    ) -> AsyncGenerator[SensorReading, None]:
        """
        Async generator for streaming sensor readings.

        Args:
            sensor_ids: Optional list of sensor IDs to filter
            equipment_id: Optional equipment ID to filter

        Yields:
            Sensor readings
        """
        while True:
            try:
                reading = await asyncio.wait_for(
                    self._message_buffer.get(),
                    timeout=1.0,
                )

                # Apply filters
                if sensor_ids and reading.sensor_id not in sensor_ids:
                    continue
                if equipment_id and reading.equipment_id != equipment_id:
                    continue

                yield reading

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    # =========================================================================
    # Batch Data Retrieval
    # =========================================================================

    async def get_sensor_readings(
        self,
        sensor_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[SensorReading]:
        """
        Get historical readings for a sensor.

        Args:
            sensor_id: Sensor ID
            start_time: Start time
            end_time: End time (default: now)
            limit: Maximum readings

        Returns:
            List of readings
        """
        if not self._iot_config.enable_batch_retrieval:
            raise ConfigurationError("Batch retrieval not enabled")

        if sensor_id not in self._sensors:
            raise ValidationError(f"Unknown sensor: {sensor_id}")

        end_time = end_time or datetime.utcnow()
        limit = limit or self._iot_config.batch_retrieval_limit

        if self._iot_config.protocol == SensorProtocol.REST_API:
            headers = {}
            if self._iot_config.gateway_config.api_key:
                headers["X-API-Key"] = self._iot_config.gateway_config.api_key

            response = await self._http_client.get(
                f"/sensors/{sensor_id}/readings",
                headers=headers,
                params={
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "limit": limit,
                },
            )
            response.raise_for_status()
            data = response.json()

            sensor = self._sensors[sensor_id]
            readings = []
            for reading_data in data.get("readings", []):
                reading = SensorReading(
                    sensor_id=sensor_id,
                    equipment_id=sensor.equipment_id,
                    sensor_type=sensor.sensor_type,
                    unit=sensor.measurement_unit,
                    **reading_data,
                )
                readings.append(reading)

            return readings

        return []

    async def get_equipment_readings(
        self,
        equipment_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        sensor_types: Optional[List[SensorType]] = None,
    ) -> Dict[str, List[SensorReading]]:
        """
        Get readings for all sensors on equipment.

        Args:
            equipment_id: Equipment ID
            start_time: Start time
            end_time: End time (default: now)
            sensor_types: Optional filter by sensor types

        Returns:
            Dictionary of sensor_id -> readings
        """
        sensors = await self.get_sensors(equipment_id=equipment_id)

        if sensor_types:
            sensors = [s for s in sensors if s.sensor_type in sensor_types]

        results = {}
        for sensor in sensors:
            try:
                readings = await self.get_sensor_readings(
                    sensor.sensor_id,
                    start_time,
                    end_time,
                )
                results[sensor.sensor_id] = readings
            except Exception as e:
                self._logger.warning(
                    f"Failed to get readings for sensor {sensor.sensor_id}: {e}"
                )

        return results

    async def get_batch_readings(
        self,
        sensor_ids: List[str],
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> List[SensorBatch]:
        """
        Get batch readings for multiple sensors.

        Args:
            sensor_ids: List of sensor IDs
            start_time: Start time
            end_time: End time

        Returns:
            List of sensor batches
        """
        end_time = end_time or datetime.utcnow()
        batches = []

        for sensor_id in sensor_ids:
            if sensor_id not in self._sensors:
                continue

            sensor = self._sensors[sensor_id]
            readings = await self.get_sensor_readings(sensor_id, start_time, end_time)

            if not readings:
                continue

            # Calculate statistics
            values = [r.value for r in readings if r.is_valid]
            import statistics

            if values:
                min_val = min(values)
                max_val = max(values)
                avg_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                min_val = max_val = avg_val = std_val = 0.0

            valid_count = sum(1 for r in readings if r.is_valid)
            invalid_count = len(readings) - valid_count
            avg_quality = statistics.mean(r.quality_score for r in readings) if readings else 0.0

            batch = SensorBatch(
                sensor_id=sensor_id,
                equipment_id=sensor.equipment_id,
                start_time=start_time,
                end_time=end_time,
                readings=readings,
                reading_count=len(readings),
                min_value=min_val,
                max_value=max_val,
                avg_value=avg_val,
                std_dev=std_val,
                valid_readings=valid_count,
                invalid_readings=invalid_count,
                avg_quality_score=avg_quality,
            )
            batches.append(batch)

        return batches

    # =========================================================================
    # Gateway Operations
    # =========================================================================

    async def get_gateways(self) -> List[GatewayStatus]:
        """Get list of IoT gateways."""
        if self._iot_config.protocol != SensorProtocol.REST_API:
            return []

        headers = {}
        if self._iot_config.gateway_config.api_key:
            headers["X-API-Key"] = self._iot_config.gateway_config.api_key

        response = await self._http_client.get("/gateways", headers=headers)
        response.raise_for_status()
        data = response.json()

        return [GatewayStatus(**gw) for gw in data.get("gateways", [])]

    async def get_gateway_status(self, gateway_id: str) -> GatewayStatus:
        """Get status of specific gateway."""
        if self._iot_config.protocol != SensorProtocol.REST_API:
            raise ConfigurationError("Gateway status only available via REST API")

        headers = {}
        if self._iot_config.gateway_config.api_key:
            headers["X-API-Key"] = self._iot_config.gateway_config.api_key

        response = await self._http_client.get(
            f"/gateways/{gateway_id}",
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        return GatewayStatus(**data)

    # =========================================================================
    # Sensor Commands
    # =========================================================================

    async def send_sensor_command(
        self,
        sensor_id: str,
        command: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send command to sensor.

        Args:
            sensor_id: Target sensor ID
            command: Command name
            parameters: Command parameters

        Returns:
            Command response
        """
        if sensor_id not in self._sensors:
            raise ValidationError(f"Unknown sensor: {sensor_id}")

        sensor = self._sensors[sensor_id]

        if self._iot_config.protocol == SensorProtocol.MQTT:
            # Publish to sensor command topic
            command_topic = f"commands/{sensor_id}"
            payload = {
                "command": command,
                "parameters": parameters or {},
                "timestamp": datetime.utcnow().isoformat(),
            }
            await self._mqtt_client.publish(command_topic, payload)
            return {"status": "sent", "topic": command_topic}

        elif self._iot_config.protocol == SensorProtocol.REST_API:
            headers = {}
            if self._iot_config.gateway_config.api_key:
                headers["X-API-Key"] = self._iot_config.gateway_config.api_key

            response = await self._http_client.post(
                f"/sensors/{sensor_id}/commands",
                headers=headers,
                json={
                    "command": command,
                    "parameters": parameters or {},
                },
            )
            response.raise_for_status()
            return response.json()

        raise ConfigurationError(f"Commands not supported for {self._iot_config.protocol.value}")

    async def calibrate_sensor(
        self,
        sensor_id: str,
        reference_value: float,
        measured_value: Optional[float] = None,
    ) -> IoTSensor:
        """
        Calibrate sensor with reference value.

        Args:
            sensor_id: Sensor ID
            reference_value: Known reference value
            measured_value: Measured value (uses current if not provided)

        Returns:
            Updated sensor with new calibration
        """
        if sensor_id not in self._sensors:
            raise ValidationError(f"Unknown sensor: {sensor_id}")

        sensor = self._sensors[sensor_id]

        # Get current reading if measured value not provided
        if measured_value is None:
            reading = await self.get_current_reading(sensor_id)
            if reading:
                measured_value = reading.value
            else:
                raise ValidationError("Could not get current sensor reading")

        # Calculate new calibration offset
        new_offset = reference_value - (measured_value * sensor.calibration_scale)

        # Update sensor calibration
        sensor.calibration_offset = new_offset
        sensor.calibration_date = datetime.utcnow()

        self._logger.info(
            f"Calibrated sensor {sensor_id}: offset={new_offset:.4f}, "
            f"reference={reference_value}, measured={measured_value}"
        )

        return sensor


# =============================================================================
# Factory Function
# =============================================================================


def create_iot_sensor_connector(
    protocol: SensorProtocol,
    connector_name: str,
    mqtt_config: Optional[MQTTConfig] = None,
    gateway_config: Optional[SensorGatewayConfig] = None,
    **kwargs,
) -> IoTSensorConnector:
    """
    Factory function to create IoT sensor connector.

    Args:
        protocol: Communication protocol
        connector_name: Connector name
        mqtt_config: MQTT configuration
        gateway_config: Gateway configuration
        **kwargs: Additional configuration options

    Returns:
        Configured IoT sensor connector
    """
    config = IoTSensorConnectorConfig(
        connector_name=connector_name,
        connector_type=ConnectorType.IOT_SENSOR,
        protocol=protocol,
        mqtt_config=mqtt_config,
        gateway_config=gateway_config,
        **kwargs,
    )

    return IoTSensorConnector(config)
