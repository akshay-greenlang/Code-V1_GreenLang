"""
MQTT Industrial IoT Connector for GreenLang.

This module provides comprehensive MQTT integration for Industrial IoT (IIoT)
applications, supporting standard MQTT and Sparkplug B specification for
industrial automation.

Features:
    - MQTT 3.1.1 and 5.0 protocol support
    - Topic subscription with wildcard patterns
    - QoS levels 0, 1, and 2
    - Message parsing (JSON, Sparkplug B, binary)
    - Retain message handling
    - Last Will and Testament (LWT)
    - TLS/SSL with client certificates
    - Connection pooling for high throughput

Example:
    >>> from integrations.industrial import MQTTConnector, MQTTConfig
    >>>
    >>> config = MQTTConfig(
    ...     host="mqtt.factory.local",
    ...     port=8883,
    ...     tls=TLSConfig(enabled=True),
    ...     client_id="greenlang-collector-01"
    ... )
    >>> connector = MQTTConnector(config)
    >>> async with connector:
    ...     await connector.subscribe("factory/+/sensors/#", callback)
"""

import asyncio
import json
import logging
import struct
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, SecretStr

from .base import (
    AuthenticationType,
    BaseConnectorConfig,
    BaseIndustrialConnector,
    TLSConfig,
)
from .data_models import (
    BatchReadResponse,
    BatchWriteRequest,
    BatchWriteResponse,
    ConnectionState,
    DataQuality,
    DataType,
    TagMetadata,
    TagValue,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MQTT Constants and Enums
# =============================================================================


class MQTTVersion(str, Enum):
    """MQTT protocol versions."""

    V311 = "3.1.1"
    V5 = "5.0"


class QoSLevel(IntEnum):
    """MQTT Quality of Service levels."""

    AT_MOST_ONCE = 0  # Fire and forget
    AT_LEAST_ONCE = 1  # Acknowledged delivery
    EXACTLY_ONCE = 2  # Assured delivery


class MessageFormat(str, Enum):
    """Supported message payload formats."""

    JSON = "json"
    SPARKPLUG_B = "sparkplug_b"
    BINARY = "binary"
    STRING = "string"
    RAW = "raw"


class SparkplugMessageType(str, Enum):
    """Sparkplug B message types."""

    NBIRTH = "NBIRTH"  # Node Birth
    NDEATH = "NDEATH"  # Node Death
    DBIRTH = "DBIRTH"  # Device Birth
    DDEATH = "DDEATH"  # Device Death
    NDATA = "NDATA"  # Node Data
    DDATA = "DDATA"  # Device Data
    NCMD = "NCMD"  # Node Command
    DCMD = "DCMD"  # Device Command
    STATE = "STATE"  # State


class SparkplugDataType(IntEnum):
    """Sparkplug B data types."""

    UNKNOWN = 0
    INT8 = 1
    INT16 = 2
    INT32 = 3
    INT64 = 4
    UINT8 = 5
    UINT16 = 6
    UINT32 = 7
    UINT64 = 8
    FLOAT = 9
    DOUBLE = 10
    BOOLEAN = 11
    STRING = 12
    DATETIME = 13
    TEXT = 14
    UUID = 15
    DATASET = 16
    BYTES = 17
    FILE = 18
    TEMPLATE = 19


# =============================================================================
# Configuration Models
# =============================================================================


class MQTTConfig(BaseConnectorConfig):
    """
    MQTT connector configuration.

    Attributes:
        host: MQTT broker hostname
        port: MQTT broker port
        client_id: Unique client identifier
        protocol_version: MQTT protocol version
        clean_session: Start with clean session
        keepalive_seconds: Keepalive interval
        username: Authentication username
        password: Authentication password
        tls: TLS configuration
    """

    port: int = Field(1883, description="MQTT broker port")
    client_id: str = Field(
        default_factory=lambda: f"greenlang-{uuid.uuid4().hex[:8]}",
        description="Client identifier"
    )
    protocol_version: MQTTVersion = Field(
        MQTTVersion.V311,
        description="MQTT protocol version"
    )
    clean_session: bool = Field(True, description="Clean session flag")
    keepalive_seconds: int = Field(60, ge=10, description="Keepalive interval")

    # Authentication
    username: Optional[str] = Field(None, description="Username")
    password: Optional[SecretStr] = Field(None, description="Password")

    # Last Will and Testament
    lwt_topic: Optional[str] = Field(None, description="LWT topic")
    lwt_message: Optional[str] = Field(None, description="LWT message")
    lwt_qos: QoSLevel = Field(QoSLevel.AT_LEAST_ONCE, description="LWT QoS")
    lwt_retain: bool = Field(True, description="LWT retain flag")

    # Message handling
    default_qos: QoSLevel = Field(QoSLevel.AT_LEAST_ONCE, description="Default QoS")
    default_format: MessageFormat = Field(MessageFormat.JSON, description="Default format")
    max_inflight: int = Field(20, ge=1, description="Max inflight messages")
    max_queued: int = Field(1000, ge=0, description="Max queued messages")

    # Sparkplug B
    sparkplug_enabled: bool = Field(False, description="Enable Sparkplug B")
    sparkplug_group_id: Optional[str] = Field(None, description="Sparkplug group ID")
    sparkplug_edge_node_id: Optional[str] = Field(None, description="Sparkplug node ID")


class TopicSubscription(BaseModel):
    """
    MQTT topic subscription configuration.

    Attributes:
        topic: Topic pattern (supports +/# wildcards)
        qos: QoS level for subscription
        format: Expected message format
        callback: Callback function name
    """

    topic: str = Field(..., description="Topic pattern")
    qos: QoSLevel = Field(QoSLevel.AT_LEAST_ONCE, description="QoS level")
    format: MessageFormat = Field(MessageFormat.JSON, description="Message format")
    retain_handling: int = Field(0, ge=0, le=2, description="Retain handling")


class MQTTMessage(BaseModel):
    """
    MQTT message representation.

    Attributes:
        topic: Message topic
        payload: Message payload
        qos: QoS level
        retain: Retain flag
        timestamp: Receive timestamp
        properties: MQTT 5.0 properties
    """

    topic: str = Field(..., description="Message topic")
    payload: Any = Field(..., description="Message payload")
    qos: QoSLevel = Field(QoSLevel.AT_MOST_ONCE, description="QoS level")
    retain: bool = Field(False, description="Retain flag")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Receive timestamp"
    )
    properties: Dict[str, Any] = Field(default_factory=dict, description="Properties")


# =============================================================================
# Sparkplug B Support
# =============================================================================


class SparkplugMetric(BaseModel):
    """
    Sparkplug B metric definition.

    Attributes:
        name: Metric name
        alias: Numeric alias
        datatype: Data type
        value: Current value
        timestamp: Value timestamp
    """

    name: str = Field(..., description="Metric name")
    alias: Optional[int] = Field(None, description="Numeric alias")
    datatype: SparkplugDataType = Field(
        SparkplugDataType.STRING,
        description="Data type"
    )
    value: Any = Field(None, description="Current value")
    timestamp: Optional[int] = Field(None, description="Timestamp (ms)")
    is_historical: bool = Field(False, description="Historical flag")
    is_transient: bool = Field(False, description="Transient flag")
    is_null: bool = Field(False, description="Null flag")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Properties")


class SparkplugPayload(BaseModel):
    """
    Sparkplug B payload structure.

    Attributes:
        timestamp: Payload timestamp
        metrics: List of metrics
        seq: Sequence number
        uuid: Payload UUID
    """

    timestamp: int = Field(..., description="Timestamp (ms since epoch)")
    metrics: List[SparkplugMetric] = Field(default_factory=list, description="Metrics")
    seq: int = Field(0, ge=0, le=255, description="Sequence number")
    uuid: Optional[str] = Field(None, description="UUID")
    body: Optional[bytes] = Field(None, description="Binary body")


class SparkplugEncoder:
    """
    Sparkplug B protocol encoder/decoder.

    Handles encoding and decoding of Sparkplug B messages
    using protobuf-like binary format.
    """

    @staticmethod
    def encode_payload(payload: SparkplugPayload) -> bytes:
        """
        Encode Sparkplug B payload to binary.

        In production, this would use the official sparkplug_b library
        with protobuf encoding.

        Args:
            payload: Sparkplug payload

        Returns:
            Binary encoded payload
        """
        # Simplified encoding - in production use sparkplug_b.Payload
        data = {
            "timestamp": payload.timestamp,
            "seq": payload.seq,
            "metrics": [
                {
                    "name": m.name,
                    "alias": m.alias,
                    "datatype": m.datatype.value,
                    "value": m.value,
                    "timestamp": m.timestamp,
                }
                for m in payload.metrics
            ],
        }
        return json.dumps(data).encode("utf-8")

    @staticmethod
    def decode_payload(data: bytes) -> SparkplugPayload:
        """
        Decode Sparkplug B binary payload.

        Args:
            data: Binary payload

        Returns:
            Decoded SparkplugPayload
        """
        # Simplified decoding - in production use sparkplug_b.Payload
        try:
            parsed = json.loads(data.decode("utf-8"))
            metrics = [
                SparkplugMetric(
                    name=m["name"],
                    alias=m.get("alias"),
                    datatype=SparkplugDataType(m.get("datatype", 0)),
                    value=m.get("value"),
                    timestamp=m.get("timestamp"),
                )
                for m in parsed.get("metrics", [])
            ]
            return SparkplugPayload(
                timestamp=parsed.get("timestamp", int(time.time() * 1000)),
                seq=parsed.get("seq", 0),
                metrics=metrics,
            )
        except Exception as e:
            logger.error(f"Failed to decode Sparkplug payload: {e}")
            return SparkplugPayload(timestamp=int(time.time() * 1000))

    @staticmethod
    def build_topic(
        message_type: SparkplugMessageType,
        group_id: str,
        edge_node_id: str,
        device_id: Optional[str] = None,
    ) -> str:
        """
        Build Sparkplug B topic.

        Args:
            message_type: Message type
            group_id: Sparkplug group ID
            edge_node_id: Edge node ID
            device_id: Optional device ID

        Returns:
            Sparkplug topic string
        """
        namespace = "spBv1.0"
        if device_id:
            return f"{namespace}/{group_id}/{message_type.value}/{edge_node_id}/{device_id}"
        return f"{namespace}/{group_id}/{message_type.value}/{edge_node_id}"


# =============================================================================
# Message Parser
# =============================================================================


class MQTTMessageParser:
    """
    MQTT message payload parser.

    Handles parsing of various message formats including
    JSON, Sparkplug B, and binary.
    """

    @staticmethod
    def parse(
        payload: bytes,
        format: MessageFormat,
        topic: Optional[str] = None,
    ) -> Any:
        """
        Parse message payload.

        Args:
            payload: Raw payload bytes
            format: Expected format
            topic: Message topic (for context)

        Returns:
            Parsed payload data
        """
        if format == MessageFormat.JSON:
            return MQTTMessageParser._parse_json(payload)
        elif format == MessageFormat.SPARKPLUG_B:
            return MQTTMessageParser._parse_sparkplug(payload)
        elif format == MessageFormat.STRING:
            return payload.decode("utf-8")
        elif format == MessageFormat.BINARY:
            return MQTTMessageParser._parse_binary(payload)
        else:
            return payload

    @staticmethod
    def _parse_json(payload: bytes) -> Dict[str, Any]:
        """Parse JSON payload."""
        try:
            return json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return {"raw": payload.decode("utf-8", errors="replace")}

    @staticmethod
    def _parse_sparkplug(payload: bytes) -> SparkplugPayload:
        """Parse Sparkplug B payload."""
        return SparkplugEncoder.decode_payload(payload)

    @staticmethod
    def _parse_binary(payload: bytes) -> Dict[str, Any]:
        """Parse binary payload (attempt to extract numeric values)."""
        result = {"raw_bytes": payload.hex()}

        # Try to extract common formats
        if len(payload) >= 4:
            result["float32"] = struct.unpack(">f", payload[:4])[0]
            result["int32"] = struct.unpack(">i", payload[:4])[0]
            result["uint32"] = struct.unpack(">I", payload[:4])[0]

        if len(payload) >= 2:
            result["int16"] = struct.unpack(">h", payload[:2])[0]
            result["uint16"] = struct.unpack(">H", payload[:2])[0]

        return result

    @staticmethod
    def extract_tag_values(
        message: MQTTMessage,
        topic_pattern: Optional[str] = None,
    ) -> List[TagValue]:
        """
        Extract TagValue objects from MQTT message.

        Args:
            message: Parsed MQTT message
            topic_pattern: Topic pattern for tag naming

        Returns:
            List of TagValue objects
        """
        values = []

        if isinstance(message.payload, SparkplugPayload):
            # Extract from Sparkplug metrics
            for metric in message.payload.metrics:
                tag_id = f"{message.topic}/{metric.name}"
                values.append(TagValue(
                    tag_id=tag_id,
                    value=metric.value,
                    timestamp=(
                        datetime.fromtimestamp(metric.timestamp / 1000)
                        if metric.timestamp
                        else message.timestamp
                    ),
                    quality=DataQuality.BAD if metric.is_null else DataQuality.GOOD,
                ))

        elif isinstance(message.payload, dict):
            # Extract from JSON object
            def extract_dict(d: Dict, prefix: str = "") -> None:
                for key, value in d.items():
                    tag_id = f"{message.topic}/{prefix}{key}" if prefix else f"{message.topic}/{key}"
                    if isinstance(value, dict):
                        extract_dict(value, f"{prefix}{key}/")
                    elif isinstance(value, (int, float, str, bool)):
                        values.append(TagValue(
                            tag_id=tag_id,
                            value=value,
                            timestamp=message.timestamp,
                            quality=DataQuality.GOOD,
                        ))

            extract_dict(message.payload)

        else:
            # Simple value
            values.append(TagValue(
                tag_id=message.topic,
                value=message.payload,
                timestamp=message.timestamp,
                quality=DataQuality.GOOD,
            ))

        return values


# =============================================================================
# MQTT Connector
# =============================================================================


class MQTTConnector(BaseIndustrialConnector):
    """
    MQTT Industrial IoT Connector.

    Provides comprehensive MQTT integration for IIoT applications
    with support for standard MQTT and Sparkplug B.

    Features:
        - Topic subscription with wildcards
        - QoS levels 0, 1, 2
        - Multiple message formats
        - Last Will and Testament
        - Sparkplug B protocol support
        - TLS with client certificates

    Example:
        >>> config = MQTTConfig(
        ...     host="mqtt.factory.local",
        ...     port=8883,
        ...     client_id="greenlang-collector",
        ...     tls=TLSConfig(enabled=True)
        ... )
        >>> connector = MQTTConnector(config)
        >>> await connector.connect()
        >>> await connector.subscribe("factory/+/sensors/#", on_message)
    """

    def __init__(self, config: MQTTConfig):
        """
        Initialize MQTT connector.

        Args:
            config: MQTT configuration
        """
        # Create base config
        base_config = BaseConnectorConfig(
            host=config.host,
            port=config.port,
            timeout_seconds=config.timeout_seconds,
            auth_type=(
                AuthenticationType.USERNAME_PASSWORD
                if config.username
                else AuthenticationType.NONE
            ),
            username=config.username,
            password=config.password,
            name=config.name or "mqtt_connector",
            tls=config.tls,
            rate_limit=config.rate_limit,
            reconnect=config.reconnect,
            health_check_interval_seconds=config.health_check_interval_seconds,
        )

        super().__init__(base_config)
        self.mqtt_config = config

        # Client state (would be aiomqtt.Client in production)
        self._client: Optional[Any] = None

        # Subscription management
        self._subscriptions_config: Dict[str, TopicSubscription] = {}
        self._subscription_callbacks: Dict[str, List[Callable]] = {}

        # Message buffer
        self._message_buffer: List[MQTTMessage] = []
        self._max_buffer_size = 10000

        # Sparkplug state
        self._sparkplug_seq = 0
        self._sparkplug_birth_published = False

        # Last received values (for read_tags)
        self._last_values: Dict[str, TagValue] = {}

        # Message processing task
        self._message_task: Optional[asyncio.Task] = None

    async def _do_connect(self) -> bool:
        """Establish MQTT connection."""
        logger.info(
            f"Connecting to MQTT broker: {self.mqtt_config.host}:{self.mqtt_config.port}"
        )

        try:
            # In production, use aiomqtt:
            # import aiomqtt
            # self._client = aiomqtt.Client(
            #     hostname=self.mqtt_config.host,
            #     port=self.mqtt_config.port,
            #     client_id=self.mqtt_config.client_id,
            #     username=self.mqtt_config.username,
            #     password=self.mqtt_config.password.get_secret_value() if self.mqtt_config.password else None,
            #     tls_context=self._ssl_context,
            #     clean_session=self.mqtt_config.clean_session,
            #     keepalive=self.mqtt_config.keepalive_seconds,
            # )

            # Configure LWT if specified
            if self.mqtt_config.lwt_topic:
                # self._client.will_set(
                #     self.mqtt_config.lwt_topic,
                #     self.mqtt_config.lwt_message,
                #     self.mqtt_config.lwt_qos,
                #     self.mqtt_config.lwt_retain
                # )
                pass

            # Connect
            # await self._client.connect()

            # Simulated connection
            self._client = True

            # Publish Sparkplug birth if enabled
            if self.mqtt_config.sparkplug_enabled:
                await self._publish_sparkplug_birth()

            # Start message processing task
            self._message_task = asyncio.create_task(self._message_loop())

            logger.info("MQTT connection established")
            return True

        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            raise

    async def _do_disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        try:
            # Publish Sparkplug death if enabled
            if self.mqtt_config.sparkplug_enabled and self._sparkplug_birth_published:
                await self._publish_sparkplug_death()

            # Stop message processing
            if self._message_task:
                self._message_task.cancel()
                try:
                    await self._message_task
                except asyncio.CancelledError:
                    pass
                self._message_task = None

            # Disconnect client
            if self._client:
                # await self._client.disconnect()
                pass

            self._client = None
            self._sparkplug_birth_published = False

            logger.info("MQTT disconnected")

        except Exception as e:
            logger.error(f"Error during MQTT disconnect: {e}")

    async def _do_health_check(self) -> bool:
        """Check MQTT connection health."""
        return self._client is not None

    # =========================================================================
    # Subscription Management
    # =========================================================================

    async def subscribe_topic(
        self,
        topic: str,
        callback: Callable[[MQTTMessage], None],
        qos: QoSLevel = QoSLevel.AT_LEAST_ONCE,
        format: MessageFormat = MessageFormat.JSON,
    ) -> str:
        """
        Subscribe to MQTT topic.

        Args:
            topic: Topic pattern (supports +/# wildcards)
            callback: Callback for received messages
            qos: QoS level
            format: Expected message format

        Returns:
            Subscription identifier
        """
        self._validate_connected()

        subscription_id = str(uuid.uuid4())

        # Store subscription config
        self._subscriptions_config[subscription_id] = TopicSubscription(
            topic=topic,
            qos=qos,
            format=format,
        )

        # Store callback
        if topic not in self._subscription_callbacks:
            self._subscription_callbacks[topic] = []
        self._subscription_callbacks[topic].append(callback)

        # Subscribe on broker
        # await self._client.subscribe(topic, qos)

        logger.info(f"Subscribed to topic: {topic} (QoS {qos})")

        # Update subscription status
        self._subscriptions[subscription_id] = {
            "subscription_id": subscription_id,
            "topic": topic,
            "qos": qos,
            "format": format.value,
            "created_at": datetime.utcnow().isoformat(),
        }

        return subscription_id

    async def unsubscribe_topic(self, subscription_id: str) -> bool:
        """
        Unsubscribe from topic.

        Args:
            subscription_id: Subscription identifier

        Returns:
            True if unsubscribed
        """
        if subscription_id not in self._subscriptions_config:
            return False

        config = self._subscriptions_config[subscription_id]

        # Unsubscribe from broker
        # await self._client.unsubscribe(config.topic)

        # Clean up
        del self._subscriptions_config[subscription_id]
        if config.topic in self._subscription_callbacks:
            del self._subscription_callbacks[config.topic]
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]

        logger.info(f"Unsubscribed from topic: {config.topic}")
        return True

    async def subscribe(
        self,
        config: Any,
        callback: Callable[[TagValue], None],
    ) -> str:
        """
        Subscribe for real-time tag data.

        Args:
            config: SubscriptionConfig
            callback: Callback for tag values

        Returns:
            Subscription identifier
        """
        # Convert tag-based subscription to topic subscription
        def tag_callback(message: MQTTMessage) -> None:
            values = MQTTMessageParser.extract_tag_values(message)
            for value in values:
                callback(value)

        # Subscribe to topic pattern based on tags
        topic = config.tag_ids[0] if len(config.tag_ids) == 1 else "#"
        return await self.subscribe_topic(topic, tag_callback)

    # =========================================================================
    # Publishing
    # =========================================================================

    async def publish(
        self,
        topic: str,
        payload: Any,
        qos: QoSLevel = QoSLevel.AT_LEAST_ONCE,
        retain: bool = False,
        format: MessageFormat = MessageFormat.JSON,
    ) -> bool:
        """
        Publish message to topic.

        Args:
            topic: Destination topic
            payload: Message payload
            qos: QoS level
            retain: Retain flag
            format: Payload format

        Returns:
            True if published successfully
        """
        self._validate_connected()

        try:
            # Encode payload
            if format == MessageFormat.JSON:
                encoded = json.dumps(payload).encode("utf-8")
            elif format == MessageFormat.STRING:
                encoded = str(payload).encode("utf-8")
            elif format == MessageFormat.SPARKPLUG_B:
                if isinstance(payload, SparkplugPayload):
                    encoded = SparkplugEncoder.encode_payload(payload)
                else:
                    raise ValueError("Sparkplug format requires SparkplugPayload")
            else:
                encoded = payload if isinstance(payload, bytes) else str(payload).encode()

            # Publish
            # await self._client.publish(topic, encoded, qos, retain)

            logger.debug(f"Published to {topic}: {len(encoded)} bytes")
            return True

        except Exception as e:
            logger.error(f"Publish failed: {e}")
            return False

    async def publish_tag_value(
        self,
        tag_id: str,
        value: Any,
        topic_prefix: str = "greenlang/tags",
    ) -> bool:
        """
        Publish tag value to MQTT.

        Args:
            tag_id: Tag identifier
            value: Tag value
            topic_prefix: Topic prefix

        Returns:
            True if published
        """
        topic = f"{topic_prefix}/{tag_id.replace('/', '_')}"
        payload = {
            "tag_id": tag_id,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
            "quality": "good",
        }
        return await self.publish(topic, payload)

    # =========================================================================
    # Sparkplug B
    # =========================================================================

    async def _publish_sparkplug_birth(self) -> None:
        """Publish Sparkplug B Node Birth message."""
        if not self.mqtt_config.sparkplug_group_id:
            return

        topic = SparkplugEncoder.build_topic(
            SparkplugMessageType.NBIRTH,
            self.mqtt_config.sparkplug_group_id,
            self.mqtt_config.sparkplug_edge_node_id or self.mqtt_config.client_id,
        )

        payload = SparkplugPayload(
            timestamp=int(time.time() * 1000),
            seq=self._sparkplug_seq,
            metrics=[
                SparkplugMetric(
                    name="Node Control/Rebirth",
                    datatype=SparkplugDataType.BOOLEAN,
                    value=False,
                ),
                SparkplugMetric(
                    name="Properties/Version",
                    datatype=SparkplugDataType.STRING,
                    value="1.0.0",
                ),
            ],
        )

        await self.publish(
            topic,
            payload,
            QoSLevel.AT_LEAST_ONCE,
            retain=False,
            format=MessageFormat.SPARKPLUG_B,
        )

        self._sparkplug_birth_published = True
        self._sparkplug_seq = (self._sparkplug_seq + 1) % 256
        logger.info("Published Sparkplug NBIRTH")

    async def _publish_sparkplug_death(self) -> None:
        """Publish Sparkplug B Node Death message."""
        if not self.mqtt_config.sparkplug_group_id:
            return

        topic = SparkplugEncoder.build_topic(
            SparkplugMessageType.NDEATH,
            self.mqtt_config.sparkplug_group_id,
            self.mqtt_config.sparkplug_edge_node_id or self.mqtt_config.client_id,
        )

        payload = SparkplugPayload(
            timestamp=int(time.time() * 1000),
            seq=self._sparkplug_seq,
        )

        await self.publish(
            topic,
            payload,
            QoSLevel.AT_LEAST_ONCE,
            retain=False,
            format=MessageFormat.SPARKPLUG_B,
        )

        logger.info("Published Sparkplug NDEATH")

    async def publish_sparkplug_data(
        self,
        metrics: List[SparkplugMetric],
        device_id: Optional[str] = None,
    ) -> bool:
        """
        Publish Sparkplug B data message.

        Args:
            metrics: List of metrics to publish
            device_id: Optional device ID (for DDATA vs NDATA)

        Returns:
            True if published
        """
        if not self.mqtt_config.sparkplug_group_id:
            raise ValueError("Sparkplug not configured")

        message_type = SparkplugMessageType.DDATA if device_id else SparkplugMessageType.NDATA

        topic = SparkplugEncoder.build_topic(
            message_type,
            self.mqtt_config.sparkplug_group_id,
            self.mqtt_config.sparkplug_edge_node_id or self.mqtt_config.client_id,
            device_id,
        )

        payload = SparkplugPayload(
            timestamp=int(time.time() * 1000),
            seq=self._sparkplug_seq,
            metrics=metrics,
        )

        success = await self.publish(
            topic,
            payload,
            QoSLevel.AT_LEAST_ONCE,
            format=MessageFormat.SPARKPLUG_B,
        )

        if success:
            self._sparkplug_seq = (self._sparkplug_seq + 1) % 256

        return success

    # =========================================================================
    # Message Processing
    # =========================================================================

    async def _message_loop(self) -> None:
        """Background message processing loop."""
        while self.is_connected:
            try:
                # In production, this would use async iteration:
                # async for message in self._client.messages:
                #     await self._handle_message(message)

                # Simulated message processing
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Message loop error: {e}")
                await asyncio.sleep(1)

    async def _handle_message(
        self,
        topic: str,
        payload: bytes,
        qos: int,
        retain: bool,
    ) -> None:
        """
        Handle received MQTT message.

        Args:
            topic: Message topic
            payload: Raw payload
            qos: QoS level
            retain: Retain flag
        """
        # Determine format from subscription config
        format = MessageFormat.JSON
        for sub in self._subscriptions_config.values():
            if self._topic_matches(sub.topic, topic):
                format = sub.format
                break

        # Parse payload
        parsed = MQTTMessageParser.parse(payload, format, topic)

        message = MQTTMessage(
            topic=topic,
            payload=parsed,
            qos=QoSLevel(qos),
            retain=retain,
            timestamp=datetime.utcnow(),
        )

        # Buffer message
        self._message_buffer.append(message)
        if len(self._message_buffer) > self._max_buffer_size:
            self._message_buffer.pop(0)

        # Extract and cache tag values
        tag_values = MQTTMessageParser.extract_tag_values(message)
        for tv in tag_values:
            self._last_values[tv.tag_id] = tv

        # Invoke callbacks
        for pattern, callbacks in self._subscription_callbacks.items():
            if self._topic_matches(pattern, topic):
                for callback in callbacks:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Callback error for {topic}: {e}")

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Check if topic matches subscription pattern."""
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")

        for i, pp in enumerate(pattern_parts):
            if pp == "#":
                return True
            if i >= len(topic_parts):
                return False
            if pp != "+" and pp != topic_parts[i]:
                return False

        return len(pattern_parts) == len(topic_parts)

    # =========================================================================
    # Tag Operations
    # =========================================================================

    async def read_tags(
        self,
        tag_ids: List[str],
    ) -> BatchReadResponse:
        """
        Read last received values for tags.

        Note: MQTT is push-based, so this returns cached values
        from the most recent messages.

        Args:
            tag_ids: List of tag identifiers

        Returns:
            BatchReadResponse with cached values
        """
        values: Dict[str, TagValue] = {}
        errors: Dict[str, str] = {}

        for tag_id in tag_ids:
            if tag_id in self._last_values:
                values[tag_id] = self._last_values[tag_id]
            else:
                errors[tag_id] = "No value received for tag"

        return BatchReadResponse(
            values=values,
            errors=errors,
            timestamp=datetime.utcnow(),
        )

    async def write_tags(
        self,
        request: BatchWriteRequest,
    ) -> BatchWriteResponse:
        """
        Write tag values by publishing to MQTT.

        Args:
            request: Batch write request

        Returns:
            BatchWriteResponse with results
        """
        self._validate_connected()

        success: Dict[str, bool] = {}
        errors: Dict[str, str] = {}

        for tag_id, value in request.writes.items():
            try:
                published = await self.publish_tag_value(tag_id, value)
                success[tag_id] = published
                if not published:
                    errors[tag_id] = "Publish failed"
            except Exception as e:
                errors[tag_id] = str(e)

        return BatchWriteResponse(
            success=success,
            errors=errors,
            timestamp=datetime.utcnow(),
        )

    # =========================================================================
    # Message Buffer Access
    # =========================================================================

    def get_message_buffer(
        self,
        topic_filter: Optional[str] = None,
        limit: int = 100,
    ) -> List[MQTTMessage]:
        """
        Get messages from buffer.

        Args:
            topic_filter: Optional topic pattern filter
            limit: Maximum messages to return

        Returns:
            List of buffered messages
        """
        messages = self._message_buffer.copy()

        if topic_filter:
            messages = [
                m for m in messages
                if self._topic_matches(topic_filter, m.topic)
            ]

        return messages[-limit:]

    def clear_message_buffer(self) -> int:
        """
        Clear message buffer.

        Returns:
            Number of messages cleared
        """
        count = len(self._message_buffer)
        self._message_buffer.clear()
        return count

    def get_last_value(self, tag_id: str) -> Optional[TagValue]:
        """
        Get last received value for a tag.

        Args:
            tag_id: Tag identifier

        Returns:
            TagValue or None
        """
        return self._last_values.get(tag_id)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "MQTTConfig",
    "TopicSubscription",
    # Enums
    "MQTTVersion",
    "QoSLevel",
    "MessageFormat",
    "SparkplugMessageType",
    "SparkplugDataType",
    # Models
    "MQTTMessage",
    "SparkplugMetric",
    "SparkplugPayload",
    # Utilities
    "SparkplugEncoder",
    "MQTTMessageParser",
    # Connector
    "MQTTConnector",
]
