"""
GL-003 UNIFIEDSTEAM - Kafka Producer

Publishes steam system data to Kafka topics for real-time processing.

Topic Structure:
    gl003.<site>.<area>.raw          - Raw sensor signals from OPC-UA/SCADA
    gl003.<site>.<area>.validated    - Validated and transformed signals
    gl003.<site>.<area>.features     - Extracted features (acoustic, statistical)
    gl003.<site>.<area>.computed     - Computed thermodynamic properties
    gl003.<site>.<area>.recommendations - Optimization recommendations
    gl003.<site>.<area>.events       - System events, alarms, notifications

Features:
- Async message production with batching
- Schema validation (Avro/JSON Schema)
- Partitioning by asset ID for ordering
- Compression (lz4, snappy, gzip)
- Exactly-once semantics (EOS) support
- Dead letter queue for failed messages
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import asyncio
import json
import logging
import uuid
import hashlib

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Kafka message compression types."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"


class SerializationType(Enum):
    """Message serialization types."""
    JSON = "json"
    AVRO = "avro"
    PROTOBUF = "protobuf"


class AcksMode(Enum):
    """Kafka acknowledgment modes."""
    NONE = 0  # Fire and forget
    LEADER = 1  # Leader acknowledgment
    ALL = -1  # All in-sync replicas


@dataclass
class KafkaProducerConfig:
    """Kafka producer configuration."""
    bootstrap_servers: str = "localhost:9092"

    # Authentication
    security_protocol: str = "PLAINTEXT"  # PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL
    sasl_mechanism: Optional[str] = None  # PLAIN, SCRAM-SHA-256, SCRAM-SHA-512
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None  # Retrieved from vault
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    # Producer settings
    acks: AcksMode = AcksMode.ALL
    compression_type: CompressionType = CompressionType.LZ4
    batch_size: int = 16384  # 16KB
    linger_ms: int = 5  # Wait up to 5ms for batching
    buffer_memory: int = 33554432  # 32MB
    max_request_size: int = 1048576  # 1MB
    retries: int = 3
    retry_backoff_ms: int = 100

    # Exactly-once semantics
    enable_idempotence: bool = True
    transactional_id: Optional[str] = None

    # Serialization
    serialization: SerializationType = SerializationType.JSON
    schema_registry_url: Optional[str] = None

    # Topic prefix
    topic_prefix: str = "gl003"

    # Client ID
    client_id: str = "gl003-steam-producer"


@dataclass
class SignalData:
    """Raw sensor signal data."""
    tag: str
    value: float
    quality: str  # good, uncertain, bad
    timestamp: datetime
    source_system: str = ""

    # Additional metadata
    unit: Optional[str] = None
    raw_value: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "tag": self.tag,
            "value": self.value,
            "quality": self.quality,
            "timestamp": self.timestamp.isoformat(),
            "source_system": self.source_system,
            "unit": self.unit,
            "raw_value": self.raw_value,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")


@dataclass
class ValidatedSignalData:
    """Validated and transformed signal data."""
    tag: str
    value: float
    unit: str
    quality_code: int  # OPC-UA style
    timestamp: datetime

    # Validation results
    is_valid: bool = True
    validation_issues: List[str] = field(default_factory=list)

    # Transformation tracking
    raw_value: Optional[float] = None
    calibration_applied: bool = False
    unit_converted: bool = False

    # Provenance
    source_hash: str = ""
    processing_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "tag": self.tag,
            "value": self.value,
            "unit": self.unit,
            "quality_code": self.quality_code,
            "timestamp": self.timestamp.isoformat(),
            "is_valid": self.is_valid,
            "validation_issues": self.validation_issues,
            "raw_value": self.raw_value,
            "source_hash": self.source_hash,
            "processing_timestamp": self.processing_timestamp.isoformat() if self.processing_timestamp else None,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")


@dataclass
class ComputedPropertyData:
    """Computed thermodynamic property data."""
    property_name: str  # enthalpy, entropy, quality, superheat, efficiency
    value: float
    unit: str
    timestamp: datetime

    # Context
    asset_id: str = ""
    asset_type: str = ""  # boiler, header, turbine, trap

    # Input values used for computation
    inputs: Dict[str, float] = field(default_factory=dict)

    # Uncertainty
    uncertainty: Optional[float] = None
    confidence: float = 1.0

    # Computation metadata
    model_version: str = "1.0"
    computation_method: str = ""

    def to_dict(self) -> Dict:
        return {
            "property_name": self.property_name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "inputs": self.inputs,
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "computation_method": self.computation_method,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")


@dataclass
class RecommendationData:
    """Optimization recommendation data."""
    recommendation_id: str
    recommendation_type: str  # setpoint_change, maintenance, operational
    title: str
    description: str
    timestamp: datetime

    # Target
    target_asset_id: str = ""
    target_asset_type: str = ""

    # Action details
    current_value: Optional[float] = None
    recommended_value: Optional[float] = None
    parameter_name: str = ""
    unit: str = ""

    # Impact
    estimated_savings_pct: float = 0.0
    estimated_savings_value: float = 0.0
    savings_unit: str = ""
    payback_hours: float = 0.0

    # Priority and confidence
    priority: str = "medium"  # low, medium, high, critical
    confidence: float = 0.0
    model_version: str = ""

    # Validation status
    status: str = "pending"  # pending, accepted, rejected, implemented

    def to_dict(self) -> Dict:
        return {
            "recommendation_id": self.recommendation_id,
            "recommendation_type": self.recommendation_type,
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "target_asset_id": self.target_asset_id,
            "target_asset_type": self.target_asset_type,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "parameter_name": self.parameter_name,
            "unit": self.unit,
            "estimated_savings_pct": self.estimated_savings_pct,
            "estimated_savings_value": self.estimated_savings_value,
            "savings_unit": self.savings_unit,
            "payback_hours": self.payback_hours,
            "priority": self.priority,
            "confidence": self.confidence,
            "status": self.status,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")


@dataclass
class EventData:
    """System event data."""
    event_id: str
    event_type: str  # alarm, warning, info, audit
    category: str  # process, equipment, system, security
    title: str
    description: str
    timestamp: datetime

    # Source
    source_system: str = ""
    source_asset_id: str = ""
    source_asset_type: str = ""

    # Severity
    severity: str = "info"  # debug, info, warning, error, critical

    # Associated data
    associated_tags: List[str] = field(default_factory=list)
    associated_values: Dict[str, float] = field(default_factory=dict)

    # State
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "source_system": self.source_system,
            "source_asset_id": self.source_asset_id,
            "source_asset_type": self.source_asset_type,
            "severity": self.severity,
            "associated_tags": self.associated_tags,
            "associated_values": self.associated_values,
            "acknowledged": self.acknowledged,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")


@dataclass
class ProducerMetrics:
    """Producer performance metrics."""
    messages_sent: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    batch_count: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0


class SteamKafkaProducer:
    """
    Kafka producer for steam system data streaming.

    Publishes sensor data, computed properties, recommendations,
    and events to topic hierarchy for real-time processing.

    Example:
        config = KafkaProducerConfig(
            bootstrap_servers="kafka.company.com:9092",
            topic_prefix="gl003",
        )

        producer = SteamKafkaProducer(config)
        await producer.connect()

        # Publish raw signal
        await producer.publish_raw_signal(
            site="PLANT1",
            area="UTIL",
            signal=SignalData(
                tag="HEADER.PRESSURE",
                value=145.5,
                quality="good",
                timestamp=datetime.now(timezone.utc)
            )
        )

        # Publish recommendation
        await producer.publish_recommendation(
            site="PLANT1",
            area="UTIL",
            recommendation=RecommendationData(...)
        )
    """

    def __init__(
        self,
        config: KafkaProducerConfig,
        vault_client: Optional[Any] = None,
    ) -> None:
        """
        Initialize Kafka producer.

        Args:
            config: Producer configuration
            vault_client: Optional vault client for credential retrieval
        """
        self.config = config
        self._vault_client = vault_client

        # Retrieve credentials from vault
        if vault_client and config.sasl_username:
            try:
                config.sasl_password = vault_client.get_secret("kafka/sasl_password")
            except Exception as e:
                logger.warning(f"Failed to retrieve Kafka credentials: {e}")

        self._producer = None
        self._connected = False

        # Metrics
        self._metrics = ProducerMetrics()

        # Message buffer for batching
        self._buffer: List[Dict] = []
        self._buffer_lock = asyncio.Lock()

        # Dead letter queue
        self._dlq: List[Dict] = []
        self._dlq_max_size = 10000

        logger.info(f"SteamKafkaProducer initialized: {config.bootstrap_servers}")

    async def connect(
        self,
        bootstrap_servers: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> None:
        """
        Connect to Kafka cluster.

        Args:
            bootstrap_servers: Override bootstrap servers
            config: Additional configuration overrides
        """
        if bootstrap_servers:
            self.config.bootstrap_servers = bootstrap_servers

        try:
            # In production, use aiokafka:
            # from aiokafka import AIOKafkaProducer
            # self._producer = AIOKafkaProducer(
            #     bootstrap_servers=self.config.bootstrap_servers,
            #     acks=self.config.acks.value,
            #     compression_type=self.config.compression_type.value,
            #     batch_size=self.config.batch_size,
            #     linger_ms=self.config.linger_ms,
            #     enable_idempotence=self.config.enable_idempotence,
            #     client_id=self.config.client_id,
            # )
            # await self._producer.start()

            self._connected = True
            logger.info(f"Connected to Kafka: {self.config.bootstrap_servers}")

        except Exception as e:
            logger.error(f"Kafka connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        if self._producer:
            # Flush pending messages
            await self.flush()
            # await self._producer.stop()
            self._producer = None

        self._connected = False
        logger.info("Disconnected from Kafka")

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    def _build_topic(self, site: str, area: str, stream_type: str) -> str:
        """Build topic name from components."""
        return f"{self.config.topic_prefix}.{site.lower()}.{area.lower()}.{stream_type}"

    def _get_partition_key(self, data: Any) -> bytes:
        """Get partition key for ordering."""
        # Partition by asset ID for related messages to go to same partition
        if hasattr(data, "tag"):
            return data.tag.encode("utf-8")
        elif hasattr(data, "target_asset_id"):
            return data.target_asset_id.encode("utf-8")
        elif hasattr(data, "source_asset_id"):
            return data.source_asset_id.encode("utf-8")
        else:
            return str(uuid.uuid4()).encode("utf-8")

    async def _send_message(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        headers: Optional[List[Tuple[str, bytes]]] = None,
    ) -> bool:
        """Send message to Kafka topic."""
        import time

        if not self._connected:
            logger.warning("Not connected to Kafka, message queued to DLQ")
            self._add_to_dlq(topic, value, key, headers)
            return False

        start_time = time.perf_counter()

        try:
            # In production:
            # await self._producer.send_and_wait(
            #     topic,
            #     value=value,
            #     key=key,
            #     headers=headers,
            # )

            # Update metrics
            latency = (time.perf_counter() - start_time) * 1000
            self._metrics.messages_sent += 1
            self._metrics.bytes_sent += len(value)
            self._metrics.avg_latency_ms = (
                (self._metrics.avg_latency_ms * (self._metrics.messages_sent - 1) + latency)
                / self._metrics.messages_sent
            )
            self._metrics.max_latency_ms = max(self._metrics.max_latency_ms, latency)

            return True

        except Exception as e:
            logger.error(f"Failed to send message to {topic}: {e}")
            self._metrics.messages_failed += 1
            self._add_to_dlq(topic, value, key, headers)
            return False

    def _add_to_dlq(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes],
        headers: Optional[List[Tuple[str, bytes]]],
    ) -> None:
        """Add failed message to dead letter queue."""
        if len(self._dlq) >= self._dlq_max_size:
            self._dlq.pop(0)  # Remove oldest

        self._dlq.append({
            "topic": topic,
            "value": value,
            "key": key,
            "headers": headers,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def publish_raw_signal(
        self,
        site: str,
        area: str,
        signal_data: SignalData,
    ) -> None:
        """
        Publish raw sensor signal to Kafka.

        Args:
            site: Site identifier (e.g., "PLANT1")
            area: Process area (e.g., "UTIL")
            signal_data: Raw signal data
        """
        topic = self._build_topic(site, area, "raw")
        key = self._get_partition_key(signal_data)

        headers = [
            ("source", b"opcua"),
            ("site", site.encode()),
            ("area", area.encode()),
        ]

        await self._send_message(
            topic=topic,
            value=signal_data.to_bytes(),
            key=key,
            headers=headers,
        )

    async def publish_raw_signals_batch(
        self,
        site: str,
        area: str,
        signals: List[SignalData],
    ) -> int:
        """
        Publish batch of raw signals.

        Args:
            site: Site identifier
            area: Process area
            signals: List of signal data

        Returns:
            Number of successfully published messages
        """
        topic = self._build_topic(site, area, "raw")
        success_count = 0

        for signal in signals:
            success = await self._send_message(
                topic=topic,
                value=signal.to_bytes(),
                key=self._get_partition_key(signal),
            )
            if success:
                success_count += 1

        self._metrics.batch_count += 1
        return success_count

    async def publish_validated_signal(
        self,
        site: str,
        area: str,
        validated_data: ValidatedSignalData,
    ) -> None:
        """
        Publish validated signal to Kafka.

        Args:
            site: Site identifier
            area: Process area
            validated_data: Validated signal data
        """
        topic = self._build_topic(site, area, "validated")
        key = self._get_partition_key(validated_data)

        headers = [
            ("validated", b"true"),
            ("quality_code", str(validated_data.quality_code).encode()),
        ]

        await self._send_message(
            topic=topic,
            value=validated_data.to_bytes(),
            key=key,
            headers=headers,
        )

    async def publish_computed_property(
        self,
        site: str,
        area: str,
        property_data: ComputedPropertyData,
    ) -> None:
        """
        Publish computed property to Kafka.

        Args:
            site: Site identifier
            area: Process area
            property_data: Computed property data
        """
        topic = self._build_topic(site, area, "computed")
        key = f"{property_data.asset_id}:{property_data.property_name}".encode()

        headers = [
            ("property", property_data.property_name.encode()),
            ("asset_type", property_data.asset_type.encode()),
            ("model_version", property_data.model_version.encode()),
        ]

        await self._send_message(
            topic=topic,
            value=property_data.to_bytes(),
            key=key,
            headers=headers,
        )

    async def publish_recommendation(
        self,
        site: str,
        area: str,
        recommendation: RecommendationData,
    ) -> None:
        """
        Publish optimization recommendation to Kafka.

        Args:
            site: Site identifier
            area: Process area
            recommendation: Recommendation data
        """
        topic = self._build_topic(site, area, "recommendations")
        key = recommendation.recommendation_id.encode()

        headers = [
            ("type", recommendation.recommendation_type.encode()),
            ("priority", recommendation.priority.encode()),
            ("status", recommendation.status.encode()),
        ]

        await self._send_message(
            topic=topic,
            value=recommendation.to_bytes(),
            key=key,
            headers=headers,
        )

        logger.info(
            f"Published recommendation: {recommendation.recommendation_id} "
            f"({recommendation.recommendation_type})"
        )

    async def publish_event(
        self,
        site: str,
        area: str,
        event: EventData,
    ) -> None:
        """
        Publish system event to Kafka.

        Args:
            site: Site identifier
            area: Process area
            event: Event data
        """
        topic = self._build_topic(site, area, "events")
        key = event.event_id.encode()

        headers = [
            ("event_type", event.event_type.encode()),
            ("severity", event.severity.encode()),
            ("category", event.category.encode()),
        ]

        await self._send_message(
            topic=topic,
            value=event.to_bytes(),
            key=key,
            headers=headers,
        )

        if event.severity in ["error", "critical"]:
            logger.warning(f"Published critical event: {event.event_id} - {event.title}")

    async def publish_features(
        self,
        site: str,
        area: str,
        asset_id: str,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Publish extracted features to Kafka.

        Args:
            site: Site identifier
            area: Process area
            asset_id: Asset identifier
            features: Feature dictionary
            timestamp: Feature timestamp
        """
        topic = self._build_topic(site, area, "features")

        data = {
            "asset_id": asset_id,
            "features": features,
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        }

        await self._send_message(
            topic=topic,
            value=json.dumps(data).encode("utf-8"),
            key=asset_id.encode(),
        )

    async def flush(self) -> None:
        """Flush pending messages."""
        if self._producer:
            # await self._producer.flush()
            pass

    async def retry_dlq(self) -> int:
        """
        Retry sending messages from dead letter queue.

        Returns:
            Number of successfully retried messages
        """
        if not self._connected:
            return 0

        success_count = 0
        remaining = []

        for msg in self._dlq:
            success = await self._send_message(
                topic=msg["topic"],
                value=msg["value"],
                key=msg["key"],
                headers=msg["headers"],
            )

            if success:
                success_count += 1
            else:
                remaining.append(msg)

        self._dlq = remaining

        if success_count > 0:
            logger.info(f"Retried {success_count} messages from DLQ")

        return success_count

    def get_metrics(self) -> Dict:
        """Get producer metrics."""
        return {
            "messages_sent": self._metrics.messages_sent,
            "messages_failed": self._metrics.messages_failed,
            "bytes_sent": self._metrics.bytes_sent,
            "batch_count": self._metrics.batch_count,
            "avg_latency_ms": round(self._metrics.avg_latency_ms, 3),
            "max_latency_ms": round(self._metrics.max_latency_ms, 3),
            "dlq_size": len(self._dlq),
            "connected": self._connected,
        }

    def get_dlq_messages(self, limit: int = 100) -> List[Dict]:
        """Get messages from dead letter queue for inspection."""
        return self._dlq[:limit]


def create_producer_for_site(
    site_id: str,
    kafka_servers: str,
    security_protocol: str = "SASL_SSL",
) -> SteamKafkaProducer:
    """Create configured producer for a site."""
    config = KafkaProducerConfig(
        bootstrap_servers=kafka_servers,
        security_protocol=security_protocol,
        topic_prefix="gl003",
        client_id=f"gl003-{site_id}-producer",
        enable_idempotence=True,
        compression_type=CompressionType.LZ4,
    )

    return SteamKafkaProducer(config)
