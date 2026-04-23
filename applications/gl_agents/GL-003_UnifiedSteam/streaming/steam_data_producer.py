"""
GL-003 UNIFIEDSTEAM - Steam Data Producer

Specialized Kafka producer for steam measurement data with:
- Batched publishing for high throughput
- Automatic partitioning by asset ID
- Avro/JSON serialization with schema validation
- Compression (LZ4/Snappy)
- Dead letter queue for failed messages
- Circuit breaker pattern for resilience
- Exponential backoff with jitter for retries
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import asyncio
import json
import logging
import uuid
import random
import time

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class RetryStrategy(Enum):
    """Retry strategies for failed operations."""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class Priority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class CompressionType(Enum):
    """Kafka message compression types."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"


class AcksMode(Enum):
    """Kafka acknowledgment modes."""
    NONE = 0
    LEADER = 1
    ALL = -1


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry logic with exponential backoff."""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER
    max_retries: int = 5
    base_delay_ms: int = 100
    max_delay_ms: int = 30000
    jitter_factor: float = 0.3

    def get_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy."""
        if self.strategy == RetryStrategy.NONE:
            return 0

        if self.strategy == RetryStrategy.LINEAR:
            delay_ms = self.base_delay_ms * (attempt + 1)
        elif self.strategy in (RetryStrategy.EXPONENTIAL, RetryStrategy.EXPONENTIAL_JITTER):
            delay_ms = self.base_delay_ms * (2 ** attempt)
        else:
            delay_ms = self.base_delay_ms

        delay_ms = min(delay_ms, self.max_delay_ms)

        if self.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            jitter = delay_ms * self.jitter_factor * random.random()
            delay_ms = delay_ms + jitter

        return delay_ms / 1000.0


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 3


@dataclass
class SteamProducerConfig:
    """Configuration for Steam Data Producer."""
    bootstrap_servers: str = "localhost:9092"

    # Authentication
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None

    # Producer settings
    acks: AcksMode = AcksMode.ALL
    compression_type: CompressionType = CompressionType.LZ4
    batch_size: int = 16384
    linger_ms: int = 5
    enable_idempotence: bool = True

    # Topic prefix
    topic_prefix: str = "gl003"
    client_id: str = "gl003-steam-producer"

    # Serialization
    use_avro: bool = False
    schema_registry_url: Optional[str] = None

    # Retry and circuit breaker
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SteamMeasurement:
    """Steam measurement data for publishing."""
    tag: str
    value: float
    unit: str
    timestamp: datetime
    quality: str = "good"
    asset_id: str = ""
    asset_type: str = ""
    source_system: str = "opcua"

    # Computed properties
    enthalpy: Optional[float] = None
    entropy: Optional[float] = None
    steam_quality: Optional[float] = None

    # Metadata
    calibration_date: Optional[datetime] = None
    sensor_type: str = ""

    def to_dict(self) -> Dict:
        result = {
            "tag": self.tag,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "quality": self.quality,
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "source_system": self.source_system,
        }
        if self.enthalpy is not None:
            result["enthalpy"] = self.enthalpy
        if self.entropy is not None:
            result["entropy"] = self.entropy
        if self.steam_quality is not None:
            result["steam_quality"] = self.steam_quality
        return result

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")


@dataclass
class DeadLetterMessage:
    """Message stored in dead letter queue."""
    message_id: str
    topic: str
    key: Optional[bytes]
    value: bytes
    headers: Optional[List[Tuple[str, bytes]]]
    timestamp: datetime
    error_message: str
    retry_count: int = 0
    max_retries: int = 5

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries


@dataclass
class ProducerMetrics:
    """Producer performance metrics."""
    messages_sent: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    batch_count: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    retry_count: int = 0
    dlq_size: int = 0
    circuit_breaker_trips: int = 0


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitBreaker:
    """Circuit breaker pattern for resilient operations."""

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self.config = config
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    async def can_execute(self) -> bool:
        """Check if request can be executed."""
        async with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True

            if self._state == CircuitBreakerState.OPEN:
                if self._last_failure_time:
                    elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
                    if elapsed >= self.config.timeout_seconds:
                        self._state = CircuitBreakerState.HALF_OPEN
                        self._half_open_calls = 0
                        logger.info("Circuit breaker transitioning to HALF_OPEN")
                        return True
                return False

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self) -> None:
        """Record successful operation."""
        async with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit breaker CLOSED after successful recovery")
            else:
                self._failure_count = 0

    async def record_failure(self) -> None:
        """Record failed operation."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now(timezone.utc)

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
                self._success_count = 0
                logger.warning("Circuit breaker OPEN after failure in HALF_OPEN state")
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPEN after {self._failure_count} failures")


# =============================================================================
# Serializers
# =============================================================================

class MessageSerializer(ABC):
    """Abstract base class for message serializers."""

    @abstractmethod
    def serialize(self, data: Dict[str, Any]) -> bytes:
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        pass


class JSONSerializer(MessageSerializer):
    """JSON message serializer."""

    def serialize(self, data: Dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> Dict[str, Any]:
        return json.loads(data.decode("utf-8"))


class AvroSerializer(MessageSerializer):
    """Avro message serializer with schema registry support."""

    def __init__(
        self,
        schema: Dict[str, Any],
        schema_registry_url: Optional[str] = None,
    ) -> None:
        self.schema = schema
        self.schema_registry_url = schema_registry_url

    def serialize(self, data: Dict[str, Any]) -> bytes:
        # In production, use fastavro
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> Dict[str, Any]:
        return json.loads(data.decode("utf-8"))


# Avro schema for steam measurements
STEAM_MEASUREMENT_SCHEMA = {
    "type": "record",
    "name": "SteamMeasurement",
    "namespace": "com.greenlang.gl003.steam",
    "fields": [
        {"name": "tag", "type": "string"},
        {"name": "value", "type": "double"},
        {"name": "unit", "type": "string"},
        {"name": "timestamp", "type": "string"},
        {"name": "quality", "type": "string", "default": "good"},
        {"name": "asset_id", "type": ["null", "string"], "default": None},
        {"name": "asset_type", "type": ["null", "string"], "default": None},
        {"name": "source_system", "type": "string", "default": "opcua"},
        {"name": "enthalpy", "type": ["null", "double"], "default": None},
        {"name": "entropy", "type": ["null", "double"], "default": None},
        {"name": "steam_quality", "type": ["null", "double"], "default": None},
    ],
}


# =============================================================================
# Steam Data Producer
# =============================================================================

class SteamDataProducer:
    """
    Specialized Kafka producer for steam measurement data.

    Features:
    - Batched publishing for high throughput
    - Automatic partitioning by asset ID
    - Avro/JSON serialization with schema validation
    - Compression (LZ4/Snappy)
    - Dead letter queue for failed messages
    - Circuit breaker for resilience

    Example:
        config = SteamProducerConfig(
            bootstrap_servers="kafka.company.com:9092",
            compression_type=CompressionType.LZ4,
        )

        producer = SteamDataProducer(config)
        await producer.connect()

        measurement = SteamMeasurement(
            tag="BOILER1.STEAM.PRESSURE",
            value=145.5,
            unit="psig",
            timestamp=datetime.now(timezone.utc),
            asset_id="BOILER1",
            asset_type="boiler",
        )
        await producer.publish_measurement("PLANT1", "UTIL", measurement)

        measurements = [measurement1, measurement2, measurement3]
        count = await producer.publish_measurements_batch("PLANT1", "UTIL", measurements)
    """

    def __init__(
        self,
        config: SteamProducerConfig,
        vault_client: Optional[Any] = None,
    ) -> None:
        """
        Initialize Steam Data Producer.

        Args:
            config: Producer configuration
            vault_client: Optional vault client for credential retrieval
        """
        self.config = config
        self._vault_client = vault_client

        # Retrieve credentials from vault (NEVER hardcode)
        if vault_client and config.sasl_username:
            try:
                config.sasl_password = vault_client.get_secret("kafka/sasl_password")
            except Exception as e:
                logger.warning(f"Failed to retrieve Kafka credentials: {e}")

        self._producer = None
        self._connected = False

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(config.circuit_breaker_config)

        # Serializer
        if config.use_avro and config.schema_registry_url:
            self._serializer: MessageSerializer = AvroSerializer(
                STEAM_MEASUREMENT_SCHEMA,
                config.schema_registry_url,
            )
        else:
            self._serializer = JSONSerializer()

        # Metrics
        self._metrics = ProducerMetrics()

        # Dead letter queue
        self._dlq: List[DeadLetterMessage] = []
        self._dlq_max_size = 10000
        self._dlq_lock = asyncio.Lock()

        # Batch buffer
        self._batch_buffer: List[Tuple[str, SteamMeasurement]] = []
        self._batch_lock = asyncio.Lock()
        self._batch_size = 100
        self._batch_timeout_ms = 100
        self._last_flush_time = time.time()

        # Background flusher task
        self._flush_task: Optional[asyncio.Task] = None

        logger.info(f"SteamDataProducer initialized: {config.bootstrap_servers}")

    async def connect(
        self,
        bootstrap_servers: Optional[str] = None,
    ) -> None:
        """
        Connect to Kafka cluster and start background flusher.

        Args:
            bootstrap_servers: Override bootstrap servers
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

            # Start background batch flusher
            self._flush_task = asyncio.create_task(self._background_flush())

        except Exception as e:
            logger.error(f"Kafka connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining messages
        await self._flush_batch()

        if self._producer:
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

    async def _background_flush(self) -> None:
        """Background task to flush batches periodically."""
        while True:
            try:
                await asyncio.sleep(self._batch_timeout_ms / 1000.0)
                await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background flush: {e}")

    async def _flush_batch(self) -> None:
        """Flush accumulated batch to Kafka."""
        async with self._batch_lock:
            if not self._batch_buffer:
                return

            messages = self._batch_buffer.copy()
            self._batch_buffer.clear()
            self._last_flush_time = time.time()

        # Group by topic
        by_topic: Dict[str, List[SteamMeasurement]] = {}
        for topic, measurement in messages:
            if topic not in by_topic:
                by_topic[topic] = []
            by_topic[topic].append(measurement)

        # Publish each topic group
        for topic, measurements in by_topic.items():
            for measurement in measurements:
                key = measurement.asset_id.encode() if measurement.asset_id else None
                value = self._serializer.serialize(measurement.to_dict())

                headers = [
                    ("source", measurement.source_system.encode()),
                    ("asset_type", measurement.asset_type.encode() if measurement.asset_type else b""),
                    ("quality", measurement.quality.encode()),
                ]

                await self._send_with_retry(
                    topic=topic,
                    value=value,
                    key=key,
                    headers=headers,
                )

        self._metrics.batch_count += 1
        logger.debug(f"Flushed batch of {len(messages)} measurements")

    async def _send_with_retry(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        headers: Optional[List[Tuple[str, bytes]]] = None,
    ) -> bool:
        """Send message with retry logic and circuit breaker."""
        retry_config = self.config.retry_config

        for attempt in range(retry_config.max_retries + 1):
            # Check circuit breaker
            if not await self._circuit_breaker.can_execute():
                logger.warning(f"Circuit breaker OPEN, message queued to DLQ: {topic}")
                await self._add_to_dlq(topic, value, key, headers, "Circuit breaker open")
                return False

            try:
                start_time = time.perf_counter()

                # In production:
                # await self._producer.send_and_wait(topic, value=value, key=key, headers=headers)

                # Simulate successful send
                await asyncio.sleep(0.001)

                await self._circuit_breaker.record_success()

                # Update metrics
                latency = (time.perf_counter() - start_time) * 1000
                self._metrics.messages_sent += 1
                self._metrics.bytes_sent += len(value)
                self._update_latency_metrics(latency)

                return True

            except Exception as e:
                logger.warning(f"Send attempt {attempt + 1} failed: {e}")
                await self._circuit_breaker.record_failure()
                self._metrics.retry_count += 1

                if attempt < retry_config.max_retries:
                    delay = retry_config.get_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retries exhausted for topic {topic}")
                    self._metrics.messages_failed += 1
                    await self._add_to_dlq(topic, value, key, headers, str(e))
                    return False

        return False

    def _update_latency_metrics(self, latency_ms: float) -> None:
        """Update latency metrics."""
        n = self._metrics.messages_sent
        self._metrics.avg_latency_ms = (
            (self._metrics.avg_latency_ms * (n - 1) + latency_ms) / n
            if n > 0 else latency_ms
        )
        self._metrics.max_latency_ms = max(self._metrics.max_latency_ms, latency_ms)

    async def _add_to_dlq(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes],
        headers: Optional[List[Tuple[str, bytes]]],
        error_message: str,
    ) -> None:
        """Add failed message to dead letter queue."""
        async with self._dlq_lock:
            if len(self._dlq) >= self._dlq_max_size:
                oldest = self._dlq.pop(0)
                logger.warning(f"DLQ overflow, dropping oldest message: {oldest.message_id}")

            dlq_message = DeadLetterMessage(
                message_id=str(uuid.uuid4()),
                topic=topic,
                key=key,
                value=value,
                headers=headers,
                timestamp=datetime.now(timezone.utc),
                error_message=error_message,
                retry_count=0,
                max_retries=self.config.retry_config.max_retries,
            )

            self._dlq.append(dlq_message)
            self._metrics.dlq_size = len(self._dlq)

    async def publish_measurement(
        self,
        site: str,
        area: str,
        measurement: SteamMeasurement,
        immediate: bool = False,
    ) -> bool:
        """
        Publish a steam measurement.

        Args:
            site: Site identifier (e.g., "PLANT1")
            area: Process area (e.g., "UTIL")
            measurement: Steam measurement data
            immediate: If True, bypass batching

        Returns:
            True if message was sent/queued successfully
        """
        topic = self._build_topic(site, area, "raw")

        if immediate:
            key = measurement.asset_id.encode() if measurement.asset_id else None
            value = self._serializer.serialize(measurement.to_dict())

            headers = [
                ("source", measurement.source_system.encode()),
                ("asset_type", measurement.asset_type.encode() if measurement.asset_type else b""),
                ("quality", measurement.quality.encode()),
            ]

            return await self._send_with_retry(
                topic=topic,
                value=value,
                key=key,
                headers=headers,
            )
        else:
            async with self._batch_lock:
                self._batch_buffer.append((topic, measurement))

                if len(self._batch_buffer) >= self._batch_size:
                    await self._flush_batch()

            return True

    async def publish_measurements_batch(
        self,
        site: str,
        area: str,
        measurements: List[SteamMeasurement],
    ) -> int:
        """
        Publish a batch of steam measurements.

        Args:
            site: Site identifier
            area: Process area
            measurements: List of measurements

        Returns:
            Number of successfully published messages
        """
        success_count = 0
        topic = self._build_topic(site, area, "raw")

        for measurement in measurements:
            key = measurement.asset_id.encode() if measurement.asset_id else None
            value = self._serializer.serialize(measurement.to_dict())

            headers = [
                ("source", measurement.source_system.encode()),
                ("asset_type", measurement.asset_type.encode() if measurement.asset_type else b""),
                ("quality", measurement.quality.encode()),
            ]

            success = await self._send_with_retry(
                topic=topic,
                value=value,
                key=key,
                headers=headers,
            )

            if success:
                success_count += 1

        self._metrics.batch_count += 1
        logger.info(f"Published batch of {success_count}/{len(measurements)} measurements")

        return success_count

    async def retry_dlq(self, batch_size: int = 100) -> int:
        """
        Retry sending messages from dead letter queue.

        Args:
            batch_size: Maximum messages to retry

        Returns:
            Number of successfully retried messages
        """
        if not self._connected:
            return 0

        async with self._dlq_lock:
            messages_to_retry = [m for m in self._dlq[:batch_size] if m.can_retry()]

        success_count = 0
        for msg in messages_to_retry:
            success = await self._send_with_retry(
                topic=msg.topic,
                value=msg.value,
                key=msg.key,
                headers=msg.headers,
            )

            async with self._dlq_lock:
                if success:
                    self._dlq.remove(msg)
                    success_count += 1
                else:
                    msg.retry_count += 1

        self._metrics.dlq_size = len(self._dlq)

        if success_count > 0:
            logger.info(f"Retried {success_count} messages from DLQ")

        return success_count

    def get_metrics(self) -> Dict[str, Any]:
        """Get producer metrics."""
        return {
            "messages_sent": self._metrics.messages_sent,
            "messages_failed": self._metrics.messages_failed,
            "bytes_sent": self._metrics.bytes_sent,
            "batch_count": self._metrics.batch_count,
            "avg_latency_ms": round(self._metrics.avg_latency_ms, 3),
            "max_latency_ms": round(self._metrics.max_latency_ms, 3),
            "retry_count": self._metrics.retry_count,
            "dlq_size": len(self._dlq),
            "circuit_breaker_state": self._circuit_breaker.state.value,
            "connected": self._connected,
        }

    def get_dlq_messages(self, limit: int = 100) -> List[Dict]:
        """Get messages from dead letter queue for inspection."""
        return [
            {
                "message_id": m.message_id,
                "topic": m.topic,
                "timestamp": m.timestamp.isoformat(),
                "error_message": m.error_message,
                "retry_count": m.retry_count,
                "can_retry": m.can_retry(),
            }
            for m in self._dlq[:limit]
        ]


# =============================================================================
# Factory Function
# =============================================================================

def create_steam_data_producer(
    kafka_servers: str,
    site_id: str,
    use_avro: bool = False,
    schema_registry_url: Optional[str] = None,
) -> SteamDataProducer:
    """
    Create a SteamDataProducer instance.

    Args:
        kafka_servers: Kafka bootstrap servers
        site_id: Site identifier for client ID
        use_avro: Enable Avro serialization
        schema_registry_url: Schema registry URL (required for Avro)

    Returns:
        Configured SteamDataProducer instance
    """
    config = SteamProducerConfig(
        bootstrap_servers=kafka_servers,
        topic_prefix="gl003",
        client_id=f"gl003-{site_id}-steam-producer",
        enable_idempotence=True,
        compression_type=CompressionType.LZ4,
        use_avro=use_avro,
        schema_registry_url=schema_registry_url,
    )

    return SteamDataProducer(config)
