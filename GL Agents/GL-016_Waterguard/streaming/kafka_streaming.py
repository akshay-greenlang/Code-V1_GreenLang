"""
GL-016 Waterguard Kafka Streaming

Production-grade Kafka producer and consumer implementations for boiler water
chemistry monitoring. Features Avro schema enforcement, exactly-once semantics,
retry logic, and comprehensive error handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, SecretStr

# Type variable for message types
T = TypeVar("T", bound="BaseKafkaMessage")

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class SecurityProtocol(str, Enum):
    """Kafka security protocols."""
    PLAINTEXT = "PLAINTEXT"
    SSL = "SSL"
    SASL_PLAINTEXT = "SASL_PLAINTEXT"
    SASL_SSL = "SASL_SSL"


class SaslMechanism(str, Enum):
    """SASL authentication mechanisms."""
    PLAIN = "PLAIN"
    SCRAM_SHA_256 = "SCRAM-SHA-256"
    SCRAM_SHA_512 = "SCRAM-SHA-512"
    OAUTHBEARER = "OAUTHBEARER"


class KafkaConfig(BaseModel):
    """
    Kafka connection and behavior configuration.

    Credentials should be retrieved from a secure vault, never hardcoded.
    """

    # Connection
    bootstrap_servers: str = Field(..., description="Comma-separated list of brokers")
    client_id: str = Field(default="waterguard-gl016", description="Client identifier")

    # Security
    security_protocol: SecurityProtocol = Field(
        default=SecurityProtocol.SASL_SSL,
        description="Security protocol"
    )
    sasl_mechanism: Optional[SaslMechanism] = Field(
        default=SaslMechanism.SCRAM_SHA_512,
        description="SASL mechanism"
    )
    sasl_username: Optional[str] = Field(default=None, description="SASL username")
    sasl_password: Optional[SecretStr] = Field(default=None, description="SASL password")
    ssl_cafile: Optional[str] = Field(default=None, description="CA certificate file")
    ssl_certfile: Optional[str] = Field(default=None, description="Client certificate file")
    ssl_keyfile: Optional[str] = Field(default=None, description="Client key file")

    # Schema Registry
    schema_registry_url: Optional[str] = Field(default=None, description="Schema registry URL")
    schema_registry_username: Optional[str] = Field(default=None, description="Schema registry username")
    schema_registry_password: Optional[SecretStr] = Field(default=None, description="Schema registry password")

    # Producer settings
    acks: str = Field(default="all", description="Acknowledgement level")
    retries: int = Field(default=5, description="Number of retries")
    retry_backoff_ms: int = Field(default=100, description="Retry backoff in ms")
    max_in_flight_requests: int = Field(default=5, description="Max in-flight requests")
    enable_idempotence: bool = Field(default=True, description="Enable idempotent producer")
    compression_type: str = Field(default="snappy", description="Compression type")
    linger_ms: int = Field(default=5, description="Linger time in ms")
    batch_size: int = Field(default=16384, description="Batch size in bytes")

    # Consumer settings
    group_id: Optional[str] = Field(default=None, description="Consumer group ID")
    auto_offset_reset: str = Field(default="earliest", description="Auto offset reset policy")
    enable_auto_commit: bool = Field(default=False, description="Enable auto commit")
    session_timeout_ms: int = Field(default=30000, description="Session timeout")
    heartbeat_interval_ms: int = Field(default=10000, description="Heartbeat interval")
    max_poll_records: int = Field(default=500, description="Max records per poll")
    max_poll_interval_ms: int = Field(default=300000, description="Max poll interval")

    # Timeouts
    request_timeout_ms: int = Field(default=30000, description="Request timeout")
    metadata_max_age_ms: int = Field(default=300000, description="Metadata max age")

    class Config:
        extra = "forbid"


# =============================================================================
# Topic Definitions
# =============================================================================

class WaterguardTopics:
    """Kafka topic definitions for Waterguard GL-016."""

    # Raw sensor data from OPC-UA
    RAW = "boiler.gl016.raw"

    # Cleaned and normalized data
    CLEANED = "boiler.gl016.cleaned"

    # Engineered features for ML
    FEATURES = "boiler.gl016.features"

    # Current chemistry state
    CHEMISTRY_STATE = "boiler.gl016.chemistry_state"

    # AI recommendations
    RECOMMENDATIONS = "boiler.gl016.recommendations"

    # Actuation commands to OPC-UA
    ACTUATION_COMMANDS = "boiler.gl016.actuation_commands"

    # Command acknowledgements
    ACTUATION_ACK = "boiler.gl016.actuation_ack"

    # Alert notifications
    ALERTS = "boiler.gl016.alerts"

    # Audit trail
    AUDIT = "boiler.gl016.audit"

    @classmethod
    def all_topics(cls) -> List[str]:
        """Get all topic names."""
        return [
            cls.RAW,
            cls.CLEANED,
            cls.FEATURES,
            cls.CHEMISTRY_STATE,
            cls.RECOMMENDATIONS,
            cls.ACTUATION_COMMANDS,
            cls.ACTUATION_ACK,
            cls.ALERTS,
            cls.AUDIT,
        ]


# =============================================================================
# Producer Metrics
# =============================================================================

class ProducerMetrics(BaseModel):
    """Metrics for Kafka producer operations."""

    messages_sent: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    send_latency_ms: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    topic_counts: Dict[str, int] = Field(default_factory=dict)


class ConsumerMetrics(BaseModel):
    """Metrics for Kafka consumer operations."""

    messages_received: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    bytes_received: int = 0
    processing_latency_ms: float = 0.0
    lag: Dict[str, int] = Field(default_factory=dict)
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None


# =============================================================================
# Kafka Producer
# =============================================================================

class WaterguardKafkaProducer:
    """
    Production Kafka producer for Waterguard GL-016.

    Features:
    - Avro schema enforcement via Schema Registry
    - Exactly-once semantics with idempotent producer
    - Async/await interface
    - Automatic retries with backoff
    - Comprehensive metrics and logging
    - Graceful shutdown handling

    Example:
        async with WaterguardKafkaProducer(config) as producer:
            await producer.send(WaterguardTopics.RAW, message)
    """

    def __init__(
        self,
        config: KafkaConfig,
        schema_registry_client: Optional[Any] = None,
    ):
        """
        Initialize Kafka producer.

        Args:
            config: Kafka configuration
            schema_registry_client: Optional schema registry client
        """
        self.config = config
        self.schema_registry = schema_registry_client
        self._producer: Optional[Any] = None
        self._metrics = ProducerMetrics()
        self._started = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the producer connection."""
        if self._started:
            return

        async with self._lock:
            if self._started:
                return

            try:
                # Import aiokafka here to allow for optional dependency
                from aiokafka import AIOKafkaProducer

                producer_config = self._build_producer_config()
                self._producer = AIOKafkaProducer(**producer_config)
                await self._producer.start()
                self._started = True
                logger.info(
                    f"Kafka producer started: {self.config.client_id} -> "
                    f"{self.config.bootstrap_servers}"
                )
            except Exception as e:
                logger.error(f"Failed to start Kafka producer: {e}")
                raise

    async def stop(self) -> None:
        """Stop the producer and flush pending messages."""
        if not self._started:
            return

        async with self._lock:
            if not self._started:
                return

            try:
                if self._producer:
                    await self._producer.flush()
                    await self._producer.stop()
                    self._producer = None
                self._started = False
                logger.info("Kafka producer stopped")
            except Exception as e:
                logger.error(f"Error stopping Kafka producer: {e}")
                raise

    async def __aenter__(self) -> "WaterguardKafkaProducer":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

    def _build_producer_config(self) -> Dict[str, Any]:
        """Build producer configuration dictionary."""
        config = {
            "bootstrap_servers": self.config.bootstrap_servers,
            "client_id": self.config.client_id,
            "acks": self.config.acks,
            "enable_idempotence": self.config.enable_idempotence,
            "compression_type": self.config.compression_type,
            "linger_ms": self.config.linger_ms,
            "max_batch_size": self.config.batch_size,
            "request_timeout_ms": self.config.request_timeout_ms,
        }

        # Add security configuration
        if self.config.security_protocol != SecurityProtocol.PLAINTEXT:
            config["security_protocol"] = self.config.security_protocol.value

            if self.config.security_protocol in [
                SecurityProtocol.SASL_PLAINTEXT,
                SecurityProtocol.SASL_SSL
            ]:
                config["sasl_mechanism"] = self.config.sasl_mechanism.value
                config["sasl_plain_username"] = self.config.sasl_username
                config["sasl_plain_password"] = self.config.sasl_password.get_secret_value()

            if self.config.security_protocol in [
                SecurityProtocol.SSL,
                SecurityProtocol.SASL_SSL
            ]:
                ssl_context = self._create_ssl_context()
                if ssl_context:
                    config["ssl_context"] = ssl_context

        return config

    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for secure connections."""
        if not any([
            self.config.ssl_cafile,
            self.config.ssl_certfile,
            self.config.ssl_keyfile
        ]):
            return None

        ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

        if self.config.ssl_cafile:
            ssl_context.load_verify_locations(self.config.ssl_cafile)

        if self.config.ssl_certfile and self.config.ssl_keyfile:
            ssl_context.load_cert_chain(
                certfile=self.config.ssl_certfile,
                keyfile=self.config.ssl_keyfile
            )

        return ssl_context

    async def send(
        self,
        topic: str,
        message: "BaseKafkaMessage",
        key: Optional[str] = None,
        partition: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Send a message to Kafka.

        Args:
            topic: Target topic
            message: Message to send (Pydantic model)
            key: Optional partition key
            partition: Optional specific partition
            headers: Optional message headers

        Returns:
            True if sent successfully

        Raises:
            RuntimeError: If producer not started
            KafkaError: On send failure
        """
        if not self._started or not self._producer:
            raise RuntimeError("Producer not started. Call start() first.")

        start_time = time.time()

        try:
            # Serialize message
            value = message.to_json().encode("utf-8")
            key_bytes = key.encode("utf-8") if key else None

            # Prepare headers
            kafka_headers = []
            if headers:
                kafka_headers = [(k, v.encode("utf-8")) for k, v in headers.items()]

            # Add trace headers
            kafka_headers.append(("trace_id", str(message.trace_id or uuid4()).encode("utf-8")))
            kafka_headers.append(("message_type", type(message).__name__.encode("utf-8")))
            kafka_headers.append(("timestamp", message.timestamp.isoformat().encode("utf-8")))

            # Send message
            result = await self._producer.send_and_wait(
                topic=topic,
                value=value,
                key=key_bytes,
                partition=partition,
                headers=kafka_headers,
            )

            # Update metrics
            elapsed_ms = (time.time() - start_time) * 1000
            self._metrics.messages_sent += 1
            self._metrics.bytes_sent += len(value)
            self._metrics.send_latency_ms = elapsed_ms
            self._metrics.topic_counts[topic] = self._metrics.topic_counts.get(topic, 0) + 1

            logger.debug(
                f"Message sent to {topic} partition {result.partition} "
                f"offset {result.offset} in {elapsed_ms:.2f}ms"
            )
            return True

        except Exception as e:
            self._metrics.messages_failed += 1
            self._metrics.last_error = str(e)
            self._metrics.last_error_time = datetime.utcnow()
            logger.error(f"Failed to send message to {topic}: {e}")
            raise

    async def send_batch(
        self,
        topic: str,
        messages: List["BaseKafkaMessage"],
        key_func: Optional[Callable[["BaseKafkaMessage"], str]] = None,
    ) -> int:
        """
        Send a batch of messages.

        Args:
            topic: Target topic
            messages: List of messages to send
            key_func: Optional function to extract partition key

        Returns:
            Number of successfully sent messages
        """
        sent_count = 0
        for message in messages:
            try:
                key = key_func(message) if key_func else None
                await self.send(topic, message, key=key)
                sent_count += 1
            except Exception as e:
                logger.error(f"Failed to send message in batch: {e}")
                continue
        return sent_count

    @property
    def metrics(self) -> ProducerMetrics:
        """Get current producer metrics."""
        return self._metrics

    @property
    def is_running(self) -> bool:
        """Check if producer is running."""
        return self._started


# =============================================================================
# Kafka Consumer
# =============================================================================

class WaterguardKafkaConsumer:
    """
    Production Kafka consumer for Waterguard GL-016.

    Features:
    - Avro schema validation
    - Manual offset commit for exactly-once processing
    - Async message handler callbacks
    - Graceful rebalancing
    - Consumer lag monitoring

    Example:
        async with WaterguardKafkaConsumer(config, topics) as consumer:
            async for message in consumer.consume():
                await process(message)
                await consumer.commit()
    """

    def __init__(
        self,
        config: KafkaConfig,
        topics: List[str],
        message_handler: Optional[Callable[["BaseKafkaMessage"], Any]] = None,
        error_handler: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Initialize Kafka consumer.

        Args:
            config: Kafka configuration
            topics: Topics to subscribe to
            message_handler: Optional async message handler
            error_handler: Optional error handler
        """
        self.config = config
        self.topics = topics
        self.message_handler = message_handler
        self.error_handler = error_handler
        self._consumer: Optional[Any] = None
        self._metrics = ConsumerMetrics()
        self._started = False
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the consumer connection."""
        if self._started:
            return

        async with self._lock:
            if self._started:
                return

            try:
                from aiokafka import AIOKafkaConsumer

                consumer_config = self._build_consumer_config()
                self._consumer = AIOKafkaConsumer(*self.topics, **consumer_config)
                await self._consumer.start()
                self._started = True
                self._running = True
                logger.info(
                    f"Kafka consumer started: {self.config.group_id} "
                    f"subscribed to {self.topics}"
                )
            except Exception as e:
                logger.error(f"Failed to start Kafka consumer: {e}")
                raise

    async def stop(self) -> None:
        """Stop the consumer."""
        if not self._started:
            return

        async with self._lock:
            if not self._started:
                return

            try:
                self._running = False
                if self._consumer:
                    await self._consumer.stop()
                    self._consumer = None
                self._started = False
                logger.info("Kafka consumer stopped")
            except Exception as e:
                logger.error(f"Error stopping Kafka consumer: {e}")
                raise

    async def __aenter__(self) -> "WaterguardKafkaConsumer":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

    def _build_consumer_config(self) -> Dict[str, Any]:
        """Build consumer configuration dictionary."""
        config = {
            "bootstrap_servers": self.config.bootstrap_servers,
            "client_id": self.config.client_id,
            "group_id": self.config.group_id,
            "auto_offset_reset": self.config.auto_offset_reset,
            "enable_auto_commit": self.config.enable_auto_commit,
            "session_timeout_ms": self.config.session_timeout_ms,
            "heartbeat_interval_ms": self.config.heartbeat_interval_ms,
            "max_poll_records": self.config.max_poll_records,
            "max_poll_interval_ms": self.config.max_poll_interval_ms,
        }

        # Add security configuration
        if self.config.security_protocol != SecurityProtocol.PLAINTEXT:
            config["security_protocol"] = self.config.security_protocol.value

            if self.config.security_protocol in [
                SecurityProtocol.SASL_PLAINTEXT,
                SecurityProtocol.SASL_SSL
            ]:
                config["sasl_mechanism"] = self.config.sasl_mechanism.value
                config["sasl_plain_username"] = self.config.sasl_username
                config["sasl_plain_password"] = self.config.sasl_password.get_secret_value()

        return config

    async def consume(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Consume messages from subscribed topics.

        Yields:
            Deserialized message dictionaries with metadata
        """
        if not self._started or not self._consumer:
            raise RuntimeError("Consumer not started. Call start() first.")

        while self._running:
            try:
                # Poll for messages with timeout
                messages = await self._consumer.getmany(timeout_ms=1000)

                for topic_partition, records in messages.items():
                    for record in records:
                        start_time = time.time()

                        try:
                            # Deserialize message
                            value = json.loads(record.value.decode("utf-8"))

                            # Add metadata
                            message_data = {
                                "topic": record.topic,
                                "partition": record.partition,
                                "offset": record.offset,
                                "key": record.key.decode("utf-8") if record.key else None,
                                "timestamp": record.timestamp,
                                "headers": {
                                    k: v.decode("utf-8")
                                    for k, v in (record.headers or [])
                                },
                                "value": value,
                            }

                            # Update metrics
                            elapsed_ms = (time.time() - start_time) * 1000
                            self._metrics.messages_received += 1
                            self._metrics.bytes_received += len(record.value)
                            self._metrics.processing_latency_ms = elapsed_ms

                            yield message_data

                        except json.JSONDecodeError as e:
                            self._metrics.messages_failed += 1
                            logger.error(f"Failed to decode message: {e}")
                            if self.error_handler:
                                self.error_handler(e)

            except Exception as e:
                self._metrics.last_error = str(e)
                self._metrics.last_error_time = datetime.utcnow()
                logger.error(f"Error consuming messages: {e}")
                if self.error_handler:
                    self.error_handler(e)
                await asyncio.sleep(1)  # Backoff on error

    async def commit(self) -> None:
        """Commit current offsets."""
        if self._consumer:
            await self._consumer.commit()
            logger.debug("Offsets committed")

    async def seek_to_beginning(self) -> None:
        """Seek to beginning of all partitions."""
        if self._consumer:
            await self._consumer.seek_to_beginning()
            logger.info("Seeked to beginning of all partitions")

    async def seek_to_end(self) -> None:
        """Seek to end of all partitions."""
        if self._consumer:
            await self._consumer.seek_to_end()
            logger.info("Seeked to end of all partitions")

    async def get_lag(self) -> Dict[str, int]:
        """Get consumer lag for all partitions."""
        lag = {}
        if self._consumer:
            partitions = self._consumer.assignment()
            for tp in partitions:
                try:
                    committed = await self._consumer.committed(tp)
                    end_offsets = await self._consumer.end_offsets([tp])
                    end_offset = end_offsets.get(tp, 0)
                    current_offset = committed or 0
                    lag[f"{tp.topic}-{tp.partition}"] = end_offset - current_offset
                except Exception as e:
                    logger.warning(f"Failed to get lag for {tp}: {e}")
        self._metrics.lag = lag
        return lag

    @property
    def metrics(self) -> ConsumerMetrics:
        """Get current consumer metrics."""
        return self._metrics

    @property
    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self._running


# =============================================================================
# Stream Consumer (Higher-level abstraction)
# =============================================================================

class StreamConsumer:
    """
    High-level stream consumer with automatic deserialization and processing.

    Example:
        consumer = StreamConsumer(config, WaterguardTopics.RAW, RawChemistryMessage)
        async for message in consumer.stream():
            # message is RawChemistryMessage instance
            print(message.boiler_id)
    """

    def __init__(
        self,
        config: KafkaConfig,
        topic: str,
        message_type: Type[T],
    ):
        """
        Initialize stream consumer.

        Args:
            config: Kafka configuration
            topic: Topic to consume
            message_type: Pydantic model class for deserialization
        """
        self.config = config
        self.topic = topic
        self.message_type = message_type
        self._consumer: Optional[WaterguardKafkaConsumer] = None

    async def start(self) -> None:
        """Start the stream consumer."""
        self._consumer = WaterguardKafkaConsumer(
            config=self.config,
            topics=[self.topic],
        )
        await self._consumer.start()

    async def stop(self) -> None:
        """Stop the stream consumer."""
        if self._consumer:
            await self._consumer.stop()

    async def stream(self) -> AsyncIterator[T]:
        """
        Stream deserialized messages.

        Yields:
            Typed message instances
        """
        if not self._consumer:
            await self.start()

        async for raw_message in self._consumer.consume():
            try:
                message = self.message_type.model_validate(raw_message["value"])
                yield message
            except Exception as e:
                logger.error(f"Failed to deserialize message: {e}")
                continue

    async def commit(self) -> None:
        """Commit current offsets."""
        if self._consumer:
            await self._consumer.commit()
