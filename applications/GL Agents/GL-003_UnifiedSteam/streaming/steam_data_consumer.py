"""
GL-003 UNIFIEDSTEAM - Steam Data Consumer

Specialized Kafka consumer for steam measurement data with:
- Consumer group management for scalability
- Offset tracking and commit strategies
- Schema validation with poison pill detection
- Batch processing support
- Rebalance handling
- Lag monitoring
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, AsyncIterator, Union
import asyncio
import json
import logging
import uuid
import time

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class OffsetResetPolicy(Enum):
    """Offset reset policy for new consumer groups."""
    EARLIEST = "earliest"
    LATEST = "latest"
    NONE = "none"


class CommitStrategy(Enum):
    """Offset commit strategies."""
    AUTO = "auto"
    MANUAL_SYNC = "manual_sync"
    MANUAL_ASYNC = "manual_async"
    BATCH = "batch"


class ProcessingState(Enum):
    """Message processing states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    POISON_PILL = "poison_pill"


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class ConsumerConfig:
    """Kafka consumer configuration."""
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "gl003-steam-consumer"

    # Authentication
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None

    # Consumer settings
    auto_offset_reset: OffsetResetPolicy = OffsetResetPolicy.LATEST
    enable_auto_commit: bool = False
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000  # 5 minutes
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000
    fetch_min_bytes: int = 1
    fetch_max_wait_ms: int = 500
    fetch_max_bytes: int = 52428800  # 50MB

    # Commit strategy
    commit_strategy: CommitStrategy = CommitStrategy.BATCH
    commit_batch_size: int = 100

    # Client ID
    client_id: str = "gl003-steam-consumer"

    # Topic prefix
    topic_prefix: str = "gl003"

    # Schema validation
    enable_schema_validation: bool = True
    poison_pill_threshold: int = 3  # Failed validations before marking as poison pill

    # Dead letter queue
    dlq_topic: Optional[str] = None  # Topic for poison pills


@dataclass
class SchemaValidationConfig:
    """Schema validation configuration."""
    required_fields: List[str] = field(default_factory=lambda: ["tag", "value", "timestamp"])
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    allowed_qualities: List[str] = field(default_factory=lambda: ["good", "uncertain", "bad"])


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConsumedMessage:
    """Consumed Kafka message with metadata."""
    message_id: str
    topic: str
    partition: int
    offset: int
    key: Optional[str]
    value: Dict[str, Any]
    timestamp: datetime
    headers: Dict[str, str]

    # Processing state
    state: ProcessingState = ProcessingState.PENDING
    processing_error: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)
    retry_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "topic": self.topic,
            "partition": self.partition,
            "offset": self.offset,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "headers": self.headers,
            "state": self.state.value,
            "validation_errors": self.validation_errors,
        }


@dataclass
class ConsumerGroupInfo:
    """Consumer group metadata."""
    group_id: str
    members: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    state: str = "stable"

    # Lag tracking
    total_lag: int = 0
    partition_lags: Dict[str, int] = field(default_factory=dict)


@dataclass
class ConsumerMetrics:
    """Consumer performance metrics."""
    messages_consumed: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    poison_pills_detected: int = 0
    bytes_consumed: int = 0
    commits: int = 0
    rebalances: int = 0
    avg_processing_time_ms: float = 0.0
    current_lag: int = 0


# Type aliases
MessageHandler = Callable[[ConsumedMessage], None]
AsyncMessageHandler = Callable[[ConsumedMessage], Any]


# =============================================================================
# Schema Validator
# =============================================================================

class SchemaValidator:
    """Validates consumed messages against schema."""

    def __init__(self, config: SchemaValidationConfig) -> None:
        self.config = config

    def validate(self, message: ConsumedMessage) -> List[str]:
        """
        Validate message against schema.

        Args:
            message: Message to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors: List[str] = []
        value = message.value

        # Check required fields
        for field_name in self.config.required_fields:
            if field_name not in value:
                errors.append(f"Missing required field: {field_name}")

        # Validate value range
        if "value" in value:
            val = value["value"]
            if not isinstance(val, (int, float)):
                errors.append(f"Value must be numeric, got {type(val).__name__}")
            else:
                if self.config.value_min is not None and val < self.config.value_min:
                    errors.append(f"Value {val} below minimum {self.config.value_min}")
                if self.config.value_max is not None and val > self.config.value_max:
                    errors.append(f"Value {val} above maximum {self.config.value_max}")

        # Validate quality
        if "quality" in value:
            if value["quality"] not in self.config.allowed_qualities:
                errors.append(f"Invalid quality: {value['quality']}")

        # Validate timestamp
        if "timestamp" in value:
            try:
                ts = value["timestamp"]
                if isinstance(ts, str):
                    datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                errors.append(f"Invalid timestamp format: {value['timestamp']}")

        return errors


# =============================================================================
# Steam Data Consumer
# =============================================================================

class SteamDataConsumer:
    """
    Specialized Kafka consumer for steam measurement data.

    Features:
    - Consumer group management for scalability
    - Offset tracking and commit strategies
    - Schema validation with poison pill detection
    - Batch processing support
    - Rebalance handling
    - Lag monitoring

    Example:
        config = ConsumerConfig(
            bootstrap_servers="kafka.company.com:9092",
            group_id="gl003-processor-1",
        )

        consumer = SteamDataConsumer(config)
        await consumer.connect()

        # Subscribe with handler
        async def process_measurement(msg: ConsumedMessage):
            print(f"Received: {msg.value['tag']} = {msg.value['value']}")

        await consumer.subscribe(
            topics=["gl003.plant1.util.raw"],
            handler=process_measurement,
        )

        # Start processing
        await consumer.start_processing()
    """

    def __init__(
        self,
        config: ConsumerConfig,
        vault_client: Optional[Any] = None,
    ) -> None:
        """
        Initialize Steam Data Consumer.

        Args:
            config: Consumer configuration
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

        self._consumer = None
        self._connected = False
        self._processing = False

        # Subscriptions
        self._subscriptions: Dict[str, AsyncMessageHandler] = {}
        self._subscription_filters: Dict[str, Callable[[ConsumedMessage], bool]] = {}

        # Offset tracking
        self._pending_offsets: Dict[Tuple[str, int], int] = {}
        self._messages_since_commit = 0

        # Schema validator
        self._validator = SchemaValidator(SchemaValidationConfig())

        # Poison pill tracking
        self._poison_pill_counts: Dict[str, int] = {}

        # Metrics
        self._metrics = ConsumerMetrics()

        # Processing task
        self._process_task: Optional[asyncio.Task] = None

        logger.info(f"SteamDataConsumer initialized: {config.group_id}")

    async def connect(
        self,
        bootstrap_servers: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> None:
        """
        Connect to Kafka cluster.

        Args:
            bootstrap_servers: Override bootstrap servers
            group_id: Override consumer group ID
        """
        if bootstrap_servers:
            self.config.bootstrap_servers = bootstrap_servers
        if group_id:
            self.config.group_id = group_id

        try:
            # In production, use aiokafka:
            # from aiokafka import AIOKafkaConsumer
            # self._consumer = AIOKafkaConsumer(
            #     bootstrap_servers=self.config.bootstrap_servers,
            #     group_id=self.config.group_id,
            #     auto_offset_reset=self.config.auto_offset_reset.value,
            #     enable_auto_commit=self.config.enable_auto_commit,
            #     max_poll_records=self.config.max_poll_records,
            #     client_id=self.config.client_id,
            # )
            # await self._consumer.start()

            self._connected = True
            logger.info(f"Connected to Kafka as group {self.config.group_id}")

        except Exception as e:
            logger.error(f"Kafka consumer connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        self._processing = False

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        # Commit pending offsets
        await self.commit_offsets()

        if self._consumer:
            # await self._consumer.stop()
            self._consumer = None

        self._connected = False
        logger.info("Disconnected from Kafka")

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    async def subscribe(
        self,
        topics: List[str],
        handler: AsyncMessageHandler,
        filter_fn: Optional[Callable[[ConsumedMessage], bool]] = None,
    ) -> None:
        """
        Subscribe to Kafka topics.

        Args:
            topics: List of topic names
            handler: Async handler function for messages
            filter_fn: Optional filter function
        """
        if not self._connected:
            raise ConnectionError("Not connected to Kafka")

        for topic in topics:
            self._subscriptions[topic] = handler
            if filter_fn:
                self._subscription_filters[topic] = filter_fn

        # In production:
        # self._consumer.subscribe(topics)

        logger.info(f"Subscribed to topics: {topics}")

    async def unsubscribe(self, topics: Optional[List[str]] = None) -> None:
        """
        Unsubscribe from topics.

        Args:
            topics: Topics to unsubscribe (None = all)
        """
        if topics is None:
            self._subscriptions.clear()
            self._subscription_filters.clear()
        else:
            for topic in topics:
                self._subscriptions.pop(topic, None)
                self._subscription_filters.pop(topic, None)

        logger.info(f"Unsubscribed from topics: {topics or 'all'}")

    async def poll(
        self,
        timeout_ms: int = 1000,
        max_records: int = 100,
    ) -> List[ConsumedMessage]:
        """
        Poll for messages.

        Args:
            timeout_ms: Poll timeout
            max_records: Maximum records to return

        Returns:
            List of consumed messages
        """
        if not self._connected:
            return []

        messages: List[ConsumedMessage] = []

        try:
            # In production:
            # records = await self._consumer.getmany(
            #     timeout_ms=timeout_ms,
            #     max_records=max_records,
            # )
            # for tp, msgs in records.items():
            #     for msg in msgs:
            #         consumed = self._parse_message(msg)
            #         messages.append(consumed)

            # Simulate for framework
            messages = self._generate_simulated_messages(min(max_records, 10))

            self._metrics.messages_consumed += len(messages)
            self._metrics.bytes_consumed += sum(len(json.dumps(m.value)) for m in messages)

        except Exception as e:
            logger.error(f"Error polling messages: {e}")

        return messages

    def _generate_simulated_messages(self, count: int) -> List[ConsumedMessage]:
        """Generate simulated messages for framework demonstration."""
        import random

        messages = []
        topics = list(self._subscriptions.keys()) or ["gl003.demo.util.raw"]

        for i in range(count):
            topic = random.choice(topics)
            timestamp = datetime.now(timezone.utc)

            value = {
                "tag": f"HEADER.PRESSURE_{i}",
                "value": 145.0 + random.gauss(0, 2),
                "quality": "good",
                "unit": "psig",
                "timestamp": timestamp.isoformat(),
                "asset_id": f"ASSET_{i % 3}",
                "source_system": "opcua",
            }

            messages.append(ConsumedMessage(
                message_id=str(uuid.uuid4()),
                topic=topic,
                partition=i % 3,
                offset=1000 + i,
                key=f"key_{i}",
                value=value,
                timestamp=timestamp,
                headers={"source": "simulation"},
            ))

        return messages

    async def process_batch(
        self,
        batch_size: int = 100,
        timeout_ms: int = 1000,
    ) -> List[ConsumedMessage]:
        """
        Poll and process a batch of messages.

        Args:
            batch_size: Maximum messages to process
            timeout_ms: Poll timeout

        Returns:
            List of processed messages
        """
        messages = await self.poll(timeout_ms, batch_size)
        processed = []

        for msg in messages:
            try:
                # Validate message
                if self.config.enable_schema_validation:
                    validation_errors = self._validator.validate(msg)
                    if validation_errors:
                        msg.validation_errors = validation_errors
                        await self._handle_validation_failure(msg)
                        continue

                # Apply filter
                filter_fn = self._subscription_filters.get(msg.topic)
                if filter_fn and not filter_fn(msg):
                    continue

                # Get handler
                handler = self._subscriptions.get(msg.topic)
                if handler:
                    msg.state = ProcessingState.PROCESSING
                    start_time = time.perf_counter()

                    await handler(msg)

                    processing_time = (time.perf_counter() - start_time) * 1000
                    msg.state = ProcessingState.COMPLETED
                    self._metrics.messages_processed += 1
                    self._update_processing_time(processing_time)

                processed.append(msg)

                # Track offset for commit
                self._pending_offsets[(msg.topic, msg.partition)] = msg.offset
                self._messages_since_commit += 1

            except Exception as e:
                msg.state = ProcessingState.FAILED
                msg.processing_error = str(e)
                self._metrics.messages_failed += 1
                logger.error(f"Error processing message: {e}")

        # Commit if batch threshold reached
        if self.config.commit_strategy == CommitStrategy.BATCH:
            if self._messages_since_commit >= self.config.commit_batch_size:
                await self.commit_offsets()

        return processed

    async def _handle_validation_failure(self, msg: ConsumedMessage) -> None:
        """Handle message that failed validation."""
        key = f"{msg.topic}:{msg.key}"
        self._poison_pill_counts[key] = self._poison_pill_counts.get(key, 0) + 1

        if self._poison_pill_counts[key] >= self.config.poison_pill_threshold:
            msg.state = ProcessingState.POISON_PILL
            self._metrics.poison_pills_detected += 1
            logger.warning(f"Poison pill detected: {key}")

            # Send to DLQ if configured
            if self.config.dlq_topic:
                await self._send_to_dlq(msg)

            # Reset counter
            del self._poison_pill_counts[key]
        else:
            msg.state = ProcessingState.FAILED

    async def _send_to_dlq(self, msg: ConsumedMessage) -> None:
        """Send poison pill message to dead letter queue."""
        # In production, use a producer to send to DLQ topic
        logger.info(f"Sending poison pill to DLQ: {msg.message_id}")

    def _update_processing_time(self, processing_time_ms: float) -> None:
        """Update average processing time metric."""
        n = self._metrics.messages_processed
        self._metrics.avg_processing_time_ms = (
            (self._metrics.avg_processing_time_ms * (n - 1) + processing_time_ms) / n
            if n > 0 else processing_time_ms
        )

    async def commit_offsets(self) -> None:
        """Commit consumed offsets."""
        if not self._connected or not self._pending_offsets:
            return

        try:
            # In production:
            # await self._consumer.commit()

            self._metrics.commits += 1
            self._pending_offsets.clear()
            self._messages_since_commit = 0

            logger.debug(f"Committed offsets (commit #{self._metrics.commits})")

        except Exception as e:
            logger.error(f"Error committing offsets: {e}")

    async def start_processing(
        self,
        batch_size: int = 100,
        commit_interval: int = 100,
    ) -> None:
        """
        Start continuous message processing.

        Args:
            batch_size: Messages per batch
            commit_interval: Messages between commits
        """
        self._processing = True
        self._messages_since_commit = 0

        logger.info("Starting continuous message processing")

        while self._processing:
            try:
                messages = await self.process_batch(batch_size)

                if not messages:
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in continuous processing: {e}")
                await asyncio.sleep(1.0)

    def stop_processing(self) -> None:
        """Stop continuous processing."""
        self._processing = False

    async def seek_to_beginning(self, topics: Optional[List[str]] = None) -> None:
        """Seek to beginning of topics."""
        logger.info(f"Seeking to beginning of topics: {topics or 'all'}")

    async def seek_to_end(self, topics: Optional[List[str]] = None) -> None:
        """Seek to end of topics."""
        logger.info(f"Seeking to end of topics: {topics or 'all'}")

    async def seek_to_timestamp(
        self,
        timestamp: datetime,
        topics: Optional[List[str]] = None,
    ) -> None:
        """Seek to specific timestamp."""
        logger.info(f"Seeking to timestamp {timestamp}")

    async def get_lag(self) -> Dict[str, int]:
        """Get consumer lag by partition."""
        # In production, calculate from end offsets and current position
        return {"total": self._metrics.current_lag}

    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        return {
            "messages_consumed": self._metrics.messages_consumed,
            "messages_processed": self._metrics.messages_processed,
            "messages_failed": self._metrics.messages_failed,
            "poison_pills_detected": self._metrics.poison_pills_detected,
            "bytes_consumed": self._metrics.bytes_consumed,
            "commits": self._metrics.commits,
            "rebalances": self._metrics.rebalances,
            "avg_processing_time_ms": round(self._metrics.avg_processing_time_ms, 3),
            "current_lag": self._metrics.current_lag,
            "subscribed_topics": list(self._subscriptions.keys()),
            "pending_offsets": len(self._pending_offsets),
            "connected": self._connected,
        }

    def set_schema_validator(self, config: SchemaValidationConfig) -> None:
        """
        Set schema validation configuration.

        Args:
            config: Schema validation configuration
        """
        self._validator = SchemaValidator(config)


# =============================================================================
# Factory Functions
# =============================================================================

def create_steam_consumer(
    kafka_servers: str,
    group_id: str,
    site_id: str,
    area: str,
    stream_type: str = "raw",
) -> SteamDataConsumer:
    """
    Create a SteamDataConsumer for a specific stream.

    Args:
        kafka_servers: Kafka bootstrap servers
        group_id: Consumer group ID
        site_id: Site identifier
        area: Process area
        stream_type: Stream type (raw, validated, computed, etc.)

    Returns:
        Configured SteamDataConsumer instance
    """
    config = ConsumerConfig(
        bootstrap_servers=kafka_servers,
        group_id=group_id,
        auto_offset_reset=OffsetResetPolicy.LATEST,
        commit_strategy=CommitStrategy.BATCH,
        commit_batch_size=100,
        client_id=f"gl003-{site_id}-{area}-consumer",
    )

    return SteamDataConsumer(config)


async def create_and_subscribe_consumer(
    kafka_servers: str,
    group_id: str,
    topics: List[str],
    handler: AsyncMessageHandler,
) -> SteamDataConsumer:
    """
    Create, connect, and subscribe a consumer.

    Args:
        kafka_servers: Kafka bootstrap servers
        group_id: Consumer group ID
        topics: Topics to subscribe
        handler: Message handler

    Returns:
        Connected and subscribed consumer
    """
    config = ConsumerConfig(
        bootstrap_servers=kafka_servers,
        group_id=group_id,
    )

    consumer = SteamDataConsumer(config)
    await consumer.connect()
    await consumer.subscribe(topics, handler)

    return consumer
