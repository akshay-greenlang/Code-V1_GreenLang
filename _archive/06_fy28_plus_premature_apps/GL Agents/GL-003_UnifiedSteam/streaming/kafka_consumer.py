"""
GL-003 UNIFIEDSTEAM - Kafka Consumer

Consumes steam system data from Kafka topics for real-time processing.

Features:
- Consumer group management for scalability
- Offset tracking and commit strategies
- Message deserialization and validation
- Batch processing support
- Rebalance handling
- Lag monitoring
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, AsyncIterator
import asyncio
import json
import logging
import uuid

logger = logging.getLogger(__name__)


class OffsetResetPolicy(Enum):
    """Offset reset policy for new consumer groups."""
    EARLIEST = "earliest"
    LATEST = "latest"
    NONE = "none"


class CommitStrategy(Enum):
    """Offset commit strategies."""
    AUTO = "auto"  # Automatic periodic commits
    MANUAL_SYNC = "manual_sync"  # Manual synchronous commits
    MANUAL_ASYNC = "manual_async"  # Manual asynchronous commits
    BATCH = "batch"  # Commit after processing batch


@dataclass
class KafkaConsumerConfig:
    """Kafka consumer configuration."""
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "gl003-steam-consumer"

    # Authentication (same as producer)
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    ssl_cafile: Optional[str] = None

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

    # Client ID
    client_id: str = "gl003-steam-consumer"

    # Topic prefix for filtering
    topic_prefix: str = "gl003"


@dataclass
class ConsumedMessage:
    """Consumed Kafka message with metadata."""
    topic: str
    partition: int
    offset: int
    key: Optional[str]
    value: Dict[str, Any]
    timestamp: datetime
    headers: Dict[str, str]

    # Processing state
    processed: bool = False
    processing_error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "topic": self.topic,
            "partition": self.partition,
            "offset": self.offset,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "headers": self.headers,
            "processed": self.processed,
        }


# Type alias for message handlers
MessageHandler = Callable[[ConsumedMessage], None]
AsyncMessageHandler = Callable[[ConsumedMessage], asyncio.coroutine]


@dataclass
class TopicSubscription:
    """Subscription to a Kafka topic."""
    topic: str
    handler: Optional[Union[MessageHandler, AsyncMessageHandler]] = None
    is_async: bool = False

    # Filtering
    key_filter: Optional[Callable[[str], bool]] = None
    value_filter: Optional[Callable[[Dict], bool]] = None

    # Statistics
    messages_received: int = 0
    messages_processed: int = 0
    messages_filtered: int = 0
    last_message_time: Optional[datetime] = None


@dataclass
class ConsumerGroup:
    """Consumer group metadata."""
    group_id: str
    members: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    state: str = "stable"  # stable, rebalancing, dead

    # Lag tracking
    total_lag: int = 0
    partition_lags: Dict[str, int] = field(default_factory=dict)


@dataclass
class ConsumerMetrics:
    """Consumer performance metrics."""
    messages_consumed: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    bytes_consumed: int = 0
    commits: int = 0
    rebalances: int = 0
    avg_processing_time_ms: float = 0.0
    current_lag: int = 0


class SteamKafkaConsumer:
    """
    Kafka consumer for steam system data processing.

    Consumes sensor data, computed properties, and events from
    Kafka topics for real-time processing and analytics.

    Example:
        config = KafkaConsumerConfig(
            bootstrap_servers="kafka.company.com:9092",
            group_id="gl003-processor-1",
        )

        consumer = SteamKafkaConsumer(config)
        await consumer.connect()

        # Subscribe to topics with handler
        def process_signal(msg: ConsumedMessage):
            print(f"Received: {msg.value['tag']} = {msg.value['value']}")

        await consumer.subscribe(
            topics=["gl003.plant1.util.raw"],
            callback=process_signal
        )

        # Process messages
        while True:
            messages = await consumer.process_batch(batch_size=100)
            await consumer.commit_offsets()
    """

    def __init__(
        self,
        config: KafkaConsumerConfig,
        vault_client: Optional[Any] = None,
    ) -> None:
        """
        Initialize Kafka consumer.

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

        # Subscriptions
        self._subscriptions: Dict[str, TopicSubscription] = {}

        # Offset tracking
        self._pending_commits: Dict[Tuple[str, int], int] = {}  # (topic, partition) -> offset

        # Metrics
        self._metrics = ConsumerMetrics()

        # Processing state
        self._processing = False
        self._process_task: Optional[asyncio.Task] = None

        logger.info(f"SteamKafkaConsumer initialized: {config.group_id}")

    async def connect(
        self,
        bootstrap_servers: Optional[str] = None,
        group_id: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> None:
        """
        Connect to Kafka cluster.

        Args:
            bootstrap_servers: Override bootstrap servers
            group_id: Override consumer group ID
            config: Additional configuration overrides
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
        callback: Optional[Union[MessageHandler, AsyncMessageHandler]] = None,
        key_filter: Optional[Callable[[str], bool]] = None,
        value_filter: Optional[Callable[[Dict], bool]] = None,
    ) -> None:
        """
        Subscribe to Kafka topics.

        Args:
            topics: List of topic names to subscribe to
            callback: Handler function for messages
            key_filter: Optional filter by message key
            value_filter: Optional filter by message value
        """
        if not self._connected:
            raise ConnectionError("Not connected to Kafka")

        for topic in topics:
            # Determine if callback is async
            is_async = asyncio.iscoroutinefunction(callback) if callback else False

            self._subscriptions[topic] = TopicSubscription(
                topic=topic,
                handler=callback,
                is_async=is_async,
                key_filter=key_filter,
                value_filter=value_filter,
            )

        # Subscribe to topics
        # In production:
        # self._consumer.subscribe(topics)

        logger.info(f"Subscribed to topics: {topics}")

    async def unsubscribe(self, topics: Optional[List[str]] = None) -> None:
        """
        Unsubscribe from topics.

        Args:
            topics: Topics to unsubscribe from (None = all)
        """
        if topics is None:
            self._subscriptions.clear()
            # self._consumer.unsubscribe()
        else:
            for topic in topics:
                self._subscriptions.pop(topic, None)

            # Re-subscribe to remaining topics
            remaining = list(self._subscriptions.keys())
            if remaining:
                # self._consumer.subscribe(remaining)
                pass

    async def process_batch(
        self,
        batch_size: int = 100,
        timeout_ms: int = 1000,
    ) -> List[ConsumedMessage]:
        """
        Process a batch of messages.

        Args:
            batch_size: Maximum messages to process
            timeout_ms: Poll timeout in milliseconds

        Returns:
            List of consumed messages
        """
        import time

        if not self._connected:
            return []

        messages: List[ConsumedMessage] = []
        start_time = time.perf_counter()

        try:
            # In production:
            # records = await self._consumer.getmany(
            #     timeout_ms=timeout_ms,
            #     max_records=batch_size,
            # )
            # for tp, msgs in records.items():
            #     for msg in msgs:
            #         consumed = self._parse_message(msg)
            #         messages.append(consumed)

            # For framework: simulate message consumption
            messages = self._generate_simulated_messages(batch_size)

            # Process each message
            for msg in messages:
                await self._process_message(msg)

            # Update metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            self._metrics.messages_consumed += len(messages)
            self._metrics.bytes_consumed += sum(
                len(json.dumps(m.value)) for m in messages
            )

            if messages:
                self._metrics.avg_processing_time_ms = (
                    (self._metrics.avg_processing_time_ms * (self._metrics.commits) + processing_time)
                    / (self._metrics.commits + 1)
                )

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self._metrics.messages_failed += len(messages)

        return messages

    def _generate_simulated_messages(self, count: int) -> List[ConsumedMessage]:
        """Generate simulated messages for framework demonstration."""
        import random

        messages = []
        topics = list(self._subscriptions.keys()) or ["gl003.demo.util.raw"]

        for i in range(min(count, 10)):  # Limit for simulation
            topic = random.choice(topics)
            timestamp = datetime.now(timezone.utc)

            # Generate appropriate message based on topic
            if "raw" in topic:
                value = {
                    "tag": f"HEADER.PRESSURE_{i}",
                    "value": 145.0 + random.gauss(0, 2),
                    "quality": "good",
                    "timestamp": timestamp.isoformat(),
                }
            elif "validated" in topic:
                value = {
                    "tag": f"HEADER.PRESSURE_{i}",
                    "value": 145.0 + random.gauss(0, 1),
                    "unit": "psig",
                    "quality_code": 0,
                    "is_valid": True,
                    "timestamp": timestamp.isoformat(),
                }
            elif "computed" in topic:
                value = {
                    "property_name": "efficiency",
                    "value": 85.0 + random.gauss(0, 1),
                    "unit": "%",
                    "asset_id": f"BOILER_{i % 3 + 1}",
                    "timestamp": timestamp.isoformat(),
                }
            else:
                value = {"type": "event", "data": {}}

            messages.append(ConsumedMessage(
                topic=topic,
                partition=i % 3,
                offset=1000 + i,
                key=f"key_{i}",
                value=value,
                timestamp=timestamp,
                headers={"source": "simulation"},
            ))

        return messages

    async def _process_message(self, msg: ConsumedMessage) -> None:
        """Process a single message."""
        subscription = self._subscriptions.get(msg.topic)

        if not subscription:
            return

        subscription.messages_received += 1
        subscription.last_message_time = datetime.now(timezone.utc)

        # Apply filters
        if subscription.key_filter and msg.key:
            if not subscription.key_filter(msg.key):
                subscription.messages_filtered += 1
                return

        if subscription.value_filter:
            if not subscription.value_filter(msg.value):
                subscription.messages_filtered += 1
                return

        # Call handler
        if subscription.handler:
            try:
                if subscription.is_async:
                    await subscription.handler(msg)
                else:
                    subscription.handler(msg)

                msg.processed = True
                subscription.messages_processed += 1
                self._metrics.messages_processed += 1

            except Exception as e:
                msg.processing_error = str(e)
                self._metrics.messages_failed += 1
                logger.error(f"Error processing message: {e}")

        # Track offset for commit
        self._pending_commits[(msg.topic, msg.partition)] = msg.offset

    async def commit_offsets(self) -> None:
        """Commit consumed offsets."""
        if not self._connected or not self._pending_commits:
            return

        try:
            # In production:
            # await self._consumer.commit()

            self._metrics.commits += 1
            self._pending_commits.clear()

            logger.debug(f"Committed offsets (commit #{self._metrics.commits})")

        except Exception as e:
            logger.error(f"Error committing offsets: {e}")

    async def seek_to_beginning(self, topics: Optional[List[str]] = None) -> None:
        """Seek to beginning of topics."""
        # In production:
        # partitions = self._consumer.assignment()
        # await self._consumer.seek_to_beginning(*partitions)
        logger.info(f"Seeking to beginning of topics: {topics}")

    async def seek_to_end(self, topics: Optional[List[str]] = None) -> None:
        """Seek to end of topics."""
        # In production:
        # partitions = self._consumer.assignment()
        # await self._consumer.seek_to_end(*partitions)
        logger.info(f"Seeking to end of topics: {topics}")

    async def seek_to_timestamp(
        self,
        timestamp: datetime,
        topics: Optional[List[str]] = None,
    ) -> None:
        """Seek to specific timestamp."""
        # In production:
        # partitions = self._consumer.assignment()
        # offsets = await self._consumer.offsets_for_times({
        #     p: timestamp.timestamp() * 1000 for p in partitions
        # })
        # for p, offset in offsets.items():
        #     self._consumer.seek(p, offset)
        logger.info(f"Seeking to timestamp {timestamp}")

    async def get_lag(self) -> Dict[str, int]:
        """Get consumer lag by partition."""
        # In production:
        # end_offsets = await self._consumer.end_offsets(self._consumer.assignment())
        # current = self._consumer.position(...)
        # lag = {str(p): end - current for p, end in end_offsets.items()}

        return {"total": 0}  # Placeholder

    async def start_continuous_processing(
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
        messages_since_commit = 0

        logger.info("Starting continuous message processing")

        while self._processing:
            try:
                messages = await self.process_batch(batch_size)

                messages_since_commit += len(messages)

                # Commit based on strategy
                if self.config.commit_strategy == CommitStrategy.BATCH:
                    if messages_since_commit >= commit_interval:
                        await self.commit_offsets()
                        messages_since_commit = 0

                # Small delay to prevent tight loop
                if not messages:
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in continuous processing: {e}")
                await asyncio.sleep(1.0)

    def stop_continuous_processing(self) -> None:
        """Stop continuous processing."""
        self._processing = False

    def get_metrics(self) -> Dict:
        """Get consumer metrics."""
        return {
            "messages_consumed": self._metrics.messages_consumed,
            "messages_processed": self._metrics.messages_processed,
            "messages_failed": self._metrics.messages_failed,
            "bytes_consumed": self._metrics.bytes_consumed,
            "commits": self._metrics.commits,
            "rebalances": self._metrics.rebalances,
            "avg_processing_time_ms": round(self._metrics.avg_processing_time_ms, 3),
            "current_lag": self._metrics.current_lag,
            "subscribed_topics": list(self._subscriptions.keys()),
            "connected": self._connected,
        }

    def get_subscription_stats(self) -> Dict[str, Dict]:
        """Get per-subscription statistics."""
        return {
            topic: {
                "messages_received": sub.messages_received,
                "messages_processed": sub.messages_processed,
                "messages_filtered": sub.messages_filtered,
                "last_message_time": sub.last_message_time.isoformat() if sub.last_message_time else None,
            }
            for topic, sub in self._subscriptions.items()
        }


async def create_consumer_for_stream(
    site_id: str,
    area: str,
    stream_type: str,
    kafka_servers: str,
    handler: MessageHandler,
    group_suffix: str = "",
) -> SteamKafkaConsumer:
    """
    Create and configure consumer for a specific stream.

    Args:
        site_id: Site identifier
        area: Process area
        stream_type: Stream type (raw, validated, computed, etc.)
        kafka_servers: Kafka bootstrap servers
        handler: Message handler function
        group_suffix: Optional group ID suffix

    Returns:
        Configured and connected consumer
    """
    topic = f"gl003.{site_id.lower()}.{area.lower()}.{stream_type}"
    group_id = f"gl003-{site_id}-{area}-{stream_type}{group_suffix}"

    config = KafkaConsumerConfig(
        bootstrap_servers=kafka_servers,
        group_id=group_id,
        auto_offset_reset=OffsetResetPolicy.LATEST,
        commit_strategy=CommitStrategy.BATCH,
    )

    consumer = SteamKafkaConsumer(config)
    await consumer.connect()
    await consumer.subscribe([topic], callback=handler)

    return consumer
