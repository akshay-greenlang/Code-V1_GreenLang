"""
Kafka Producer for GreenLang Agent Events.

This module provides an async Kafka producer for publishing agent events
with support for exactly-once semantics, batching, and dead letter queue handling.

Features:
- Async event publishing with aiokafka
- Exactly-once semantics with transactions
- Automatic batching for throughput
- Configurable partitioning strategies
- Dead letter queue for failed messages
- Metrics and monitoring
- Graceful shutdown

Usage:
    from connectors.kafka.producer import AgentEventProducer
    from connectors.kafka.config import create_production_config
    from connectors.kafka.events import AgentCalculationCompleted

    config = create_production_config(
        bootstrap_servers=["kafka-1:9092"],
        sasl_username="producer",
        sasl_password="secret",
    )

    async with AgentEventProducer(config) as producer:
        event = AgentCalculationCompleted.create(...)
        await producer.send_event("gl.agent.calculations", event)
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .config import (
    KafkaConfig,
    PartitionStrategy,
    GreenLangTopics,
    DeadLetterQueueConfig,
)
from .events import AgentEvent, EventMetadata
from .serializers import (
    AgentEventSerializer,
    SerializationFormat,
    CompressionFormat,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Producer Statistics
# =============================================================================


@dataclass
class ProducerStatistics:
    """Statistics for producer monitoring."""

    messages_sent: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    batches_sent: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    messages_to_dlq: int = 0
    transactions_started: int = 0
    transactions_committed: int = 0
    transactions_aborted: int = 0


@dataclass
class SendResult:
    """Result of sending an event."""

    success: bool
    topic: str
    partition: int
    offset: int
    timestamp: datetime
    event_id: str
    latency_ms: float
    error: Optional[str] = None


# =============================================================================
# Partitioner
# =============================================================================


class AgentEventPartitioner:
    """
    Partitioner for agent events.

    Supports multiple partitioning strategies to control message distribution.
    """

    def __init__(
        self,
        strategy: PartitionStrategy = PartitionStrategy.BY_AGENT_ID,
        num_partitions: int = 6,
    ):
        """
        Initialize partitioner.

        Args:
            strategy: Partitioning strategy
            num_partitions: Number of topic partitions
        """
        self.strategy = strategy
        self.num_partitions = num_partitions
        self._round_robin_counter = 0

    def get_partition(
        self,
        event: AgentEvent,
        key: Optional[bytes] = None,
    ) -> Optional[int]:
        """
        Get partition for an event.

        Args:
            event: Event to partition
            key: Optional partition key

        Returns:
            Partition number or None for default partitioning
        """
        if self.strategy == PartitionStrategy.ROUND_ROBIN:
            self._round_robin_counter = (self._round_robin_counter + 1) % self.num_partitions
            return self._round_robin_counter

        elif self.strategy == PartitionStrategy.BY_AGENT_ID:
            partition_key = event.agent_id

        elif self.strategy == PartitionStrategy.BY_TENANT_ID:
            partition_key = event.metadata.tenant_id or "default"

        elif self.strategy == PartitionStrategy.BY_EVENT_TYPE:
            partition_key = event.event_type

        elif self.strategy == PartitionStrategy.BY_KEY:
            if key:
                partition_key = key.decode("utf-8")
            else:
                return None  # Use default partitioner

        else:
            return None

        # Hash the key to get partition
        key_hash = int(hashlib.md5(partition_key.encode()).hexdigest(), 16)
        return key_hash % self.num_partitions


# =============================================================================
# Callback Handler
# =============================================================================


class ProducerCallbackHandler:
    """Handles producer callbacks for success and error cases."""

    def __init__(
        self,
        on_success: Optional[Callable[[SendResult], None]] = None,
        on_error: Optional[Callable[[Exception, AgentEvent], None]] = None,
        stats: Optional[ProducerStatistics] = None,
    ):
        """
        Initialize callback handler.

        Args:
            on_success: Callback for successful sends
            on_error: Callback for failed sends
            stats: Statistics object to update
        """
        self.on_success = on_success
        self.on_error = on_error
        self.stats = stats or ProducerStatistics()

    def handle_success(self, result: SendResult) -> None:
        """Handle successful send."""
        self.stats.messages_sent += 1

        # Update latency stats
        if self.stats.max_latency_ms < result.latency_ms:
            self.stats.max_latency_ms = result.latency_ms

        # Running average
        n = self.stats.messages_sent
        self.stats.avg_latency_ms = (
            (self.stats.avg_latency_ms * (n - 1) + result.latency_ms) / n
        )

        if self.on_success:
            self.on_success(result)

        logger.debug(
            f"Event sent: topic={result.topic}, partition={result.partition}, "
            f"offset={result.offset}, latency={result.latency_ms:.2f}ms"
        )

    def handle_error(self, error: Exception, event: AgentEvent) -> None:
        """Handle failed send."""
        self.stats.messages_failed += 1
        self.stats.last_error = str(error)
        self.stats.last_error_time = datetime.utcnow()

        if self.on_error:
            self.on_error(error, event)

        logger.error(
            f"Event send failed: event_id={event.event_id}, error={error}",
            exc_info=True,
        )


# =============================================================================
# Agent Event Producer
# =============================================================================


class AgentEventProducer:
    """
    Kafka producer for GreenLang agent events.

    This producer provides:
    - Async event publishing with aiokafka
    - Exactly-once semantics with transactional support
    - Automatic batching for improved throughput
    - Configurable partitioning strategies
    - Dead letter queue handling for failed messages
    - Comprehensive monitoring and statistics
    - Graceful shutdown with message draining

    Example:
        config = KafkaConfig(
            bootstrap_servers=["kafka:9092"],
            producer=KafkaProducerConfig(
                enable_idempotence=True,
                transactional_id="gl-producer-001",
            ),
        )

        async with AgentEventProducer(config) as producer:
            # Send single event
            result = await producer.send_event(
                topic="gl.agent.calculations",
                event=calculation_event,
            )

            # Send batch
            results = await producer.send_batch(
                topic="gl.agent.events",
                events=[event1, event2, event3],
            )

            # Transactional send
            async with producer.transaction():
                await producer.send_event("topic1", event1)
                await producer.send_event("topic2", event2)
    """

    def __init__(
        self,
        config: KafkaConfig,
        serialization_format: SerializationFormat = SerializationFormat.JSON,
        compression: CompressionFormat = CompressionFormat.NONE,
        on_success: Optional[Callable[[SendResult], None]] = None,
        on_error: Optional[Callable[[Exception, AgentEvent], None]] = None,
    ):
        """
        Initialize the producer.

        Args:
            config: Kafka configuration
            serialization_format: Event serialization format
            compression: Compression format
            on_success: Callback for successful sends
            on_error: Callback for failed sends
        """
        self.config = config
        self.serialization_format = serialization_format
        self.compression = compression

        # Initialize components
        self._producer = None
        self._started = False
        self._in_transaction = False

        # Serializer
        self._serializer = AgentEventSerializer(
            format=serialization_format,
            compression=compression,
            schema_registry_url=(
                config.schema_registry.url if config.schema_registry else None
            ),
        )

        # Partitioner
        self._partitioner = AgentEventPartitioner(
            strategy=PartitionStrategy.BY_AGENT_ID,
            num_partitions=config.default_partitions,
        )

        # Statistics and callbacks
        self._stats = ProducerStatistics()
        self._callback_handler = ProducerCallbackHandler(
            on_success=on_success,
            on_error=on_error,
            stats=self._stats,
        )

        # Pending messages for tracking
        self._pending_messages: Dict[str, AgentEvent] = {}

        logger.info(
            f"AgentEventProducer initialized: "
            f"bootstrap_servers={config.bootstrap_servers}, "
            f"format={serialization_format.value}"
        )

    async def start(self) -> None:
        """Start the producer and connect to Kafka."""
        if self._started:
            logger.warning("Producer already started")
            return

        try:
            from aiokafka import AIOKafkaProducer

            # Get producer config
            producer_config = self.config.to_aiokafka_producer_config()

            self._producer = AIOKafkaProducer(**producer_config)

            await self._producer.start()
            self._started = True

            logger.info("AgentEventProducer started successfully")

        except ImportError:
            logger.warning(
                "aiokafka not installed, using mock producer. "
                "Install with: pip install aiokafka"
            )
            self._producer = MockKafkaProducer()
            self._started = True

        except Exception as e:
            logger.error(f"Failed to start producer: {e}")
            raise

    async def stop(self) -> None:
        """Stop the producer and flush pending messages."""
        if not self._started:
            return

        try:
            if self._producer:
                # Flush pending messages
                await self._producer.flush()
                await self._producer.stop()

            self._started = False
            logger.info("AgentEventProducer stopped")

        except Exception as e:
            logger.error(f"Error stopping producer: {e}")
            raise

    async def send_event(
        self,
        topic: str,
        event: AgentEvent,
        partition: Optional[int] = None,
        key: Optional[bytes] = None,
        headers: Optional[List[Tuple[str, bytes]]] = None,
    ) -> SendResult:
        """
        Send a single event to Kafka.

        Args:
            topic: Target topic
            event: Event to send
            partition: Optional specific partition
            key: Optional partition key
            headers: Optional additional headers

        Returns:
            SendResult with send details

        Raises:
            KafkaError: If send fails and DLQ is disabled
        """
        if not self._started:
            await self.start()

        start_time = time.time()

        try:
            # Serialize event
            key_bytes, value_bytes, event_headers = await self._serializer.serialize(event)

            # Use provided key if given
            if key:
                key_bytes = key

            # Merge headers
            if headers:
                event_headers.extend(headers)

            # Determine partition
            if partition is None:
                partition = self._partitioner.get_partition(event, key_bytes)

            # Track pending message
            self._pending_messages[event.event_id] = event

            # Send to Kafka
            result = await self._producer.send_and_wait(
                topic=topic,
                value=value_bytes,
                key=key_bytes,
                partition=partition,
                headers=event_headers,
            )

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Build result
            send_result = SendResult(
                success=True,
                topic=topic,
                partition=result.partition if hasattr(result, "partition") else partition or 0,
                offset=result.offset if hasattr(result, "offset") else 0,
                timestamp=datetime.utcnow(),
                event_id=event.event_id,
                latency_ms=latency_ms,
            )

            # Update stats
            self._stats.bytes_sent += len(value_bytes)
            self._callback_handler.handle_success(send_result)

            # Remove from pending
            self._pending_messages.pop(event.event_id, None)

            return send_result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Handle error
            self._callback_handler.handle_error(e, event)

            # Try to send to DLQ if configured
            if self.config.dlq.enabled:
                await self._send_to_dlq(topic, event, str(e))

            # Remove from pending
            self._pending_messages.pop(event.event_id, None)

            return SendResult(
                success=False,
                topic=topic,
                partition=-1,
                offset=-1,
                timestamp=datetime.utcnow(),
                event_id=event.event_id,
                latency_ms=latency_ms,
                error=str(e),
            )

    async def send_batch(
        self,
        topic: str,
        events: List[AgentEvent],
        partition: Optional[int] = None,
    ) -> List[SendResult]:
        """
        Send multiple events as a batch.

        Events are sent concurrently for improved throughput.

        Args:
            topic: Target topic
            events: Events to send
            partition: Optional specific partition for all events

        Returns:
            List of SendResult for each event
        """
        if not self._started:
            await self.start()

        if not events:
            return []

        self._stats.batches_sent += 1

        # Send all events concurrently
        tasks = [
            self.send_event(topic, event, partition)
            for event in events
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        send_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                send_results.append(SendResult(
                    success=False,
                    topic=topic,
                    partition=-1,
                    offset=-1,
                    timestamp=datetime.utcnow(),
                    event_id=events[i].event_id,
                    latency_ms=0,
                    error=str(result),
                ))
            else:
                send_results.append(result)

        return send_results

    async def send_transactional(
        self,
        messages: List[Tuple[str, AgentEvent]],
    ) -> List[SendResult]:
        """
        Send multiple messages in a transaction.

        All messages are sent atomically - either all succeed or all fail.

        Args:
            messages: List of (topic, event) tuples

        Returns:
            List of SendResult for each message

        Raises:
            KafkaError: If transaction fails
        """
        if not self._started:
            await self.start()

        if not self.config.producer.transactional_id:
            raise ValueError("Transactional ID must be configured for transactions")

        try:
            # Begin transaction
            await self._producer.begin_transaction()
            self._stats.transactions_started += 1
            self._in_transaction = True

            results = []
            for topic, event in messages:
                result = await self.send_event(topic, event)
                results.append(result)

                # Abort if any send fails
                if not result.success:
                    await self._producer.abort_transaction()
                    self._stats.transactions_aborted += 1
                    self._in_transaction = False
                    return results

            # Commit transaction
            await self._producer.commit_transaction()
            self._stats.transactions_committed += 1
            self._in_transaction = False

            return results

        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            if self._in_transaction:
                await self._producer.abort_transaction()
                self._stats.transactions_aborted += 1
                self._in_transaction = False
            raise

    def configure_partitioner(self, strategy: PartitionStrategy) -> None:
        """
        Configure the partitioning strategy.

        Args:
            strategy: Partitioning strategy to use
        """
        self._partitioner = AgentEventPartitioner(
            strategy=strategy,
            num_partitions=self.config.default_partitions,
        )
        logger.info(f"Partitioner configured: strategy={strategy.value}")

    async def _send_to_dlq(
        self,
        original_topic: str,
        event: AgentEvent,
        error: str,
    ) -> None:
        """Send failed message to dead letter queue."""
        dlq_topic = original_topic + self.config.dlq.topic_suffix

        try:
            # Add error metadata
            dlq_headers = [
                ("original_topic", original_topic.encode("utf-8")),
                ("error", error.encode("utf-8")),
                ("failed_at", datetime.utcnow().isoformat().encode("utf-8")),
            ]

            # Serialize and send
            key_bytes, value_bytes, event_headers = await self._serializer.serialize(event)
            dlq_headers.extend(event_headers)

            await self._producer.send_and_wait(
                topic=dlq_topic,
                value=value_bytes,
                key=key_bytes,
                headers=dlq_headers,
            )

            self._stats.messages_to_dlq += 1
            logger.warning(
                f"Message sent to DLQ: topic={dlq_topic}, event_id={event.event_id}"
            )

        except Exception as dlq_error:
            logger.error(
                f"Failed to send to DLQ: {dlq_error}",
                exc_info=True,
            )

    async def flush(self, timeout_ms: int = 10000) -> None:
        """
        Flush all pending messages.

        Args:
            timeout_ms: Maximum time to wait
        """
        if self._producer:
            await self._producer.flush()

    def get_statistics(self) -> ProducerStatistics:
        """Get producer statistics."""
        return self._stats

    def get_pending_count(self) -> int:
        """Get count of pending messages."""
        return len(self._pending_messages)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health check result dictionary
        """
        return {
            "status": "healthy" if self._started else "not_started",
            "connected": self._started,
            "pending_messages": len(self._pending_messages),
            "messages_sent": self._stats.messages_sent,
            "messages_failed": self._stats.messages_failed,
            "avg_latency_ms": round(self._stats.avg_latency_ms, 2),
            "last_error": self._stats.last_error,
        }

    # Context manager support
    async def __aenter__(self) -> "AgentEventProducer":
        """Enter async context."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        await self.stop()


# =============================================================================
# Mock Producer for Testing
# =============================================================================


class MockKafkaProducer:
    """Mock Kafka producer for testing without Kafka."""

    def __init__(self):
        self._messages: List[Dict[str, Any]] = []
        self._offset = 0

    async def start(self) -> None:
        """Start mock producer."""
        pass

    async def stop(self) -> None:
        """Stop mock producer."""
        pass

    async def send_and_wait(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        partition: Optional[int] = None,
        headers: Optional[List[Tuple[str, bytes]]] = None,
    ):
        """Send message and return mock result."""
        self._offset += 1
        message = {
            "topic": topic,
            "value": value,
            "key": key,
            "partition": partition or 0,
            "offset": self._offset,
            "headers": headers,
        }
        self._messages.append(message)

        # Return mock result
        return type("MockResult", (), {
            "partition": partition or 0,
            "offset": self._offset,
        })()

    async def flush(self) -> None:
        """Flush mock producer."""
        pass

    async def begin_transaction(self) -> None:
        """Begin mock transaction."""
        pass

    async def commit_transaction(self) -> None:
        """Commit mock transaction."""
        pass

    async def abort_transaction(self) -> None:
        """Abort mock transaction."""
        pass

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all sent messages (for testing)."""
        return self._messages


# =============================================================================
# Factory Function
# =============================================================================


def create_producer(
    bootstrap_servers: List[str],
    client_id: str = "gl-agent-producer",
    format: SerializationFormat = SerializationFormat.JSON,
    enable_exactly_once: bool = False,
    transactional_id: Optional[str] = None,
) -> AgentEventProducer:
    """
    Factory function to create a producer with common settings.

    Args:
        bootstrap_servers: Kafka broker addresses
        client_id: Client identifier
        format: Serialization format
        enable_exactly_once: Enable exactly-once semantics
        transactional_id: Transactional ID (required for exactly-once)

    Returns:
        Configured AgentEventProducer
    """
    from .config import KafkaConfig, KafkaProducerConfig, AcksMode

    producer_config = KafkaProducerConfig(
        acks=AcksMode.ALL if enable_exactly_once else AcksMode.LEADER_ONLY,
        enable_idempotence=enable_exactly_once,
        transactional_id=transactional_id,
    )

    config = KafkaConfig(
        bootstrap_servers=bootstrap_servers,
        client_id=client_id,
        producer=producer_config,
    )

    return AgentEventProducer(
        config=config,
        serialization_format=format,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AgentEventProducer",
    "AgentEventPartitioner",
    "ProducerStatistics",
    "SendResult",
    "ProducerCallbackHandler",
    "MockKafkaProducer",
    "create_producer",
]
