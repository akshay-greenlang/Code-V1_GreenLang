# -*- coding: utf-8 -*-
"""
Abstract Broker Interface

This module defines the abstract interface for all message brokers.
Supports pluggable implementations (Redis Streams, Kafka, RabbitMQ).

Example:
    >>> broker = RedisStreamsBroker(config)
    >>> await broker.connect()
    >>> await broker.publish("agent.tasks", message)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncIterator, Callable, Any
from datetime import datetime
import logging

from greenlang.determinism import DeterministicClock
from .message import (
    Message,
    MessageBatch,
    MessageAck,
    DeadLetterMessage,
    MessagePriority,
)

logger = logging.getLogger(__name__)


class BrokerMetrics(ABC):
    """
    Metrics tracking for broker operations.

    Tracks throughput, latency, errors for observability.
    """

    def __init__(self):
        """Initialize metrics counters."""
        self.messages_published = 0
        self.messages_consumed = 0
        self.messages_failed = 0
        self.messages_dlq = 0
        self.total_publish_time_ms = 0.0
        self.total_consume_time_ms = 0.0
        self.start_time = DeterministicClock.utcnow()

    def record_publish(self, count: int, duration_ms: float) -> None:
        """Record published messages."""
        self.messages_published += count
        self.total_publish_time_ms += duration_ms

    def record_consume(self, count: int, duration_ms: float) -> None:
        """Record consumed messages."""
        self.messages_consumed += count
        self.total_consume_time_ms += duration_ms

    def record_failure(self) -> None:
        """Record failed message."""
        self.messages_failed += 1

    def record_dlq(self) -> None:
        """Record dead letter queue movement."""
        self.messages_dlq += 1

    def get_throughput_per_second(self) -> Dict[str, float]:
        """Calculate messages per second."""
        elapsed_seconds = (DeterministicClock.utcnow() - self.start_time).total_seconds()
        if elapsed_seconds == 0:
            return {"publish": 0.0, "consume": 0.0}

        return {
            "publish": self.messages_published / elapsed_seconds,
            "consume": self.messages_consumed / elapsed_seconds,
        }

    def get_average_latency_ms(self) -> Dict[str, float]:
        """Calculate average latency."""
        return {
            "publish": (
                self.total_publish_time_ms / self.messages_published
                if self.messages_published > 0 else 0.0
            ),
            "consume": (
                self.total_consume_time_ms / self.messages_consumed
                if self.messages_consumed > 0 else 0.0
            ),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get complete metrics summary."""
        throughput = self.get_throughput_per_second()
        latency = self.get_average_latency_ms()

        return {
            "messages_published": self.messages_published,
            "messages_consumed": self.messages_consumed,
            "messages_failed": self.messages_failed,
            "messages_dlq": self.messages_dlq,
            "throughput_per_second": throughput,
            "average_latency_ms": latency,
            "uptime_seconds": (DeterministicClock.utcnow() - self.start_time).total_seconds(),
        }


class MessageBrokerInterface(ABC):
    """
    Abstract interface for message brokers.

    All broker implementations (Redis, Kafka, RabbitMQ) must implement this interface.
    Provides consistent API for agent communication regardless of underlying technology.

    Features:
        - Async publish/subscribe
        - Consumer groups for parallel processing
        - Dead letter queue (DLQ) for failed messages
        - Request-reply pattern support
        - Batch operations
        - Health monitoring
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize message broker.

        Args:
            config: Broker configuration dictionary
        """
        self.config = config
        self.metrics = BrokerMetrics()
        self._connected = False
        self._consumers: Dict[str, Callable] = {}

    @abstractmethod
    async def connect(self) -> None:
        """
        Connect to message broker.

        Establishes connection and initializes resources.
        Raises ConnectionError if connection fails.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from message broker.

        Closes connections and cleans up resources.
        """
        pass

    @abstractmethod
    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        headers: Optional[Dict[str, str]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Publish single message to topic.

        Args:
            topic: Target topic/stream
            payload: Message data
            priority: Message priority (high, normal, low)
            headers: Optional metadata headers
            ttl_seconds: Message time-to-live

        Returns:
            Message ID

        Raises:
            PublishError: If publish fails
        """
        pass

    @abstractmethod
    async def publish_batch(
        self,
        topic: str,
        payloads: List[Dict[str, Any]],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> List[str]:
        """
        Publish batch of messages (100+ messages).

        Reduces network overhead by 80% compared to individual publishes.

        Args:
            topic: Target topic/stream
            payloads: List of message payloads
            priority: Message priority

        Returns:
            List of message IDs

        Raises:
            PublishError: If batch publish fails
        """
        pass

    @abstractmethod
    async def consume(
        self,
        topic: str,
        consumer_group: str,
        consumer_id: Optional[str] = None,
        batch_size: int = 10,
        timeout_ms: int = 5000,
    ) -> AsyncIterator[Message]:
        """
        Consume messages from topic.

        Uses consumer groups for parallel processing and load balancing.

        Args:
            topic: Source topic/stream
            consumer_group: Consumer group name
            consumer_id: Optional consumer identifier
            batch_size: Messages per batch (default 10)
            timeout_ms: Polling timeout in milliseconds

        Yields:
            Message instances

        Raises:
            ConsumeError: If consumption fails
        """
        pass

    @abstractmethod
    async def acknowledge(self, message: Message) -> None:
        """
        Acknowledge message processing.

        Removes message from pending list.

        Args:
            message: Message to acknowledge

        Raises:
            AckError: If acknowledgment fails
        """
        pass

    @abstractmethod
    async def nack(
        self,
        message: Message,
        error_message: str,
        requeue: bool = True,
    ) -> None:
        """
        Negative acknowledge - message processing failed.

        Args:
            message: Failed message
            error_message: Failure reason
            requeue: Whether to retry or move to DLQ

        Raises:
            NackError: If nack fails
        """
        pass

    @abstractmethod
    async def subscribe(
        self,
        pattern: str,
        handler: Callable[[Message], None],
    ) -> None:
        """
        Subscribe to topic pattern with handler.

        Supports wildcard patterns (e.g., "agent.events.*").

        Args:
            pattern: Topic pattern to subscribe to
            handler: Async handler function

        Raises:
            SubscribeError: If subscription fails
        """
        pass

    @abstractmethod
    async def request(
        self,
        topic: str,
        payload: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Optional[Message]:
        """
        Request-reply pattern - send request and wait for response.

        Args:
            topic: Request topic
            payload: Request data
            timeout: Response timeout in seconds

        Returns:
            Response message or None if timeout

        Raises:
            RequestError: If request fails
        """
        pass

    @abstractmethod
    async def reply(
        self,
        original_message: Message,
        response_payload: Dict[str, Any],
    ) -> None:
        """
        Reply to request message.

        Args:
            original_message: Original request message
            response_payload: Response data

        Raises:
            ReplyError: If reply fails
        """
        pass

    @abstractmethod
    async def create_consumer_group(
        self,
        topic: str,
        group_name: str,
    ) -> None:
        """
        Create consumer group for topic.

        Args:
            topic: Topic name
            group_name: Consumer group name

        Raises:
            GroupCreationError: If creation fails
        """
        pass

    @abstractmethod
    async def delete_consumer_group(
        self,
        topic: str,
        group_name: str,
    ) -> None:
        """
        Delete consumer group.

        Args:
            topic: Topic name
            group_name: Consumer group name

        Raises:
            GroupDeletionError: If deletion fails
        """
        pass

    @abstractmethod
    async def get_consumer_lag(
        self,
        topic: str,
        consumer_group: str,
    ) -> int:
        """
        Get consumer lag (pending messages count).

        Args:
            topic: Topic name
            consumer_group: Consumer group name

        Returns:
            Number of pending messages

        Raises:
            LagCheckError: If lag check fails
        """
        pass

    @abstractmethod
    async def get_dead_letter_messages(
        self,
        topic: str,
        limit: int = 100,
    ) -> List[DeadLetterMessage]:
        """
        Get messages from dead letter queue.

        Args:
            topic: Original topic name
            limit: Maximum messages to retrieve

        Returns:
            List of dead letter messages

        Raises:
            DLQError: If retrieval fails
        """
        pass

    @abstractmethod
    async def reprocess_dead_letter_message(
        self,
        dlq_message: DeadLetterMessage,
    ) -> None:
        """
        Reprocess message from DLQ.

        Moves message back to original topic for retry.

        Args:
            dlq_message: Dead letter message to reprocess

        Raises:
            ReprocessError: If reprocessing fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check broker health.

        Returns:
            Health status dictionary with connection info

        Example:
            {
                "status": "healthy",
                "connected": True,
                "latency_ms": 5.2,
                "uptime_seconds": 3600
            }
        """
        pass

    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self._connected

    def get_metrics(self) -> Dict[str, Any]:
        """Get broker metrics summary."""
        return self.metrics.get_summary()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
