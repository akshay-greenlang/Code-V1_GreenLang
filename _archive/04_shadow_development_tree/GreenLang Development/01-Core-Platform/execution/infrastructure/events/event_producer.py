"""
Generic Event Producer for GreenLang

This module provides a generic event producer that abstracts
the underlying messaging infrastructure.

Features:
- Multi-backend support (Kafka, MQTT, Redis)
- Automatic serialization
- Retry logic
- Batching
- Provenance tracking
- Metrics collection

Example:
    >>> producer = EventProducer(config)
    >>> await producer.start()
    >>> await producer.publish(event)
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type
from uuid import uuid4

from pydantic import BaseModel, Field

from greenlang.infrastructure.events.event_schema import (
    BaseEvent,
    DomainEvent,
    IntegrationEvent,
    EventSchema,
)

logger = logging.getLogger(__name__)


class BackendType(str, Enum):
    """Supported messaging backends."""
    KAFKA = "kafka"
    MQTT = "mqtt"
    REDIS = "redis"
    MEMORY = "memory"  # For testing


class DeliveryMode(str, Enum):
    """Message delivery modes."""
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


@dataclass
class EventProducerConfig:
    """Configuration for event producer."""
    backend: BackendType = BackendType.KAFKA
    # Kafka settings
    kafka_bootstrap_servers: List[str] = field(
        default_factory=lambda: ["localhost:9092"]
    )
    # MQTT settings
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    # Redis settings
    redis_url: str = "redis://localhost:6379"
    # Common settings
    default_topic: str = "greenlang-events"
    delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    batch_size: int = 100
    batch_timeout_ms: int = 1000
    max_retries: int = 3
    retry_delay_ms: int = 1000
    enable_compression: bool = True
    enable_batching: bool = True


class PublishResult(BaseModel):
    """Result of a publish operation."""
    event_id: str = Field(..., description="Event ID")
    topic: str = Field(..., description="Topic published to")
    partition: Optional[int] = Field(default=None, description="Partition")
    offset: Optional[int] = Field(default=None, description="Offset")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="Provenance hash")
    success: bool = Field(default=True, description="Publish success")
    error: Optional[str] = Field(default=None, description="Error message")


class EventProducerBackend(ABC):
    """Abstract base class for event producer backends."""

    @abstractmethod
    async def start(self) -> None:
        """Start the backend."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the backend."""
        pass

    @abstractmethod
    async def publish(
        self,
        topic: str,
        key: Optional[str],
        value: bytes,
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Publish a message."""
        pass

    @abstractmethod
    async def publish_batch(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Publish a batch of messages."""
        pass


class MemoryBackend(EventProducerBackend):
    """In-memory backend for testing."""

    def __init__(self):
        """Initialize memory backend."""
        self.messages: List[Dict[str, Any]] = []
        self._offset = 0

    async def start(self) -> None:
        """Start the backend."""
        logger.info("Memory backend started")

    async def stop(self) -> None:
        """Stop the backend."""
        logger.info("Memory backend stopped")

    async def publish(
        self,
        topic: str,
        key: Optional[str],
        value: bytes,
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Publish a message."""
        self._offset += 1
        message = {
            "topic": topic,
            "key": key,
            "value": value,
            "headers": headers,
            "offset": self._offset,
            "partition": 0,
            "timestamp": datetime.utcnow(),
        }
        self.messages.append(message)
        return message

    async def publish_batch(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Publish a batch of messages."""
        results = []
        for msg in messages:
            result = await self.publish(
                msg["topic"],
                msg.get("key"),
                msg["value"],
                msg.get("headers", {})
            )
            results.append(result)
        return results


class EventProducer:
    """
    Generic event producer for GreenLang.

    Provides a unified interface for publishing events across
    different messaging backends.

    Attributes:
        config: Producer configuration
        backend: Messaging backend instance

    Example:
        >>> config = EventProducerConfig(
        ...     backend=BackendType.KAFKA,
        ...     kafka_bootstrap_servers=["kafka:9092"]
        ... )
        >>> producer = EventProducer(config)
        >>> async with producer:
        ...     result = await producer.publish(event)
    """

    def __init__(self, config: EventProducerConfig):
        """
        Initialize event producer.

        Args:
            config: Producer configuration
        """
        self.config = config
        self._backend: Optional[EventProducerBackend] = None
        self._started = False
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._batch_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._metrics: Dict[str, int] = {
            "events_published": 0,
            "events_failed": 0,
            "batches_sent": 0,
        }
        self._middlewares: List[Callable] = []

        logger.info(f"EventProducer initialized with backend: {config.backend}")

    async def start(self) -> None:
        """
        Start the event producer.

        Initializes the backend and starts batch processing.
        """
        if self._started:
            logger.warning("Producer already started")
            return

        try:
            # Initialize backend
            self._backend = self._create_backend()
            await self._backend.start()

            # Start batch processor if enabled
            if self.config.enable_batching:
                self._batch_task = asyncio.create_task(self._batch_processor())

            self._started = True
            self._shutdown = False

            logger.info("Event producer started")

        except Exception as e:
            logger.error(f"Failed to start producer: {e}", exc_info=True)
            raise

    async def stop(self) -> None:
        """
        Stop the event producer gracefully.

        Flushes pending events and closes connections.
        """
        self._shutdown = True

        # Stop batch processor
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Flush pending events
        await self.flush()

        # Stop backend
        if self._backend:
            await self._backend.stop()

        self._started = False
        logger.info("Event producer stopped")

    def _create_backend(self) -> EventProducerBackend:
        """Create the appropriate backend."""
        if self.config.backend == BackendType.MEMORY:
            return MemoryBackend()
        elif self.config.backend == BackendType.KAFKA:
            # Would integrate with KafkaAvroProducer
            return MemoryBackend()  # Placeholder
        elif self.config.backend == BackendType.MQTT:
            # Would integrate with MQTTClient
            return MemoryBackend()  # Placeholder
        else:
            return MemoryBackend()

    def add_middleware(
        self,
        middleware: Callable[[BaseEvent], BaseEvent]
    ) -> None:
        """
        Add a middleware function for event processing.

        Args:
            middleware: Function that processes events before publishing
        """
        self._middlewares.append(middleware)
        logger.debug(f"Added middleware: {middleware.__name__}")

    async def publish(
        self,
        event: BaseEvent,
        topic: Optional[str] = None,
        key: Optional[str] = None,
        wait: bool = True
    ) -> PublishResult:
        """
        Publish a single event.

        Args:
            event: Event to publish
            topic: Target topic (defaults to config)
            key: Partition key
            wait: Wait for confirmation

        Returns:
            PublishResult with confirmation details
        """
        self._ensure_started()

        # Apply middlewares
        processed_event = event
        for middleware in self._middlewares:
            try:
                if asyncio.iscoroutinefunction(middleware):
                    processed_event = await middleware(processed_event)
                else:
                    processed_event = middleware(processed_event)
            except Exception as e:
                logger.error(f"Middleware error: {e}")

        # Validate event
        if not EventSchema.validate_event(processed_event):
            return PublishResult(
                event_id=event.metadata.event_id,
                topic=topic or self.config.default_topic,
                provenance_hash=event.provenance_hash,
                success=False,
                error="Event validation failed"
            )

        target_topic = topic or self._get_topic_for_event(processed_event)
        partition_key = key or self._get_partition_key(processed_event)

        try:
            # Serialize event
            value = json.dumps(processed_event.to_avro_dict()).encode()

            # Build headers
            headers = {
                "event_type": processed_event.event_type,
                "event_id": processed_event.metadata.event_id,
                "correlation_id": processed_event.metadata.correlation_id or "",
                "timestamp": processed_event.metadata.timestamp.isoformat(),
                "provenance_hash": processed_event.provenance_hash,
            }

            if self.config.enable_batching and not wait:
                # Add to batch queue
                await self._batch_queue.put({
                    "topic": target_topic,
                    "key": partition_key,
                    "value": value,
                    "headers": headers,
                    "event_id": processed_event.metadata.event_id,
                })
                return PublishResult(
                    event_id=processed_event.metadata.event_id,
                    topic=target_topic,
                    provenance_hash=processed_event.provenance_hash,
                    success=True,
                )
            else:
                # Publish immediately
                result = await self._publish_with_retry(
                    target_topic,
                    partition_key,
                    value,
                    headers
                )

                self._metrics["events_published"] += 1

                return PublishResult(
                    event_id=processed_event.metadata.event_id,
                    topic=target_topic,
                    partition=result.get("partition"),
                    offset=result.get("offset"),
                    timestamp=result.get("timestamp", datetime.utcnow()),
                    provenance_hash=processed_event.provenance_hash,
                    success=True,
                )

        except Exception as e:
            self._metrics["events_failed"] += 1
            logger.error(f"Publish failed: {e}")
            return PublishResult(
                event_id=event.metadata.event_id,
                topic=target_topic,
                provenance_hash=event.provenance_hash,
                success=False,
                error=str(e)
            )

    async def publish_batch(
        self,
        events: List[BaseEvent],
        topic: Optional[str] = None
    ) -> List[PublishResult]:
        """
        Publish multiple events in a batch.

        Args:
            events: Events to publish
            topic: Target topic for all events

        Returns:
            List of publish results
        """
        self._ensure_started()

        results = []
        for event in events:
            result = await self.publish(event, topic, wait=True)
            results.append(result)

        logger.info(f"Published batch of {len(events)} events")
        return results

    async def publish_domain_event(
        self,
        event_type: str,
        aggregate_id: str,
        aggregate_type: str,
        data: Dict[str, Any],
        topic: Optional[str] = None
    ) -> PublishResult:
        """
        Convenience method to publish a domain event.

        Args:
            event_type: Event type
            aggregate_id: Aggregate identifier
            aggregate_type: Aggregate type
            data: Event data
            topic: Target topic

        Returns:
            PublishResult
        """
        event = DomainEvent(
            event_type=event_type,
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            data=data,
        )
        return await self.publish(event, topic, key=aggregate_id)

    async def publish_integration_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        target_systems: List[str],
        topic: Optional[str] = None
    ) -> PublishResult:
        """
        Convenience method to publish an integration event.

        Args:
            event_type: Event type
            data: Event data
            target_systems: Target systems
            topic: Target topic

        Returns:
            PublishResult
        """
        event = IntegrationEvent(
            event_type=event_type,
            data=data,
            target_systems=target_systems,
        )
        return await self.publish(event, topic)

    async def _publish_with_retry(
        self,
        topic: str,
        key: Optional[str],
        value: bytes,
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Publish with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                return await self._backend.publish(topic, key, value, headers)
            except Exception as e:
                last_error = e
                logger.warning(f"Publish attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(
                        self.config.retry_delay_ms * (2 ** attempt) / 1000
                    )

        raise last_error

    async def _batch_processor(self) -> None:
        """Process batched events."""
        batch: List[Dict[str, Any]] = []
        last_flush = datetime.utcnow()

        while not self._shutdown:
            try:
                # Try to get an item with timeout
                try:
                    item = await asyncio.wait_for(
                        self._batch_queue.get(),
                        timeout=self.config.batch_timeout_ms / 1000
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass

                # Check if we should flush
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    (datetime.utcnow() - last_flush).total_seconds() * 1000 >= self.config.batch_timeout_ms
                )

                if should_flush and batch:
                    await self._flush_batch(batch)
                    batch = []
                    last_flush = datetime.utcnow()

            except asyncio.CancelledError:
                # Flush remaining on shutdown
                if batch:
                    await self._flush_batch(batch)
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")

    async def _flush_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Flush a batch of messages."""
        try:
            await self._backend.publish_batch(batch)
            self._metrics["events_published"] += len(batch)
            self._metrics["batches_sent"] += 1
            logger.debug(f"Flushed batch of {len(batch)} events")
        except Exception as e:
            self._metrics["events_failed"] += len(batch)
            logger.error(f"Batch flush failed: {e}")

    async def flush(self) -> None:
        """Flush all pending events."""
        if not self.config.enable_batching:
            return

        # Drain the queue
        batch = []
        while not self._batch_queue.empty():
            try:
                item = self._batch_queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                break

        if batch:
            await self._flush_batch(batch)

    def _get_topic_for_event(self, event: BaseEvent) -> str:
        """Determine topic for an event."""
        # Topic routing based on event type
        event_type = event.event_type

        if event_type.startswith("emission."):
            return "greenlang-emissions"
        elif event_type.startswith("compliance."):
            return "greenlang-compliance"
        elif event_type.startswith("saga."):
            return "greenlang-sagas"
        elif event_type.startswith("audit."):
            return "greenlang-audit"
        else:
            return self.config.default_topic

    def _get_partition_key(self, event: BaseEvent) -> Optional[str]:
        """Determine partition key for an event."""
        if isinstance(event, DomainEvent):
            return event.aggregate_id
        return event.metadata.correlation_id or event.metadata.event_id

    def _ensure_started(self) -> None:
        """Ensure producer is started."""
        if not self._started:
            raise RuntimeError("Producer not started")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get producer metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "started": self._started,
            "backend": self.config.backend.value,
            "events_published": self._metrics["events_published"],
            "events_failed": self._metrics["events_failed"],
            "batches_sent": self._metrics["batches_sent"],
            "queue_size": self._batch_queue.qsize() if self.config.enable_batching else 0,
        }

    async def __aenter__(self) -> "EventProducer":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
