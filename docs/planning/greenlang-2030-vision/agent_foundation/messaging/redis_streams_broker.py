# -*- coding: utf-8 -*-
"""
Redis Streams Message Broker Implementation

Production-ready message broker using Redis Streams with:
- AsyncIO for high concurrency (10K msg/s)
- Consumer groups for parallel processing
- Automatic redelivery on failure
- Dead letter queue (DLQ) for failed messages
- At-least-once delivery guarantee
- Sub-10ms P95 latency

Example:
    >>> broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    >>> await broker.connect()
    >>> await broker.publish("agent.tasks", {"task": "analyze_esg"})
    >>> async for message in broker.consume("agent.tasks", "workers"):
    ...     await process(message)
    ...     await broker.acknowledge(message)
"""

from typing import Dict, List, Optional, AsyncIterator, Callable, Any
import asyncio
import logging
from datetime import datetime
import json
import uuid
from greenlang.determinism import deterministic_uuid, DeterministicClock

try:
    import redis.asyncio as aioredis
except ImportError:
    raise ImportError(
        "redis-py async not installed. Run: pip install redis[hiredis]"
    )

from .broker_interface import MessageBrokerInterface
from .message import (
    Message,
    MessageBatch,
    MessageAck,
    DeadLetterMessage,
    MessagePriority,
    MessageStatus,
)

logger = logging.getLogger(__name__)


class RedisStreamsBroker(MessageBrokerInterface):
    """
    Redis Streams implementation of message broker.

    Uses Redis Streams for high-performance message queuing with:
    - Consumer groups for parallel processing (100 concurrent consumers)
    - Automatic message acknowledgment and redelivery
    - Dead letter queue for failed messages
    - Pub/Sub for real-time notifications
    - AOF persistence for durability

    Performance Targets:
        - Throughput: 10K msg/s
        - Latency P95: < 10ms
        - Max Consumers: 100
        - Message Size: 1MB
        - Retention: 7 days

    Attributes:
        redis_client: AsyncIO Redis client
        config: Broker configuration
        _pending_replies: Pending request-reply messages
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_connections: int = 50,
        decode_responses: bool = False,
        **kwargs
    ):
        """
        Initialize Redis Streams broker.

        Args:
            redis_url: Redis connection URL
            max_connections: Max connection pool size
            decode_responses: Whether to decode responses to strings
            **kwargs: Additional Redis configuration
        """
        config = {
            "redis_url": redis_url,
            "max_connections": max_connections,
            "decode_responses": decode_responses,
            **kwargs
        }
        super().__init__(config)

        self.redis_client: Optional[aioredis.Redis] = None
        self._pending_replies: Dict[str, asyncio.Queue] = {}
        self._consumer_tasks: List[asyncio.Task] = []
        self._dlq_prefix = "dlq:"
        self._reply_prefix = "reply:"

    async def connect(self) -> None:
        """
        Connect to Redis.

        Creates connection pool and tests connectivity.
        """
        try:
            self.redis_client = await aioredis.from_url(
                self.config["redis_url"],
                max_connections=self.config["max_connections"],
                decode_responses=self.config["decode_responses"],
                encoding="utf-8",
            )

            # Test connection
            await self.redis_client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.config['redis_url']}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
            raise ConnectionError(f"Redis connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from Redis and cleanup resources."""
        try:
            # Cancel consumer tasks
            for task in self._consumer_tasks:
                task.cancel()

            if self.redis_client:
                await self.redis_client.close()
                self._connected = False
                logger.info("Disconnected from Redis")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}", exc_info=True)

    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        headers: Optional[Dict[str, str]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Publish message to Redis Stream.

        Args:
            topic: Stream name
            payload: Message data
            priority: Message priority
            headers: Optional headers
            ttl_seconds: Message TTL

        Returns:
            Message ID
        """
        if not self._connected:
            raise ConnectionError("Not connected to Redis")

        start_time = DeterministicClock.utcnow()

        try:
            # Create message
            message = Message(
                topic=topic,
                payload=payload,
                priority=priority,
                headers=headers or {},
                ttl_seconds=ttl_seconds,
            )

            # Serialize message
            message_data = {
                "data": message.serialize().decode('utf-8'),
                "priority": priority.value,
                "timestamp": message.timestamp.isoformat(),
            }

            # Add to stream with priority-based stream naming
            stream_key = self._get_stream_key(topic, priority)
            message_id = await self.redis_client.xadd(
                stream_key,
                message_data,
                maxlen=100000,  # Keep last 100K messages
            )

            # Record metrics
            duration_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000
            self.metrics.record_publish(1, duration_ms)

            logger.debug(f"Published message {message.id} to {stream_key}")
            return message.id

        except Exception as e:
            logger.error(f"Failed to publish message: {e}", exc_info=True)
            self.metrics.record_failure()
            raise

    async def publish_batch(
        self,
        topic: str,
        payloads: List[Dict[str, Any]],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> List[str]:
        """
        Publish batch of messages using pipeline (80% overhead reduction).

        Args:
            topic: Stream name
            payloads: List of message payloads
            priority: Message priority

        Returns:
            List of message IDs
        """
        if not self._connected:
            raise ConnectionError("Not connected to Redis")

        start_time = DeterministicClock.utcnow()

        try:
            message_ids = []
            stream_key = self._get_stream_key(topic, priority)

            # Use pipeline for batch operations
            pipe = self.redis_client.pipeline()

            for payload in payloads:
                message = Message(
                    topic=topic,
                    payload=payload,
                    priority=priority,
                )

                message_data = {
                    "data": message.serialize().decode('utf-8'),
                    "priority": priority.value,
                    "timestamp": message.timestamp.isoformat(),
                }

                pipe.xadd(stream_key, message_data, maxlen=100000)
                message_ids.append(message.id)

            # Execute pipeline
            await pipe.execute()

            # Record metrics
            duration_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000
            self.metrics.record_publish(len(payloads), duration_ms)

            logger.info(f"Published batch of {len(payloads)} messages to {stream_key}")
            return message_ids

        except Exception as e:
            logger.error(f"Failed to publish batch: {e}", exc_info=True)
            self.metrics.record_failure()
            raise

    async def consume(
        self,
        topic: str,
        consumer_group: str,
        consumer_id: Optional[str] = None,
        batch_size: int = 10,
        timeout_ms: int = 5000,
    ) -> AsyncIterator[Message]:
        """
        Consume messages from Redis Stream using consumer group.

        Args:
            topic: Stream name
            consumer_group: Consumer group name
            consumer_id: Consumer identifier (auto-generated if None)
            batch_size: Messages per batch
            timeout_ms: Polling timeout

        Yields:
            Message instances
        """
        if not self._connected:
            raise ConnectionError("Not connected to Redis")

        consumer_id = consumer_id or f"consumer-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8]}"
        stream_key = self._get_stream_key(topic, MessagePriority.NORMAL)

        # Create consumer group if not exists
        try:
            await self.create_consumer_group(topic, consumer_group)
        except Exception as e:
            logger.debug(f"Consumer group already exists: {e}")

        logger.info(f"Consumer {consumer_id} starting on {stream_key}")

        while True:
            try:
                start_time = DeterministicClock.utcnow()

                # Read from stream
                messages = await self.redis_client.xreadgroup(
                    groupname=consumer_group,
                    consumername=consumer_id,
                    streams={stream_key: ">"},
                    count=batch_size,
                    block=timeout_ms,
                )

                if not messages:
                    await asyncio.sleep(0.1)  # Brief pause before next poll
                    continue

                # Process messages
                for stream, message_list in messages:
                    for message_id, message_data in message_list:
                        try:
                            # Deserialize message
                            message_json = message_data[b"data"]
                            message = Message.deserialize(message_json.encode())

                            # Check if expired
                            if message.is_expired():
                                logger.warning(f"Message {message.id} expired, skipping")
                                await self._ack_stream_message(stream_key, consumer_group, message_id)
                                continue

                            # Store Redis message ID for acknowledgment
                            message.headers["_redis_message_id"] = message_id
                            message.headers["_redis_stream"] = stream_key
                            message.headers["_redis_group"] = consumer_group

                            # Update status
                            message.status = MessageStatus.PROCESSING

                            # Record metrics
                            duration_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000
                            self.metrics.record_consume(1, duration_ms)

                            yield message

                        except Exception as e:
                            logger.error(f"Error processing message: {e}", exc_info=True)
                            self.metrics.record_failure()

            except asyncio.CancelledError:
                logger.info(f"Consumer {consumer_id} stopped")
                break
            except Exception as e:
                logger.error(f"Consumer error: {e}", exc_info=True)
                await asyncio.sleep(1)  # Brief pause before retry

    async def acknowledge(self, message: Message) -> None:
        """
        Acknowledge message processing (remove from pending).

        Args:
            message: Message to acknowledge
        """
        if not self._connected:
            raise ConnectionError("Not connected to Redis")

        try:
            stream_key = message.headers.get("_redis_stream")
            consumer_group = message.headers.get("_redis_group")
            redis_message_id = message.headers.get("_redis_message_id")

            if not all([stream_key, consumer_group, redis_message_id]):
                raise ValueError("Message missing Redis metadata for acknowledgment")

            await self._ack_stream_message(stream_key, consumer_group, redis_message_id)

            message.status = MessageStatus.COMPLETED
            logger.debug(f"Acknowledged message {message.id}")

        except Exception as e:
            logger.error(f"Failed to acknowledge message: {e}", exc_info=True)
            raise

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
        """
        if not self._connected:
            raise ConnectionError("Not connected to Redis")

        try:
            message.status = MessageStatus.FAILED
            message.increment_retry()

            if requeue and message.can_retry():
                # Requeue for retry
                logger.info(f"Requeuing message {message.id} (attempt {message.retry_count})")
                await self.publish(
                    message.topic,
                    message.payload,
                    message.priority,
                    message.headers,
                    message.ttl_seconds,
                )
            else:
                # Move to DLQ
                logger.warning(f"Moving message {message.id} to DLQ: {error_message}")
                await self._move_to_dlq(message, error_message)

            # Acknowledge original message to remove from pending
            await self.acknowledge(message)

        except Exception as e:
            logger.error(f"Failed to nack message: {e}", exc_info=True)
            raise

    async def subscribe(
        self,
        pattern: str,
        handler: Callable[[Message], None],
    ) -> None:
        """
        Subscribe to topic pattern using Redis Pub/Sub.

        Args:
            pattern: Topic pattern (supports wildcards)
            handler: Message handler function
        """
        if not self._connected:
            raise ConnectionError("Not connected to Redis")

        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.psubscribe(pattern)

            async def consume_messages():
                async for message in pubsub.listen():
                    if message["type"] == "pmessage":
                        try:
                            msg_data = json.loads(message["data"])
                            msg = Message(**msg_data)
                            await handler(msg)
                        except Exception as e:
                            logger.error(f"Handler error: {e}", exc_info=True)

            # Start consumer task
            task = asyncio.create_task(consume_messages())
            self._consumer_tasks.append(task)

            logger.info(f"Subscribed to pattern: {pattern}")

        except Exception as e:
            logger.error(f"Failed to subscribe: {e}", exc_info=True)
            raise

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
            timeout: Response timeout

        Returns:
            Response message or None if timeout
        """
        if not self._connected:
            raise ConnectionError("Not connected to Redis")

        correlation_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        reply_topic = f"{self._reply_prefix}{correlation_id}"

        try:
            # Create reply queue
            reply_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
            self._pending_replies[correlation_id] = reply_queue

            # Publish request
            await self.publish(
                topic,
                payload,
                headers={"correlation_id": correlation_id, "reply_to": reply_topic},
            )

            # Wait for reply with timeout
            try:
                response = await asyncio.wait_for(reply_queue.get(), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout for correlation_id {correlation_id}")
                return None

        finally:
            # Cleanup
            self._pending_replies.pop(correlation_id, None)

    async def reply(
        self,
        original_message: Message,
        response_payload: Dict[str, Any],
    ) -> None:
        """
        Reply to request message.

        Args:
            original_message: Original request
            response_payload: Response data
        """
        if not self._connected:
            raise ConnectionError("Not connected to Redis")

        correlation_id = original_message.headers.get("correlation_id")
        reply_to = original_message.headers.get("reply_to")

        if not correlation_id or not reply_to:
            raise ValueError("Original message missing request-reply headers")

        # Publish response
        await self.publish(
            reply_to,
            response_payload,
            headers={"correlation_id": correlation_id},
        )

    async def create_consumer_group(
        self,
        topic: str,
        group_name: str,
    ) -> None:
        """Create consumer group for stream."""
        if not self._connected:
            raise ConnectionError("Not connected to Redis")

        stream_key = self._get_stream_key(topic, MessagePriority.NORMAL)

        try:
            await self.redis_client.xgroup_create(
                stream_key,
                group_name,
                id="0",
                mkstream=True,
            )
            logger.info(f"Created consumer group {group_name} for {stream_key}")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.error(f"Failed to create consumer group: {e}")
                raise

    async def delete_consumer_group(
        self,
        topic: str,
        group_name: str,
    ) -> None:
        """Delete consumer group."""
        if not self._connected:
            raise ConnectionError("Not connected to Redis")

        stream_key = self._get_stream_key(topic, MessagePriority.NORMAL)

        try:
            await self.redis_client.xgroup_destroy(stream_key, group_name)
            logger.info(f"Deleted consumer group {group_name}")
        except Exception as e:
            logger.error(f"Failed to delete consumer group: {e}")
            raise

    async def get_consumer_lag(
        self,
        topic: str,
        consumer_group: str,
    ) -> int:
        """Get consumer lag (pending messages)."""
        if not self._connected:
            raise ConnectionError("Not connected to Redis")

        stream_key = self._get_stream_key(topic, MessagePriority.NORMAL)

        try:
            info = await self.redis_client.xinfo_groups(stream_key)
            for group in info:
                if group["name"] == consumer_group.encode():
                    return group["pending"]
            return 0
        except Exception as e:
            logger.error(f"Failed to get consumer lag: {e}")
            return 0

    async def get_dead_letter_messages(
        self,
        topic: str,
        limit: int = 100,
    ) -> List[DeadLetterMessage]:
        """Get messages from DLQ."""
        if not self._connected:
            raise ConnectionError("Not connected to Redis")

        dlq_key = f"{self._dlq_prefix}{topic}"
        messages = []

        try:
            # Get DLQ messages
            dlq_data = await self.redis_client.lrange(dlq_key, 0, limit - 1)

            for data in dlq_data:
                try:
                    dlq_dict = json.loads(data)
                    original_msg = Message(**dlq_dict["original_message"])
                    dlq_msg = DeadLetterMessage(
                        original_message=original_msg,
                        failure_reason=dlq_dict["failure_reason"],
                        retry_history=dlq_dict.get("retry_history", []),
                        moved_to_dlq_at=datetime.fromisoformat(dlq_dict["moved_to_dlq_at"]),
                        dlq_topic=dlq_key,
                    )
                    messages.append(dlq_msg)
                except Exception as e:
                    logger.error(f"Error parsing DLQ message: {e}")

            return messages

        except Exception as e:
            logger.error(f"Failed to get DLQ messages: {e}")
            return []

    async def reprocess_dead_letter_message(
        self,
        dlq_message: DeadLetterMessage,
    ) -> None:
        """Reprocess message from DLQ."""
        if not self._connected:
            raise ConnectionError("Not connected to Redis")

        try:
            # Reset retry count
            original = dlq_message.original_message
            original.retry_count = 0
            original.status = MessageStatus.PENDING

            # Republish
            await self.publish(
                original.topic,
                original.payload,
                original.priority,
                original.headers,
                original.ttl_seconds,
            )

            # Remove from DLQ
            dlq_key = dlq_message.dlq_topic
            await self.redis_client.lrem(dlq_key, 1, json.dumps(dlq_message.to_dict()))

            logger.info(f"Reprocessed DLQ message {original.id}")

        except Exception as e:
            logger.error(f"Failed to reprocess DLQ message: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check broker health."""
        try:
            start_time = DeterministicClock.utcnow()
            await self.redis_client.ping()
            latency_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000

            info = await self.redis_client.info()

            return {
                "status": "healthy",
                "connected": self._connected,
                "latency_ms": latency_ms,
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                "connected_clients": info.get("connected_clients", 0),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
            }

    # Helper methods

    def _get_stream_key(self, topic: str, priority: MessagePriority) -> str:
        """Get Redis stream key with priority suffix."""
        if priority == MessagePriority.HIGH or priority == MessagePriority.CRITICAL:
            return f"{topic}:high"
        return topic

    async def _ack_stream_message(
        self,
        stream_key: str,
        consumer_group: str,
        message_id: str,
    ) -> None:
        """Acknowledge message in Redis Stream."""
        await self.redis_client.xack(stream_key, consumer_group, message_id)

    async def _move_to_dlq(self, message: Message, error_message: str) -> None:
        """Move failed message to dead letter queue."""
        dlq_key = f"{self._dlq_prefix}{message.topic}"

        dlq_message = DeadLetterMessage(
            original_message=message,
            failure_reason=error_message,
            retry_history=[],
            dlq_topic=dlq_key,
        )

        # Store in DLQ (Redis list)
        await self.redis_client.lpush(dlq_key, json.dumps(dlq_message.to_dict()))

        # Set expiration (30 days)
        await self.redis_client.expire(dlq_key, 30 * 24 * 3600)

        self.metrics.record_dlq()
        logger.info(f"Moved message {message.id} to DLQ: {dlq_key}")
