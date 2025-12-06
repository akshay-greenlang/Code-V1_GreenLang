"""
Redis Streams Integration for Agent Execution Pipeline.

This module provides Redis Streams-based message queue functionality
for async agent execution, event publishing, and workflow orchestration.

Features:
- Agent execution queuing
- Event publishing/subscription
- Dead letter queue handling
- Consumer group management
- Backpressure handling

Example:
    >>> queue = RedisStreamQueue(redis_url="redis://localhost:6379")
    >>> await queue.connect()
    >>> await queue.publish_execution("emissions/carbon_calculator_v1", {"fuel_type": "natural_gas"})
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import redis.asyncio as redis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StreamName(str, Enum):
    """Standard stream names for the platform."""

    AGENT_EXECUTIONS = "gl:executions"
    AGENT_RESULTS = "gl:results"
    EVENTS = "gl:events"
    AUDIT_LOG = "gl:audit"
    DEAD_LETTER = "gl:dlq"


class MessagePriority(str, Enum):
    """Message priority levels."""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ExecutionMessage(BaseModel):
    """Message format for agent execution requests."""

    execution_id: str = Field(..., description="Unique execution ID")
    agent_id: str = Field(..., description="Agent identifier")
    input_data: Dict[str, Any] = Field(..., description="Agent input data")
    tenant_id: str = Field(..., description="Tenant ID for multi-tenancy")
    user_id: str = Field(..., description="Requesting user ID")
    priority: MessagePriority = Field(MessagePriority.NORMAL, description="Priority")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    timeout_seconds: int = Field(300, description="Execution timeout")
    retry_count: int = Field(0, description="Current retry count")
    max_retries: int = Field(3, description="Maximum retry attempts")


class ResultMessage(BaseModel):
    """Message format for agent execution results."""

    execution_id: str = Field(..., description="Execution ID")
    agent_id: str = Field(..., description="Agent identifier")
    status: str = Field(..., description="Status: completed, failed, timeout")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    provenance_hash: Optional[str] = Field(None, description="Provenance hash")
    processing_time_ms: float = Field(..., description="Processing duration")
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class EventMessage(BaseModel):
    """Message format for system events."""

    event_id: str = Field(..., description="Unique event ID")
    event_type: str = Field(..., description="Event type")
    source: str = Field(..., description="Event source")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = Field(None, description="Correlation ID")


class RedisStreamQueue:
    """
    Redis Streams-based message queue for agent execution.

    This class provides:
    - Publish/subscribe to Redis Streams
    - Consumer group management
    - Dead letter queue handling
    - Backpressure and rate limiting
    - Message acknowledgment

    Attributes:
        redis: Redis client connection
        consumer_name: Unique consumer identifier
        group_name: Consumer group name

    Example:
        >>> queue = RedisStreamQueue("redis://localhost:6379")
        >>> await queue.connect()
        >>> await queue.publish_execution(...)
        >>> async for message in queue.consume_executions():
        ...     await process(message)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        consumer_name: Optional[str] = None,
        group_name: str = "gl-workers",
        max_connections: int = 10,
    ):
        """
        Initialize Redis Stream Queue.

        Args:
            redis_url: Redis connection URL
            consumer_name: Unique consumer identifier
            group_name: Consumer group name
            max_connections: Max pool connections
        """
        self.redis_url = redis_url
        self.consumer_name = consumer_name or f"worker-{datetime.utcnow().timestamp()}"
        self.group_name = group_name
        self.max_connections = max_connections
        self._redis: Optional[redis.Redis] = None
        self._running = False

        logger.info(f"RedisStreamQueue initialized: consumer={self.consumer_name}")

    async def connect(self) -> None:
        """Establish connection to Redis."""
        self._redis = redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=self.max_connections,
        )

        # Verify connection
        await self._redis.ping()
        logger.info("Connected to Redis")

        # Create consumer groups for all streams
        await self._ensure_consumer_groups()

    async def disconnect(self) -> None:
        """Close Redis connection."""
        self._running = False
        if self._redis:
            await self._redis.close()
            logger.info("Disconnected from Redis")

    async def _ensure_consumer_groups(self) -> None:
        """Create consumer groups if they don't exist."""
        streams = [
            StreamName.AGENT_EXECUTIONS,
            StreamName.AGENT_RESULTS,
            StreamName.EVENTS,
        ]

        for stream in streams:
            try:
                await self._redis.xgroup_create(
                    stream.value,
                    self.group_name,
                    id="0",
                    mkstream=True,
                )
                logger.info(f"Created consumer group for {stream.value}")
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
                # Group already exists

    # ==================== PUBLISHING ====================

    async def publish_execution(
        self,
        execution_id: str,
        agent_id: str,
        input_data: Dict[str, Any],
        tenant_id: str,
        user_id: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        timeout_seconds: int = 300,
    ) -> str:
        """
        Publish an agent execution request to the queue.

        Args:
            execution_id: Unique execution ID
            agent_id: Agent to execute
            input_data: Agent input data
            tenant_id: Tenant ID
            user_id: User ID
            priority: Message priority
            correlation_id: Optional correlation ID
            timeout_seconds: Execution timeout

        Returns:
            Redis stream message ID
        """
        message = ExecutionMessage(
            execution_id=execution_id,
            agent_id=agent_id,
            input_data=input_data,
            tenant_id=tenant_id,
            user_id=user_id,
            priority=priority,
            correlation_id=correlation_id,
            timeout_seconds=timeout_seconds,
        )

        # Convert to Redis-compatible format
        data = {
            "payload": message.json(),
            "priority": priority.value,
            "created_at": datetime.utcnow().isoformat(),
        }

        message_id = await self._redis.xadd(
            StreamName.AGENT_EXECUTIONS.value,
            data,
            maxlen=100000,  # Limit stream size
        )

        logger.info(
            f"Published execution: id={execution_id}, agent={agent_id}, "
            f"message_id={message_id}"
        )

        return message_id

    async def publish_result(
        self,
        execution_id: str,
        agent_id: str,
        status: str,
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        provenance_hash: Optional[str] = None,
        processing_time_ms: float = 0,
    ) -> str:
        """
        Publish an agent execution result.

        Args:
            execution_id: Execution ID
            agent_id: Agent identifier
            status: Execution status
            output_data: Result data
            error: Error message if failed
            provenance_hash: Calculation provenance hash
            processing_time_ms: Processing time

        Returns:
            Redis stream message ID
        """
        message = ResultMessage(
            execution_id=execution_id,
            agent_id=agent_id,
            status=status,
            output_data=output_data,
            error=error,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
        )

        data = {
            "payload": message.json(),
            "status": status,
            "completed_at": datetime.utcnow().isoformat(),
        }

        message_id = await self._redis.xadd(
            StreamName.AGENT_RESULTS.value,
            data,
            maxlen=100000,
        )

        logger.info(
            f"Published result: execution_id={execution_id}, status={status}, "
            f"message_id={message_id}"
        )

        return message_id

    async def publish_event(
        self,
        event_type: str,
        source: str,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Publish a system event.

        Args:
            event_type: Type of event
            source: Event source
            data: Event data
            correlation_id: Correlation ID

        Returns:
            Redis stream message ID
        """
        import uuid

        event = EventMessage(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source=source,
            data=data,
            correlation_id=correlation_id,
        )

        stream_data = {
            "payload": event.json(),
            "event_type": event_type,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
        }

        message_id = await self._redis.xadd(
            StreamName.EVENTS.value,
            stream_data,
            maxlen=500000,
        )

        logger.debug(f"Published event: type={event_type}, source={source}")

        return message_id

    # ==================== CONSUMING ====================

    async def consume_executions(
        self,
        batch_size: int = 10,
        block_ms: int = 5000,
    ) -> List[Tuple[str, ExecutionMessage]]:
        """
        Consume execution messages from the queue.

        Args:
            batch_size: Number of messages to fetch
            block_ms: Block time in milliseconds

        Returns:
            List of (message_id, ExecutionMessage) tuples
        """
        messages = await self._redis.xreadgroup(
            self.group_name,
            self.consumer_name,
            {StreamName.AGENT_EXECUTIONS.value: ">"},
            count=batch_size,
            block=block_ms,
        )

        result = []
        if messages:
            for stream_name, stream_messages in messages:
                for message_id, data in stream_messages:
                    try:
                        payload = json.loads(data["payload"])
                        execution_msg = ExecutionMessage(**payload)
                        result.append((message_id, execution_msg))
                    except Exception as e:
                        logger.error(f"Failed to parse message {message_id}: {e}")
                        await self.move_to_dlq(
                            StreamName.AGENT_EXECUTIONS.value,
                            message_id,
                            data,
                            str(e),
                        )

        return result

    async def consume_results(
        self,
        batch_size: int = 10,
        block_ms: int = 5000,
    ) -> List[Tuple[str, ResultMessage]]:
        """
        Consume result messages from the queue.

        Args:
            batch_size: Number of messages to fetch
            block_ms: Block time in milliseconds

        Returns:
            List of (message_id, ResultMessage) tuples
        """
        messages = await self._redis.xreadgroup(
            self.group_name,
            self.consumer_name,
            {StreamName.AGENT_RESULTS.value: ">"},
            count=batch_size,
            block=block_ms,
        )

        result = []
        if messages:
            for stream_name, stream_messages in messages:
                for message_id, data in stream_messages:
                    try:
                        payload = json.loads(data["payload"])
                        result_msg = ResultMessage(**payload)
                        result.append((message_id, result_msg))
                    except Exception as e:
                        logger.error(f"Failed to parse result {message_id}: {e}")

        return result

    async def consume_events(
        self,
        event_types: Optional[List[str]] = None,
        batch_size: int = 50,
        block_ms: int = 1000,
    ) -> List[Tuple[str, EventMessage]]:
        """
        Consume event messages from the queue.

        Args:
            event_types: Filter by event types
            batch_size: Number of messages to fetch
            block_ms: Block time in milliseconds

        Returns:
            List of (message_id, EventMessage) tuples
        """
        messages = await self._redis.xreadgroup(
            self.group_name,
            self.consumer_name,
            {StreamName.EVENTS.value: ">"},
            count=batch_size,
            block=block_ms,
        )

        result = []
        if messages:
            for stream_name, stream_messages in messages:
                for message_id, data in stream_messages:
                    try:
                        # Filter by event type if specified
                        if event_types and data.get("event_type") not in event_types:
                            await self.acknowledge(StreamName.EVENTS.value, message_id)
                            continue

                        payload = json.loads(data["payload"])
                        event_msg = EventMessage(**payload)
                        result.append((message_id, event_msg))
                    except Exception as e:
                        logger.error(f"Failed to parse event {message_id}: {e}")

        return result

    # ==================== ACKNOWLEDGMENT ====================

    async def acknowledge(
        self,
        stream: str,
        message_id: str,
    ) -> None:
        """
        Acknowledge a processed message.

        Args:
            stream: Stream name
            message_id: Message ID to acknowledge
        """
        await self._redis.xack(stream, self.group_name, message_id)
        logger.debug(f"Acknowledged message: stream={stream}, id={message_id}")

    async def acknowledge_batch(
        self,
        stream: str,
        message_ids: List[str],
    ) -> None:
        """
        Acknowledge multiple messages.

        Args:
            stream: Stream name
            message_ids: Message IDs to acknowledge
        """
        if message_ids:
            await self._redis.xack(stream, self.group_name, *message_ids)
            logger.debug(f"Acknowledged {len(message_ids)} messages from {stream}")

    # ==================== DEAD LETTER QUEUE ====================

    async def move_to_dlq(
        self,
        source_stream: str,
        message_id: str,
        data: Dict[str, Any],
        error: str,
    ) -> str:
        """
        Move a failed message to the dead letter queue.

        Args:
            source_stream: Original stream name
            message_id: Original message ID
            data: Message data
            error: Error description

        Returns:
            DLQ message ID
        """
        dlq_data = {
            "original_stream": source_stream,
            "original_message_id": message_id,
            "payload": json.dumps(data),
            "error": error,
            "moved_at": datetime.utcnow().isoformat(),
        }

        dlq_message_id = await self._redis.xadd(
            StreamName.DEAD_LETTER.value,
            dlq_data,
            maxlen=50000,
        )

        # Acknowledge original message
        await self.acknowledge(source_stream, message_id)

        logger.warning(
            f"Moved message to DLQ: stream={source_stream}, id={message_id}, "
            f"dlq_id={dlq_message_id}, error={error}"
        )

        return dlq_message_id

    async def reprocess_dlq(
        self,
        count: int = 10,
    ) -> int:
        """
        Reprocess messages from the dead letter queue.

        Args:
            count: Number of messages to reprocess

        Returns:
            Number of messages reprocessed
        """
        messages = await self._redis.xrange(
            StreamName.DEAD_LETTER.value,
            count=count,
        )

        reprocessed = 0
        for message_id, data in messages:
            try:
                original_stream = data.get("original_stream")
                payload = json.loads(data.get("payload", "{}"))

                # Re-add to original stream
                await self._redis.xadd(original_stream, payload)

                # Remove from DLQ
                await self._redis.xdel(StreamName.DEAD_LETTER.value, message_id)

                reprocessed += 1
                logger.info(f"Reprocessed DLQ message: {message_id}")

            except Exception as e:
                logger.error(f"Failed to reprocess DLQ message {message_id}: {e}")

        return reprocessed

    # ==================== MONITORING ====================

    async def get_stream_info(self, stream: str) -> Dict[str, Any]:
        """
        Get information about a stream.

        Args:
            stream: Stream name

        Returns:
            Stream information dictionary
        """
        info = await self._redis.xinfo_stream(stream)
        return {
            "length": info.get("length", 0),
            "first_entry": info.get("first-entry"),
            "last_entry": info.get("last-entry"),
            "groups": info.get("groups", 0),
        }

    async def get_pending_count(self, stream: str) -> int:
        """
        Get count of pending messages for this consumer.

        Args:
            stream: Stream name

        Returns:
            Number of pending messages
        """
        try:
            pending = await self._redis.xpending(stream, self.group_name)
            return pending.get("pending", 0) if pending else 0
        except redis.ResponseError:
            return 0

    async def get_consumer_lag(self, stream: str) -> Dict[str, Any]:
        """
        Get consumer lag information.

        Args:
            stream: Stream name

        Returns:
            Lag information dictionary
        """
        try:
            groups = await self._redis.xinfo_groups(stream)
            for group in groups:
                if group.get("name") == self.group_name:
                    return {
                        "pending": group.get("pending", 0),
                        "consumers": group.get("consumers", 0),
                        "last_delivered_id": group.get("last-delivered-id"),
                    }
        except redis.ResponseError:
            pass

        return {"pending": 0, "consumers": 0, "last_delivered_id": None}

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Redis connection.

        Returns:
            Health check result dictionary
        """
        try:
            start = datetime.utcnow()
            await self._redis.ping()
            latency_ms = (datetime.utcnow() - start).total_seconds() * 1000

            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "connected": True,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False,
            }


class ExecutionWorker:
    """
    Worker that processes agent execution messages.

    This worker:
    - Consumes execution requests from Redis Streams
    - Executes agents with proper error handling
    - Publishes results back to the results stream
    - Handles retries and dead letter queue

    Example:
        >>> worker = ExecutionWorker(queue, agent_registry)
        >>> await worker.start()
    """

    def __init__(
        self,
        queue: RedisStreamQueue,
        agent_registry: Dict[str, Any],
        max_concurrent: int = 10,
    ):
        """
        Initialize Execution Worker.

        Args:
            queue: Redis stream queue instance
            agent_registry: Dictionary of agent_id -> agent_class
            max_concurrent: Maximum concurrent executions
        """
        self.queue = queue
        self.agent_registry = agent_registry
        self.max_concurrent = max_concurrent
        self._running = False
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(f"ExecutionWorker initialized: max_concurrent={max_concurrent}")

    async def start(self) -> None:
        """Start the worker loop."""
        self._running = True
        logger.info("ExecutionWorker started")

        while self._running:
            try:
                messages = await self.queue.consume_executions(
                    batch_size=self.max_concurrent,
                    block_ms=5000,
                )

                if messages:
                    tasks = [
                        self._process_message(msg_id, msg)
                        for msg_id, msg in messages
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        self._running = False
        logger.info("ExecutionWorker stopping...")

    async def _process_message(
        self,
        message_id: str,
        message: ExecutionMessage,
    ) -> None:
        """
        Process a single execution message.

        Args:
            message_id: Redis message ID
            message: Execution message
        """
        async with self._semaphore:
            start_time = datetime.utcnow()
            logger.info(
                f"Processing execution: id={message.execution_id}, "
                f"agent={message.agent_id}"
            )

            try:
                # Get agent class
                agent_class = self.agent_registry.get(message.agent_id)
                if not agent_class:
                    raise ValueError(f"Unknown agent: {message.agent_id}")

                # Execute agent
                agent = agent_class()
                result = agent.run(message.input_data)

                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

                # Publish success result
                await self.queue.publish_result(
                    execution_id=message.execution_id,
                    agent_id=message.agent_id,
                    status="completed",
                    output_data=result.dict() if hasattr(result, "dict") else result,
                    provenance_hash=getattr(result, "provenance_hash", None),
                    processing_time_ms=processing_time,
                )

                # Acknowledge message
                await self.queue.acknowledge(
                    StreamName.AGENT_EXECUTIONS.value,
                    message_id,
                )

                logger.info(
                    f"Execution completed: id={message.execution_id}, "
                    f"time={processing_time:.2f}ms"
                )

            except Exception as e:
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                logger.error(
                    f"Execution failed: id={message.execution_id}, error={e}",
                    exc_info=True,
                )

                # Check retry count
                if message.retry_count < message.max_retries:
                    # Retry by republishing with incremented count
                    message.retry_count += 1
                    await self.queue.publish_execution(
                        execution_id=message.execution_id,
                        agent_id=message.agent_id,
                        input_data=message.input_data,
                        tenant_id=message.tenant_id,
                        user_id=message.user_id,
                        priority=message.priority,
                        correlation_id=message.correlation_id,
                        timeout_seconds=message.timeout_seconds,
                    )
                    logger.info(
                        f"Retrying execution: id={message.execution_id}, "
                        f"retry={message.retry_count}/{message.max_retries}"
                    )
                else:
                    # Max retries exceeded - publish failure
                    await self.queue.publish_result(
                        execution_id=message.execution_id,
                        agent_id=message.agent_id,
                        status="failed",
                        error=str(e),
                        processing_time_ms=processing_time,
                    )

                    # Move to DLQ
                    await self.queue.move_to_dlq(
                        StreamName.AGENT_EXECUTIONS.value,
                        message_id,
                        message.dict(),
                        str(e),
                    )

                # Acknowledge original message
                await self.queue.acknowledge(
                    StreamName.AGENT_EXECUTIONS.value,
                    message_id,
                )
