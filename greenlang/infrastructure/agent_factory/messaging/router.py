# -*- coding: utf-8 -*-
"""
Message Router - Dual-transport message routing for inter-agent communication.

Routes ``MessageEnvelope`` instances to the correct transport based on
``channel_type``:

- **DurableTransport**: Wraps Redis Streams (XADD, XREADGROUP, XACK,
  XPENDING, XCLAIM) for at-least-once delivery with consumer groups.
- **EphemeralTransport**: Wraps Redis Pub/Sub for best-effort,
  fire-and-forget delivery.
- **MessageRouter**: Facade that selects the transport and provides
  convenience methods including request/response with timeout.

Classes:
    - DurableTransport: Redis Streams wrapper.
    - EphemeralTransport: Redis Pub/Sub wrapper.
    - MessageRouter: High-level routing facade.
    - RouterMetrics: Observable counters.

Example:
    >>> router = MessageRouter(redis_client)
    >>> await router.initialize()
    >>> receipt = await router.send_durable(
    ...     MessageEnvelope.command("intake", "carbon", {"scope": 1})
    ... )
    >>> await router.publish_event(
    ...     MessageEnvelope.event("carbon", "agent.health", {"status": "ok"})
    ... )
    >>> await router.close()

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from greenlang.infrastructure.agent_factory.messaging.protocol import (
    ChannelType,
    DeliveryStatus,
    MessageEnvelope,
    MessageReceipt,
    MessageType,
    PendingMessage,
)
from greenlang.infrastructure.agent_factory.messaging.acknowledgment import (
    AcknowledgmentTracker,
    AckTrackerConfig,
)

logger = logging.getLogger(__name__)

# Type alias for subscription handlers
MessageHandler = Callable[[MessageEnvelope], Awaitable[None]]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class RouterMetrics:
    """Observable counters for the message router.

    Attributes:
        durable_sent: Total durable messages sent.
        durable_consumed: Total durable messages consumed.
        durable_acknowledged: Total durable messages acknowledged.
        durable_failed: Total durable message failures.
        ephemeral_published: Total ephemeral messages published.
        ephemeral_received: Total ephemeral messages received.
        request_response_count: Total request/response round-trips.
        request_response_timeouts: Total request/response timeouts.
    """

    durable_sent: int = 0
    durable_consumed: int = 0
    durable_acknowledged: int = 0
    durable_failed: int = 0
    ephemeral_published: int = 0
    ephemeral_received: int = 0
    request_response_count: int = 0
    request_response_timeouts: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Serialize metrics to a plain dictionary.

        Returns:
            Dictionary of metric names to values.
        """
        return {
            "durable_sent": self.durable_sent,
            "durable_consumed": self.durable_consumed,
            "durable_acknowledged": self.durable_acknowledged,
            "durable_failed": self.durable_failed,
            "ephemeral_published": self.ephemeral_published,
            "ephemeral_received": self.ephemeral_received,
            "request_response_count": self.request_response_count,
            "request_response_timeouts": self.request_response_timeouts,
        }


# ---------------------------------------------------------------------------
# DurableTransport (Redis Streams)
# ---------------------------------------------------------------------------


class DurableTransport:
    """Redis Streams transport for at-least-once durable message delivery.

    Wraps the core Redis Stream commands (XADD, XREADGROUP, XACK,
    XPENDING, XCLAIM) behind a typed async API.  Each logical channel
    maps to a Redis Stream key with the configured prefix.

    Stream key format: ``{prefix}{channel_name}``

    Attributes:
        prefix: Stream key prefix (default ``gl:msg:``).
    """

    def __init__(
        self,
        redis_client: Any,
        prefix: str = "gl:msg:",
        max_stream_length: int = 100_000,
    ) -> None:
        """Initialize the durable transport.

        Args:
            redis_client: Async Redis client.
            prefix: Prefix for Redis Stream keys.
            max_stream_length: Approximate maximum entries per stream.
        """
        self._redis = redis_client
        self.prefix = prefix
        self._max_stream_length = max_stream_length

    # ------------------------------------------------------------------
    # Stream Name Helpers
    # ------------------------------------------------------------------

    def stream_key(self, channel_name: str) -> str:
        """Build the full Redis Stream key for a channel.

        Args:
            channel_name: Logical channel name.

        Returns:
            Fully qualified Redis Stream key.
        """
        return f"{self.prefix}{channel_name}"

    def consumer_group_name(self, channel_name: str) -> str:
        """Build the default consumer group name for a channel.

        Args:
            channel_name: Logical channel name.

        Returns:
            Consumer group name.
        """
        return f"{channel_name}-consumers"

    # ------------------------------------------------------------------
    # Consumer Group Setup
    # ------------------------------------------------------------------

    async def setup_consumer_group(
        self,
        stream_name: str,
        group: str,
    ) -> None:
        """Create a consumer group for a stream (idempotent).

        If the group already exists, the BUSYGROUP error is silently
        ignored.

        Args:
            stream_name: Redis Stream key.
            group: Consumer group name.
        """
        try:
            await self._redis.xgroup_create(
                stream_name,
                group,
                id="0",
                mkstream=True,
            )
            logger.info(
                "Consumer group '%s' created on stream %s",
                group,
                stream_name,
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
            logger.debug(
                "Consumer group '%s' already exists on stream %s",
                group,
                stream_name,
            )

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    async def send(
        self,
        stream_name: str,
        envelope: MessageEnvelope,
    ) -> str:
        """Send a message to a Redis Stream (XADD).

        Args:
            stream_name: Redis Stream key.
            envelope: Message envelope to send.

        Returns:
            Redis Stream entry ID (e.g. ``"1706000000000-0"``).
        """
        entry_id = await self._redis.xadd(
            stream_name,
            envelope.to_flat_dict(),
            maxlen=self._max_stream_length,
        )
        logger.debug(
            "XADD %s -> %s (msg=%s, type=%s)",
            stream_name,
            entry_id,
            envelope.id,
            envelope.message_type.value,
        )
        return entry_id

    # ------------------------------------------------------------------
    # Consume
    # ------------------------------------------------------------------

    async def consume(
        self,
        stream_name: str,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 2000,
    ) -> List[tuple[str, MessageEnvelope]]:
        """Read messages from a stream via consumer group (XREADGROUP).

        Each returned message must be acknowledged with ``acknowledge``
        after processing.

        Args:
            stream_name: Redis Stream key.
            group: Consumer group name.
            consumer: Consumer identifier within the group.
            count: Maximum messages to read per call.
            block_ms: Block duration in milliseconds (0 = non-blocking).

        Returns:
            List of ``(entry_id, envelope)`` tuples.
        """
        try:
            messages = await self._redis.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={stream_name: ">"},
                count=count,
                block=block_ms,
            )
        except Exception as exc:
            logger.error(
                "XREADGROUP error on stream %s (group=%s, consumer=%s): %s",
                stream_name,
                group,
                consumer,
                exc,
            )
            return []

        if not messages:
            return []

        results: List[tuple[str, MessageEnvelope]] = []
        for _stream, entries in messages:
            for entry_id, fields in entries:
                try:
                    envelope = MessageEnvelope.from_flat_dict(fields)
                    results.append((entry_id, envelope))
                except Exception as exc:
                    logger.warning(
                        "Failed to deserialize stream entry %s: %s",
                        entry_id,
                        exc,
                    )

        return results

    # ------------------------------------------------------------------
    # Acknowledge
    # ------------------------------------------------------------------

    async def acknowledge(
        self,
        stream_name: str,
        group: str,
        message_ids: List[str],
    ) -> int:
        """Acknowledge one or more messages in a consumer group (XACK).

        Args:
            stream_name: Redis Stream key.
            group: Consumer group name.
            message_ids: List of stream entry IDs to acknowledge.

        Returns:
            Number of messages successfully acknowledged.
        """
        if not message_ids:
            return 0
        try:
            acked = await self._redis.xack(stream_name, group, *message_ids)
            logger.debug(
                "XACK %s group=%s ids=%d acked=%d",
                stream_name,
                group,
                len(message_ids),
                acked,
            )
            return acked
        except Exception as exc:
            logger.error(
                "XACK failed on stream %s group %s: %s",
                stream_name,
                group,
                exc,
            )
            return 0

    # ------------------------------------------------------------------
    # Stale Reclamation
    # ------------------------------------------------------------------

    async def claim_stale(
        self,
        stream_name: str,
        group: str,
        consumer: str,
        min_idle_ms: int = 60_000,
        count: int = 100,
    ) -> List[tuple[str, MessageEnvelope]]:
        """Reclaim stale messages from crashed consumers (XCLAIM).

        Finds pending messages that have been idle for at least
        ``min_idle_ms`` and transfers ownership to the specified consumer.

        Args:
            stream_name: Redis Stream key.
            group: Consumer group name.
            consumer: Consumer that will take ownership.
            min_idle_ms: Minimum idle time in milliseconds.
            count: Maximum messages to reclaim.

        Returns:
            List of ``(entry_id, envelope)`` tuples.
        """
        # Find pending messages
        try:
            pending = await self._redis.xpending_range(
                stream_name,
                group,
                min="-",
                max="+",
                count=count,
            )
        except Exception as exc:
            logger.error(
                "XPENDING failed on stream %s group %s: %s",
                stream_name,
                group,
                exc,
            )
            return []

        stale_ids = [
            entry.get("message_id", "")
            for entry in pending
            if entry.get("time_since_delivered", 0) >= min_idle_ms
        ]
        if not stale_ids:
            return []

        # XCLAIM
        try:
            claimed = await self._redis.xclaim(
                stream_name,
                group,
                consumer,
                min_idle_time=min_idle_ms,
                message_ids=stale_ids,
            )
        except Exception as exc:
            logger.error(
                "XCLAIM failed on stream %s group %s: %s",
                stream_name,
                group,
                exc,
            )
            return []

        results: List[tuple[str, MessageEnvelope]] = []
        for entry_id, fields in claimed:
            try:
                envelope = MessageEnvelope.from_flat_dict(fields)
                results.append((entry_id, envelope))
            except Exception as exc:
                logger.warning(
                    "Failed to deserialize claimed entry %s: %s",
                    entry_id,
                    exc,
                )

        if results:
            logger.info(
                "Claimed %d stale messages from stream %s (min_idle=%dms)",
                len(results),
                stream_name,
                min_idle_ms,
            )
        return results

    # ------------------------------------------------------------------
    # Stream Info
    # ------------------------------------------------------------------

    async def stream_length(self, stream_name: str) -> int:
        """Get the number of entries in a stream.

        Args:
            stream_name: Redis Stream key.

        Returns:
            Number of entries.
        """
        try:
            return await self._redis.xlen(stream_name)
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# EphemeralTransport (Redis Pub/Sub)
# ---------------------------------------------------------------------------


class EphemeralTransport:
    """Redis Pub/Sub transport for best-effort ephemeral messaging.

    Messages published via this transport are fire-and-forget.  If no
    subscribers are listening, the message is silently dropped.

    Channel key format: ``{prefix}{channel_name}``

    Attributes:
        prefix: Channel key prefix (default ``gl:evt:``).
    """

    def __init__(
        self,
        redis_client: Any,
        prefix: str = "gl:evt:",
    ) -> None:
        """Initialize the ephemeral transport.

        Args:
            redis_client: Async Redis client.
            prefix: Prefix for Pub/Sub channel keys.
        """
        self._redis = redis_client
        self.prefix = prefix
        self._subscriptions: Dict[str, Any] = {}
        self._subscription_tasks: Dict[str, asyncio.Task[None]] = {}
        self._next_sub_id: int = 0

    # ------------------------------------------------------------------
    # Channel Name Helper
    # ------------------------------------------------------------------

    def channel_key(self, channel_name: str) -> str:
        """Build the full Redis Pub/Sub channel key.

        Args:
            channel_name: Logical channel name.

        Returns:
            Fully qualified Redis Pub/Sub channel key.
        """
        return f"{self.prefix}{channel_name}"

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish(
        self,
        channel: str,
        envelope: MessageEnvelope,
    ) -> int:
        """Publish a message to a Pub/Sub channel.

        Args:
            channel: Full Redis Pub/Sub channel key.
            envelope: Message envelope to publish.

        Returns:
            Number of subscribers that received the message.
        """
        import json
        data = json.dumps(envelope.to_dict(), default=str)
        subscriber_count = await self._redis.publish(channel, data)
        logger.debug(
            "Published to %s (subscribers=%d, msg=%s)",
            channel,
            subscriber_count,
            envelope.id,
        )
        return subscriber_count

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        channel: str,
        handler: MessageHandler,
    ) -> str:
        """Subscribe to a Pub/Sub channel with an async handler.

        The handler is invoked for each message received on the channel.
        Messages that fail deserialization are logged and skipped.

        Args:
            channel: Full Redis Pub/Sub channel key.
            handler: Async callback invoked with each MessageEnvelope.

        Returns:
            Subscription ID for use with ``unsubscribe``.
        """
        self._next_sub_id += 1
        sub_id = f"sub-{self._next_sub_id}-{channel}"

        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        self._subscriptions[sub_id] = pubsub

        async def _listener() -> None:
            """Background listener task for a Pub/Sub subscription."""
            import json
            try:
                async for message in pubsub.listen():
                    if message["type"] != "message":
                        continue
                    try:
                        raw_data = message["data"]
                        if isinstance(raw_data, bytes):
                            raw_data = raw_data.decode("utf-8")
                        parsed = json.loads(raw_data)
                        envelope = MessageEnvelope.from_dict(parsed)
                        await handler(envelope)
                    except Exception as exc:
                        logger.warning(
                            "Failed to process Pub/Sub message on %s: %s",
                            channel,
                            exc,
                        )
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error(
                    "Pub/Sub listener for %s terminated: %s", channel, exc
                )
            finally:
                try:
                    await pubsub.unsubscribe(channel)
                    await pubsub.close()
                except Exception:
                    pass

        task = asyncio.create_task(_listener(), name=f"pubsub-{sub_id}")
        self._subscription_tasks[sub_id] = task

        logger.info("Subscribed to %s (sub_id=%s)", channel, sub_id)
        return sub_id

    # ------------------------------------------------------------------
    # Unsubscribe
    # ------------------------------------------------------------------

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from a Pub/Sub channel.

        Cancels the background listener task and cleans up the pubsub
        connection.

        Args:
            subscription_id: ID returned by ``subscribe``.
        """
        task = self._subscription_tasks.pop(subscription_id, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        pubsub = self._subscriptions.pop(subscription_id, None)
        if pubsub is not None:
            try:
                await pubsub.close()
            except Exception:
                pass

        logger.info("Unsubscribed (sub_id=%s)", subscription_id)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close all active subscriptions."""
        sub_ids = list(self._subscription_tasks.keys())
        for sub_id in sub_ids:
            await self.unsubscribe(sub_id)
        logger.debug("EphemeralTransport closed (%d subscriptions)", len(sub_ids))


# ---------------------------------------------------------------------------
# MessageRouter
# ---------------------------------------------------------------------------


class MessageRouter:
    """High-level message routing facade for inter-agent communication.

    The router selects the correct transport based on the envelope's
    ``channel_type`` and provides convenience methods for common patterns
    including request/response with timeout.

    Attributes:
        metrics: Observable counters.
        durable: Durable transport instance.
        ephemeral: Ephemeral transport instance.
    """

    def __init__(
        self,
        redis_client: Any,
        stream_prefix: str = "gl:msg:",
        pubsub_prefix: str = "gl:evt:",
        max_stream_length: int = 100_000,
        ack_tracker_config: Optional[AckTrackerConfig] = None,
    ) -> None:
        """Initialize the message router.

        Args:
            redis_client: Async Redis client (dependency injection).
            stream_prefix: Prefix for Redis Stream keys.
            pubsub_prefix: Prefix for Redis Pub/Sub channel keys.
            max_stream_length: Approximate max entries per stream.
            ack_tracker_config: Configuration for the acknowledgment tracker.
        """
        self._redis = redis_client
        self.durable = DurableTransport(
            redis_client, stream_prefix, max_stream_length
        )
        self.ephemeral = EphemeralTransport(redis_client, pubsub_prefix)
        self._ack_tracker = AcknowledgmentTracker(
            redis_client, ack_tracker_config
        )
        self.metrics = RouterMetrics()
        self._initialized = False
        logger.debug(
            "MessageRouter constructed (stream_prefix=%s, pubsub_prefix=%s)",
            stream_prefix,
            pubsub_prefix,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Verify Redis connectivity and mark the router as ready."""
        try:
            await self._redis.ping()
        except Exception as exc:
            raise RuntimeError(
                f"MessageRouter initialization failed: Redis ping error: {exc}"
            ) from exc
        self._initialized = True
        logger.info("MessageRouter initialized")

    async def close(self) -> None:
        """Close all transports and subscriptions."""
        await self.ephemeral.close()
        self._initialized = False
        logger.info("MessageRouter closed")

    # ------------------------------------------------------------------
    # Durable Messaging
    # ------------------------------------------------------------------

    async def send_durable(
        self,
        envelope: MessageEnvelope,
        channel_name: Optional[str] = None,
    ) -> MessageReceipt:
        """Send a message via Redis Streams (at-least-once delivery).

        The envelope's ``channel_type`` is set to DURABLE.  A consumer
        group is created automatically if it does not exist.

        Args:
            envelope: Message envelope to send.
            channel_name: Override the channel name.  Defaults to
                ``envelope.target_agent``.

        Returns:
            MessageReceipt with the stream entry ID.
        """
        self._ensure_initialized()
        envelope.channel_type = ChannelType.DURABLE

        channel = channel_name or envelope.target_agent
        stream_name = self.durable.stream_key(channel)
        group = self.durable.consumer_group_name(channel)

        # Ensure consumer group exists
        await self.durable.setup_consumer_group(stream_name, group)

        # XADD
        start = time.monotonic()
        entry_id = await self.durable.send(stream_name, envelope)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Track for acknowledgment
        await self._ack_tracker.track(stream_name, entry_id, envelope)

        self.metrics.durable_sent += 1
        logger.debug(
            "Durable send: stream=%s, entry=%s, msg=%s (%.1fms)",
            stream_name,
            entry_id,
            envelope.id,
            elapsed_ms,
        )

        return MessageReceipt(
            message_id=envelope.id,
            stream_id=entry_id,
            delivery_status=DeliveryStatus.DELIVERED,
            delivered_at=datetime.now(timezone.utc),
        )

    async def consume_durable(
        self,
        channel_name: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 2000,
    ) -> List[tuple[str, MessageEnvelope]]:
        """Consume messages from a durable channel.

        Args:
            channel_name: Logical channel name.
            consumer: Consumer identifier.
            count: Maximum messages per read.
            block_ms: Block duration in milliseconds.

        Returns:
            List of ``(entry_id, envelope)`` tuples.
        """
        self._ensure_initialized()
        stream_name = self.durable.stream_key(channel_name)
        group = self.durable.consumer_group_name(channel_name)

        results = await self.durable.consume(
            stream_name, group, consumer, count, block_ms
        )

        # Filter expired messages
        valid: List[tuple[str, MessageEnvelope]] = []
        for entry_id, envelope in results:
            if envelope.is_expired():
                await self.durable.acknowledge(stream_name, group, [entry_id])
                logger.info(
                    "Discarded expired message %s (ttl=%ds)",
                    envelope.id,
                    envelope.ttl_seconds,
                )
                continue
            valid.append((entry_id, envelope))

        self.metrics.durable_consumed += len(valid)
        return valid

    async def acknowledge_durable(
        self,
        channel_name: str,
        message_ids: List[str],
    ) -> int:
        """Acknowledge durable messages.

        Args:
            channel_name: Logical channel name.
            message_ids: Stream entry IDs to acknowledge.

        Returns:
            Number acknowledged.
        """
        self._ensure_initialized()
        stream_name = self.durable.stream_key(channel_name)
        group = self.durable.consumer_group_name(channel_name)

        acked = await self.durable.acknowledge(stream_name, group, message_ids)
        self.metrics.durable_acknowledged += acked

        # Update tracking
        for mid in message_ids:
            await self._ack_tracker.acknowledge(stream_name, group, mid)

        return acked

    # ------------------------------------------------------------------
    # Ephemeral Messaging
    # ------------------------------------------------------------------

    async def publish_event(
        self,
        envelope: MessageEnvelope,
        channel_name: Optional[str] = None,
    ) -> int:
        """Publish a message via Redis Pub/Sub (fire-and-forget).

        Args:
            envelope: Message envelope to publish.
            channel_name: Override the channel name.  Defaults to
                ``envelope.target_agent``.

        Returns:
            Number of subscribers that received the message.
        """
        self._ensure_initialized()
        envelope.channel_type = ChannelType.EPHEMERAL

        channel = channel_name or envelope.target_agent
        full_channel = self.ephemeral.channel_key(channel)

        subscriber_count = await self.ephemeral.publish(full_channel, envelope)
        self.metrics.ephemeral_published += 1

        return subscriber_count

    async def subscribe_events(
        self,
        channel_name: str,
        handler: MessageHandler,
    ) -> str:
        """Subscribe to ephemeral events on a channel.

        Args:
            channel_name: Logical channel name.
            handler: Async callback for each message.

        Returns:
            Subscription ID for use with ``unsubscribe_events``.
        """
        self._ensure_initialized()
        full_channel = self.ephemeral.channel_key(channel_name)

        async def _wrapped_handler(envelope: MessageEnvelope) -> None:
            self.metrics.ephemeral_received += 1
            await handler(envelope)

        return await self.ephemeral.subscribe(full_channel, _wrapped_handler)

    async def unsubscribe_events(self, subscription_id: str) -> None:
        """Unsubscribe from ephemeral events.

        Args:
            subscription_id: ID returned by ``subscribe_events``.
        """
        await self.ephemeral.unsubscribe(subscription_id)

    # ------------------------------------------------------------------
    # Request / Response
    # ------------------------------------------------------------------

    async def request_response(
        self,
        envelope: MessageEnvelope,
        timeout_s: float = 30.0,
    ) -> MessageEnvelope:
        """Send a durable request and wait for a response on the reply stream.

        Creates a temporary reply stream, sends the request, and blocks
        until a correlated response arrives or the timeout expires.

        Args:
            envelope: REQUEST or QUERY envelope with ``reply_to`` set.
            timeout_s: Maximum seconds to wait for a response.

        Returns:
            The response MessageEnvelope.

        Raises:
            TimeoutError: If no response arrives within ``timeout_s``.
            ValueError: If the envelope is not a REQUEST or QUERY.
        """
        self._ensure_initialized()

        if envelope.message_type not in (MessageType.REQUEST, MessageType.QUERY):
            raise ValueError(
                f"request_response requires REQUEST or QUERY, got {envelope.message_type}"
            )

        # Ensure a reply channel is set
        if not envelope.reply_to:
            envelope.reply_to = (
                f"reply:{envelope.source_agent}:{envelope.correlation_id}"
            )

        reply_stream = self.durable.stream_key(envelope.reply_to)
        reply_group = f"reply-{envelope.correlation_id}-consumers"

        # Setup reply consumer group
        await self.durable.setup_consumer_group(reply_stream, reply_group)

        # Send the request
        await self.send_durable(envelope)
        self.metrics.request_response_count += 1

        # Wait for response
        consumer_name = f"requester-{envelope.source_agent}"
        deadline = time.monotonic() + timeout_s
        poll_block_ms = min(1000, int(timeout_s * 1000))

        while time.monotonic() < deadline:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            block = min(poll_block_ms, max(100, remaining_ms))

            results = await self.durable.consume(
                reply_stream, reply_group, consumer_name, count=1, block_ms=block
            )

            for entry_id, response_env in results:
                if response_env.correlation_id == envelope.correlation_id:
                    # Acknowledge the response
                    await self.durable.acknowledge(
                        reply_stream, reply_group, [entry_id]
                    )
                    logger.debug(
                        "Received response for request %s (correlation=%s)",
                        envelope.id,
                        envelope.correlation_id,
                    )
                    return response_env

        self.metrics.request_response_timeouts += 1
        raise TimeoutError(
            f"No response received for request {envelope.id} "
            f"(correlation={envelope.correlation_id}) within {timeout_s}s"
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot of the router state.

        Returns:
            Dictionary with metrics and initialization status.
        """
        return {
            "initialized": self._initialized,
            "metrics": self.metrics.to_dict(),
            "ack_tracker_metrics": self._ack_tracker.metrics.to_dict(),
        }

    @property
    def ack_tracker(self) -> AcknowledgmentTracker:
        """Expose the acknowledgment tracker for advanced usage.

        Returns:
            The AcknowledgmentTracker instance.
        """
        return self._ack_tracker

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """Raise if the router has not been initialized."""
        if not self._initialized:
            raise RuntimeError(
                "MessageRouter is not initialized; call initialize() first"
            )


__all__ = [
    "DurableTransport",
    "EphemeralTransport",
    "MessageHandler",
    "MessageRouter",
    "RouterMetrics",
]
