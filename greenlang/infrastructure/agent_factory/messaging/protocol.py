# -*- coding: utf-8 -*-
"""
Inter-Agent Messaging Protocol - Core message envelope and type definitions.

Defines the canonical message format for all inter-agent communication in the
GreenLang Agent Factory.  Every message is wrapped in a ``MessageEnvelope``
that carries routing information, provenance metadata, and serialization
hints.  Two channel types coexist:

- **Durable** (Redis Streams): at-least-once delivery with consumer groups,
  acknowledgment tracking, and dead-letter queues.
- **Ephemeral** (Redis Pub/Sub): best-effort, fire-and-forget delivery for
  telemetry signals, config broadcasts, and health pings.

Classes:
    - MessageType: Semantic type of the message payload.
    - ChannelType: Transport selection (durable vs. ephemeral).
    - DeliveryStatus: Lifecycle state of a durable message.
    - MessageEnvelope: Canonical message wrapper.
    - MessageReceipt: Confirmation of message delivery.
    - PendingMessage: Metadata about an unacknowledged durable message.

Example:
    >>> envelope = MessageEnvelope.request(
    ...     source_agent="intake-agent",
    ...     target_agent="carbon-agent",
    ...     payload={"scope": 1, "year": 2025},
    ... )
    >>> assert envelope.message_type == MessageType.REQUEST
    >>> assert envelope.channel_type == ChannelType.DURABLE

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MessageType(str, Enum):
    """Semantic type of the message payload.

    Determines how the receiving agent should interpret and handle the
    message:
        - REQUEST: Expects a RESPONSE on the reply channel.
        - RESPONSE: Answer to a previous REQUEST.
        - EVENT: Informational broadcast (no response expected).
        - COMMAND: Imperative instruction (no response expected).
        - QUERY: Read-only request (expects a RESPONSE).
    """

    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"
    QUERY = "query"


class ChannelType(str, Enum):
    """Transport selection for a message.

    Attributes:
        DURABLE: Redis Streams -- at-least-once delivery with XACK.
        EPHEMERAL: Redis Pub/Sub -- best-effort, fire-and-forget.
    """

    DURABLE = "durable"
    EPHEMERAL = "ephemeral"


class DeliveryStatus(str, Enum):
    """Lifecycle state of a durable message.

    Tracks the delivery lifecycle from submission through final disposition:
        PENDING -> DELIVERED -> ACKNOWLEDGED
        PENDING -> DELIVERED -> FAILED -> (retry or DLQ)
        PENDING -> EXPIRED
    """

    PENDING = "pending"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------
# Value Objects
# ---------------------------------------------------------------------------


@dataclass
class MessageReceipt:
    """Confirmation of message delivery or acknowledgment.

    Returned by durable send operations to give the caller a handle for
    tracking message disposition.

    Attributes:
        message_id: UUID of the message envelope.
        stream_id: Redis Stream entry ID (e.g. ``"1706000000000-0"``).
        delivery_status: Current delivery status.
        delivered_at: UTC timestamp when the message was appended to the stream.
        acknowledged_at: UTC timestamp when XACK was received (None until acked).
        error: Error description if delivery failed.
    """

    message_id: uuid.UUID
    stream_id: str = ""
    delivery_status: DeliveryStatus = DeliveryStatus.PENDING
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt to a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "message_id": str(self.message_id),
            "stream_id": self.stream_id,
            "delivery_status": self.delivery_status.value,
            "delivered_at": (
                self.delivered_at.isoformat() if self.delivered_at else None
            ),
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
            "error": self.error,
        }


@dataclass
class PendingMessage:
    """Metadata about an unacknowledged durable message.

    Populated from Redis ``XPENDING`` responses for stale-message
    reclamation and monitoring.

    Attributes:
        stream_id: Redis Stream entry ID.
        consumer: Name of the consumer that owns the message.
        idle_ms: Milliseconds since the message was last delivered.
        delivery_count: Number of times this message has been delivered.
    """

    stream_id: str
    consumer: str
    idle_ms: int
    delivery_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "stream_id": self.stream_id,
            "consumer": self.consumer,
            "idle_ms": self.idle_ms,
            "delivery_count": self.delivery_count,
        }


# ---------------------------------------------------------------------------
# MessageEnvelope
# ---------------------------------------------------------------------------


@dataclass
class MessageEnvelope:
    """Canonical message wrapper for inter-agent communication.

    Every message exchanged between GreenLang agents -- whether via durable
    Redis Streams or ephemeral Pub/Sub -- is wrapped in a ``MessageEnvelope``.
    The envelope carries routing, correlation, and provenance metadata
    alongside the application payload.

    Attributes:
        id: Unique message identifier (UUID4, auto-generated).
        correlation_id: Correlation identifier for request/response chains.
        source_agent: Key of the sending agent.
        target_agent: Key of the receiving agent or channel name.
        message_type: Semantic type (request, response, event, command, query).
        channel_type: Transport selection (durable or ephemeral).
        payload: Application-specific data (must be JSON-serializable).
        metadata: Auxiliary key-value pairs (tenant_id, priority, ttl, etc.).
        timestamp: UTC creation timestamp.
        schema_version: Envelope schema version for forward compatibility.
        reply_to: Reply channel name for request/response patterns.
        ttl_seconds: Time-to-live in seconds (0 = no expiry).
        attempt: Current delivery attempt number (durable only).
        max_retries: Maximum delivery retries before dead-lettering.

    Example:
        >>> env = MessageEnvelope(
        ...     source_agent="intake-agent",
        ...     target_agent="carbon-agent",
        ...     message_type=MessageType.REQUEST,
        ...     channel_type=ChannelType.DURABLE,
        ...     payload={"scope": 1},
        ... )
        >>> data = env.to_dict()
        >>> restored = MessageEnvelope.from_dict(data)
        >>> assert restored.id == env.id
    """

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    correlation_id: uuid.UUID = field(default_factory=uuid.uuid4)
    source_agent: str = ""
    target_agent: str = ""
    message_type: MessageType = MessageType.EVENT
    channel_type: ChannelType = ChannelType.EPHEMERAL
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    schema_version: str = "1.0"
    reply_to: Optional[str] = None
    ttl_seconds: int = 0
    attempt: int = 0
    max_retries: int = 3

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the envelope to a plain dictionary.

        All values are converted to JSON-safe primitives (strings, ints,
        dicts).  UUIDs and datetimes are serialized to ISO strings.

        Returns:
            Dictionary representation suitable for JSON or MessagePack encoding.
        """
        return {
            "id": str(self.id),
            "correlation_id": str(self.correlation_id),
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "message_type": self.message_type.value,
            "channel_type": self.channel_type.value,
            "payload": self.payload,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "schema_version": self.schema_version,
            "reply_to": self.reply_to,
            "ttl_seconds": self.ttl_seconds,
            "attempt": self.attempt,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MessageEnvelope:
        """Deserialize an envelope from a plain dictionary.

        Handles both raw dicts and dicts produced by ``to_dict()``.
        Missing fields are filled with safe defaults.

        Args:
            data: Dictionary with envelope fields.

        Returns:
            Reconstructed MessageEnvelope instance.

        Raises:
            ValueError: If required fields contain unparseable values.
        """
        # Parse UUID fields
        msg_id = data.get("id")
        if msg_id is not None:
            msg_id = uuid.UUID(str(msg_id)) if not isinstance(msg_id, uuid.UUID) else msg_id
        else:
            msg_id = uuid.uuid4()

        corr_id = data.get("correlation_id")
        if corr_id is not None:
            corr_id = uuid.UUID(str(corr_id)) if not isinstance(corr_id, uuid.UUID) else corr_id
        else:
            corr_id = uuid.uuid4()

        # Parse timestamp
        ts = data.get("timestamp")
        if isinstance(ts, str):
            timestamp = datetime.fromisoformat(ts)
        elif isinstance(ts, datetime):
            timestamp = ts
        else:
            timestamp = datetime.now(timezone.utc)

        # Parse enums with safe fallback
        message_type = MessageType(data.get("message_type", MessageType.EVENT.value))
        channel_type = ChannelType(data.get("channel_type", ChannelType.EPHEMERAL.value))

        return cls(
            id=msg_id,
            correlation_id=corr_id,
            source_agent=data.get("source_agent", ""),
            target_agent=data.get("target_agent", ""),
            message_type=message_type,
            channel_type=channel_type,
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            timestamp=timestamp,
            schema_version=data.get("schema_version", "1.0"),
            reply_to=data.get("reply_to"),
            ttl_seconds=int(data.get("ttl_seconds", 0)),
            attempt=int(data.get("attempt", 0)),
            max_retries=int(data.get("max_retries", 3)),
        )

    # ------------------------------------------------------------------
    # Flat serialization for Redis Streams (XADD requires str values)
    # ------------------------------------------------------------------

    def to_flat_dict(self) -> Dict[str, str]:
        """Serialize the envelope to a flat dict of strings for XADD.

        Redis Streams require all field values to be strings.  Complex
        fields (payload, metadata) are JSON-encoded.

        Returns:
            Dictionary with string keys and string values.
        """
        import json
        return {
            "id": str(self.id),
            "correlation_id": str(self.correlation_id),
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "message_type": self.message_type.value,
            "channel_type": self.channel_type.value,
            "payload": json.dumps(self.payload),
            "metadata": json.dumps(self.metadata),
            "timestamp": self.timestamp.isoformat(),
            "schema_version": self.schema_version,
            "reply_to": self.reply_to or "",
            "ttl_seconds": str(self.ttl_seconds),
            "attempt": str(self.attempt),
            "max_retries": str(self.max_retries),
        }

    @classmethod
    def from_flat_dict(cls, data: Dict[str, str]) -> MessageEnvelope:
        """Deserialize from a flat dict of strings (Redis Stream entry).

        This is the inverse of ``to_flat_dict()``.  JSON-encoded fields
        are parsed back to their native types.

        Args:
            data: Flat dictionary from XREADGROUP.

        Returns:
            Reconstructed MessageEnvelope instance.
        """
        import json
        parsed: Dict[str, Any] = {
            "id": data.get("id", ""),
            "correlation_id": data.get("correlation_id", ""),
            "source_agent": data.get("source_agent", ""),
            "target_agent": data.get("target_agent", ""),
            "message_type": data.get("message_type", "event"),
            "channel_type": data.get("channel_type", "ephemeral"),
            "payload": json.loads(data.get("payload", "{}")),
            "metadata": json.loads(data.get("metadata", "{}")),
            "timestamp": data.get("timestamp", ""),
            "schema_version": data.get("schema_version", "1.0"),
            "reply_to": data.get("reply_to") or None,
            "ttl_seconds": int(data.get("ttl_seconds", "0")),
            "attempt": int(data.get("attempt", "0")),
            "max_retries": int(data.get("max_retries", "3")),
        }
        return cls.from_dict(parsed)

    # ------------------------------------------------------------------
    # Expiry check
    # ------------------------------------------------------------------

    def is_expired(self) -> bool:
        """Check whether the message has exceeded its TTL.

        Returns:
            True if the message is expired, False otherwise.
        """
        if self.ttl_seconds <= 0:
            return False
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > self.ttl_seconds

    # ------------------------------------------------------------------
    # Factory Methods
    # ------------------------------------------------------------------

    @classmethod
    def request(
        cls,
        source_agent: str,
        target_agent: str,
        payload: Dict[str, Any],
        *,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        ttl_seconds: int = 300,
        max_retries: int = 3,
    ) -> MessageEnvelope:
        """Create a durable REQUEST envelope.

        Requests are always sent via the durable transport and expect a
        RESPONSE on the ``reply_to`` channel.  If no reply channel is
        given, one is auto-generated from the source agent key.

        Args:
            source_agent: Sending agent key.
            target_agent: Receiving agent key.
            payload: Request payload.
            reply_to: Explicit reply channel name.
            metadata: Additional key-value metadata.
            ttl_seconds: Time-to-live for the request.
            max_retries: Maximum delivery retries.

        Returns:
            Configured MessageEnvelope.
        """
        correlation_id = uuid.uuid4()
        return cls(
            correlation_id=correlation_id,
            source_agent=source_agent,
            target_agent=target_agent,
            message_type=MessageType.REQUEST,
            channel_type=ChannelType.DURABLE,
            payload=payload,
            metadata=metadata or {},
            reply_to=reply_to or f"reply:{source_agent}:{correlation_id}",
            ttl_seconds=ttl_seconds,
            max_retries=max_retries,
        )

    @classmethod
    def response(
        cls,
        source_agent: str,
        original: MessageEnvelope,
        payload: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, str]] = None,
    ) -> MessageEnvelope:
        """Create a durable RESPONSE envelope correlated to an original request.

        The response inherits the ``correlation_id`` of the original request
        and targets its ``reply_to`` channel.

        Args:
            source_agent: Responding agent key.
            original: The original REQUEST envelope.
            payload: Response payload.
            metadata: Additional key-value metadata.

        Returns:
            Configured MessageEnvelope.
        """
        return cls(
            correlation_id=original.correlation_id,
            source_agent=source_agent,
            target_agent=original.source_agent,
            message_type=MessageType.RESPONSE,
            channel_type=ChannelType.DURABLE,
            payload=payload,
            metadata=metadata or {},
            reply_to=None,
        )

    @classmethod
    def event(
        cls,
        source_agent: str,
        channel: str,
        payload: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, str]] = None,
    ) -> MessageEnvelope:
        """Create an ephemeral EVENT envelope for broadcast.

        Events are fire-and-forget notifications published via Redis
        Pub/Sub.  They are not persisted or retried.

        Args:
            source_agent: Publishing agent key.
            channel: Target Pub/Sub channel name.
            payload: Event data.
            metadata: Additional key-value metadata.

        Returns:
            Configured MessageEnvelope.
        """
        return cls(
            source_agent=source_agent,
            target_agent=channel,
            message_type=MessageType.EVENT,
            channel_type=ChannelType.EPHEMERAL,
            payload=payload,
            metadata=metadata or {},
        )

    @classmethod
    def command(
        cls,
        source_agent: str,
        target_agent: str,
        payload: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, str]] = None,
        ttl_seconds: int = 600,
        max_retries: int = 3,
    ) -> MessageEnvelope:
        """Create a durable COMMAND envelope.

        Commands are imperative instructions delivered via Redis Streams.
        They do not expect a response but are guaranteed at-least-once
        delivery.

        Args:
            source_agent: Sending agent key.
            target_agent: Receiving agent key.
            payload: Command data.
            metadata: Additional key-value metadata.
            ttl_seconds: Time-to-live for the command.
            max_retries: Maximum delivery retries.

        Returns:
            Configured MessageEnvelope.
        """
        return cls(
            source_agent=source_agent,
            target_agent=target_agent,
            message_type=MessageType.COMMAND,
            channel_type=ChannelType.DURABLE,
            payload=payload,
            metadata=metadata or {},
            ttl_seconds=ttl_seconds,
            max_retries=max_retries,
        )

    @classmethod
    def query(
        cls,
        source_agent: str,
        target_agent: str,
        payload: Dict[str, Any],
        *,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        ttl_seconds: int = 120,
    ) -> MessageEnvelope:
        """Create a durable QUERY envelope.

        Queries are read-only requests that expect a RESPONSE.  They use
        the durable transport for reliability.

        Args:
            source_agent: Sending agent key.
            target_agent: Receiving agent key.
            payload: Query parameters.
            reply_to: Explicit reply channel name.
            metadata: Additional key-value metadata.
            ttl_seconds: Time-to-live for the query.

        Returns:
            Configured MessageEnvelope.
        """
        correlation_id = uuid.uuid4()
        return cls(
            correlation_id=correlation_id,
            source_agent=source_agent,
            target_agent=target_agent,
            message_type=MessageType.QUERY,
            channel_type=ChannelType.DURABLE,
            payload=payload,
            metadata=metadata or {},
            reply_to=reply_to or f"reply:{source_agent}:{correlation_id}",
            ttl_seconds=ttl_seconds,
        )


__all__ = [
    "ChannelType",
    "DeliveryStatus",
    "MessageEnvelope",
    "MessageReceipt",
    "MessageType",
    "PendingMessage",
]
