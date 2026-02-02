"""
GL-016 Waterguard Event Envelope

Standard event wrapper providing consistent metadata, serialization,
and traceability for all messages in the Waterguard streaming pipeline.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Envelope Enums
# =============================================================================

class EventPriority(str, Enum):
    """Event priority levels for processing."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EnvelopeVersion(str, Enum):
    """Envelope schema versions."""
    V1 = "1.0"
    V2 = "2.0"


class CompressionType(str, Enum):
    """Payload compression types."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"


# =============================================================================
# Event Metadata
# =============================================================================

class EventMetadata(BaseModel):
    """
    Metadata for event tracing and processing.

    Provides complete lineage and context for each event in the system.
    """

    # Identification
    event_id: UUID = Field(default_factory=uuid4, description="Unique event ID")
    event_type: str = Field(..., description="Event type name (e.g., 'RawChemistryMessage')")
    event_version: str = Field(default="1.0", description="Event schema version")

    # Timing
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event creation timestamp (UTC)"
    )
    ingestion_time: Optional[datetime] = Field(
        default=None,
        description="Time when event entered the system"
    )
    processing_time: Optional[datetime] = Field(
        default=None,
        description="Time when event was processed"
    )

    # Source
    source_system: str = Field(..., description="Source system identifier")
    source_component: str = Field(default="unknown", description="Source component")
    source_host: Optional[str] = Field(default=None, description="Source hostname/IP")

    # Tracing
    trace_id: UUID = Field(
        default_factory=uuid4,
        description="Distributed trace ID for request correlation"
    )
    span_id: Optional[UUID] = Field(default=None, description="Current span ID")
    parent_span_id: Optional[UUID] = Field(default=None, description="Parent span ID")
    correlation_id: Optional[UUID] = Field(
        default=None,
        description="Correlation ID for request/response matching"
    )

    # Lineage
    causation_id: Optional[UUID] = Field(
        default=None,
        description="ID of event that caused this event"
    )
    upstream_ids: list[UUID] = Field(
        default_factory=list,
        description="IDs of upstream events that contributed to this event"
    )

    # Processing
    priority: EventPriority = Field(
        default=EventPriority.NORMAL,
        description="Processing priority"
    )
    retry_count: int = Field(default=0, description="Number of processing retries")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # Tags and Context
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom key-value tags"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context data"
    )

    # Security
    tenant_id: Optional[str] = Field(default=None, description="Multi-tenant identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")

    def with_span(self) -> "EventMetadata":
        """Create a new child span."""
        return self.model_copy(
            update={
                "parent_span_id": self.span_id,
                "span_id": uuid4(),
            }
        )

    def with_causation(self, causation_id: UUID) -> "EventMetadata":
        """Set causation ID."""
        return self.model_copy(update={"causation_id": causation_id})

    def add_tag(self, key: str, value: str) -> "EventMetadata":
        """Add a tag."""
        new_tags = {**self.tags, key: value}
        return self.model_copy(update={"tags": new_tags})

    def increment_retry(self) -> "EventMetadata":
        """Increment retry count."""
        return self.model_copy(update={"retry_count": self.retry_count + 1})


# =============================================================================
# Event Envelope
# =============================================================================

class EventEnvelope(BaseModel, Generic[T]):
    """
    Standard event envelope wrapping all Waterguard events.

    Provides:
    - Consistent metadata across all events
    - Serialization/deserialization
    - Payload compression
    - Integrity verification
    - Schema versioning

    Example:
        # Create envelope for raw chemistry message
        message = RawChemistryMessage(...)
        envelope = EventEnvelope.wrap(
            payload=message,
            source_system="opc-ua-connector",
        )

        # Serialize for Kafka
        json_bytes = envelope.serialize()

        # Deserialize
        envelope = EventEnvelope.deserialize(json_bytes, RawChemistryMessage)
        message = envelope.payload
    """

    # Envelope metadata
    envelope_version: EnvelopeVersion = Field(
        default=EnvelopeVersion.V2,
        description="Envelope schema version"
    )
    envelope_id: UUID = Field(
        default_factory=uuid4,
        description="Unique envelope identifier"
    )

    # Event metadata
    metadata: EventMetadata = Field(..., description="Event metadata")

    # Payload
    payload_type: str = Field(..., description="Payload class name")
    payload: Dict[str, Any] = Field(..., description="Serialized payload")
    payload_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of payload for integrity"
    )

    # Compression
    compression: CompressionType = Field(
        default=CompressionType.NONE,
        description="Payload compression type"
    )
    compressed_size: Optional[int] = Field(
        default=None,
        description="Compressed payload size in bytes"
    )
    original_size: Optional[int] = Field(
        default=None,
        description="Original payload size in bytes"
    )

    # Routing
    topic: Optional[str] = Field(default=None, description="Target Kafka topic")
    partition_key: Optional[str] = Field(
        default=None,
        description="Partition key for consistent routing"
    )

    class Config:
        extra = "forbid"
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str,
        }

    @classmethod
    def wrap(
        cls,
        payload: T,
        source_system: str,
        source_component: str = "unknown",
        trace_id: Optional[UUID] = None,
        priority: EventPriority = EventPriority.NORMAL,
        topic: Optional[str] = None,
        partition_key: Optional[str] = None,
        compression: CompressionType = CompressionType.NONE,
        tags: Optional[Dict[str, str]] = None,
    ) -> "EventEnvelope[T]":
        """
        Wrap a payload in an event envelope.

        Args:
            payload: Pydantic model to wrap
            source_system: Source system identifier
            source_component: Source component name
            trace_id: Optional trace ID (generated if not provided)
            priority: Event priority
            topic: Target Kafka topic
            partition_key: Partition key
            compression: Compression type
            tags: Custom tags

        Returns:
            EventEnvelope instance
        """
        payload_dict = payload.model_dump()
        payload_json = json.dumps(payload_dict, default=str)
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()

        metadata = EventMetadata(
            event_type=type(payload).__name__,
            source_system=source_system,
            source_component=source_component,
            trace_id=trace_id or uuid4(),
            priority=priority,
            tags=tags or {},
            ingestion_time=datetime.utcnow(),
        )

        return cls(
            metadata=metadata,
            payload_type=type(payload).__name__,
            payload=payload_dict,
            payload_hash=payload_hash,
            topic=topic,
            partition_key=partition_key,
            compression=compression,
            original_size=len(payload_json),
        )

    def unwrap(self, payload_class: Type[T]) -> T:
        """
        Unwrap and deserialize the payload.

        Args:
            payload_class: Expected payload class

        Returns:
            Deserialized payload instance

        Raises:
            ValueError: If payload type doesn't match
        """
        if self.payload_type != payload_class.__name__:
            raise ValueError(
                f"Payload type mismatch: expected {payload_class.__name__}, "
                f"got {self.payload_type}"
            )

        return payload_class.model_validate(self.payload)

    def verify_integrity(self) -> bool:
        """
        Verify payload integrity using hash.

        Returns:
            True if integrity check passes
        """
        if not self.payload_hash:
            return True  # No hash to verify

        payload_json = json.dumps(self.payload, default=str)
        computed_hash = hashlib.sha256(payload_json.encode()).hexdigest()
        return computed_hash == self.payload_hash

    def serialize(self, compress: bool = False) -> bytes:
        """
        Serialize envelope to bytes.

        Args:
            compress: Whether to apply gzip compression

        Returns:
            Serialized bytes
        """
        json_str = self.model_dump_json()
        data = json_str.encode("utf-8")

        if compress:
            data = gzip.compress(data)

        return data

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        payload_class: Optional[Type[T]] = None,
        decompress: bool = False,
    ) -> "EventEnvelope[T]":
        """
        Deserialize envelope from bytes.

        Args:
            data: Serialized bytes
            payload_class: Optional payload class for validation
            decompress: Whether to decompress gzip data

        Returns:
            EventEnvelope instance
        """
        if decompress:
            data = gzip.decompress(data)

        json_str = data.decode("utf-8")
        envelope = cls.model_validate_json(json_str)

        # Verify integrity
        if not envelope.verify_integrity():
            logger.warning(f"Integrity check failed for envelope {envelope.envelope_id}")

        return envelope

    def to_kafka_headers(self) -> list[tuple[str, bytes]]:
        """
        Extract key metadata as Kafka headers.

        Returns:
            List of (key, value) header tuples
        """
        headers = [
            ("envelope_id", str(self.envelope_id).encode()),
            ("event_type", self.payload_type.encode()),
            ("trace_id", str(self.metadata.trace_id).encode()),
            ("source_system", self.metadata.source_system.encode()),
            ("timestamp", self.metadata.timestamp.isoformat().encode()),
            ("priority", self.metadata.priority.value.encode()),
        ]

        if self.metadata.correlation_id:
            headers.append(("correlation_id", str(self.metadata.correlation_id).encode()))

        if self.metadata.causation_id:
            headers.append(("causation_id", str(self.metadata.causation_id).encode()))

        return headers

    @classmethod
    def from_kafka_message(
        cls,
        value: bytes,
        headers: Optional[list[tuple[str, bytes]]] = None,
        payload_class: Optional[Type[T]] = None,
    ) -> "EventEnvelope[T]":
        """
        Create envelope from Kafka message.

        Args:
            value: Message value bytes
            headers: Optional Kafka headers
            payload_class: Optional payload class

        Returns:
            EventEnvelope instance
        """
        # Check for compression indicator in headers
        decompress = False
        if headers:
            for key, val in headers:
                if key == "compression" and val.decode() == "gzip":
                    decompress = True
                    break

        return cls.deserialize(value, payload_class, decompress)

    def with_updated_metadata(self, **kwargs) -> "EventEnvelope[T]":
        """
        Create new envelope with updated metadata.

        Args:
            **kwargs: Metadata fields to update

        Returns:
            New EventEnvelope with updated metadata
        """
        updated_metadata = self.metadata.model_copy(update=kwargs)
        return self.model_copy(update={"metadata": updated_metadata})

    def child_envelope(
        self,
        payload: T,
        source_component: str,
    ) -> "EventEnvelope[T]":
        """
        Create a child envelope maintaining trace lineage.

        Args:
            payload: New payload
            source_component: New source component

        Returns:
            Child EventEnvelope
        """
        return EventEnvelope.wrap(
            payload=payload,
            source_system=self.metadata.source_system,
            source_component=source_component,
            trace_id=self.metadata.trace_id,
            priority=self.metadata.priority,
            topic=self.topic,
            tags={
                **self.metadata.tags,
                "parent_envelope_id": str(self.envelope_id),
            },
        )


# =============================================================================
# Envelope Builder
# =============================================================================

class EnvelopeBuilder(Generic[T]):
    """
    Builder pattern for constructing EventEnvelopes.

    Example:
        envelope = (
            EnvelopeBuilder(message)
            .from_source("opc-ua-connector", "tag-reader")
            .with_trace(existing_trace_id)
            .with_priority(EventPriority.HIGH)
            .for_topic("boiler.gl016.raw")
            .with_tags({"boiler_id": "B001"})
            .build()
        )
    """

    def __init__(self, payload: T):
        self._payload = payload
        self._source_system: str = "unknown"
        self._source_component: str = "unknown"
        self._trace_id: Optional[UUID] = None
        self._correlation_id: Optional[UUID] = None
        self._causation_id: Optional[UUID] = None
        self._priority: EventPriority = EventPriority.NORMAL
        self._topic: Optional[str] = None
        self._partition_key: Optional[str] = None
        self._compression: CompressionType = CompressionType.NONE
        self._tags: Dict[str, str] = {}
        self._context: Dict[str, Any] = {}

    def from_source(self, system: str, component: str = "unknown") -> "EnvelopeBuilder[T]":
        """Set source system and component."""
        self._source_system = system
        self._source_component = component
        return self

    def with_trace(self, trace_id: UUID) -> "EnvelopeBuilder[T]":
        """Set trace ID."""
        self._trace_id = trace_id
        return self

    def with_correlation(self, correlation_id: UUID) -> "EnvelopeBuilder[T]":
        """Set correlation ID."""
        self._correlation_id = correlation_id
        return self

    def with_causation(self, causation_id: UUID) -> "EnvelopeBuilder[T]":
        """Set causation ID."""
        self._causation_id = causation_id
        return self

    def with_priority(self, priority: EventPriority) -> "EnvelopeBuilder[T]":
        """Set priority."""
        self._priority = priority
        return self

    def for_topic(self, topic: str) -> "EnvelopeBuilder[T]":
        """Set target topic."""
        self._topic = topic
        return self

    def with_partition_key(self, key: str) -> "EnvelopeBuilder[T]":
        """Set partition key."""
        self._partition_key = key
        return self

    def with_compression(self, compression: CompressionType) -> "EnvelopeBuilder[T]":
        """Set compression type."""
        self._compression = compression
        return self

    def with_tags(self, tags: Dict[str, str]) -> "EnvelopeBuilder[T]":
        """Add tags."""
        self._tags.update(tags)
        return self

    def with_tag(self, key: str, value: str) -> "EnvelopeBuilder[T]":
        """Add single tag."""
        self._tags[key] = value
        return self

    def with_context(self, context: Dict[str, Any]) -> "EnvelopeBuilder[T]":
        """Add context data."""
        self._context.update(context)
        return self

    def build(self) -> EventEnvelope[T]:
        """Build the envelope."""
        envelope = EventEnvelope.wrap(
            payload=self._payload,
            source_system=self._source_system,
            source_component=self._source_component,
            trace_id=self._trace_id,
            priority=self._priority,
            topic=self._topic,
            partition_key=self._partition_key,
            compression=self._compression,
            tags=self._tags,
        )

        # Add additional metadata
        if self._correlation_id:
            envelope = envelope.with_updated_metadata(correlation_id=self._correlation_id)

        if self._causation_id:
            envelope = envelope.with_updated_metadata(causation_id=self._causation_id)

        if self._context:
            envelope = envelope.with_updated_metadata(context=self._context)

        return envelope
