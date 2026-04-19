"""
Event Envelope Module - GL-001 ThermalCommand

This module provides the standard event envelope wrapper for all Kafka messages
in the ThermalCommand system. It ensures consistent metadata, correlation tracking,
schema versioning, and provenance across all event types.

The EventEnvelope follows CloudEvents specification principles with extensions
specific to industrial process control requirements.

Example:
    >>> from datetime import datetime
    >>> envelope = EventEnvelope.create(
    ...     event_type="gl001.telemetry.normalized",
    ...     source="opc-ua-collector-01",
    ...     payload={"sensor_id": "T-101", "value": 450.5},
    ...     correlation_id="corr-123"
    ... )
    >>> print(envelope.metadata.envelope_id)
    'env-abc123...'

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator


class SchemaCompatibility(str, Enum):
    """Schema compatibility modes for Kafka Schema Registry."""

    BACKWARD = "BACKWARD"  # New schema can read old data
    FORWARD = "FORWARD"  # Old schema can read new data
    FULL = "FULL"  # Both backward and forward compatible
    NONE = "NONE"  # No compatibility check


class SchemaVersion(BaseModel):
    """
    Schema version information for event payloads.

    Supports semantic versioning with compatibility tracking
    for schema evolution in the Kafka Schema Registry.

    Attributes:
        major: Major version (breaking changes)
        minor: Minor version (backward-compatible additions)
        patch: Patch version (backward-compatible fixes)
        schema_id: Registry schema ID (assigned by Schema Registry)
        compatibility: Schema compatibility mode
        fingerprint: SHA-256 fingerprint of the schema
    """

    major: int = Field(1, ge=0, description="Major version number")
    minor: int = Field(0, ge=0, description="Minor version number")
    patch: int = Field(0, ge=0, description="Patch version number")
    schema_id: Optional[int] = Field(
        None, ge=1, description="Schema Registry assigned ID"
    )
    compatibility: SchemaCompatibility = Field(
        SchemaCompatibility.BACKWARD,
        description="Schema compatibility mode",
    )
    fingerprint: Optional[str] = Field(
        None,
        min_length=64,
        max_length=64,
        description="SHA-256 fingerprint of schema",
    )

    @property
    def version_string(self) -> str:
        """Return semantic version string."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible_with(self, other: SchemaVersion) -> bool:
        """
        Check if this schema version is compatible with another.

        Args:
            other: The other schema version to compare against

        Returns:
            True if compatible based on compatibility mode
        """
        if self.compatibility == SchemaCompatibility.NONE:
            return self.major == other.major and self.minor == other.minor

        if self.compatibility == SchemaCompatibility.BACKWARD:
            # New can read old: same major, new minor >= old minor
            return self.major == other.major and self.minor >= other.minor

        if self.compatibility == SchemaCompatibility.FORWARD:
            # Old can read new: same major, old minor >= new minor
            return self.major == other.major and self.minor <= other.minor

        if self.compatibility == SchemaCompatibility.FULL:
            # Both directions: must be same major version
            return self.major == other.major

        return False

    def model_post_init(self, __context: Any) -> None:
        """Generate fingerprint if not provided."""
        if self.fingerprint is None:
            content = f"{self.major}.{self.minor}.{self.patch}"
            self.fingerprint = hashlib.sha256(content.encode()).hexdigest()


class EnvelopeMetadata(BaseModel):
    """
    Metadata for event envelope tracking and correlation.

    Provides comprehensive metadata for event tracing, debugging,
    and audit trail construction across distributed systems.

    Attributes:
        envelope_id: Unique identifier for this envelope instance
        correlation_id: ID linking related events across the system
        causation_id: ID of the event that caused this event
        timestamp: Event creation timestamp (UTC)
        source: Originating system/component identifier
        event_type: Fully qualified event type name
        schema_version: Schema version information
        trace_id: Distributed tracing ID (OpenTelemetry compatible)
        span_id: Distributed tracing span ID
        partition_key: Kafka partition key for ordering
        idempotency_key: Key for exactly-once processing
        priority: Event priority (1=highest, 10=lowest)
        ttl_seconds: Time-to-live before expiration
        retry_count: Number of processing retries
        tags: Additional metadata tags
    """

    envelope_id: str = Field(
        default_factory=lambda: f"env-{uuid.uuid4().hex}",
        description="Unique envelope identifier",
    )
    correlation_id: str = Field(
        default_factory=lambda: f"corr-{uuid.uuid4().hex}",
        description="Correlation ID for related events",
    )
    causation_id: Optional[str] = Field(
        None,
        description="ID of the causing event",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp in UTC",
    )
    source: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Event source identifier",
    )
    event_type: str = Field(
        ...,
        pattern=r"^gl001\.[a-z]+\.[a-z]+$",
        description="Fully qualified event type",
    )
    schema_version: SchemaVersion = Field(
        default_factory=SchemaVersion,
        description="Schema version information",
    )
    trace_id: Optional[str] = Field(
        None,
        min_length=32,
        max_length=32,
        description="OpenTelemetry trace ID",
    )
    span_id: Optional[str] = Field(
        None,
        min_length=16,
        max_length=16,
        description="OpenTelemetry span ID",
    )
    partition_key: Optional[str] = Field(
        None,
        max_length=256,
        description="Kafka partition key",
    )
    idempotency_key: Optional[str] = Field(
        None,
        max_length=256,
        description="Idempotency key for exactly-once semantics",
    )
    priority: int = Field(
        5,
        ge=1,
        le=10,
        description="Event priority (1=highest)",
    )
    ttl_seconds: Optional[int] = Field(
        None,
        ge=1,
        description="Time-to-live in seconds",
    )
    retry_count: int = Field(
        0,
        ge=0,
        description="Processing retry count",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata tags",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        """Ensure timestamp is UTC timezone-aware."""
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        return v

    def with_retry(self) -> EnvelopeMetadata:
        """Create a copy with incremented retry count."""
        return self.model_copy(update={"retry_count": self.retry_count + 1})

    def is_expired(self) -> bool:
        """Check if the event has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > self.ttl_seconds


PayloadT = TypeVar("PayloadT", bound=BaseModel)


class EventEnvelope(BaseModel, Generic[PayloadT]):
    """
    Standard event envelope wrapper for all Kafka messages.

    The EventEnvelope provides a consistent structure for all events
    in the ThermalCommand system, ensuring proper correlation tracking,
    schema versioning, and provenance.

    Type Parameters:
        PayloadT: The Pydantic model type for the event payload

    Attributes:
        metadata: Envelope metadata for tracking and correlation
        payload: The actual event data
        provenance_hash: SHA-256 hash for audit trail verification

    Example:
        >>> from gl001_streaming.kafka_schemas import TelemetryNormalizedEvent
        >>> envelope = EventEnvelope[TelemetryNormalizedEvent].create(
        ...     event_type="gl001.telemetry.normalized",
        ...     source="opc-ua-collector",
        ...     payload=telemetry_data
        ... )
    """

    metadata: EnvelopeMetadata = Field(
        ...,
        description="Event metadata",
    )
    payload: Any = Field(
        ...,
        description="Event payload data",
    )
    provenance_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of envelope for audit",
    )

    @model_validator(mode="before")
    @classmethod
    def compute_provenance_hash(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Compute provenance hash if not provided."""
        if "provenance_hash" not in values or values["provenance_hash"] is None:
            # Create deterministic hash of metadata and payload
            metadata = values.get("metadata", {})
            payload = values.get("payload", {})

            # Serialize for hashing (exclude provenance_hash itself)
            if isinstance(metadata, EnvelopeMetadata):
                metadata_dict = metadata.model_dump(exclude={"retry_count"})
            else:
                metadata_dict = {k: v for k, v in metadata.items() if k != "retry_count"}

            if hasattr(payload, "model_dump"):
                payload_dict = payload.model_dump()
            else:
                payload_dict = payload

            hash_content = json.dumps(
                {"metadata": metadata_dict, "payload": payload_dict},
                sort_keys=True,
                default=str,
            )
            values["provenance_hash"] = hashlib.sha256(hash_content.encode()).hexdigest()

        return values

    @classmethod
    def create(
        cls,
        event_type: str,
        source: str,
        payload: PayloadT,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        partition_key: Optional[str] = None,
        priority: int = 5,
        ttl_seconds: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> EventEnvelope[PayloadT]:
        """
        Factory method to create a new EventEnvelope.

        Args:
            event_type: Fully qualified event type (e.g., gl001.telemetry.normalized)
            source: Event source identifier
            payload: Event payload data
            correlation_id: Optional correlation ID (auto-generated if not provided)
            causation_id: Optional ID of the causing event
            partition_key: Optional Kafka partition key
            priority: Event priority (1=highest, 10=lowest)
            ttl_seconds: Optional time-to-live in seconds
            tags: Optional metadata tags
            trace_id: Optional OpenTelemetry trace ID
            span_id: Optional OpenTelemetry span ID

        Returns:
            New EventEnvelope instance with computed provenance hash
        """
        metadata = EnvelopeMetadata(
            correlation_id=correlation_id or f"corr-{uuid.uuid4().hex}",
            causation_id=causation_id,
            source=source,
            event_type=event_type,
            partition_key=partition_key,
            priority=priority,
            ttl_seconds=ttl_seconds,
            tags=tags or {},
            trace_id=trace_id,
            span_id=span_id,
        )

        return cls(
            metadata=metadata,
            payload=payload,
            provenance_hash="",  # Will be computed by validator
        )

    @classmethod
    def from_causation(
        cls,
        causing_envelope: EventEnvelope,
        event_type: str,
        source: str,
        payload: PayloadT,
        partition_key: Optional[str] = None,
    ) -> EventEnvelope[PayloadT]:
        """
        Create a new envelope causally linked to an existing envelope.

        Maintains correlation chain by inheriting correlation_id
        and setting causation_id to the causing envelope's ID.

        Args:
            causing_envelope: The envelope that caused this event
            event_type: New event type
            source: New event source
            payload: New event payload
            partition_key: Optional partition key (defaults to causing envelope's)

        Returns:
            New EventEnvelope linked to the causing envelope
        """
        return cls.create(
            event_type=event_type,
            source=source,
            payload=payload,
            correlation_id=causing_envelope.metadata.correlation_id,
            causation_id=causing_envelope.metadata.envelope_id,
            partition_key=partition_key or causing_envelope.metadata.partition_key,
            trace_id=causing_envelope.metadata.trace_id,
            span_id=causing_envelope.metadata.span_id,
            tags=causing_envelope.metadata.tags.copy(),
        )

    def to_kafka_message(self) -> Dict[str, Any]:
        """
        Convert envelope to Kafka message format.

        Returns:
            Dictionary suitable for Kafka producer with key, value, and headers
        """
        headers = [
            ("correlation_id", self.metadata.correlation_id.encode()),
            ("event_type", self.metadata.event_type.encode()),
            ("source", self.metadata.source.encode()),
            ("schema_version", self.metadata.schema_version.version_string.encode()),
            ("provenance_hash", self.provenance_hash.encode()),
        ]

        if self.metadata.trace_id:
            headers.append(("trace_id", self.metadata.trace_id.encode()))
        if self.metadata.span_id:
            headers.append(("span_id", self.metadata.span_id.encode()))
        if self.metadata.causation_id:
            headers.append(("causation_id", self.metadata.causation_id.encode()))

        return {
            "key": (self.metadata.partition_key or self.metadata.envelope_id).encode(),
            "value": self.model_dump_json().encode(),
            "headers": headers,
            "timestamp_ms": int(self.metadata.timestamp.timestamp() * 1000),
        }

    @classmethod
    def from_kafka_message(
        cls,
        value: bytes,
        headers: Optional[List[tuple]] = None,
    ) -> EventEnvelope:
        """
        Create envelope from Kafka message.

        Args:
            value: Raw message value (JSON bytes)
            headers: Optional Kafka headers

        Returns:
            Deserialized EventEnvelope
        """
        data = json.loads(value.decode())
        return cls.model_validate(data)

    def verify_provenance(self) -> bool:
        """
        Verify the provenance hash matches the envelope content.

        Returns:
            True if provenance hash is valid, False otherwise
        """
        # Recompute hash without retry_count
        metadata_dict = self.metadata.model_dump(exclude={"retry_count"})

        if hasattr(self.payload, "model_dump"):
            payload_dict = self.payload.model_dump()
        else:
            payload_dict = self.payload

        hash_content = json.dumps(
            {"metadata": metadata_dict, "payload": payload_dict},
            sort_keys=True,
            default=str,
        )
        expected_hash = hashlib.sha256(hash_content.encode()).hexdigest()

        return self.provenance_hash == expected_hash


class EnvelopeBatch(BaseModel):
    """
    Batch of event envelopes for bulk operations.

    Supports efficient batch processing with correlation tracking
    across all envelopes in the batch.

    Attributes:
        batch_id: Unique batch identifier
        correlation_id: Shared correlation ID for the batch
        envelopes: List of event envelopes
        created_at: Batch creation timestamp
        batch_hash: SHA-256 hash of all envelope hashes
    """

    batch_id: str = Field(
        default_factory=lambda: f"batch-{uuid.uuid4().hex}",
        description="Unique batch identifier",
    )
    correlation_id: str = Field(
        default_factory=lambda: f"corr-{uuid.uuid4().hex}",
        description="Batch correlation ID",
    )
    envelopes: List[EventEnvelope] = Field(
        default_factory=list,
        description="Envelopes in this batch",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Batch creation timestamp",
    )
    batch_hash: Optional[str] = Field(
        None,
        description="Combined hash of all envelopes",
    )

    def model_post_init(self, __context: Any) -> None:
        """Compute batch hash after initialization."""
        if self.batch_hash is None and self.envelopes:
            combined = "".join(e.provenance_hash for e in self.envelopes)
            self.batch_hash = hashlib.sha256(combined.encode()).hexdigest()

    def add_envelope(self, envelope: EventEnvelope) -> None:
        """Add an envelope to the batch and update batch hash."""
        self.envelopes.append(envelope)
        combined = "".join(e.provenance_hash for e in self.envelopes)
        self.batch_hash = hashlib.sha256(combined.encode()).hexdigest()

    @property
    def size(self) -> int:
        """Return number of envelopes in batch."""
        return len(self.envelopes)

    def verify_batch(self) -> bool:
        """Verify all envelopes in batch have valid provenance."""
        return all(e.verify_provenance() for e in self.envelopes)
