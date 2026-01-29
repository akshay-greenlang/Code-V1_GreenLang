"""
Event serialization for GL-FOUND-X-003 audit events.

This module provides serialization capabilities for audit events,
supporting multiple output formats for different use cases:
- JSON serialization for API responses and general storage
- Kafka message format for event streaming (FR-080)
- Parquet schema for cold storage and analytics

Key Features:
    - Deterministic JSON serialization for hash reproducibility
    - Kafka message format with headers and metadata
    - Parquet schema generation for columnar storage
    - Batch serialization for high-throughput scenarios

Example:
    >>> from gl_normalizer_core.audit.serializer import AuditEventSerializer
    >>> serializer = AuditEventSerializer()
    >>> json_str = serializer.to_json(event)
    >>> kafka_msg = serializer.to_kafka_message(event)
    >>> parquet_schema = serializer.get_parquet_schema()
"""

import gzip
import json
import logging
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

from .schema import (
    AuditError,
    AuditWarning,
    EntityAudit,
    EventStatus,
    MeasurementAudit,
    NormalizationEvent,
)

logger = logging.getLogger(__name__)


class KafkaMessage:
    """
    Kafka message representation for audit events.

    Encapsulates the key, value, headers, and metadata for a Kafka message.

    Attributes:
        key: Message key (typically org_id + event_id).
        value: Message value (serialized event).
        headers: List of (key, value) header tuples.
        topic: Target Kafka topic.
        partition: Optional target partition.
        timestamp: Message timestamp in milliseconds.

    Example:
        >>> msg = KafkaMessage(
        ...     key=b"org-acme:norm-evt-001",
        ...     value=b'{"event_id": "norm-evt-001", ...}',
        ...     headers=[("content-type", b"application/json")],
        ...     topic="gl-normalizer-audit"
        ... )
    """

    def __init__(
        self,
        key: bytes,
        value: bytes,
        headers: List[Tuple[str, bytes]],
        topic: str,
        partition: Optional[int] = None,
        timestamp: Optional[int] = None,
    ):
        self.key = key
        self.value = value
        self.headers = headers
        self.topic = topic
        self.partition = partition
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return (
            f"KafkaMessage(key={self.key!r}, topic={self.topic!r}, "
            f"value_size={len(self.value)}, headers={len(self.headers)})"
        )


class ParquetColumn:
    """
    Definition of a Parquet column for audit events.

    Attributes:
        name: Column name.
        type: Parquet data type.
        nullable: Whether the column is nullable.
        nested: Whether this is a nested/struct column.
        description: Human-readable description.
    """

    def __init__(
        self,
        name: str,
        type: str,
        nullable: bool = True,
        nested: bool = False,
        description: str = "",
    ):
        self.name = name
        self.type = type
        self.nullable = nullable
        self.nested = nested
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type,
            "nullable": self.nullable,
            "nested": self.nested,
            "description": self.description,
        }


class ParquetSchema:
    """
    Parquet schema definition for audit events.

    Provides a schema definition that can be used with PyArrow or other
    Parquet libraries to create strongly-typed columnar storage.

    Attributes:
        columns: List of column definitions.
        compression: Compression codec (snappy, gzip, zstd).
        row_group_size: Target row group size.

    Example:
        >>> schema = ParquetSchema()
        >>> columns = schema.get_columns()
        >>> pyarrow_schema = schema.to_pyarrow()  # If PyArrow available
    """

    # Default compression and settings
    DEFAULT_COMPRESSION = "snappy"
    DEFAULT_ROW_GROUP_SIZE = 100_000

    def __init__(
        self,
        compression: str = DEFAULT_COMPRESSION,
        row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
    ):
        self.compression = compression
        self.row_group_size = row_group_size
        self.columns = self._build_schema()

    def _build_schema(self) -> List[ParquetColumn]:
        """Build the Parquet schema for NormalizationEvent."""
        return [
            # Event identification
            ParquetColumn(
                "event_id",
                "STRING",
                nullable=False,
                description="Unique identifier for this audit event",
            ),
            ParquetColumn(
                "event_ts",
                "TIMESTAMP",
                nullable=False,
                description="Timestamp when the event was created (UTC)",
            ),
            ParquetColumn(
                "prev_event_hash",
                "STRING",
                nullable=True,
                description="SHA-256 hash of the previous event in the chain",
            ),

            # Request context
            ParquetColumn(
                "request_id",
                "STRING",
                nullable=False,
                description="Correlation ID for the normalization request",
            ),
            ParquetColumn(
                "source_record_id",
                "STRING",
                nullable=False,
                description="ID of the source record being normalized",
            ),
            ParquetColumn(
                "org_id",
                "STRING",
                nullable=False,
                description="Organization ID for multi-tenant environments",
            ),
            ParquetColumn(
                "policy_mode",
                "STRING",
                nullable=False,
                description="Policy mode used: STRICT or LENIENT",
            ),
            ParquetColumn(
                "status",
                "STRING",
                nullable=False,
                description="Final status: success, warning, or failed",
            ),

            # Version tracking
            ParquetColumn(
                "vocab_version",
                "STRING",
                nullable=False,
                description="Version of the controlled vocabulary used",
            ),
            ParquetColumn(
                "policy_version",
                "STRING",
                nullable=False,
                description="Version of the policy configuration used",
            ),
            ParquetColumn(
                "unit_registry_version",
                "STRING",
                nullable=False,
                description="Version of the unit registry used",
            ),
            ParquetColumn(
                "validator_version",
                "STRING",
                nullable=False,
                description="Version of the validator used",
            ),
            ParquetColumn(
                "api_revision",
                "STRING",
                nullable=False,
                description="API revision of the normalizer service",
            ),

            # Payloads (stored as JSON strings for flexibility)
            ParquetColumn(
                "measurements_json",
                "STRING",
                nullable=True,
                description="JSON array of measurement audit records",
            ),
            ParquetColumn(
                "entities_json",
                "STRING",
                nullable=True,
                description="JSON array of entity audit records",
            ),
            ParquetColumn(
                "errors_json",
                "STRING",
                nullable=True,
                description="JSON array of error records",
            ),
            ParquetColumn(
                "warnings_json",
                "STRING",
                nullable=True,
                description="JSON array of warning records",
            ),

            # Derived metrics for analytics
            ParquetColumn(
                "measurement_count",
                "INT32",
                nullable=False,
                description="Number of measurements in this event",
            ),
            ParquetColumn(
                "entity_count",
                "INT32",
                nullable=False,
                description="Number of entities in this event",
            ),
            ParquetColumn(
                "error_count",
                "INT32",
                nullable=False,
                description="Number of errors in this event",
            ),
            ParquetColumn(
                "warning_count",
                "INT32",
                nullable=False,
                description="Number of warnings in this event",
            ),
            ParquetColumn(
                "needs_review_count",
                "INT32",
                nullable=False,
                description="Number of entities flagged for human review",
            ),

            # Integrity hashes
            ParquetColumn(
                "payload_hash",
                "STRING",
                nullable=False,
                description="SHA-256 hash of the payload",
            ),
            ParquetColumn(
                "event_hash",
                "STRING",
                nullable=False,
                description="SHA-256 hash of the complete event",
            ),

            # Partition columns
            ParquetColumn(
                "event_date",
                "DATE",
                nullable=False,
                description="Date partition key (derived from event_ts)",
            ),
            ParquetColumn(
                "event_hour",
                "INT32",
                nullable=False,
                description="Hour partition key (derived from event_ts)",
            ),
        ]

    def get_columns(self) -> List[ParquetColumn]:
        """Get list of column definitions."""
        return self.columns

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary representation."""
        return {
            "columns": [c.to_dict() for c in self.columns],
            "compression": self.compression,
            "row_group_size": self.row_group_size,
        }

    def to_pyarrow(self) -> Any:
        """
        Convert to PyArrow schema (requires pyarrow).

        Returns:
            pyarrow.Schema if pyarrow is available.

        Raises:
            ImportError: If pyarrow is not installed.
        """
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError(
                "PyArrow is required for Parquet schema conversion. "
                "Install with: pip install pyarrow"
            )

        type_mapping = {
            "STRING": pa.string(),
            "TIMESTAMP": pa.timestamp("us", tz="UTC"),
            "DATE": pa.date32(),
            "INT32": pa.int32(),
            "INT64": pa.int64(),
            "FLOAT": pa.float32(),
            "DOUBLE": pa.float64(),
            "BOOLEAN": pa.bool_(),
        }

        fields = []
        for col in self.columns:
            pa_type = type_mapping.get(col.type, pa.string())
            fields.append(pa.field(col.name, pa_type, nullable=col.nullable))

        return pa.schema(fields)


class AuditEventSerializer:
    """
    Serializer for audit events.

    Provides methods for converting NormalizationEvent objects to various
    output formats including JSON, Kafka messages, and Parquet rows.

    Thread Safety:
        This class is thread-safe for concurrent serialization operations.

    Attributes:
        _kafka_topic: Default Kafka topic for audit events.
        _parquet_schema: Cached Parquet schema.

    Example:
        >>> serializer = AuditEventSerializer()
        >>> json_str = serializer.to_json(event)
        >>> kafka_msg = serializer.to_kafka_message(event)
        >>> parquet_row = serializer.to_parquet_row(event)
    """

    # Default Kafka topic
    DEFAULT_KAFKA_TOPIC = "gl-normalizer-audit"

    # JSON encoding settings
    JSON_ENSURE_ASCII = False
    JSON_INDENT = None  # Compact by default

    def __init__(
        self,
        kafka_topic: Optional[str] = None,
        json_indent: Optional[int] = None,
    ):
        """
        Initialize the serializer.

        Args:
            kafka_topic: Override default Kafka topic.
            json_indent: JSON indentation (None for compact).
        """
        self._kafka_topic = kafka_topic or self.DEFAULT_KAFKA_TOPIC
        self._json_indent = json_indent
        self._parquet_schema: Optional[ParquetSchema] = None

    def to_json(
        self,
        event: NormalizationEvent,
        *,
        pretty: bool = False,
        sort_keys: bool = True,
    ) -> str:
        """
        Serialize event to JSON string.

        Args:
            event: NormalizationEvent to serialize.
            pretty: Whether to use pretty formatting (indented).
            sort_keys: Whether to sort keys (for deterministic output).

        Returns:
            JSON string representation of the event.

        Example:
            >>> json_str = serializer.to_json(event)
            >>> json_pretty = serializer.to_json(event, pretty=True)
        """
        indent = 2 if pretty else self._json_indent
        data = event.model_dump(mode="json")

        json_str = json.dumps(
            data,
            indent=indent,
            ensure_ascii=self.JSON_ENSURE_ASCII,
            sort_keys=sort_keys,
            default=self._json_default,
        )

        logger.debug(
            "Serialized event %s to JSON (%d bytes)",
            event.event_id,
            len(json_str),
        )

        return json_str

    def to_json_bytes(
        self,
        event: NormalizationEvent,
        *,
        sort_keys: bool = True,
        compress: bool = False,
    ) -> bytes:
        """
        Serialize event to JSON bytes.

        Args:
            event: NormalizationEvent to serialize.
            sort_keys: Whether to sort keys.
            compress: Whether to gzip compress the output.

        Returns:
            JSON bytes (optionally compressed).

        Example:
            >>> json_bytes = serializer.to_json_bytes(event)
            >>> compressed = serializer.to_json_bytes(event, compress=True)
        """
        json_str = self.to_json(event, sort_keys=sort_keys)
        json_bytes = json_str.encode("utf-8")

        if compress:
            buffer = BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
                f.write(json_bytes)
            json_bytes = buffer.getvalue()
            logger.debug(
                "Compressed event %s: %d -> %d bytes",
                event.event_id,
                len(json_str),
                len(json_bytes),
            )

        return json_bytes

    def to_dict(self, event: NormalizationEvent) -> Dict[str, Any]:
        """
        Convert event to dictionary.

        Args:
            event: NormalizationEvent to convert.

        Returns:
            Dictionary representation of the event.

        Example:
            >>> data = serializer.to_dict(event)
        """
        return event.model_dump(mode="json")

    def to_kafka_message(
        self,
        event: NormalizationEvent,
        *,
        topic: Optional[str] = None,
        partition: Optional[int] = None,
        compress: bool = False,
    ) -> KafkaMessage:
        """
        Convert event to Kafka message format.

        Creates a Kafka message with:
        - Key: {org_id}:{event_id} for partitioning
        - Value: JSON-encoded event (optionally compressed)
        - Headers: Content-type, event metadata

        Args:
            event: NormalizationEvent to convert.
            topic: Override default topic.
            partition: Explicit partition (None for key-based).
            compress: Whether to gzip compress the value.

        Returns:
            KafkaMessage ready for producer.

        Example:
            >>> msg = serializer.to_kafka_message(event)
            >>> producer.send(msg.topic, msg.value, msg.key, msg.headers)
        """
        # Create message key
        key = f"{event.org_id}:{event.event_id}".encode("utf-8")

        # Serialize value
        value = self.to_json_bytes(event, compress=compress)

        # Build headers
        content_type = "application/gzip" if compress else "application/json"
        headers = [
            ("content-type", content_type.encode("utf-8")),
            ("event-type", b"normalization-event"),
            ("event-id", event.event_id.encode("utf-8")),
            ("org-id", event.org_id.encode("utf-8")),
            ("status", event.status.value.encode("utf-8")),
            ("vocab-version", event.vocab_version.encode("utf-8")),
            ("event-hash", event.event_hash.encode("utf-8")),
        ]

        # Add prev_event_hash if present
        if event.prev_event_hash:
            headers.append(("prev-event-hash", event.prev_event_hash.encode("utf-8")))

        # Compute timestamp in milliseconds
        timestamp_ms = int(event.event_ts.timestamp() * 1000)

        msg = KafkaMessage(
            key=key,
            value=value,
            headers=headers,
            topic=topic or self._kafka_topic,
            partition=partition,
            timestamp=timestamp_ms,
        )

        logger.debug(
            "Created Kafka message for event %s: topic=%s, key=%s, value_size=%d",
            event.event_id,
            msg.topic,
            key.decode("utf-8"),
            len(value),
        )

        return msg

    def to_parquet_row(self, event: NormalizationEvent) -> Dict[str, Any]:
        """
        Convert event to Parquet row format.

        Creates a flattened row dictionary suitable for Parquet writing,
        with JSON-encoded nested structures and derived metrics.

        Args:
            event: NormalizationEvent to convert.

        Returns:
            Dictionary with flattened row data.

        Example:
            >>> row = serializer.to_parquet_row(event)
            >>> df = pd.DataFrame([row])
        """
        # Count entities needing review
        needs_review_count = sum(
            1 for e in event.entities if e.needs_review
        )

        # Create row dictionary
        row = {
            # Event identification
            "event_id": event.event_id,
            "event_ts": event.event_ts,
            "prev_event_hash": event.prev_event_hash,

            # Request context
            "request_id": event.request_id,
            "source_record_id": event.source_record_id,
            "org_id": event.org_id,
            "policy_mode": event.policy_mode,
            "status": event.status.value,

            # Version tracking
            "vocab_version": event.vocab_version,
            "policy_version": event.policy_version,
            "unit_registry_version": event.unit_registry_version,
            "validator_version": event.validator_version,
            "api_revision": event.api_revision,

            # Payloads as JSON strings
            "measurements_json": json.dumps(
                [m.model_dump(mode="json") for m in event.measurements],
                default=self._json_default,
            ),
            "entities_json": json.dumps(
                [e.model_dump(mode="json") for e in event.entities],
                default=self._json_default,
            ),
            "errors_json": json.dumps(
                self._serialize_errors_warnings(event.errors),
                default=self._json_default,
            ),
            "warnings_json": json.dumps(
                self._serialize_errors_warnings(event.warnings),
                default=self._json_default,
            ),

            # Derived metrics
            "measurement_count": len(event.measurements),
            "entity_count": len(event.entities),
            "error_count": len(event.errors),
            "warning_count": len(event.warnings),
            "needs_review_count": needs_review_count,

            # Integrity hashes
            "payload_hash": event.payload_hash,
            "event_hash": event.event_hash,

            # Partition columns
            "event_date": event.event_ts.date(),
            "event_hour": event.event_ts.hour,
        }

        logger.debug(
            "Created Parquet row for event %s: %d columns",
            event.event_id,
            len(row),
        )

        return row

    def to_parquet_rows(
        self,
        events: List[NormalizationEvent],
    ) -> List[Dict[str, Any]]:
        """
        Convert multiple events to Parquet rows.

        Args:
            events: List of events to convert.

        Returns:
            List of row dictionaries.

        Example:
            >>> rows = serializer.to_parquet_rows(events)
            >>> df = pd.DataFrame(rows)
        """
        return [self.to_parquet_row(event) for event in events]

    def get_parquet_schema(self) -> ParquetSchema:
        """
        Get the Parquet schema for audit events.

        Returns:
            ParquetSchema object with column definitions.

        Example:
            >>> schema = serializer.get_parquet_schema()
            >>> columns = schema.get_columns()
        """
        if self._parquet_schema is None:
            self._parquet_schema = ParquetSchema()
        return self._parquet_schema

    def from_json(self, json_str: str) -> NormalizationEvent:
        """
        Deserialize event from JSON string.

        Args:
            json_str: JSON string to deserialize.

        Returns:
            NormalizationEvent object.

        Example:
            >>> event = serializer.from_json(json_str)
        """
        data = json.loads(json_str)
        return NormalizationEvent.model_validate(data)

    def from_json_bytes(
        self,
        json_bytes: bytes,
        *,
        compressed: bool = False,
    ) -> NormalizationEvent:
        """
        Deserialize event from JSON bytes.

        Args:
            json_bytes: JSON bytes to deserialize.
            compressed: Whether the bytes are gzip compressed.

        Returns:
            NormalizationEvent object.

        Example:
            >>> event = serializer.from_json_bytes(json_bytes)
        """
        if compressed:
            buffer = BytesIO(json_bytes)
            with gzip.GzipFile(fileobj=buffer, mode="rb") as f:
                json_bytes = f.read()

        json_str = json_bytes.decode("utf-8")
        return self.from_json(json_str)

    def from_kafka_message(self, msg: KafkaMessage) -> NormalizationEvent:
        """
        Deserialize event from Kafka message.

        Args:
            msg: KafkaMessage to deserialize.

        Returns:
            NormalizationEvent object.

        Example:
            >>> event = serializer.from_kafka_message(kafka_msg)
        """
        # Check if compressed
        compressed = False
        for key, value in msg.headers:
            if key == "content-type" and value == b"application/gzip":
                compressed = True
                break

        return self.from_json_bytes(msg.value, compressed=compressed)

    def _serialize_errors_warnings(
        self,
        items: List[Union[AuditError, AuditWarning, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Serialize errors or warnings to dictionaries."""
        result = []
        for item in items:
            if isinstance(item, (AuditError, AuditWarning)):
                result.append(item.model_dump(mode="json"))
            else:
                result.append(item)
        return result

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, datetime):
            return obj.isoformat() + "Z" if obj.tzinfo is None else obj.isoformat()
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# Default serializer instance
_default_serializer: Optional[AuditEventSerializer] = None


def get_default_serializer() -> AuditEventSerializer:
    """
    Get the default serializer singleton.

    Returns:
        Default AuditEventSerializer instance.

    Example:
        >>> serializer = get_default_serializer()
        >>> json_str = serializer.to_json(event)
    """
    global _default_serializer
    if _default_serializer is None:
        _default_serializer = AuditEventSerializer()
    return _default_serializer


def to_json(event: NormalizationEvent, **kwargs) -> str:
    """
    Convenience function to serialize event to JSON.

    Args:
        event: Event to serialize.
        **kwargs: Additional arguments passed to to_json().

    Returns:
        JSON string.

    Example:
        >>> from gl_normalizer_core.audit.serializer import to_json
        >>> json_str = to_json(event)
    """
    return get_default_serializer().to_json(event, **kwargs)


def to_kafka_message(event: NormalizationEvent, **kwargs) -> KafkaMessage:
    """
    Convenience function to convert event to Kafka message.

    Args:
        event: Event to convert.
        **kwargs: Additional arguments passed to to_kafka_message().

    Returns:
        KafkaMessage.

    Example:
        >>> from gl_normalizer_core.audit.serializer import to_kafka_message
        >>> msg = to_kafka_message(event)
    """
    return get_default_serializer().to_kafka_message(event, **kwargs)


def to_parquet_row(event: NormalizationEvent) -> Dict[str, Any]:
    """
    Convenience function to convert event to Parquet row.

    Args:
        event: Event to convert.

    Returns:
        Row dictionary.

    Example:
        >>> from gl_normalizer_core.audit.serializer import to_parquet_row
        >>> row = to_parquet_row(event)
    """
    return get_default_serializer().to_parquet_row(event)


def from_json(json_str: str) -> NormalizationEvent:
    """
    Convenience function to deserialize event from JSON.

    Args:
        json_str: JSON string to deserialize.

    Returns:
        NormalizationEvent.

    Example:
        >>> from gl_normalizer_core.audit.serializer import from_json
        >>> event = from_json(json_str)
    """
    return get_default_serializer().from_json(json_str)
