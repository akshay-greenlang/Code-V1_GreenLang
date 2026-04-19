"""
Serializers and Deserializers for Kafka Events.

This module provides serialization and deserialization for agent events,
supporting both Avro (with Schema Registry) and JSON formats.

Features:
- Avro serialization with Schema Registry integration
- JSON serialization with validation
- Automatic schema evolution handling
- Compression support
- Error handling with fallback strategies

Serialization Flow:
1. Event -> Pydantic model validation
2. Model -> Dictionary conversion
3. Dictionary -> Avro/JSON bytes
4. Optional compression
5. Add headers (schema ID, content type)

Usage:
    from connectors.kafka.serializers import (
        AgentEventSerializer,
        AgentEventDeserializer,
        SerializationFormat,
    )

    # Create serializer
    serializer = AgentEventSerializer(format=SerializationFormat.AVRO)

    # Serialize event
    key, value, headers = await serializer.serialize(event)

    # Deserialize event
    event = await deserializer.deserialize(value, headers)
"""

import gzip
import json
import logging
import struct
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel

from .events import (
    AgentEvent,
    AgentCalculationCompleted,
    AgentAlertRaised,
    AgentRecommendationGenerated,
    AgentHealthCheck,
    AgentConfigurationChanged,
    EventType,
)
from .schemas import AgentEventSchemas, get_avro_schema

logger = logging.getLogger(__name__)


# =============================================================================
# Serialization Format
# =============================================================================


class SerializationFormat(str, Enum):
    """Supported serialization formats."""

    JSON = "json"
    AVRO = "avro"
    AVRO_JSON = "avro_json"  # Avro schema with JSON encoding


class CompressionFormat(str, Enum):
    """Supported compression formats."""

    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"


# =============================================================================
# Serialization Configuration
# =============================================================================


class SerializerConfig(BaseModel):
    """Configuration for event serialization."""

    format: SerializationFormat = SerializationFormat.JSON
    compression: CompressionFormat = CompressionFormat.NONE
    schema_registry_url: Optional[str] = None
    schema_registry_auth: Optional[Tuple[str, str]] = None
    auto_register_schemas: bool = True
    validate_on_serialize: bool = True
    include_schema_id: bool = True
    subject_name_strategy: str = "topic_name"


# =============================================================================
# Schema Registry Client
# =============================================================================


class SchemaRegistryClient:
    """
    Client for Confluent Schema Registry.

    Handles schema registration, retrieval, and caching.
    """

    def __init__(
        self,
        url: str,
        auth: Optional[Tuple[str, str]] = None,
        ssl_cafile: Optional[str] = None,
    ):
        """
        Initialize Schema Registry client.

        Args:
            url: Schema Registry URL
            auth: Optional (username, password) tuple
            ssl_cafile: Path to CA certificate
        """
        self.url = url.rstrip("/")
        self.auth = auth
        self.ssl_cafile = ssl_cafile
        self._schema_cache: Dict[int, Dict[str, Any]] = {}
        self._id_cache: Dict[str, int] = {}

    async def register_schema(
        self,
        subject: str,
        schema: Dict[str, Any],
    ) -> int:
        """
        Register a schema with the registry.

        Args:
            subject: Schema subject name
            schema: Avro schema dictionary

        Returns:
            Schema ID
        """
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/subjects/{subject}/versions",
                    json={"schema": json.dumps(schema)},
                    headers={"Content-Type": "application/vnd.schemaregistry.v1+json"},
                    auth=self.auth,
                )
                response.raise_for_status()
                result = response.json()
                schema_id = result["id"]

                # Cache the schema
                self._schema_cache[schema_id] = schema
                self._id_cache[f"{subject}:{json.dumps(schema, sort_keys=True)}"] = schema_id

                logger.info(f"Registered schema for subject '{subject}' with ID {schema_id}")
                return schema_id

        except ImportError:
            logger.warning("httpx not installed, using mock schema ID")
            return 1
        except Exception as e:
            logger.error(f"Failed to register schema: {e}")
            raise

    async def get_schema(self, schema_id: int) -> Dict[str, Any]:
        """
        Get schema by ID.

        Args:
            schema_id: Schema ID

        Returns:
            Avro schema dictionary
        """
        # Check cache first
        if schema_id in self._schema_cache:
            return self._schema_cache[schema_id]

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.url}/schemas/ids/{schema_id}",
                    auth=self.auth,
                )
                response.raise_for_status()
                result = response.json()
                schema = json.loads(result["schema"])

                # Cache the schema
                self._schema_cache[schema_id] = schema
                return schema

        except ImportError:
            logger.warning("httpx not installed, returning base schema")
            return AgentEventSchemas.AGENT_EVENT_AVRO
        except Exception as e:
            logger.error(f"Failed to get schema {schema_id}: {e}")
            raise

    async def get_latest_schema(self, subject: str) -> Tuple[int, Dict[str, Any]]:
        """
        Get latest schema for a subject.

        Args:
            subject: Schema subject name

        Returns:
            Tuple of (schema_id, schema)
        """
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.url}/subjects/{subject}/versions/latest",
                    auth=self.auth,
                )
                response.raise_for_status()
                result = response.json()
                schema_id = result["id"]
                schema = json.loads(result["schema"])

                # Cache the schema
                self._schema_cache[schema_id] = schema
                return schema_id, schema

        except ImportError:
            logger.warning("httpx not installed, using default schema")
            return 1, AgentEventSchemas.AGENT_EVENT_AVRO
        except Exception as e:
            logger.error(f"Failed to get latest schema for {subject}: {e}")
            raise


# =============================================================================
# Base Serializer
# =============================================================================


class BaseSerializer(ABC):
    """Abstract base class for event serializers."""

    @abstractmethod
    async def serialize(
        self,
        event: AgentEvent,
    ) -> Tuple[bytes, bytes, List[Tuple[str, bytes]]]:
        """
        Serialize an event.

        Args:
            event: Event to serialize

        Returns:
            Tuple of (key_bytes, value_bytes, headers)
        """
        pass


class BaseDeserializer(ABC):
    """Abstract base class for event deserializers."""

    @abstractmethod
    async def deserialize(
        self,
        data: bytes,
        headers: Optional[List[Tuple[str, bytes]]] = None,
    ) -> AgentEvent:
        """
        Deserialize bytes to an event.

        Args:
            data: Serialized event bytes
            headers: Optional Kafka headers

        Returns:
            Deserialized event
        """
        pass


# =============================================================================
# JSON Serializer
# =============================================================================


class JSONEventSerializer(BaseSerializer):
    """JSON serializer for agent events."""

    def __init__(
        self,
        compression: CompressionFormat = CompressionFormat.NONE,
        validate: bool = True,
    ):
        """
        Initialize JSON serializer.

        Args:
            compression: Compression format
            validate: Whether to validate before serializing
        """
        self.compression = compression
        self.validate = validate

    async def serialize(
        self,
        event: AgentEvent,
    ) -> Tuple[bytes, bytes, List[Tuple[str, bytes]]]:
        """Serialize event to JSON bytes."""
        # Get partition key
        key = event.get_partition_key()
        key_bytes = key.encode("utf-8")

        # Convert to dictionary
        event_dict = event.model_dump(mode="json")

        # Serialize to JSON
        json_str = json.dumps(event_dict, default=self._json_serializer)
        value_bytes = json_str.encode("utf-8")

        # Apply compression
        if self.compression == CompressionFormat.GZIP:
            value_bytes = gzip.compress(value_bytes)

        # Build headers
        headers = event.to_kafka_headers()
        headers.append(("content-type", b"application/json"))
        if self.compression != CompressionFormat.NONE:
            headers.append(("content-encoding", self.compression.value.encode("utf-8")))

        return key_bytes, value_bytes, headers

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class JSONEventDeserializer(BaseDeserializer):
    """JSON deserializer for agent events."""

    def __init__(self, compression: CompressionFormat = CompressionFormat.NONE):
        """
        Initialize JSON deserializer.

        Args:
            compression: Expected compression format
        """
        self.compression = compression
        self._event_type_map = self._build_event_type_map()

    def _build_event_type_map(self) -> Dict[str, Type[AgentEvent]]:
        """Build mapping from event type to event class."""
        return {
            EventType.CALCULATION_COMPLETED.value: AgentCalculationCompleted,
            EventType.ALERT_RAISED.value: AgentAlertRaised,
            EventType.RECOMMENDATION_GENERATED.value: AgentRecommendationGenerated,
            EventType.HEALTH_CHECK.value: AgentHealthCheck,
            EventType.CONFIG_CHANGED.value: AgentConfigurationChanged,
        }

    async def deserialize(
        self,
        data: bytes,
        headers: Optional[List[Tuple[str, bytes]]] = None,
    ) -> AgentEvent:
        """Deserialize JSON bytes to event."""
        # Check for compression
        compression = self._get_compression_from_headers(headers)
        if compression == CompressionFormat.GZIP:
            data = gzip.decompress(data)

        # Parse JSON
        json_str = data.decode("utf-8")
        event_dict = json.loads(json_str)

        # Determine event type and instantiate
        event_type = event_dict.get("event_type", "")
        event_class = self._event_type_map.get(event_type, AgentEvent)

        return event_class.model_validate(event_dict)

    def _get_compression_from_headers(
        self,
        headers: Optional[List[Tuple[str, bytes]]],
    ) -> CompressionFormat:
        """Extract compression format from headers."""
        if not headers:
            return CompressionFormat.NONE

        for key, value in headers:
            if key == "content-encoding":
                encoding = value.decode("utf-8")
                try:
                    return CompressionFormat(encoding)
                except ValueError:
                    pass

        return CompressionFormat.NONE


# =============================================================================
# Avro Serializer
# =============================================================================


class AvroEventSerializer(BaseSerializer):
    """Avro serializer for agent events with Schema Registry support."""

    # Magic byte for Confluent wire format
    MAGIC_BYTE = 0

    def __init__(
        self,
        schema_registry: Optional[SchemaRegistryClient] = None,
        auto_register: bool = True,
        compression: CompressionFormat = CompressionFormat.NONE,
    ):
        """
        Initialize Avro serializer.

        Args:
            schema_registry: Schema Registry client
            auto_register: Auto-register schemas
            compression: Compression format
        """
        self.schema_registry = schema_registry
        self.auto_register = auto_register
        self.compression = compression
        self._schema_id_cache: Dict[str, int] = {}

    async def serialize(
        self,
        event: AgentEvent,
    ) -> Tuple[bytes, bytes, List[Tuple[str, bytes]]]:
        """Serialize event to Avro bytes."""
        # Get partition key
        key = event.get_partition_key()
        key_bytes = key.encode("utf-8")

        # Get schema for event type
        schema = get_avro_schema(event.event_type)

        # Get or register schema ID
        schema_id = await self._get_or_register_schema(event.event_type, schema)

        # Convert event to dictionary
        event_dict = event.model_dump(mode="json")

        # Serialize to Avro bytes
        try:
            import fastavro
            from io import BytesIO

            # Write Confluent wire format
            buffer = BytesIO()
            buffer.write(struct.pack(">bI", self.MAGIC_BYTE, schema_id))

            # Write Avro data
            fastavro.schemaless_writer(buffer, schema, event_dict)
            value_bytes = buffer.getvalue()

        except ImportError:
            # Fallback to JSON if fastavro not available
            logger.warning("fastavro not installed, falling back to JSON")
            json_str = json.dumps(event_dict, default=str)
            value_bytes = json_str.encode("utf-8")

        # Apply compression
        if self.compression == CompressionFormat.GZIP:
            value_bytes = gzip.compress(value_bytes)

        # Build headers
        headers = event.to_kafka_headers()
        headers.append(("content-type", b"application/avro"))
        headers.append(("schema-id", str(schema_id).encode("utf-8")))
        if self.compression != CompressionFormat.NONE:
            headers.append(("content-encoding", self.compression.value.encode("utf-8")))

        return key_bytes, value_bytes, headers

    async def _get_or_register_schema(
        self,
        event_type: str,
        schema: Dict[str, Any],
    ) -> int:
        """Get schema ID from cache or register new schema."""
        cache_key = event_type

        if cache_key in self._schema_id_cache:
            return self._schema_id_cache[cache_key]

        if self.schema_registry and self.auto_register:
            subject = f"gl.agent.events-{event_type.replace('.', '_')}-value"
            schema_id = await self.schema_registry.register_schema(subject, schema)
            self._schema_id_cache[cache_key] = schema_id
            return schema_id

        # Return placeholder ID if no registry
        return 1


class AvroEventDeserializer(BaseDeserializer):
    """Avro deserializer for agent events with Schema Registry support."""

    MAGIC_BYTE = 0

    def __init__(
        self,
        schema_registry: Optional[SchemaRegistryClient] = None,
    ):
        """
        Initialize Avro deserializer.

        Args:
            schema_registry: Schema Registry client
        """
        self.schema_registry = schema_registry
        self._event_type_map = self._build_event_type_map()

    def _build_event_type_map(self) -> Dict[str, Type[AgentEvent]]:
        """Build mapping from event type to event class."""
        return {
            EventType.CALCULATION_COMPLETED.value: AgentCalculationCompleted,
            EventType.ALERT_RAISED.value: AgentAlertRaised,
            EventType.RECOMMENDATION_GENERATED.value: AgentRecommendationGenerated,
            EventType.HEALTH_CHECK.value: AgentHealthCheck,
            EventType.CONFIG_CHANGED.value: AgentConfigurationChanged,
        }

    async def deserialize(
        self,
        data: bytes,
        headers: Optional[List[Tuple[str, bytes]]] = None,
    ) -> AgentEvent:
        """Deserialize Avro bytes to event."""
        # Check for compression
        compression = self._get_compression_from_headers(headers)
        if compression == CompressionFormat.GZIP:
            data = gzip.decompress(data)

        try:
            import fastavro
            from io import BytesIO

            buffer = BytesIO(data)

            # Read Confluent wire format
            magic, schema_id = struct.unpack(">bI", buffer.read(5))

            if magic != self.MAGIC_BYTE:
                raise ValueError(f"Invalid magic byte: {magic}")

            # Get schema from registry
            if self.schema_registry:
                schema = await self.schema_registry.get_schema(schema_id)
            else:
                # Try to get from headers
                event_type = self._get_event_type_from_headers(headers)
                schema = get_avro_schema(event_type)

            # Read Avro data
            event_dict = fastavro.schemaless_reader(buffer, schema)

        except ImportError:
            # Fallback to JSON parsing
            logger.warning("fastavro not installed, falling back to JSON parsing")
            # Skip wire format header if present
            if data[:1] == struct.pack(">b", self.MAGIC_BYTE):
                data = data[5:]
            json_str = data.decode("utf-8")
            event_dict = json.loads(json_str)

        # Determine event type and instantiate
        event_type = event_dict.get("event_type", "")
        event_class = self._event_type_map.get(event_type, AgentEvent)

        return event_class.model_validate(event_dict)

    def _get_compression_from_headers(
        self,
        headers: Optional[List[Tuple[str, bytes]]],
    ) -> CompressionFormat:
        """Extract compression format from headers."""
        if not headers:
            return CompressionFormat.NONE

        for key, value in headers:
            if key == "content-encoding":
                encoding = value.decode("utf-8")
                try:
                    return CompressionFormat(encoding)
                except ValueError:
                    pass

        return CompressionFormat.NONE

    def _get_event_type_from_headers(
        self,
        headers: Optional[List[Tuple[str, bytes]]],
    ) -> str:
        """Extract event type from headers."""
        if not headers:
            return ""

        for key, value in headers:
            if key == "event_type":
                return value.decode("utf-8")

        return ""


# =============================================================================
# Unified Serializer
# =============================================================================


class AgentEventSerializer:
    """
    Unified serializer for agent events.

    Supports multiple formats and provides a consistent interface
    for serializing events to Kafka.

    Example:
        serializer = AgentEventSerializer(
            format=SerializationFormat.AVRO,
            schema_registry_url="http://localhost:8081",
        )

        key, value, headers = await serializer.serialize(event)
    """

    def __init__(
        self,
        format: SerializationFormat = SerializationFormat.JSON,
        compression: CompressionFormat = CompressionFormat.NONE,
        schema_registry_url: Optional[str] = None,
        schema_registry_auth: Optional[Tuple[str, str]] = None,
        auto_register_schemas: bool = True,
    ):
        """
        Initialize serializer.

        Args:
            format: Serialization format
            compression: Compression format
            schema_registry_url: Schema Registry URL (for Avro)
            schema_registry_auth: Schema Registry auth
            auto_register_schemas: Auto-register schemas
        """
        self.format = format
        self.compression = compression

        # Initialize schema registry if needed
        self.schema_registry = None
        if schema_registry_url and format == SerializationFormat.AVRO:
            self.schema_registry = SchemaRegistryClient(
                url=schema_registry_url,
                auth=schema_registry_auth,
            )

        # Create appropriate serializer
        if format == SerializationFormat.JSON:
            self._serializer = JSONEventSerializer(
                compression=compression,
            )
        elif format == SerializationFormat.AVRO:
            self._serializer = AvroEventSerializer(
                schema_registry=self.schema_registry,
                auto_register=auto_register_schemas,
                compression=compression,
            )
        else:
            self._serializer = JSONEventSerializer(compression=compression)

    async def serialize(
        self,
        event: AgentEvent,
    ) -> Tuple[bytes, bytes, List[Tuple[str, bytes]]]:
        """
        Serialize an event.

        Args:
            event: Event to serialize

        Returns:
            Tuple of (key_bytes, value_bytes, headers)
        """
        return await self._serializer.serialize(event)

    async def serialize_batch(
        self,
        events: List[AgentEvent],
    ) -> List[Tuple[bytes, bytes, List[Tuple[str, bytes]]]]:
        """
        Serialize multiple events.

        Args:
            events: Events to serialize

        Returns:
            List of (key_bytes, value_bytes, headers) tuples
        """
        return [await self._serializer.serialize(event) for event in events]


class AgentEventDeserializer:
    """
    Unified deserializer for agent events.

    Automatically detects format from headers and deserializes appropriately.

    Example:
        deserializer = AgentEventDeserializer(
            schema_registry_url="http://localhost:8081",
        )

        event = await deserializer.deserialize(value, headers)
    """

    def __init__(
        self,
        schema_registry_url: Optional[str] = None,
        schema_registry_auth: Optional[Tuple[str, str]] = None,
    ):
        """
        Initialize deserializer.

        Args:
            schema_registry_url: Schema Registry URL
            schema_registry_auth: Schema Registry auth
        """
        # Initialize schema registry if provided
        self.schema_registry = None
        if schema_registry_url:
            self.schema_registry = SchemaRegistryClient(
                url=schema_registry_url,
                auth=schema_registry_auth,
            )

        # Create deserializers
        self._json_deserializer = JSONEventDeserializer()
        self._avro_deserializer = AvroEventDeserializer(
            schema_registry=self.schema_registry,
        )

    async def deserialize(
        self,
        data: bytes,
        headers: Optional[List[Tuple[str, bytes]]] = None,
    ) -> AgentEvent:
        """
        Deserialize bytes to an event.

        Automatically detects format from headers or data.

        Args:
            data: Serialized event bytes
            headers: Optional Kafka headers

        Returns:
            Deserialized event
        """
        # Detect format from headers
        format = self._detect_format(data, headers)

        if format == SerializationFormat.AVRO:
            return await self._avro_deserializer.deserialize(data, headers)
        else:
            return await self._json_deserializer.deserialize(data, headers)

    def _detect_format(
        self,
        data: bytes,
        headers: Optional[List[Tuple[str, bytes]]],
    ) -> SerializationFormat:
        """Detect serialization format from headers or data."""
        # Check headers first
        if headers:
            for key, value in headers:
                if key == "content-type":
                    content_type = value.decode("utf-8")
                    if "avro" in content_type:
                        return SerializationFormat.AVRO
                    if "json" in content_type:
                        return SerializationFormat.JSON

        # Check for Avro magic byte
        if len(data) >= 5 and data[0] == 0:
            return SerializationFormat.AVRO

        # Check if it looks like JSON
        try:
            # Decompress if needed
            try:
                decompressed = gzip.decompress(data)
                data = decompressed
            except gzip.BadGzipFile:
                pass

            if data[0:1] in (b"{", b"["):
                return SerializationFormat.JSON
        except Exception:
            pass

        # Default to JSON
        return SerializationFormat.JSON


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enumerations
    "SerializationFormat",
    "CompressionFormat",
    # Configuration
    "SerializerConfig",
    # Schema Registry
    "SchemaRegistryClient",
    # Serializers
    "JSONEventSerializer",
    "AvroEventSerializer",
    "AgentEventSerializer",
    # Deserializers
    "JSONEventDeserializer",
    "AvroEventDeserializer",
    "AgentEventDeserializer",
]
