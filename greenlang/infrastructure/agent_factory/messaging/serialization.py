# -*- coding: utf-8 -*-
"""
Message Serialization - JSON and MessagePack codecs for message envelopes.

Provides a ``MessageSerializer`` that encodes and decodes ``MessageEnvelope``
instances to/from bytes using either JSON (default, human-readable) or
MessagePack (compact, binary).  MessagePack is optional -- if the ``msgpack``
package is not installed, the MessagePack methods fall back to JSON with a
logged warning.

Custom type handlers are included for ``uuid.UUID`` and ``datetime`` so that
these Python types survive a round-trip through both codecs.

Classes:
    - MessageSerializer: Static serialization facade.
    - SerializationFormat: Enum of supported formats.

Example:
    >>> from greenlang.infrastructure.agent_factory.messaging.protocol import (
    ...     MessageEnvelope, MessageType,
    ... )
    >>> envelope = MessageEnvelope.event("agent-a", "agent.health", {"status": "ok"})
    >>> data = MessageSerializer.serialize(envelope)
    >>> restored = MessageSerializer.deserialize(data)
    >>> assert restored.id == envelope.id

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Union

from greenlang.infrastructure.agent_factory.messaging.protocol import (
    MessageEnvelope,
)

logger = logging.getLogger(__name__)

# Optional msgpack import with graceful fallback
try:
    import msgpack

    MSGPACK_AVAILABLE = True
except ImportError:
    msgpack = None  # type: ignore[assignment]
    MSGPACK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Format Enum
# ---------------------------------------------------------------------------


class SerializationFormat(str, Enum):
    """Supported serialization formats.

    Attributes:
        JSON: Human-readable JSON encoding (default).
        MSGPACK: Compact binary MessagePack encoding.
    """

    JSON = "json"
    MSGPACK = "msgpack"


# ---------------------------------------------------------------------------
# Custom JSON Encoder
# ---------------------------------------------------------------------------


class _EnvelopeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles UUID and datetime types.

    Used internally by ``MessageSerializer.serialize_json`` to produce
    clean JSON output without requiring callers to pre-convert types.
    """

    def default(self, obj: Any) -> Any:
        """Encode non-standard types.

        Args:
            obj: Object to encode.

        Returns:
            JSON-safe primitive.
        """
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# ---------------------------------------------------------------------------
# MessagePack Custom Hooks
# ---------------------------------------------------------------------------


def _msgpack_default(obj: Any) -> Any:
    """Custom packer for msgpack: convert UUID and datetime to primitives.

    Args:
        obj: Python object that msgpack cannot natively serialize.

    Returns:
        A dict with ``__type__`` marker for deserialization.

    Raises:
        TypeError: If the object type is not handled.
    """
    if isinstance(obj, uuid.UUID):
        return {"__type__": "uuid", "value": str(obj)}
    if isinstance(obj, datetime):
        return {"__type__": "datetime", "value": obj.isoformat()}
    raise TypeError(f"Object of type {type(obj).__name__} is not msgpack-serializable")


def _msgpack_object_hook(obj: Any) -> Any:
    """Custom unpacker for msgpack: restore UUID and datetime from markers.

    Args:
        obj: Deserialized value from msgpack.

    Returns:
        Restored Python object if a type marker is found, otherwise the
        original value.
    """
    if isinstance(obj, dict):
        type_marker = obj.get("__type__")
        if type_marker == "uuid":
            return uuid.UUID(obj["value"])
        if type_marker == "datetime":
            return datetime.fromisoformat(obj["value"])
    return obj


def _msgpack_ext_hook(code: int, data: bytes) -> Any:
    """Handle unknown ext types gracefully.

    Args:
        code: MessagePack ext type code.
        data: Raw ext data.

    Returns:
        Raw data as-is.
    """
    return data


# ---------------------------------------------------------------------------
# MessageSerializer
# ---------------------------------------------------------------------------


class MessageSerializer:
    """Facade for message envelope serialization.

    All methods are static -- no instance state is needed.  The class
    provides format-agnostic ``serialize`` / ``deserialize`` entry points
    as well as format-specific helpers.

    JSON is always available.  MessagePack requires the ``msgpack`` package;
    if absent, MessagePack methods fall back to JSON and emit a warning.

    Attributes:
        HEADER_SIZE: Number of bytes reserved for the format header.
    """

    HEADER_SIZE: int = 1

    # Format header bytes (prepended to serialized output for auto-detection)
    _JSON_HEADER: bytes = b"\x00"
    _MSGPACK_HEADER: bytes = b"\x01"

    # ------------------------------------------------------------------
    # Format-agnostic API
    # ------------------------------------------------------------------

    @staticmethod
    def serialize(
        envelope: MessageEnvelope,
        fmt: SerializationFormat = SerializationFormat.JSON,
    ) -> bytes:
        """Serialize an envelope to bytes with format auto-detection header.

        A one-byte header is prepended so that ``deserialize`` can
        automatically detect the format without external hints.

        Args:
            envelope: The message envelope to serialize.
            fmt: Desired serialization format.

        Returns:
            Bytes with format header prefix.
        """
        if fmt == SerializationFormat.MSGPACK:
            payload = MessageSerializer.serialize_msgpack(envelope)
            return MessageSerializer._MSGPACK_HEADER + payload
        payload = MessageSerializer.serialize_json(envelope)
        return MessageSerializer._JSON_HEADER + payload

    @staticmethod
    def deserialize(data: bytes) -> MessageEnvelope:
        """Deserialize an envelope from bytes with format auto-detection.

        Reads the one-byte header to determine the format, then delegates
        to the appropriate format-specific deserializer.

        Args:
            data: Bytes previously produced by ``serialize``.

        Returns:
            Reconstructed MessageEnvelope.

        Raises:
            ValueError: If the format header is unrecognized.
        """
        if len(data) < MessageSerializer.HEADER_SIZE:
            raise ValueError("Data too short to contain format header")

        header = data[:MessageSerializer.HEADER_SIZE]
        payload = data[MessageSerializer.HEADER_SIZE:]

        if header == MessageSerializer._MSGPACK_HEADER:
            return MessageSerializer.deserialize_msgpack(payload)
        if header == MessageSerializer._JSON_HEADER:
            return MessageSerializer.deserialize_json(payload)

        raise ValueError(f"Unrecognized format header: {header!r}")

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_json(envelope: MessageEnvelope) -> bytes:
        """Serialize an envelope to JSON bytes.

        Args:
            envelope: The message envelope to serialize.

        Returns:
            UTF-8 encoded JSON bytes (no format header).
        """
        data = envelope.to_dict()
        return json.dumps(
            data,
            cls=_EnvelopeJSONEncoder,
            separators=(",", ":"),
            sort_keys=False,
        ).encode("utf-8")

    @staticmethod
    def deserialize_json(data: bytes) -> MessageEnvelope:
        """Deserialize an envelope from JSON bytes.

        Args:
            data: UTF-8 encoded JSON bytes (no format header).

        Returns:
            Reconstructed MessageEnvelope.

        Raises:
            json.JSONDecodeError: If data is not valid JSON.
            ValueError: If required fields are missing or malformed.
        """
        parsed = json.loads(data.decode("utf-8"))
        return MessageEnvelope.from_dict(parsed)

    # ------------------------------------------------------------------
    # MessagePack
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_msgpack(envelope: MessageEnvelope) -> bytes:
        """Serialize an envelope to MessagePack bytes.

        Falls back to JSON if the ``msgpack`` package is not installed.

        Args:
            envelope: The message envelope to serialize.

        Returns:
            MessagePack (or JSON fallback) encoded bytes (no format header).
        """
        if not MSGPACK_AVAILABLE:
            logger.warning(
                "msgpack not installed; falling back to JSON serialization. "
                "Install msgpack for binary encoding: pip install msgpack"
            )
            return MessageSerializer.serialize_json(envelope)

        data = envelope.to_dict()
        return msgpack.packb(data, default=_msgpack_default, use_bin_type=True)

    @staticmethod
    def deserialize_msgpack(data: bytes) -> MessageEnvelope:
        """Deserialize an envelope from MessagePack bytes.

        Falls back to JSON deserialization if ``msgpack`` is not installed.

        Args:
            data: MessagePack (or JSON fallback) encoded bytes (no format header).

        Returns:
            Reconstructed MessageEnvelope.

        Raises:
            ValueError: If data cannot be decoded.
        """
        if not MSGPACK_AVAILABLE:
            logger.warning(
                "msgpack not installed; falling back to JSON deserialization"
            )
            return MessageSerializer.deserialize_json(data)

        parsed = msgpack.unpackb(
            data,
            object_hook=_msgpack_object_hook,
            ext_hook=_msgpack_ext_hook,
            raw=False,
        )
        return MessageEnvelope.from_dict(parsed)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def detect_format(data: bytes) -> SerializationFormat:
        """Detect the serialization format from the header byte.

        Args:
            data: Serialized bytes with format header.

        Returns:
            The detected SerializationFormat.

        Raises:
            ValueError: If the header is unrecognized.
        """
        if len(data) < MessageSerializer.HEADER_SIZE:
            raise ValueError("Data too short to contain format header")

        header = data[:MessageSerializer.HEADER_SIZE]
        if header == MessageSerializer._MSGPACK_HEADER:
            return SerializationFormat.MSGPACK
        if header == MessageSerializer._JSON_HEADER:
            return SerializationFormat.JSON
        raise ValueError(f"Unrecognized format header: {header!r}")

    @staticmethod
    def is_msgpack_available() -> bool:
        """Check whether the msgpack package is installed.

        Returns:
            True if msgpack is importable.
        """
        return MSGPACK_AVAILABLE

    @staticmethod
    def estimate_size(envelope: MessageEnvelope) -> Dict[str, int]:
        """Estimate the serialized size of an envelope in each format.

        Useful for deciding whether to use JSON or MessagePack based on
        message size constraints.

        Args:
            envelope: The message envelope to measure.

        Returns:
            Dictionary with ``json_bytes`` and ``msgpack_bytes`` keys.
        """
        json_bytes = len(MessageSerializer.serialize_json(envelope))
        msgpack_bytes = len(MessageSerializer.serialize_msgpack(envelope))
        return {
            "json_bytes": json_bytes,
            "msgpack_bytes": msgpack_bytes,
        }


__all__ = [
    "MessageSerializer",
    "SerializationFormat",
]
