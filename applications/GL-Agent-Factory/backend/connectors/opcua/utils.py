"""
OPC-UA Utility Functions for GreenLang Process Heat Agents.

This module provides helper functions for OPC-UA operations including:
- Node ID parsing and formatting
- Data type conversions
- Quality code interpretation
- Address space navigation helpers
- Endpoint discovery utilities
- Data validation and normalization

Usage:
    from connectors.opcua.utils import (
        parse_node_id,
        format_node_id,
        convert_to_opcua_type,
        get_quality_description,
    )

    # Parse a node ID string
    ns, identifier = parse_node_id("ns=2;s=Temperature.PV")

    # Get human-readable quality description
    desc = get_quality_description(OPCUAQuality.BAD_SENSOR_FAILURE)
"""

import re
import struct
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from urllib.parse import urlparse
import uuid

from .types import (
    OPCUAQuality,
    QualityLevel,
    OPCUADataType,
    NodeClass,
    SecurityPolicy,
    MessageSecurityMode,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Node ID Utilities
# =============================================================================


class NodeIdType(str, Enum):
    """OPC-UA Node ID types."""

    NUMERIC = "i"  # Numeric identifier
    STRING = "s"  # String identifier
    GUID = "g"  # GUID identifier
    OPAQUE = "b"  # Opaque (byte string) identifier


def parse_node_id(node_id: str) -> Tuple[int, Union[int, str, uuid.UUID, bytes], NodeIdType]:
    """
    Parse an OPC-UA Node ID string.

    Args:
        node_id: Node ID string (e.g., "ns=2;s=Temperature.PV")

    Returns:
        Tuple of (namespace_index, identifier, identifier_type)

    Raises:
        ValueError: If node ID format is invalid
    """
    # Default namespace
    namespace = 0

    # Remove whitespace
    node_id = node_id.strip()

    # Parse namespace
    ns_match = re.match(r"ns=(\d+);", node_id)
    if ns_match:
        namespace = int(ns_match.group(1))
        node_id = node_id[ns_match.end():]

    # Parse identifier
    if node_id.startswith("i="):
        # Numeric identifier
        identifier = int(node_id[2:])
        id_type = NodeIdType.NUMERIC

    elif node_id.startswith("s="):
        # String identifier
        identifier = node_id[2:]
        id_type = NodeIdType.STRING

    elif node_id.startswith("g="):
        # GUID identifier
        identifier = uuid.UUID(node_id[2:])
        id_type = NodeIdType.GUID

    elif node_id.startswith("b="):
        # Opaque (byte string) identifier
        import base64
        identifier = base64.b64decode(node_id[2:])
        id_type = NodeIdType.OPAQUE

    else:
        # Try to parse as numeric (default)
        try:
            identifier = int(node_id)
            id_type = NodeIdType.NUMERIC
        except ValueError:
            # Treat as string identifier
            identifier = node_id
            id_type = NodeIdType.STRING

    return namespace, identifier, id_type


def format_node_id(
    namespace: int,
    identifier: Union[int, str, uuid.UUID, bytes],
    id_type: Optional[NodeIdType] = None,
) -> str:
    """
    Format a Node ID as a string.

    Args:
        namespace: Namespace index
        identifier: Node identifier
        id_type: Identifier type (auto-detected if not specified)

    Returns:
        Formatted Node ID string
    """
    # Auto-detect type if not specified
    if id_type is None:
        if isinstance(identifier, int):
            id_type = NodeIdType.NUMERIC
        elif isinstance(identifier, uuid.UUID):
            id_type = NodeIdType.GUID
        elif isinstance(identifier, bytes):
            id_type = NodeIdType.OPAQUE
        else:
            id_type = NodeIdType.STRING

    # Format identifier
    if id_type == NodeIdType.NUMERIC:
        id_str = f"i={identifier}"
    elif id_type == NodeIdType.STRING:
        id_str = f"s={identifier}"
    elif id_type == NodeIdType.GUID:
        id_str = f"g={identifier}"
    elif id_type == NodeIdType.OPAQUE:
        import base64
        id_str = f"b={base64.b64encode(identifier).decode()}"
    else:
        id_str = str(identifier)

    # Format full Node ID
    if namespace == 0:
        return id_str
    else:
        return f"ns={namespace};{id_str}"


def is_valid_node_id(node_id: str) -> bool:
    """
    Check if a Node ID string is valid.

    Args:
        node_id: Node ID string to validate

    Returns:
        True if valid
    """
    try:
        parse_node_id(node_id)
        return True
    except (ValueError, Exception):
        return False


def get_node_id_namespace(node_id: str) -> int:
    """
    Extract namespace index from a Node ID.

    Args:
        node_id: Node ID string

    Returns:
        Namespace index
    """
    namespace, _, _ = parse_node_id(node_id)
    return namespace


# =============================================================================
# Data Type Conversions
# =============================================================================


def convert_to_opcua_type(value: Any, target_type: OPCUADataType) -> Any:
    """
    Convert a Python value to the specified OPC-UA data type.

    Args:
        value: Value to convert
        target_type: Target OPC-UA data type

    Returns:
        Converted value

    Raises:
        ValueError: If conversion is not possible
    """
    if value is None:
        return None

    try:
        if target_type == OPCUADataType.BOOLEAN:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)

        elif target_type in (OPCUADataType.SBYTE, OPCUADataType.INT16,
                             OPCUADataType.INT32, OPCUADataType.INT64):
            return int(value)

        elif target_type in (OPCUADataType.BYTE, OPCUADataType.UINT16,
                             OPCUADataType.UINT32, OPCUADataType.UINT64):
            val = int(value)
            if val < 0:
                raise ValueError(f"Cannot convert negative value to unsigned type: {value}")
            return val

        elif target_type in (OPCUADataType.FLOAT, OPCUADataType.DOUBLE):
            return float(value)

        elif target_type == OPCUADataType.STRING:
            return str(value)

        elif target_type == OPCUADataType.DATETIME:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(value, tz=timezone.utc)
            raise ValueError(f"Cannot convert to DateTime: {type(value)}")

        elif target_type == OPCUADataType.GUID:
            if isinstance(value, uuid.UUID):
                return value
            return uuid.UUID(str(value))

        elif target_type == OPCUADataType.BYTESTRING:
            if isinstance(value, bytes):
                return value
            if isinstance(value, str):
                return value.encode("utf-8")
            raise ValueError(f"Cannot convert to ByteString: {type(value)}")

        else:
            # Return as-is for complex types
            return value

    except Exception as e:
        raise ValueError(f"Failed to convert {value} to {target_type}: {e}")


def convert_from_opcua_type(value: Any, data_type: str) -> Any:
    """
    Convert an OPC-UA value to a Python native type.

    Args:
        value: OPC-UA value
        data_type: OPC-UA data type name

    Returns:
        Python native value
    """
    if value is None:
        return None

    # Handle common OPC-UA specific types
    type_name = data_type.lower()

    if type_name in ("datetime", "date"):
        if isinstance(value, datetime):
            return value
        # OPC-UA DateTime is 100ns ticks since 1601-01-01
        if isinstance(value, int):
            # Convert from OPC-UA DateTime format
            epoch_diff = 11644473600  # Seconds between 1601 and 1970
            seconds = value / 10_000_000 - epoch_diff
            return datetime.fromtimestamp(seconds, tz=timezone.utc)

    elif type_name == "guid":
        if isinstance(value, uuid.UUID):
            return str(value)
        return value

    elif type_name == "bytestring":
        if isinstance(value, bytes):
            return value
        return bytes(value)

    elif type_name in ("localizedtext",):
        # Extract text from LocalizedText
        if hasattr(value, "Text"):
            return value.Text
        return str(value)

    elif type_name in ("qualifiedname",):
        # Extract name from QualifiedName
        if hasattr(value, "Name"):
            return value.Name
        return str(value)

    # Return as-is for basic types
    return value


def get_python_type_for_opcua(data_type: OPCUADataType) -> type:
    """
    Get the Python type corresponding to an OPC-UA data type.

    Args:
        data_type: OPC-UA data type

    Returns:
        Python type
    """
    type_mapping = {
        OPCUADataType.BOOLEAN: bool,
        OPCUADataType.SBYTE: int,
        OPCUADataType.BYTE: int,
        OPCUADataType.INT16: int,
        OPCUADataType.UINT16: int,
        OPCUADataType.INT32: int,
        OPCUADataType.UINT32: int,
        OPCUADataType.INT64: int,
        OPCUADataType.UINT64: int,
        OPCUADataType.FLOAT: float,
        OPCUADataType.DOUBLE: float,
        OPCUADataType.STRING: str,
        OPCUADataType.DATETIME: datetime,
        OPCUADataType.GUID: uuid.UUID,
        OPCUADataType.BYTESTRING: bytes,
    }
    return type_mapping.get(data_type, object)


# =============================================================================
# Quality Code Utilities
# =============================================================================


# Quality code descriptions
QUALITY_DESCRIPTIONS = {
    OPCUAQuality.GOOD: "Good - The value is good",
    OPCUAQuality.GOOD_LOCAL_OVERRIDE: "Good - Local override applied",
    OPCUAQuality.GOOD_SUB_NORMAL: "Good - Sub-normal, some limits exceeded",
    OPCUAQuality.GOOD_CLAMPED: "Good - Value is clamped to limits",
    OPCUAQuality.UNCERTAIN: "Uncertain - Quality is uncertain",
    OPCUAQuality.UNCERTAIN_INITIAL_VALUE: "Uncertain - Initial value after startup",
    OPCUAQuality.UNCERTAIN_SENSOR_NOT_ACCURATE: "Uncertain - Sensor not accurate",
    OPCUAQuality.UNCERTAIN_ENGINEERING_UNITS_EXCEEDED: "Uncertain - Engineering units exceeded",
    OPCUAQuality.UNCERTAIN_SUB_NORMAL: "Uncertain - Sub-normal conditions",
    OPCUAQuality.BAD: "Bad - The value is bad",
    OPCUAQuality.BAD_CONFIG_ERROR: "Bad - Configuration error",
    OPCUAQuality.BAD_NOT_CONNECTED: "Bad - Not connected to data source",
    OPCUAQuality.BAD_DEVICE_FAILURE: "Bad - Device failure",
    OPCUAQuality.BAD_SENSOR_FAILURE: "Bad - Sensor failure",
    OPCUAQuality.BAD_LAST_KNOWN_VALUE: "Bad - Last known value",
    OPCUAQuality.BAD_COMM_FAILURE: "Bad - Communication failure",
    OPCUAQuality.BAD_OUT_OF_SERVICE: "Bad - Out of service",
    OPCUAQuality.BAD_WAITING_FOR_INITIAL_DATA: "Bad - Waiting for initial data",
}


def get_quality_description(quality: OPCUAQuality) -> str:
    """
    Get a human-readable description of a quality code.

    Args:
        quality: OPC-UA quality code

    Returns:
        Description string
    """
    return QUALITY_DESCRIPTIONS.get(quality, f"Unknown quality code: {quality}")


def get_quality_level(quality: OPCUAQuality) -> QualityLevel:
    """
    Get the quality level (good/uncertain/bad) from a quality code.

    Args:
        quality: OPC-UA quality code

    Returns:
        Quality level
    """
    return QualityLevel.from_quality_code(quality)


def is_quality_good(quality: OPCUAQuality) -> bool:
    """Check if quality is good."""
    return get_quality_level(quality) == QualityLevel.GOOD


def is_quality_usable(quality: OPCUAQuality) -> bool:
    """Check if quality is usable (good or uncertain)."""
    level = get_quality_level(quality)
    return level in (QualityLevel.GOOD, QualityLevel.UNCERTAIN)


def is_quality_bad(quality: OPCUAQuality) -> bool:
    """Check if quality is bad."""
    return get_quality_level(quality) == QualityLevel.BAD


def parse_status_code(status_code: int) -> Dict[str, Any]:
    """
    Parse an OPC-UA status code into its components.

    Args:
        status_code: 32-bit status code

    Returns:
        Dictionary with severity, sub-code, and flags
    """
    # Severity (bits 30-31)
    severity = (status_code >> 30) & 0x3
    severity_names = {0: "Good", 1: "Uncertain", 2: "Bad", 3: "Bad"}

    # Sub-code (bits 16-29)
    sub_code = (status_code >> 16) & 0x3FFF

    # Info type (bits 10-11)
    info_type = (status_code >> 10) & 0x3
    info_types = {0: "NotUsed", 1: "DataValue", 2: "Reserved", 3: "Reserved"}

    # Limit bits (bits 8-9)
    limit_bits = (status_code >> 8) & 0x3
    limit_names = {0: "None", 1: "Low", 2: "High", 3: "Constant"}

    # Overflow (bit 7)
    overflow = bool((status_code >> 7) & 0x1)

    return {
        "raw_code": status_code,
        "hex_code": f"0x{status_code:08X}",
        "severity": severity_names.get(severity, "Unknown"),
        "severity_code": severity,
        "sub_code": sub_code,
        "info_type": info_types.get(info_type, "Unknown"),
        "limit": limit_names.get(limit_bits, "Unknown"),
        "overflow": overflow,
    }


# =============================================================================
# Endpoint Utilities
# =============================================================================


def parse_endpoint_url(url: str) -> Dict[str, Any]:
    """
    Parse an OPC-UA endpoint URL.

    Args:
        url: OPC-UA endpoint URL

    Returns:
        Dictionary with parsed components
    """
    parsed = urlparse(url)

    # Determine transport
    transport = "tcp"
    if parsed.scheme == "opc.tcp":
        transport = "tcp"
    elif parsed.scheme == "opc.https":
        transport = "https"
    elif parsed.scheme == "opc.wss":
        transport = "websocket"

    # Default port based on transport
    default_ports = {"tcp": 4840, "https": 443, "websocket": 443}
    port = parsed.port or default_ports.get(transport, 4840)

    return {
        "scheme": parsed.scheme,
        "transport": transport,
        "host": parsed.hostname or "localhost",
        "port": port,
        "path": parsed.path or "/",
        "query": parsed.query,
        "full_url": url,
    }


def build_endpoint_url(
    host: str,
    port: int = 4840,
    path: str = "",
    transport: str = "tcp",
) -> str:
    """
    Build an OPC-UA endpoint URL.

    Args:
        host: Server hostname or IP
        port: Server port
        path: Server path
        transport: Transport type (tcp, https, wss)

    Returns:
        Formatted endpoint URL
    """
    scheme = f"opc.{transport}"
    if path and not path.startswith("/"):
        path = "/" + path
    return f"{scheme}://{host}:{port}{path}"


def get_security_level(
    policy: SecurityPolicy,
    mode: MessageSecurityMode,
) -> int:
    """
    Calculate relative security level for endpoint selection.

    Args:
        policy: Security policy
        mode: Message security mode

    Returns:
        Security level (0-100, higher is more secure)
    """
    # Policy scores
    policy_scores = {
        SecurityPolicy.NONE: 0,
        SecurityPolicy.BASIC128RSA15: 20,
        SecurityPolicy.BASIC256: 40,
        SecurityPolicy.BASIC256SHA256: 60,
        SecurityPolicy.AES128_SHA256_RSAOAEP: 80,
        SecurityPolicy.AES256_SHA256_RSAPSS: 100,
    }

    # Mode multipliers
    mode_multipliers = {
        MessageSecurityMode.NONE: 0,
        MessageSecurityMode.SIGN: 0.5,
        MessageSecurityMode.SIGN_AND_ENCRYPT: 1.0,
    }

    policy_score = policy_scores.get(policy, 0)
    mode_mult = mode_multipliers.get(mode, 0)

    return int(policy_score * mode_mult)


# =============================================================================
# Time Utilities
# =============================================================================


# OPC-UA epoch: January 1, 1601
OPCUA_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)

# Difference between OPC-UA epoch and Unix epoch in 100ns ticks
EPOCH_DIFF_TICKS = 116444736000000000


def datetime_to_opcua(dt: datetime) -> int:
    """
    Convert a Python datetime to OPC-UA DateTime (100ns ticks since 1601).

    Args:
        dt: Python datetime

    Returns:
        OPC-UA DateTime value
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Calculate delta from OPC-UA epoch
    delta = dt - OPCUA_EPOCH
    total_seconds = delta.total_seconds()

    # Convert to 100ns ticks
    return int(total_seconds * 10_000_000)


def opcua_to_datetime(ticks: int) -> datetime:
    """
    Convert an OPC-UA DateTime to Python datetime.

    Args:
        ticks: OPC-UA DateTime (100ns ticks since 1601)

    Returns:
        Python datetime
    """
    # Convert ticks to seconds
    seconds = ticks / 10_000_000

    # Add to OPC-UA epoch
    return OPCUA_EPOCH + timedelta(seconds=seconds)


def get_opcua_now() -> int:
    """Get current time in OPC-UA DateTime format."""
    return datetime_to_opcua(datetime.now(timezone.utc))


# =============================================================================
# Validation Utilities
# =============================================================================


def validate_sampling_interval(interval_ms: int) -> int:
    """
    Validate and normalize a sampling interval.

    Args:
        interval_ms: Requested interval in milliseconds

    Returns:
        Validated interval (clamped to valid range)
    """
    MIN_INTERVAL = 10  # 10ms minimum
    MAX_INTERVAL = 3600000  # 1 hour maximum

    if interval_ms < MIN_INTERVAL:
        logger.warning(f"Sampling interval {interval_ms}ms below minimum, using {MIN_INTERVAL}ms")
        return MIN_INTERVAL

    if interval_ms > MAX_INTERVAL:
        logger.warning(f"Sampling interval {interval_ms}ms above maximum, using {MAX_INTERVAL}ms")
        return MAX_INTERVAL

    return interval_ms


def validate_queue_size(size: int) -> int:
    """
    Validate and normalize a queue size.

    Args:
        size: Requested queue size

    Returns:
        Validated queue size
    """
    MIN_SIZE = 1
    MAX_SIZE = 10000

    if size < MIN_SIZE:
        return MIN_SIZE
    if size > MAX_SIZE:
        return MAX_SIZE
    return size


def validate_node_id_list(node_ids: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate a list of node IDs.

    Args:
        node_ids: List of node ID strings

    Returns:
        Tuple of (valid_node_ids, invalid_node_ids)
    """
    valid = []
    invalid = []

    for node_id in node_ids:
        if is_valid_node_id(node_id):
            valid.append(node_id)
        else:
            invalid.append(node_id)
            logger.warning(f"Invalid node ID: {node_id}")

    return valid, invalid


# =============================================================================
# Browse Path Utilities
# =============================================================================


def parse_browse_path(path: str) -> List[Tuple[int, str]]:
    """
    Parse a browse path string into components.

    Browse paths use forward slashes to separate path elements.
    Each element can optionally include a namespace index prefix.

    Example: "2:Objects/Server/ServerStatus/CurrentTime"

    Args:
        path: Browse path string

    Returns:
        List of (namespace_index, browse_name) tuples
    """
    components = []

    for element in path.split("/"):
        element = element.strip()
        if not element:
            continue

        # Check for namespace prefix
        ns_match = re.match(r"(\d+):(.*)", element)
        if ns_match:
            namespace = int(ns_match.group(1))
            name = ns_match.group(2)
        else:
            namespace = 0
            name = element

        components.append((namespace, name))

    return components


def format_browse_path(components: List[Tuple[int, str]]) -> str:
    """
    Format browse path components into a path string.

    Args:
        components: List of (namespace_index, browse_name) tuples

    Returns:
        Browse path string
    """
    elements = []
    for namespace, name in components:
        if namespace == 0:
            elements.append(name)
        else:
            elements.append(f"{namespace}:{name}")

    return "/".join(elements)


# =============================================================================
# Well-Known Node IDs
# =============================================================================


class WellKnownNodes:
    """Well-known OPC-UA node IDs from Part 3 - Address Space Model."""

    # Root nodes
    ROOT = "i=84"
    OBJECTS = "i=85"
    TYPES = "i=86"
    VIEWS = "i=87"

    # Object types
    BASE_OBJECT_TYPE = "i=58"
    FOLDER_TYPE = "i=61"
    BASE_DATA_VARIABLE_TYPE = "i=63"

    # Server object
    SERVER = "i=2253"
    SERVER_ARRAY = "i=2254"
    NAMESPACE_ARRAY = "i=2255"
    SERVER_STATUS = "i=2256"
    SERVICE_LEVEL = "i=2267"
    AUDITING = "i=2994"

    # Server status
    SERVER_STATUS_STATE = "i=2259"
    SERVER_STATUS_BUILD_INFO = "i=2260"
    SERVER_STATUS_CURRENT_TIME = "i=2258"
    SERVER_STATUS_START_TIME = "i=2257"

    # Server capabilities
    SERVER_CAPABILITIES = "i=2268"
    OPERATION_LIMITS = "i=11704"

    # Reference types
    REFERENCES = "i=31"
    HIERARCHICAL_REFERENCES = "i=33"
    NON_HIERARCHICAL_REFERENCES = "i=32"
    HAS_CHILD = "i=34"
    ORGANIZES = "i=35"
    HAS_EVENT_SOURCE = "i=36"
    HAS_MODELLING_RULE = "i=37"
    HAS_ENCODING = "i=38"
    HAS_DESCRIPTION = "i=39"
    HAS_TYPE_DEFINITION = "i=40"
    GENERATES_EVENT = "i=41"
    AGGREGATES = "i=44"
    HAS_SUBTYPE = "i=45"
    HAS_PROPERTY = "i=46"
    HAS_COMPONENT = "i=47"
    HAS_NOTIFIER = "i=48"
    HAS_ORDERED_COMPONENT = "i=49"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Node ID utilities
    "NodeIdType",
    "parse_node_id",
    "format_node_id",
    "is_valid_node_id",
    "get_node_id_namespace",
    # Data type conversions
    "convert_to_opcua_type",
    "convert_from_opcua_type",
    "get_python_type_for_opcua",
    # Quality code utilities
    "get_quality_description",
    "get_quality_level",
    "is_quality_good",
    "is_quality_usable",
    "is_quality_bad",
    "parse_status_code",
    # Endpoint utilities
    "parse_endpoint_url",
    "build_endpoint_url",
    "get_security_level",
    # Time utilities
    "datetime_to_opcua",
    "opcua_to_datetime",
    "get_opcua_now",
    # Validation utilities
    "validate_sampling_interval",
    "validate_queue_size",
    "validate_node_id_list",
    # Browse path utilities
    "parse_browse_path",
    "format_browse_path",
    # Well-known nodes
    "WellKnownNodes",
]
