# -*- coding: utf-8 -*-
"""
Safe YAML/JSON Parser for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module implements secure YAML/JSON parsing with comprehensive protection against:
- Large payloads (size limit enforcement before parsing)
- Deep nesting (depth limit enforcement during parsing)
- YAML bombs (billion laughs attack prevention via anchor/alias disabling)
- Excessive nodes (total node count limit enforcement)
- Malformed UTF-8 (graceful error handling)

The parser uses yaml.SafeLoader only and completely disables YAML anchors/aliases
to prevent memory exhaustion attacks.

Security Features:
    1. Size Check: Content size verified before any parsing begins
    2. Safe YAML Loader: Custom loader that disables all dangerous features
    3. Anchor/Alias Rejection: YAML anchors and aliases raise immediate errors
    4. Depth Tracking: Nesting depth monitored during parsing
    5. Node Counting: Total node count tracked post-parse with limits
    6. UTF-8 Handling: Graceful handling of encoding errors

Example:
    >>> from greenlang.schema.compiler.parser import parse_payload, detect_format
    >>> result = parse_payload('{"name": "test", "value": 42}')
    >>> print(result.format)
    json
    >>> print(result.data)
    {'name': 'test', 'value': 42}
    >>> print(result.node_count)
    3
    >>> print(result.max_depth)
    1

Error Handling:
    >>> from greenlang.schema.compiler.parser import parse_payload, ParseError
    >>> try:
    ...     result = parse_payload('{"deeply": ' * 100 + '{}' + '}' * 100)
    ... except ParseError as e:
    ...     print(f"Error: {e.code}")
    Error: GLSCHEMA-E801

Author: GreenLang Team
Date: 2026-01-29
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field

from greenlang.schema.constants import (
    MAX_OBJECT_DEPTH,
    MAX_PAYLOAD_BYTES,
    MAX_TOTAL_NODES,
    DEFAULT_ENCODING,
)
from greenlang.schema.errors import ErrorCode, format_error_message

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class ParseResult(BaseModel):
    """
    Result of parsing a YAML/JSON payload.

    This model contains both the parsed data and metadata about the parsing
    process, useful for validation, auditing, and performance monitoring.

    Attributes:
        data: The parsed data structure (dictionary at root level)
        format: The detected format ("yaml" or "json")
        size_bytes: Size of the input in bytes
        node_count: Total number of nodes in the parsed structure
        max_depth: Maximum nesting depth found in the structure
        parse_time_ms: Time taken to parse in milliseconds

    Example:
        >>> result = ParseResult(
        ...     data={"key": "value"},
        ...     format="json",
        ...     size_bytes=16,
        ...     node_count=2,
        ...     max_depth=1,
        ...     parse_time_ms=0.5
        ... )
        >>> result.model_dump()
        {'data': {'key': 'value'}, 'format': 'json', ...}
    """

    data: Dict[str, Any] = Field(
        ...,
        description="The parsed data structure"
    )
    format: Literal["yaml", "json"] = Field(
        ...,
        description="The detected input format"
    )
    size_bytes: int = Field(
        ...,
        ge=0,
        description="Size of the input in bytes"
    )
    node_count: int = Field(
        ...,
        ge=0,
        description="Total number of nodes in the parsed structure"
    )
    max_depth: int = Field(
        ...,
        ge=0,
        description="Maximum nesting depth found"
    )
    parse_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Time taken to parse in milliseconds"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }


class ParseError(Exception):
    """
    Exception raised when parsing fails.

    This exception provides structured error information compatible with
    the GLSCHEMA-* error code taxonomy.

    Attributes:
        code: GLSCHEMA-* error code string (e.g., "GLSCHEMA-E800")
        message: Human-readable error message
        details: Additional context about the error

    Example:
        >>> raise ParseError(
        ...     code="GLSCHEMA-E800",
        ...     message="Payload too large",
        ...     details={"size_bytes": 2000000, "max_bytes": 1048576}
        ... )
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ParseError.

        Args:
            code: GLSCHEMA-* error code string
            message: Human-readable error message
            details: Additional context dictionary
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return formatted error string."""
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"ParseError(code={self.code!r}, message={self.message!r}, details={self.details!r})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary for serialization.

        Returns:
            Dictionary with code, message, and details
        """
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Safe YAML Loader
# =============================================================================


class SafeYAMLLoader(yaml.SafeLoader):
    """
    Custom YAML loader that disables dangerous features.

    This loader extends yaml.SafeLoader to completely disable YAML anchors
    and aliases, preventing "billion laughs" style memory exhaustion attacks.

    Security Features:
        - Inherits all SafeLoader restrictions (no arbitrary Python objects)
        - Rejects all anchor definitions
        - Rejects all alias references
        - Provides clear error messages when these features are detected

    Example:
        >>> import yaml
        >>> yaml_content = "key: value"
        >>> data = yaml.load(yaml_content, Loader=SafeYAMLLoader)
        >>> print(data)
        {'key': 'value'}

    Attack Prevention:
        >>> yaml_bomb = '''
        ... a: &a ["lol","lol","lol","lol","lol","lol","lol","lol","lol"]
        ... b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]
        ... c: [*b,*b,*b,*b,*b,*b,*b,*b,*b]
        ... '''
        >>> yaml.load(yaml_bomb, Loader=SafeYAMLLoader)
        # Raises ParseError with GLSCHEMA-E507
    """

    def __init__(self, stream: Any):
        """
        Initialize SafeYAMLLoader.

        Args:
            stream: Input stream (string, bytes, or file-like object)
        """
        super().__init__(stream)
        # Track if we've seen any anchors (for better error messages)
        self._anchor_count = 0
        self._alias_count = 0

    def compose_node(self, parent: Any, index: Any) -> Any:
        """
        Override compose_node to detect and reject aliases.

        Args:
            parent: Parent node
            index: Index in parent

        Returns:
            Composed node

        Raises:
            ParseError: If an alias reference is detected
        """
        # Check for alias marker (*)
        if self.check_event(yaml.AliasEvent):
            event = self.peek_event()
            self._alias_count += 1
            # Consume the event to get details
            alias_event = self.get_event()
            raise _create_yaml_alias_error(
                alias_name=alias_event.anchor,
                line=alias_event.start_mark.line + 1 if alias_event.start_mark else None,
                column=alias_event.start_mark.column + 1 if alias_event.start_mark else None
            )

        return super().compose_node(parent, index)

    def compose_mapping_node(self, anchor: Optional[str]) -> Any:
        """
        Override compose_mapping_node to detect and reject anchors on mappings.

        Args:
            anchor: Anchor name if present

        Returns:
            Composed mapping node

        Raises:
            ParseError: If an anchor definition is detected
        """
        if anchor is not None:
            self._anchor_count += 1
            raise _create_yaml_anchor_error(anchor_name=anchor)

        return super().compose_mapping_node(anchor)

    def compose_sequence_node(self, anchor: Optional[str]) -> Any:
        """
        Override compose_sequence_node to detect and reject anchors on sequences.

        Args:
            anchor: Anchor name if present

        Returns:
            Composed sequence node

        Raises:
            ParseError: If an anchor definition is detected
        """
        if anchor is not None:
            self._anchor_count += 1
            raise _create_yaml_anchor_error(anchor_name=anchor)

        return super().compose_sequence_node(anchor)

    def compose_scalar_node(self, anchor: Optional[str]) -> Any:
        """
        Override compose_scalar_node to detect and reject anchors on scalars.

        Args:
            anchor: Anchor name if present

        Returns:
            Composed scalar node

        Raises:
            ParseError: If an anchor definition is detected
        """
        if anchor is not None:
            self._anchor_count += 1
            raise _create_yaml_anchor_error(anchor_name=anchor)

        return super().compose_scalar_node(anchor)


def _create_yaml_anchor_error(anchor_name: str) -> ParseError:
    """
    Create ParseError for YAML anchor detection.

    Args:
        anchor_name: The anchor name that was detected

    Returns:
        ParseError with appropriate code and message
    """
    return ParseError(
        code=ErrorCode.SCHEMA_PARSE_ERROR.value,
        message=f"YAML anchors are disabled for security. Found anchor '&{anchor_name}'",
        details={
            "anchor_name": anchor_name,
            "security_reason": "YAML anchors can be exploited in 'billion laughs' attacks",
            "suggestion": "Remove all YAML anchors (&name) from the document"
        }
    )


def _create_yaml_alias_error(
    alias_name: str,
    line: Optional[int] = None,
    column: Optional[int] = None
) -> ParseError:
    """
    Create ParseError for YAML alias detection.

    Args:
        alias_name: The alias name that was detected
        line: Line number where alias was found
        column: Column number where alias was found

    Returns:
        ParseError with appropriate code and message
    """
    location = ""
    if line is not None:
        location = f" at line {line}"
        if column is not None:
            location += f", column {column}"

    return ParseError(
        code=ErrorCode.SCHEMA_PARSE_ERROR.value,
        message=f"YAML aliases are disabled for security. Found alias '*{alias_name}'{location}",
        details={
            "alias_name": alias_name,
            "line": line,
            "column": column,
            "security_reason": "YAML aliases can be exploited in 'billion laughs' attacks",
            "suggestion": "Remove all YAML aliases (*name) and inline the values"
        }
    )


# =============================================================================
# Format Detection
# =============================================================================


def detect_format(content: Union[str, bytes]) -> Literal["yaml", "json"]:
    """
    Detect if content is YAML or JSON.

    JSON detection is attempted first by checking for valid JSON structure.
    If JSON parsing fails, YAML is assumed (since valid JSON is also valid YAML).

    Detection Strategy:
        1. Strip leading/trailing whitespace
        2. Check if content starts with JSON structural characters
        3. Attempt to parse as JSON
        4. If JSON fails, classify as YAML

    Args:
        content: The content to analyze (string or bytes)

    Returns:
        "json" if content is valid JSON, "yaml" otherwise

    Example:
        >>> detect_format('{"key": "value"}')
        'json'
        >>> detect_format('key: value')
        'yaml'
        >>> detect_format('[1, 2, 3]')
        'json'
        >>> detect_format('- item1\\n- item2')
        'yaml'

    Note:
        This function does NOT validate the content, it only detects the format.
        Invalid JSON that looks like JSON (e.g., '{"key": }') will still be
        classified as JSON and will fail during actual parsing.
    """
    # Convert bytes to string if needed
    if isinstance(content, bytes):
        try:
            content = content.decode(DEFAULT_ENCODING)
        except UnicodeDecodeError:
            # If we can't decode, try with error replacement
            content = content.decode(DEFAULT_ENCODING, errors="replace")

    # Strip whitespace for analysis
    stripped = content.strip()

    if not stripped:
        # Empty content - default to JSON (empty object)
        return "json"

    # Check for JSON structural indicators
    first_char = stripped[0]

    # JSON typically starts with { or [
    if first_char in ('{', '['):
        # Try to parse as JSON to confirm
        try:
            json.loads(stripped)
            return "json"
        except json.JSONDecodeError:
            # Looks like JSON but isn't valid - could be YAML
            pass

    # Check for other JSON indicators (string, number, boolean, null literals)
    if first_char == '"':
        # Could be a JSON string at top level
        try:
            json.loads(stripped)
            return "json"
        except json.JSONDecodeError:
            pass

    # Check for JSON literals
    json_literals = ('true', 'false', 'null')
    for literal in json_literals:
        if stripped == literal:
            return "json"

    # Check for JSON number at top level
    if first_char.isdigit() or first_char == '-':
        try:
            json.loads(stripped)
            return "json"
        except json.JSONDecodeError:
            pass

    # Default to YAML
    return "yaml"


# =============================================================================
# Node Counting and Depth Tracking
# =============================================================================


def _count_nodes_and_depth(
    data: Any,
    current_depth: int = 0,
    max_depth: int = MAX_OBJECT_DEPTH,
    max_nodes: int = MAX_TOTAL_NODES,
    node_count: Optional[List[int]] = None,
    path: str = ""
) -> Tuple[int, int]:
    """
    Count nodes and track maximum depth in a data structure.

    This function recursively traverses the data structure to count all nodes
    and find the maximum nesting depth. It also enforces limits during traversal.

    Node Definition:
        - Each scalar value (string, number, boolean, null) = 1 node
        - Each dictionary = 1 node (the dict itself, not counting keys as separate nodes)
        - Each list = 1 node (the list itself)
        - Keys in dictionaries are counted as part of the dictionary node

    Depth Definition:
        - Root level = depth 0
        - Each nested object/array increases depth by 1

    Args:
        data: The data structure to analyze
        current_depth: Current nesting depth (for recursion)
        max_depth: Maximum allowed depth (raises error if exceeded)
        max_nodes: Maximum allowed nodes (raises error if exceeded)
        node_count: Mutable counter for node tracking (internal use)
        path: Current JSON pointer path (for error messages)

    Returns:
        Tuple of (total_node_count, maximum_depth_found)

    Raises:
        ParseError: If depth or node count limits are exceeded

    Example:
        >>> _count_nodes_and_depth({"a": {"b": 1}})
        (3, 2)  # dict + nested dict + value = 3 nodes, depth 2
        >>> _count_nodes_and_depth([1, 2, 3])
        (4, 1)  # list + 3 values = 4 nodes, depth 1
    """
    # Initialize node count on first call
    if node_count is None:
        node_count = [0]

    # Check depth limit
    if current_depth > max_depth:
        raise ParseError(
            code=ErrorCode.DEPTH_EXCEEDED.value,
            message=format_error_message(
                ErrorCode.DEPTH_EXCEEDED,
                depth=current_depth,
                path=path or "/",
                max_depth=max_depth
            ),
            details={
                "depth": current_depth,
                "max_depth": max_depth,
                "path": path or "/"
            }
        )

    # Increment node count
    node_count[0] += 1

    # Check node count limit
    if node_count[0] > max_nodes:
        raise ParseError(
            code=ErrorCode.NODES_EXCEEDED.value,
            message=format_error_message(
                ErrorCode.NODES_EXCEEDED,
                count=node_count[0],
                max_nodes=max_nodes
            ),
            details={
                "count": node_count[0],
                "max_nodes": max_nodes
            }
        )

    max_depth_found = current_depth

    if isinstance(data, dict):
        for key, value in data.items():
            child_path = f"{path}/{key}" if path else f"/{key}"
            _, child_max_depth = _count_nodes_and_depth(
                value,
                current_depth + 1,
                max_depth,
                max_nodes,
                node_count,
                child_path
            )
            max_depth_found = max(max_depth_found, child_max_depth)

    elif isinstance(data, list):
        for i, item in enumerate(data):
            child_path = f"{path}/{i}"
            _, child_max_depth = _count_nodes_and_depth(
                item,
                current_depth + 1,
                max_depth,
                max_nodes,
                node_count,
                child_path
            )
            max_depth_found = max(max_depth_found, child_max_depth)

    # For scalar values (str, int, float, bool, None), just return current info

    return node_count[0], max_depth_found


# =============================================================================
# Content Preprocessing
# =============================================================================


def _prepare_content(content: Union[str, bytes]) -> Tuple[str, int]:
    """
    Prepare content for parsing by handling encoding and measuring size.

    This function:
        1. Determines the size in bytes
        2. Converts bytes to string if needed
        3. Handles encoding errors gracefully

    Args:
        content: Input content (string or bytes)

    Returns:
        Tuple of (string_content, size_in_bytes)

    Raises:
        ParseError: If content cannot be decoded (malformed UTF-8)

    Example:
        >>> _prepare_content('{"key": "value"}')
        ('{"key": "value"}', 16)
        >>> _prepare_content(b'{"key": "value"}')
        ('{"key": "value"}', 16)
    """
    # Calculate size in bytes
    if isinstance(content, bytes):
        size_bytes = len(content)
        try:
            string_content = content.decode(DEFAULT_ENCODING)
        except UnicodeDecodeError as e:
            # Log the encoding error
            logger.warning(f"UTF-8 decode error at position {e.start}: {e.reason}")

            # Try with error replacement to provide more details
            try:
                # Attempt partial decode to identify problematic bytes
                partial = content[:e.start].decode(DEFAULT_ENCODING, errors="replace")
                context_start = max(0, e.start - 20)
                context_bytes = content[context_start:e.start + 20]

                raise ParseError(
                    code=ErrorCode.SCHEMA_PARSE_ERROR.value,
                    message=f"Malformed UTF-8 encoding at byte position {e.start}: {e.reason}",
                    details={
                        "position": e.start,
                        "reason": e.reason,
                        "context_hex": context_bytes.hex() if context_bytes else None,
                        "encoding": DEFAULT_ENCODING
                    }
                )
            except ParseError:
                raise
            except Exception:
                raise ParseError(
                    code=ErrorCode.SCHEMA_PARSE_ERROR.value,
                    message=f"Malformed UTF-8 encoding: {e.reason}",
                    details={
                        "reason": e.reason,
                        "encoding": DEFAULT_ENCODING
                    }
                )
    else:
        # String content
        string_content = content
        size_bytes = len(content.encode(DEFAULT_ENCODING))

    return string_content, size_bytes


# =============================================================================
# Main Parsing Function
# =============================================================================


def parse_payload(
    content: Union[str, bytes],
    max_bytes: Optional[int] = None,
    max_depth: Optional[int] = None,
    max_nodes: Optional[int] = None
) -> ParseResult:
    """
    Parse YAML/JSON content with comprehensive safety limits.

    This is the main entry point for safely parsing YAML or JSON payloads.
    It enforces multiple security limits to prevent denial-of-service attacks.

    Security Enforcement Order:
        1. Payload size check (before any parsing)
        2. Format detection
        3. Parsing with SafeYAMLLoader (no anchors/aliases)
        4. Node counting and depth validation (post-parse)

    Args:
        content: The YAML or JSON content to parse (string or bytes)
        max_bytes: Maximum allowed content size in bytes (default: MAX_PAYLOAD_BYTES)
        max_depth: Maximum allowed nesting depth (default: MAX_OBJECT_DEPTH)
        max_nodes: Maximum allowed total node count (default: MAX_TOTAL_NODES)

    Returns:
        ParseResult containing:
            - data: Parsed dictionary
            - format: "yaml" or "json"
            - size_bytes: Input size
            - node_count: Total nodes
            - max_depth: Maximum depth found
            - parse_time_ms: Parsing duration

    Raises:
        ParseError: With appropriate GLSCHEMA-E8xx codes:
            - GLSCHEMA-E800: Payload exceeds size limit
            - GLSCHEMA-E801: Depth exceeds limit
            - GLSCHEMA-E805: Node count exceeds limit
            - GLSCHEMA-E507: Parse error (invalid syntax, anchors/aliases)

    Example:
        >>> result = parse_payload('{"name": "test", "value": 42}')
        >>> print(result.format)
        json
        >>> print(result.data)
        {'name': 'test', 'value': 42}
        >>> print(result.node_count)
        3

    Security Example:
        >>> # Large payload rejection
        >>> large = '{"data": "x" * 10000000}'
        >>> parse_payload(large, max_bytes=1000)  # Raises ParseError E800

        >>> # Deep nesting rejection
        >>> deep = '{"a": ' * 100 + '{}' + '}' * 100
        >>> parse_payload(deep, max_depth=50)  # Raises ParseError E801

        >>> # YAML bomb rejection
        >>> bomb = 'a: &a [1,2,3]\\nb: *a'
        >>> parse_payload(bomb)  # Raises ParseError E507

    Performance Notes:
        - Size check is O(1) for bytes, O(n) for strings
        - Format detection is fast (simple checks + single JSON parse attempt)
        - Node counting is O(n) where n is number of nodes
        - Overall complexity is O(n)
    """
    start_time = time.perf_counter()

    # Apply defaults
    if max_bytes is None:
        max_bytes = MAX_PAYLOAD_BYTES
    if max_depth is None:
        max_depth = MAX_OBJECT_DEPTH
    if max_nodes is None:
        max_nodes = MAX_TOTAL_NODES

    # Step 1: Prepare content and check size
    try:
        string_content, size_bytes = _prepare_content(content)
    except ParseError:
        raise
    except Exception as e:
        logger.error(f"Content preparation failed: {e}", exc_info=True)
        raise ParseError(
            code=ErrorCode.SCHEMA_PARSE_ERROR.value,
            message=f"Failed to prepare content for parsing: {str(e)}",
            details={"error": str(e)}
        )

    # Step 2: Enforce size limit BEFORE parsing
    if size_bytes > max_bytes:
        raise ParseError(
            code=ErrorCode.PAYLOAD_TOO_LARGE.value,
            message=format_error_message(
                ErrorCode.PAYLOAD_TOO_LARGE,
                size_bytes=size_bytes,
                max_bytes=max_bytes
            ),
            details={
                "size_bytes": size_bytes,
                "max_bytes": max_bytes,
                "excess_bytes": size_bytes - max_bytes
            }
        )

    # Step 3: Detect format
    detected_format = detect_format(string_content)
    logger.debug(f"Detected format: {detected_format} for {size_bytes} bytes")

    # Handle empty content early
    if not string_content.strip():
        end_time = time.perf_counter()
        parse_time_ms = (end_time - start_time) * 1000
        return ParseResult(
            data={},
            format=detected_format,
            size_bytes=size_bytes,
            node_count=1,
            max_depth=0,
            parse_time_ms=parse_time_ms
        )

    # Step 4: Parse content
    parsed_data: Dict[str, Any]

    try:
        if detected_format == "json":
            # Parse as JSON
            parsed_data = _parse_json(string_content)
        else:
            # Parse as YAML with safe loader
            parsed_data = _parse_yaml(string_content)
    except ParseError:
        # Re-raise our own errors
        raise
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        raise ParseError(
            code=ErrorCode.SCHEMA_PARSE_ERROR.value,
            message=f"JSON parse error at line {e.lineno}, column {e.colno}: {e.msg}",
            details={
                "line": e.lineno,
                "column": e.colno,
                "error": e.msg,
                "format": "json"
            }
        )
    except yaml.YAMLError as e:
        logger.warning(f"YAML parse error: {e}")
        error_details: Dict[str, Any] = {"format": "yaml", "error": str(e)}

        # Extract position information if available
        if hasattr(e, 'problem_mark') and e.problem_mark:
            error_details["line"] = e.problem_mark.line + 1
            error_details["column"] = e.problem_mark.column + 1

        raise ParseError(
            code=ErrorCode.SCHEMA_PARSE_ERROR.value,
            message=f"YAML parse error: {str(e)}",
            details=error_details
        )
    except Exception as e:
        logger.error(f"Unexpected parse error: {e}", exc_info=True)
        raise ParseError(
            code=ErrorCode.SCHEMA_PARSE_ERROR.value,
            message=f"Unexpected error during parsing: {str(e)}",
            details={"error": str(e), "format": detected_format}
        )

    # Step 5: Validate parsed data is a dictionary
    if not isinstance(parsed_data, dict):
        # Wrap non-dict values in a dict for consistent handling
        # This handles JSON arrays at root level
        logger.debug(f"Wrapping non-dict parsed data of type {type(parsed_data).__name__}")
        parsed_data = {"_root": parsed_data}
    else:
        # Ensure all keys are strings (YAML may have non-string keys)
        if any(not isinstance(k, str) for k in parsed_data.keys()):
            parsed_data = {str(k): v for k, v in parsed_data.items()}

    # Step 6: Count nodes and validate depth
    try:
        node_count, actual_max_depth = _count_nodes_and_depth(
            parsed_data,
            current_depth=0,
            max_depth=max_depth,
            max_nodes=max_nodes
        )
    except ParseError:
        raise
    except Exception as e:
        logger.error(f"Node counting failed: {e}", exc_info=True)
        raise ParseError(
            code=ErrorCode.SCHEMA_PARSE_ERROR.value,
            message=f"Failed to analyze parsed structure: {str(e)}",
            details={"error": str(e)}
        )

    # Calculate parse time
    end_time = time.perf_counter()
    parse_time_ms = (end_time - start_time) * 1000

    logger.debug(
        f"Parsed {detected_format} payload: {size_bytes} bytes, "
        f"{node_count} nodes, depth {actual_max_depth}, "
        f"{parse_time_ms:.2f}ms"
    )

    return ParseResult(
        data=parsed_data,
        format=detected_format,
        size_bytes=size_bytes,
        node_count=node_count,
        max_depth=actual_max_depth,
        parse_time_ms=parse_time_ms
    )


def _parse_json(content: str) -> Any:
    """
    Parse JSON content.

    Args:
        content: JSON string to parse

    Returns:
        Parsed data structure

    Raises:
        json.JSONDecodeError: If JSON is invalid
    """
    return json.loads(content)


def _parse_yaml(content: str) -> Any:
    """
    Parse YAML content with safe loader.

    This function uses our custom SafeYAMLLoader which disables
    anchors and aliases.

    Args:
        content: YAML string to parse

    Returns:
        Parsed data structure

    Raises:
        ParseError: If anchors or aliases are found
        yaml.YAMLError: If YAML is invalid
    """
    result = yaml.load(content, Loader=SafeYAMLLoader)

    # Handle empty YAML (returns None)
    if result is None:
        return {}

    return result


# =============================================================================
# Utility Functions
# =============================================================================


def validate_payload_size(
    content: Union[str, bytes],
    max_bytes: int = MAX_PAYLOAD_BYTES
) -> Tuple[bool, int]:
    """
    Validate payload size without parsing.

    This is a quick check to reject oversized payloads before any processing.

    Args:
        content: Content to check (string or bytes)
        max_bytes: Maximum allowed size in bytes

    Returns:
        Tuple of (is_valid, actual_size_bytes)

    Example:
        >>> is_valid, size = validate_payload_size('{"key": "value"}')
        >>> print(f"Valid: {is_valid}, Size: {size}")
        Valid: True, Size: 16
    """
    if isinstance(content, bytes):
        size_bytes = len(content)
    else:
        size_bytes = len(content.encode(DEFAULT_ENCODING))

    return size_bytes <= max_bytes, size_bytes


def is_valid_json(content: str) -> bool:
    """
    Check if content is valid JSON.

    Args:
        content: String to check

    Returns:
        True if valid JSON, False otherwise

    Example:
        >>> is_valid_json('{"key": "value"}')
        True
        >>> is_valid_json('key: value')
        False
    """
    try:
        json.loads(content)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def is_valid_yaml(content: str) -> bool:
    """
    Check if content is valid YAML (without anchors/aliases).

    Args:
        content: String to check

    Returns:
        True if valid YAML, False otherwise

    Example:
        >>> is_valid_yaml('key: value')
        True
        >>> is_valid_yaml('key: value: invalid')
        False
    """
    try:
        yaml.load(content, Loader=SafeYAMLLoader)
        return True
    except (yaml.YAMLError, ParseError):
        return False


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Main classes
    "ParseResult",
    "ParseError",
    "SafeYAMLLoader",
    # Main functions
    "parse_payload",
    "detect_format",
    # Utility functions
    "validate_payload_size",
    "is_valid_json",
    "is_valid_yaml",
]
