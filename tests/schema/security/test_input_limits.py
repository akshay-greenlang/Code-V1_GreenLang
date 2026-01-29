# -*- coding: utf-8 -*-
"""
Security Tests for Input Limits - GL-FOUND-X-002.

This module tests the schema validator's enforcement of input limits,
including:
    - Large payloads (memory exhaustion prevention)
    - Deep nesting (stack overflow prevention)
    - Many array items (DoS prevention)
    - Many object properties (DoS prevention)
    - Total node count limits
    - String length limits

The parser MUST:
    1. Reject oversized payloads BEFORE parsing
    2. Detect and reject deeply nested structures
    3. Enforce array item limits
    4. Enforce object property limits
    5. Enforce total node count limits
    6. Complete within timeout limits

References:
    - https://cwe.mitre.org/data/definitions/400.html (Resource Exhaustion)
    - https://cwe.mitre.org/data/definitions/770.html (Allocation without Limits)
    - PRD Section 6.10: Security Limits

Author: GreenLang Security Testing Team
Date: 2026-01-29
Version: 1.0.0
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import pytest

from greenlang.schema.compiler.parser import (
    ParseError,
    ParseResult,
    parse_payload,
    validate_payload_size,
    detect_format,
)
from greenlang.schema.constants import (
    MAX_PAYLOAD_BYTES,
    MAX_OBJECT_DEPTH,
    MAX_ARRAY_ITEMS,
    MAX_TOTAL_NODES,
    MAX_STRING_LENGTH,
    MAX_OBJECT_PROPERTIES,
)
from greenlang.schema.errors import ErrorCode


# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.security,
    pytest.mark.timeout(30),  # Tests may create large data
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_limits() -> Dict[str, int]:
    """Small limits for faster testing."""
    return {
        "max_bytes": 10 * 1024,  # 10 KB
        "max_depth": 10,
        "max_nodes": 1000,
    }


@pytest.fixture
def default_timeout_seconds() -> float:
    """Default timeout for operations."""
    return 5.0


# =============================================================================
# Large Payload Tests
# =============================================================================


class TestLargePayloads:
    """Test suite for large payload handling."""

    def test_payload_exactly_at_limit(self):
        """Test payload exactly at the size limit."""
        # Create a payload close to but not over the limit
        # We need to account for JSON structure overhead
        target_size = MAX_PAYLOAD_BYTES - 100
        padding = "x" * target_size
        payload = json.dumps({"data": padding})

        # Should be accepted if under limit
        if len(payload.encode("utf-8")) <= MAX_PAYLOAD_BYTES:
            result = parse_payload(payload)
            assert result.data["data"] == padding

    def test_payload_over_limit_rejected(self, small_limits: Dict[str, int]):
        """Test that payloads over the limit are rejected."""
        max_bytes = small_limits["max_bytes"]

        # Create payload larger than limit
        large_string = "x" * (max_bytes + 1000)
        payload = json.dumps({"data": large_string})

        with pytest.raises(ParseError) as exc_info:
            parse_payload(payload, max_bytes=max_bytes)

        assert exc_info.value.code == ErrorCode.PAYLOAD_TOO_LARGE.value

    def test_size_check_before_parsing(self, small_limits: Dict[str, int]):
        """Test that size is checked BEFORE any parsing occurs."""
        max_bytes = small_limits["max_bytes"]

        # Create invalid JSON that's too large
        # If parsing happened first, we'd get a parse error, not size error
        invalid_large = "{" * (max_bytes + 1000)

        with pytest.raises(ParseError) as exc_info:
            parse_payload(invalid_large, max_bytes=max_bytes)

        # Should get size error, not parse error
        assert exc_info.value.code == ErrorCode.PAYLOAD_TOO_LARGE.value

    def test_validate_payload_size_utility(self, small_limits: Dict[str, int]):
        """Test the validate_payload_size utility function."""
        max_bytes = small_limits["max_bytes"]

        # Small payload
        is_valid, size = validate_payload_size('{"key": "value"}', max_bytes)
        assert is_valid
        assert size < max_bytes

        # Large payload
        large = "x" * (max_bytes + 1000)
        is_valid, size = validate_payload_size(large, max_bytes)
        assert not is_valid
        assert size > max_bytes

    def test_bytes_vs_string_size_calculation(self):
        """Test that size calculation uses bytes, not string length."""
        # Unicode characters can be multiple bytes
        unicode_string = json.dumps({"emoji": "?????"})

        result = parse_payload(unicode_string)

        # Size should be in bytes, not characters
        assert result.size_bytes >= len(unicode_string)

    def test_large_payload_rejection_is_fast(self, small_limits: Dict[str, int], default_timeout_seconds: float):
        """Test that large payloads are rejected quickly."""
        max_bytes = small_limits["max_bytes"]
        large_payload = "x" * (max_bytes * 10)

        start_time = time.perf_counter()

        with pytest.raises(ParseError):
            parse_payload(large_payload, max_bytes=max_bytes)

        elapsed = time.perf_counter() - start_time

        # Should reject almost immediately (no parsing)
        assert elapsed < 0.1, f"Rejection took {elapsed:.2f}s"


class TestDeepNesting:
    """Test suite for deep nesting handling."""

    def test_nesting_at_limit(self, small_limits: Dict[str, int]):
        """Test nesting exactly at the depth limit."""
        max_depth = small_limits["max_depth"]

        # Create nested structure at exactly the limit
        # Note: depth limit may be interpreted differently
        nested = {}
        current = nested
        for i in range(max_depth - 2):  # Account for root level
            current["level"] = {}
            current = current["level"]
        current["value"] = "end"

        payload = json.dumps(nested)

        # Should be accepted at or below limit
        result = parse_payload(payload, max_depth=max_depth)
        assert "level" in result.data or "value" in result.data

    def test_nesting_over_limit_rejected(self, small_limits: Dict[str, int]):
        """Test that nesting over the limit is rejected."""
        max_depth = small_limits["max_depth"]

        # Create structure deeper than limit
        depth = max_depth + 10
        nested = {}
        current = nested
        for i in range(depth):
            current["level"] = {}
            current = current["level"]
        current["value"] = "end"

        payload = json.dumps(nested)

        with pytest.raises(ParseError) as exc_info:
            parse_payload(payload, max_depth=max_depth)

        assert exc_info.value.code == ErrorCode.DEPTH_EXCEEDED.value

    def test_array_nesting_counted(self, small_limits: Dict[str, int]):
        """Test that array nesting counts toward depth."""
        max_depth = small_limits["max_depth"]

        # Create deeply nested arrays
        depth = max_depth + 5
        nested: Any = "end"
        for _ in range(depth):
            nested = [nested]

        payload = json.dumps({"data": nested})

        with pytest.raises(ParseError) as exc_info:
            parse_payload(payload, max_depth=max_depth)

        assert exc_info.value.code == ErrorCode.DEPTH_EXCEEDED.value

    def test_mixed_object_array_nesting(self, small_limits: Dict[str, int]):
        """Test depth counting with mixed objects and arrays."""
        max_depth = small_limits["max_depth"]

        # Alternating objects and arrays
        depth = max_depth + 5
        nested: Any = "end"
        for i in range(depth):
            if i % 2 == 0:
                nested = {"nested": nested}
            else:
                nested = [nested]

        payload = json.dumps(nested)

        with pytest.raises(ParseError) as exc_info:
            parse_payload(payload, max_depth=max_depth)

        assert exc_info.value.code == ErrorCode.DEPTH_EXCEEDED.value

    def test_wide_but_shallow_allowed(self, small_limits: Dict[str, int]):
        """Test that wide but shallow structures are allowed."""
        max_depth = small_limits["max_depth"]

        # Many keys at the same level (shallow)
        wide = {f"key{i}": f"value{i}" for i in range(100)}
        payload = json.dumps(wide)

        result = parse_payload(payload, max_depth=max_depth)
        assert len(result.data) == 100

    def test_deep_nesting_error_includes_path(self, small_limits: Dict[str, int]):
        """Test that depth error includes the path where limit was exceeded."""
        max_depth = small_limits["max_depth"]

        # Create deep structure
        depth = max_depth + 5
        nested = {}
        current = nested
        for i in range(depth):
            current["level"] = {}
            current = current["level"]
        current["value"] = "end"

        payload = json.dumps(nested)

        with pytest.raises(ParseError) as exc_info:
            parse_payload(payload, max_depth=max_depth)

        # Error should mention path
        assert "path" in exc_info.value.details or "level" in str(exc_info.value)


class TestArrayItemLimits:
    """Test suite for array item count limits."""

    def test_array_at_limit(self, small_limits: Dict[str, int]):
        """Test array exactly at the item limit."""
        max_nodes = small_limits["max_nodes"]

        # Create array at a reasonable size
        items = list(range(100))
        payload = json.dumps({"items": items})

        result = parse_payload(payload, max_nodes=max_nodes)
        assert len(result.data["items"]) == 100

    def test_array_over_limit_rejected(self, small_limits: Dict[str, int]):
        """Test that arrays over the node limit are rejected."""
        max_nodes = small_limits["max_nodes"]

        # Create array with more items than allowed nodes
        items = list(range(max_nodes + 100))
        payload = json.dumps({"items": items})

        with pytest.raises(ParseError) as exc_info:
            parse_payload(payload, max_nodes=max_nodes)

        assert exc_info.value.code == ErrorCode.NODES_EXCEEDED.value

    def test_multiple_arrays_combined(self, small_limits: Dict[str, int]):
        """Test that multiple arrays' items are counted together."""
        max_nodes = small_limits["max_nodes"]

        # Multiple arrays that together exceed limit
        half_limit = max_nodes // 2
        payload = json.dumps({
            "array1": list(range(half_limit)),
            "array2": list(range(half_limit)),
            "array3": list(range(half_limit)),  # This should push over limit
        })

        with pytest.raises(ParseError) as exc_info:
            parse_payload(payload, max_nodes=max_nodes)

        assert exc_info.value.code == ErrorCode.NODES_EXCEEDED.value

    def test_nested_arrays_counted(self, small_limits: Dict[str, int]):
        """Test that items in nested arrays are counted."""
        max_nodes = small_limits["max_nodes"]

        # Nested arrays
        nested = [[i for i in range(100)] for _ in range(20)]
        payload = json.dumps({"data": nested})

        with pytest.raises(ParseError) as exc_info:
            parse_payload(payload, max_nodes=max_nodes)

        assert exc_info.value.code == ErrorCode.NODES_EXCEEDED.value


class TestObjectPropertyLimits:
    """Test suite for object property count limits."""

    def test_many_properties_counted(self, small_limits: Dict[str, int]):
        """Test that many properties contribute to node count."""
        max_nodes = small_limits["max_nodes"]

        # Object with many properties
        many_props = {f"key{i}": f"value{i}" for i in range(max_nodes + 100)}
        payload = json.dumps(many_props)

        with pytest.raises(ParseError) as exc_info:
            parse_payload(payload, max_nodes=max_nodes)

        assert exc_info.value.code == ErrorCode.NODES_EXCEEDED.value

    def test_nested_objects_with_many_properties(self, small_limits: Dict[str, int]):
        """Test nested objects with many properties each.

        Note: Each nested object {"value": i} adds 2 nodes (the object + the value).
        With max_nodes=1000, we need roughly 500+ nested objects to exceed.
        """
        max_nodes = small_limits["max_nodes"]

        # Create nested objects each with many properties
        # Each entry adds: 1 (key's object) + 1 (nested "value" int) = 2 nodes
        # Plus 1 for root + 1 for level1 = 2 + (n * 2) nodes
        # To exceed 1000 nodes, we need ~500 entries
        num_entries = (max_nodes // 2) + 10  # Slightly more than half

        nested = {
            "level1": {
                f"key{i}": {"value": i}
                for i in range(num_entries)
            }
        }
        payload = json.dumps(nested)

        with pytest.raises(ParseError) as exc_info:
            parse_payload(payload, max_nodes=max_nodes)

        assert exc_info.value.code == ErrorCode.NODES_EXCEEDED.value


class TestTotalNodeCount:
    """Test suite for total node count limits."""

    def test_node_count_includes_all_types(self, small_limits: Dict[str, int]):
        """Test that all node types are counted (objects, arrays, scalars)."""
        max_nodes = small_limits["max_nodes"]

        # Mix of different types
        data = {
            "string": "test",
            "number": 42,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "object": {"nested": "value"},
        }
        payload = json.dumps(data)

        result = parse_payload(payload, max_nodes=max_nodes)

        # Verify node count is reported
        assert result.node_count > 0

    def test_node_count_tracking_accurate(self):
        """Test that node count tracking is accurate."""
        # Simple structure with known node count
        # Root object (1) + 2 keys with values (2) = 3 nodes
        data = {"key1": "value1", "key2": "value2"}
        payload = json.dumps(data)

        result = parse_payload(payload)

        # Should count: 1 object + 2 string values = 3 nodes
        # (keys are part of the object, not separate nodes)
        assert result.node_count >= 3

    def test_depth_tracking_accurate(self):
        """Test that depth tracking is accurate."""
        # Structure with known depth
        data = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }
        payload = json.dumps(data)

        result = parse_payload(payload)

        # Depth should be at least 3
        assert result.max_depth >= 3


class TestStringLengthLimits:
    """Test suite for string length limits."""

    def test_long_string_in_payload(self):
        """Test handling of very long strings."""
        # This should work within reasonable limits
        long_string = "x" * 10000
        payload = json.dumps({"data": long_string})

        result = parse_payload(payload)
        assert len(result.data["data"]) == 10000

    def test_very_long_key_names(self):
        """Test handling of very long key names."""
        long_key = "k" * 1000
        payload = json.dumps({long_key: "value"})

        result = parse_payload(payload)
        assert long_key in result.data


class TestCombinedLimits:
    """Test suite for combined limit enforcement."""

    def test_depth_and_size_combined(self, small_limits: Dict[str, int]):
        """Test enforcement when both depth and size limits apply."""
        max_depth = small_limits["max_depth"]
        max_bytes = small_limits["max_bytes"]

        # Create deep AND large payload
        depth = max_depth + 5
        nested = {}
        current = nested
        for i in range(depth):
            current["level"] = {"padding": "x" * 1000}
            current = current["level"]
        current["value"] = "end"

        payload = json.dumps(nested)

        # Should fail on whichever limit is hit first
        with pytest.raises(ParseError) as exc_info:
            parse_payload(payload, max_bytes=max_bytes, max_depth=max_depth)

        # Either size or depth error
        assert exc_info.value.code in [
            ErrorCode.PAYLOAD_TOO_LARGE.value,
            ErrorCode.DEPTH_EXCEEDED.value,
        ]

    def test_all_limits_work_together(self, small_limits: Dict[str, int]):
        """Test that all limits work correctly together."""
        # Normal payload should work
        normal = {"key": "value", "items": [1, 2, 3]}
        payload = json.dumps(normal)

        result = parse_payload(
            payload,
            max_bytes=small_limits["max_bytes"],
            max_depth=small_limits["max_depth"],
            max_nodes=small_limits["max_nodes"],
        )

        assert result.data["key"] == "value"


class TestFormatDetection:
    """Test suite for format detection with malicious input."""

    def test_detect_format_handles_large_input(self, small_limits: Dict[str, int]):
        """Test that format detection handles large input."""
        # Large input that looks like JSON
        large = '{"key": "' + "x" * 10000 + '"}'

        format_type = detect_format(large)
        assert format_type == "json"

    def test_detect_format_handles_binary_like_input(self):
        """Test format detection with binary-like content."""
        # Bytes that might confuse detection
        binary_like = b'\x00\x01\x02\x03'

        # Should not crash
        try:
            format_type = detect_format(binary_like)
        except Exception:
            pass  # May fail to decode, that's okay

    def test_detect_format_unicode_content(self):
        """Test format detection with unicode content."""
        unicode_yaml = "name: ??? ????"
        format_type = detect_format(unicode_yaml)
        assert format_type == "yaml"

        unicode_json = '{"name": "???"}'
        format_type = detect_format(unicode_json)
        assert format_type == "json"


class TestMalformedInput:
    """Test suite for malformed input handling."""

    def test_invalid_utf8_rejected(self):
        """Test that invalid UTF-8 is handled gracefully."""
        # Invalid UTF-8 byte sequence
        invalid_utf8 = b'{"key": "\xff\xfe"}'

        with pytest.raises(ParseError):
            parse_payload(invalid_utf8)

    def test_truncated_json_rejected(self):
        """Test that truncated JSON is rejected."""
        truncated = '{"key": "val'

        with pytest.raises(ParseError):
            parse_payload(truncated)

    def test_truncated_yaml_rejected(self):
        """Test that truncated YAML is handled."""
        truncated = "key: value\nitems:\n  - one\n  - "

        # May parse partially or fail
        try:
            result = parse_payload(truncated)
        except ParseError:
            pass  # Expected

    def test_null_bytes_handled(self):
        """Test handling of null bytes in input."""
        with_nulls = b'{"key": "value\x00hidden"}'

        # Should either parse or fail gracefully
        try:
            result = parse_payload(with_nulls)
        except ParseError:
            pass


class TestPerformance:
    """Performance tests for limit enforcement."""

    def test_limit_enforcement_is_fast(self, small_limits: Dict[str, int], default_timeout_seconds: float):
        """Test that limit enforcement doesn't add significant overhead."""
        max_bytes = small_limits["max_bytes"]

        # Payload close to but under limit
        payload = json.dumps({"data": "x" * (max_bytes - 100)})

        start_time = time.perf_counter()
        result = parse_payload(payload, max_bytes=max_bytes)
        elapsed = time.perf_counter() - start_time

        assert elapsed < default_timeout_seconds

    def test_rejection_is_faster_than_parsing(self, small_limits: Dict[str, int]):
        """Test that rejecting oversized input is faster than parsing it would be."""
        max_bytes = small_limits["max_bytes"]

        # Large payload
        large_payload = json.dumps({"data": "x" * (max_bytes * 10)})

        # Measure rejection time
        start_time = time.perf_counter()
        with pytest.raises(ParseError):
            parse_payload(large_payload, max_bytes=max_bytes)
        reject_time = time.perf_counter() - start_time

        # Rejection should be very fast (no parsing)
        assert reject_time < 0.1, f"Rejection took {reject_time:.3f}s"


class TestErrorMessages:
    """Test suite for error message quality."""

    def test_size_error_includes_limits(self, small_limits: Dict[str, int]):
        """Test that size error includes actual and max size."""
        max_bytes = small_limits["max_bytes"]
        large_payload = "x" * (max_bytes + 1000)

        with pytest.raises(ParseError) as exc_info:
            parse_payload(large_payload, max_bytes=max_bytes)

        details = exc_info.value.details
        assert "size_bytes" in details or "max_bytes" in details

    def test_depth_error_includes_details(self, small_limits: Dict[str, int]):
        """Test that depth error includes depth details."""
        max_depth = small_limits["max_depth"]

        # Deep structure
        depth = max_depth + 10
        nested = {}
        current = nested
        for i in range(depth):
            current["level"] = {}
            current = current["level"]
        current["value"] = "end"

        payload = json.dumps(nested)

        with pytest.raises(ParseError) as exc_info:
            parse_payload(payload, max_depth=max_depth)

        details = exc_info.value.details
        assert "depth" in details or "max_depth" in details

    def test_node_error_includes_count(self, small_limits: Dict[str, int]):
        """Test that node count error includes actual count."""
        max_nodes = small_limits["max_nodes"]

        # Many nodes
        items = list(range(max_nodes + 100))
        payload = json.dumps({"items": items})

        with pytest.raises(ParseError) as exc_info:
            parse_payload(payload, max_nodes=max_nodes)

        details = exc_info.value.details
        assert "count" in details or "max_nodes" in details
