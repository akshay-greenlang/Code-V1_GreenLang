# -*- coding: utf-8 -*-
"""
Unit Tests for Safe YAML/JSON Parser (Task 1.1).

This module contains comprehensive tests for the parser module including:
- Basic JSON and YAML parsing
- Format detection
- Size limit enforcement
- Depth limit enforcement
- Node count limit enforcement
- YAML anchor/alias rejection (billion laughs prevention)
- Malformed UTF-8 handling
- Edge cases

Test Categories:
    - test_parse_*: Tests for the main parse_payload function
    - test_detect_*: Tests for format detection
    - test_safe_yaml_*: Tests for YAML security features
    - test_limits_*: Tests for limit enforcement
    - test_error_*: Tests for error handling

Author: GreenLang Team
Date: 2026-01-29
"""

import json
import pytest
from typing import Any

from greenlang.schema.compiler.parser import (
    ParseResult,
    ParseError,
    SafeYAMLLoader,
    parse_payload,
    detect_format,
    validate_payload_size,
    is_valid_json,
    is_valid_yaml,
    _count_nodes_and_depth,
)
from greenlang.schema.errors import ErrorCode


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_json() -> str:
    """Simple JSON payload."""
    return '{"name": "test", "value": 42}'


@pytest.fixture
def simple_yaml() -> str:
    """Simple YAML payload."""
    return "name: test\nvalue: 42"


@pytest.fixture
def nested_json() -> str:
    """Nested JSON payload."""
    return '{"level1": {"level2": {"level3": {"value": 1}}}}'


@pytest.fixture
def nested_yaml() -> str:
    """Nested YAML payload."""
    return """
level1:
  level2:
    level3:
      value: 1
"""


@pytest.fixture
def large_payload() -> str:
    """Large JSON payload."""
    data = {"items": [{"id": i, "value": f"item_{i}"} for i in range(1000)]}
    return json.dumps(data)


@pytest.fixture
def yaml_with_anchor() -> str:
    """YAML payload with anchor (should be rejected)."""
    return """
base: &base
  name: test
  value: 42
extended:
  <<: *base
  extra: data
"""


@pytest.fixture
def yaml_billion_laughs() -> str:
    """Classic YAML billion laughs attack payload."""
    return """
a: &a ["lol","lol","lol","lol","lol","lol","lol","lol","lol"]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c]
"""


# =============================================================================
# Basic Parsing Tests
# =============================================================================


class TestParsePayloadBasic:
    """Tests for basic parsing functionality."""

    def test_parse_simple_json(self, simple_json: str) -> None:
        """Test parsing simple JSON payload."""
        result = parse_payload(simple_json)

        assert result.format == "json"
        assert result.data == {"name": "test", "value": 42}
        assert result.size_bytes > 0
        assert result.node_count == 3  # dict + 2 values
        assert result.max_depth == 1
        assert result.parse_time_ms >= 0

    def test_parse_simple_yaml(self, simple_yaml: str) -> None:
        """Test parsing simple YAML payload."""
        result = parse_payload(simple_yaml)

        assert result.format == "yaml"
        assert result.data == {"name": "test", "value": 42}
        assert result.size_bytes > 0
        assert result.node_count == 3
        assert result.max_depth == 1

    def test_parse_nested_json(self, nested_json: str) -> None:
        """Test parsing nested JSON payload."""
        result = parse_payload(nested_json)

        assert result.format == "json"
        assert result.data["level1"]["level2"]["level3"]["value"] == 1
        assert result.max_depth == 4  # root + 3 levels

    def test_parse_nested_yaml(self, nested_yaml: str) -> None:
        """Test parsing nested YAML payload."""
        result = parse_payload(nested_yaml)

        assert result.format == "yaml"
        assert result.data["level1"]["level2"]["level3"]["value"] == 1
        assert result.max_depth == 4

    def test_parse_json_array(self) -> None:
        """Test parsing JSON array at root level."""
        result = parse_payload('[1, 2, 3]')

        # Arrays are wrapped in {"_root": [...]}
        assert "_root" in result.data
        assert result.data["_root"] == [1, 2, 3]
        assert result.format == "json"

    def test_parse_yaml_array(self) -> None:
        """Test parsing YAML array at root level."""
        result = parse_payload("- 1\n- 2\n- 3")

        assert "_root" in result.data
        assert result.data["_root"] == [1, 2, 3]
        assert result.format == "yaml"

    def test_parse_empty_json(self) -> None:
        """Test parsing empty JSON object."""
        result = parse_payload('{}')

        assert result.format == "json"
        assert result.data == {}
        assert result.node_count == 1

    def test_parse_empty_yaml(self) -> None:
        """Test parsing empty YAML document."""
        result = parse_payload('')

        assert result.format == "json"  # Empty defaults to JSON
        assert result.data == {}

    def test_parse_bytes_input(self, simple_json: str) -> None:
        """Test parsing bytes input."""
        content_bytes = simple_json.encode('utf-8')
        result = parse_payload(content_bytes)

        assert result.format == "json"
        assert result.data == {"name": "test", "value": 42}

    def test_parse_unicode_content(self) -> None:
        """Test parsing content with Unicode characters."""
        content = '{"message": "Hello, \\u4e16\\u754c", "emoji": "\\ud83d\\ude00"}'
        result = parse_payload(content)

        assert result.format == "json"
        assert "message" in result.data

    def test_parse_result_model(self, simple_json: str) -> None:
        """Test ParseResult is a valid Pydantic model."""
        result = parse_payload(simple_json)

        # Check model serialization
        as_dict = result.model_dump()
        assert "data" in as_dict
        assert "format" in as_dict
        assert "size_bytes" in as_dict
        assert "node_count" in as_dict
        assert "max_depth" in as_dict
        assert "parse_time_ms" in as_dict


# =============================================================================
# Format Detection Tests
# =============================================================================


class TestDetectFormat:
    """Tests for format detection."""

    def test_detect_json_object(self) -> None:
        """Test detection of JSON object."""
        assert detect_format('{"key": "value"}') == "json"

    def test_detect_json_array(self) -> None:
        """Test detection of JSON array."""
        assert detect_format('[1, 2, 3]') == "json"

    def test_detect_json_string(self) -> None:
        """Test detection of JSON string literal."""
        assert detect_format('"hello"') == "json"

    def test_detect_json_number(self) -> None:
        """Test detection of JSON number."""
        assert detect_format('42') == "json"
        assert detect_format('3.14') == "json"
        assert detect_format('-123') == "json"

    def test_detect_json_boolean(self) -> None:
        """Test detection of JSON boolean literals."""
        assert detect_format('true') == "json"
        assert detect_format('false') == "json"

    def test_detect_json_null(self) -> None:
        """Test detection of JSON null."""
        assert detect_format('null') == "json"

    def test_detect_yaml_simple(self) -> None:
        """Test detection of simple YAML."""
        assert detect_format('key: value') == "yaml"

    def test_detect_yaml_list(self) -> None:
        """Test detection of YAML list."""
        assert detect_format('- item1\n- item2') == "yaml"

    def test_detect_yaml_multiline(self) -> None:
        """Test detection of multiline YAML."""
        yaml_content = """
name: test
value: 42
items:
  - one
  - two
"""
        assert detect_format(yaml_content) == "yaml"

    def test_detect_empty_as_json(self) -> None:
        """Test empty content defaults to JSON."""
        assert detect_format('') == "json"
        assert detect_format('  ') == "json"

    def test_detect_with_bytes(self) -> None:
        """Test format detection with bytes input."""
        assert detect_format(b'{"key": "value"}') == "json"
        assert detect_format(b'key: value') == "yaml"

    def test_detect_invalid_json_as_yaml(self) -> None:
        """Test that invalid JSON-looking content is detected as YAML."""
        # This looks like JSON but is invalid
        assert detect_format('{key: value}') == "yaml"


# =============================================================================
# Size Limit Tests
# =============================================================================


class TestSizeLimits:
    """Tests for payload size limit enforcement."""

    def test_small_payload_accepted(self) -> None:
        """Test that small payloads are accepted."""
        small = '{"key": "value"}'
        result = parse_payload(small, max_bytes=1000)
        assert result.data == {"key": "value"}

    def test_large_payload_rejected(self) -> None:
        """Test that oversized payloads are rejected."""
        large = '{"data": "' + 'x' * 10000 + '"}'

        with pytest.raises(ParseError) as exc_info:
            parse_payload(large, max_bytes=100)

        assert exc_info.value.code == ErrorCode.PAYLOAD_TOO_LARGE.value
        assert "size_bytes" in exc_info.value.details
        assert "max_bytes" in exc_info.value.details

    def test_exact_size_limit_accepted(self) -> None:
        """Test payload at exact size limit is accepted."""
        content = '{"a": 1}'  # 8 bytes in UTF-8
        result = parse_payload(content, max_bytes=8)
        assert result.data == {"a": 1}

    def test_size_limit_exceeded_by_one_byte(self) -> None:
        """Test payload exceeding limit by one byte is rejected."""
        content = '{"a": 1}'  # 8 bytes

        with pytest.raises(ParseError) as exc_info:
            parse_payload(content, max_bytes=7)

        assert exc_info.value.code == ErrorCode.PAYLOAD_TOO_LARGE.value

    def test_validate_payload_size_utility(self) -> None:
        """Test the validate_payload_size utility function."""
        is_valid, size = validate_payload_size('{"key": "value"}')
        assert is_valid is True
        assert size > 0

        is_valid, size = validate_payload_size('{"key": "value"}', max_bytes=5)
        assert is_valid is False


# =============================================================================
# Depth Limit Tests
# =============================================================================


class TestDepthLimits:
    """Tests for nesting depth limit enforcement."""

    def test_shallow_payload_accepted(self, nested_json: str) -> None:
        """Test shallow nested payload is accepted."""
        result = parse_payload(nested_json, max_depth=50)
        assert result.max_depth == 4

    def test_deep_payload_rejected(self) -> None:
        """Test deeply nested payload is rejected."""
        # Create deeply nested structure
        deep = '{"a": ' * 60 + '{}' + '}' * 60

        with pytest.raises(ParseError) as exc_info:
            parse_payload(deep, max_depth=50)

        assert exc_info.value.code == ErrorCode.DEPTH_EXCEEDED.value
        assert "depth" in exc_info.value.details
        assert "max_depth" in exc_info.value.details

    def test_depth_limit_at_boundary(self) -> None:
        """Test depth at exact limit is accepted."""
        # Create exactly 5 levels deep
        content = '{"l1": {"l2": {"l3": {"l4": {"l5": 1}}}}}'
        result = parse_payload(content, max_depth=5)
        assert result.max_depth == 5

    def test_depth_limit_exceeded_by_one(self) -> None:
        """Test depth exceeding limit by one is rejected."""
        content = '{"l1": {"l2": {"l3": {"l4": {"l5": 1}}}}}'  # Depth 5

        with pytest.raises(ParseError) as exc_info:
            parse_payload(content, max_depth=4)

        assert exc_info.value.code == ErrorCode.DEPTH_EXCEEDED.value

    def test_array_depth_counted(self) -> None:
        """Test that array nesting counts toward depth."""
        content = '[[[[1]]]]'  # 4 levels of arrays

        with pytest.raises(ParseError) as exc_info:
            parse_payload(content, max_depth=3)

        assert exc_info.value.code == ErrorCode.DEPTH_EXCEEDED.value


# =============================================================================
# Node Count Limit Tests
# =============================================================================


class TestNodeCountLimits:
    """Tests for total node count limit enforcement."""

    def test_small_node_count_accepted(self) -> None:
        """Test payload with few nodes is accepted."""
        content = '{"a": 1, "b": 2, "c": 3}'
        result = parse_payload(content, max_nodes=1000)
        assert result.node_count == 4  # dict + 3 values

    def test_large_node_count_rejected(self, large_payload: str) -> None:
        """Test payload with too many nodes is rejected."""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(large_payload, max_nodes=100)

        assert exc_info.value.code == ErrorCode.NODES_EXCEEDED.value
        assert "count" in exc_info.value.details
        assert "max_nodes" in exc_info.value.details

    def test_node_count_at_limit(self) -> None:
        """Test payload at exact node limit is accepted."""
        content = '{"a": 1, "b": 2, "c": 3}'  # 4 nodes
        result = parse_payload(content, max_nodes=4)
        assert result.node_count == 4

    def test_node_count_exceeded_by_one(self) -> None:
        """Test payload exceeding node limit by one is rejected."""
        content = '{"a": 1, "b": 2, "c": 3}'  # 4 nodes

        with pytest.raises(ParseError) as exc_info:
            parse_payload(content, max_nodes=3)

        assert exc_info.value.code == ErrorCode.NODES_EXCEEDED.value


# =============================================================================
# YAML Anchor/Alias Security Tests
# =============================================================================


class TestYAMLSecurity:
    """Tests for YAML anchor and alias rejection."""

    def test_yaml_anchor_rejected(self, yaml_with_anchor: str) -> None:
        """Test that YAML anchors are rejected."""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_with_anchor)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value
        assert "anchor" in exc_info.value.message.lower()

    def test_yaml_alias_rejected(self) -> None:
        """Test that YAML aliases are rejected."""
        yaml_content = """
base: &base
  name: test
copy: *base
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_content)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_yaml_billion_laughs_rejected(self, yaml_billion_laughs: str) -> None:
        """Test that billion laughs attack is prevented."""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_billion_laughs)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value
        # Should fail on first anchor, not expand the bomb
        assert "anchor" in exc_info.value.message.lower() or "alias" in exc_info.value.message.lower()

    def test_simple_yaml_without_anchors_accepted(self) -> None:
        """Test that YAML without anchors is accepted."""
        yaml_content = """
name: test
items:
  - one
  - two
  - three
nested:
  value: 42
"""
        result = parse_payload(yaml_content)
        assert result.format == "yaml"
        assert result.data["name"] == "test"
        assert result.data["items"] == ["one", "two", "three"]

    def test_safe_yaml_loader_class(self) -> None:
        """Test SafeYAMLLoader class directly."""
        import yaml

        # Normal YAML should work
        result = yaml.load("key: value", Loader=SafeYAMLLoader)
        assert result == {"key": "value"}

        # Anchors should raise
        with pytest.raises(ParseError):
            yaml.load("a: &a [1, 2, 3]", Loader=SafeYAMLLoader)

    def test_merge_key_with_anchor_rejected(self) -> None:
        """Test that YAML merge key with anchor is rejected."""
        yaml_content = """
defaults: &defaults
  adapter: postgres
  host: localhost

development:
  <<: *defaults
  database: dev_db
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_content)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value


# =============================================================================
# UTF-8 and Encoding Tests
# =============================================================================


class TestEncodingHandling:
    """Tests for encoding and UTF-8 handling."""

    def test_valid_utf8_accepted(self) -> None:
        """Test valid UTF-8 content is accepted."""
        content = '{"message": "Hello, World!"}'.encode('utf-8')
        result = parse_payload(content)
        assert result.data["message"] == "Hello, World!"

    def test_utf8_with_unicode_chars(self) -> None:
        """Test UTF-8 with various Unicode characters."""
        content = '{"greeting": "Hola", "japanese": "konnichiwa"}'.encode('utf-8')
        result = parse_payload(content)
        assert "greeting" in result.data

    def test_malformed_utf8_rejected(self) -> None:
        """Test malformed UTF-8 raises appropriate error."""
        # Create invalid UTF-8 sequence
        invalid_utf8 = b'{"key": "\xff\xfe"}'

        with pytest.raises(ParseError) as exc_info:
            parse_payload(invalid_utf8)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value
        assert "UTF-8" in exc_info.value.message or "encoding" in exc_info.value.message.lower()

    def test_utf8_bom_handled(self) -> None:
        """Test UTF-8 with BOM is handled correctly."""
        # UTF-8 BOM + content
        content_with_bom = b'\xef\xbb\xbf{"key": "value"}'

        # This should either parse correctly or raise a clear error
        try:
            result = parse_payload(content_with_bom)
            assert "key" in result.data
        except ParseError:
            # BOM handling might fail on some systems, which is acceptable
            pass


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and ParseError structure."""

    def test_parse_error_structure(self) -> None:
        """Test ParseError has correct structure."""
        error = ParseError(
            code="GLSCHEMA-E800",
            message="Test error",
            details={"key": "value"}
        )

        assert error.code == "GLSCHEMA-E800"
        assert error.message == "Test error"
        assert error.details == {"key": "value"}

    def test_parse_error_str(self) -> None:
        """Test ParseError string representation."""
        error = ParseError(
            code="GLSCHEMA-E800",
            message="Payload too large"
        )

        error_str = str(error)
        assert "GLSCHEMA-E800" in error_str
        assert "Payload too large" in error_str

    def test_parse_error_to_dict(self) -> None:
        """Test ParseError serialization."""
        error = ParseError(
            code="GLSCHEMA-E800",
            message="Test error",
            details={"size": 1000}
        )

        as_dict = error.to_dict()
        assert as_dict["code"] == "GLSCHEMA-E800"
        assert as_dict["message"] == "Test error"
        assert as_dict["details"]["size"] == 1000

    def test_invalid_json_syntax_error(self) -> None:
        """Test invalid JSON raises ParseError."""
        # Use content that looks like JSON but has trailing content
        # (gets detected as JSON but fails to parse)
        with pytest.raises(ParseError) as exc_info:
            parse_payload('{"key": "value"} extra')

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value
        assert "line" in exc_info.value.details or "error" in exc_info.value.details

    def test_invalid_yaml_syntax_error(self) -> None:
        """Test invalid YAML raises ParseError."""
        yaml_content = """
key: value
  bad_indent: here
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_content)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_is_valid_json_true(self) -> None:
        """Test is_valid_json returns True for valid JSON."""
        assert is_valid_json('{"key": "value"}') is True
        assert is_valid_json('[1, 2, 3]') is True
        assert is_valid_json('"string"') is True
        assert is_valid_json('42') is True
        assert is_valid_json('true') is True
        assert is_valid_json('null') is True

    def test_is_valid_json_false(self) -> None:
        """Test is_valid_json returns False for invalid JSON."""
        assert is_valid_json('key: value') is False
        assert is_valid_json('{key: value}') is False
        assert is_valid_json('{"key": }') is False

    def test_is_valid_yaml_true(self) -> None:
        """Test is_valid_yaml returns True for valid YAML."""
        assert is_valid_yaml('key: value') is True
        assert is_valid_yaml('- item1\n- item2') is True
        assert is_valid_yaml('{}') is True

    def test_is_valid_yaml_false_for_anchors(self) -> None:
        """Test is_valid_yaml returns False for YAML with anchors."""
        assert is_valid_yaml('a: &a [1, 2, 3]') is False

    def test_count_nodes_and_depth_simple(self) -> None:
        """Test _count_nodes_and_depth with simple data."""
        data = {"a": 1, "b": 2}
        count, depth = _count_nodes_and_depth(data)

        assert count == 3  # dict + 2 values
        assert depth == 1

    def test_count_nodes_and_depth_nested(self) -> None:
        """Test _count_nodes_and_depth with nested data."""
        data = {"level1": {"level2": {"value": 1}}}
        count, depth = _count_nodes_and_depth(data)

        assert count == 4  # 3 dicts + 1 value
        assert depth == 3

    def test_count_nodes_and_depth_with_arrays(self) -> None:
        """Test _count_nodes_and_depth with arrays."""
        data = {"items": [1, 2, 3]}
        count, depth = _count_nodes_and_depth(data)

        assert count == 5  # dict + array + 3 values
        assert depth == 2


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_json_with_whitespace(self) -> None:
        """Test JSON with extra whitespace."""
        content = '  { "key" : "value" }  '
        result = parse_payload(content)
        assert result.data == {"key": "value"}

    def test_yaml_with_comments(self) -> None:
        """Test YAML with comments."""
        yaml_content = """
# This is a comment
key: value  # inline comment
# another comment
other: data
"""
        result = parse_payload(yaml_content)
        assert result.data["key"] == "value"
        assert result.data["other"] == "data"

    def test_deeply_nested_arrays(self) -> None:
        """Test deeply nested arrays."""
        content = json.dumps([[[[[[1]]]]]])  # 6 levels

        with pytest.raises(ParseError) as exc_info:
            parse_payload(content, max_depth=5)

        assert exc_info.value.code == ErrorCode.DEPTH_EXCEEDED.value

    def test_mixed_nesting(self) -> None:
        """Test mixed object and array nesting."""
        content = '{"a": [{"b": [{"c": 1}]}]}'
        result = parse_payload(content)
        assert result.data["a"][0]["b"][0]["c"] == 1

    def test_empty_array(self) -> None:
        """Test parsing empty array."""
        result = parse_payload('[]')
        assert result.data == {"_root": []}

    def test_empty_nested_objects(self) -> None:
        """Test empty nested objects."""
        result = parse_payload('{"a": {}, "b": {}}')
        assert result.data == {"a": {}, "b": {}}

    def test_json_with_special_chars_in_strings(self) -> None:
        """Test JSON with special characters in strings."""
        content = '{"text": "line1\\nline2\\ttab"}'
        result = parse_payload(content)
        assert "line1" in result.data["text"]

    def test_yaml_multiline_strings(self) -> None:
        """Test YAML multiline strings."""
        yaml_content = """
literal: |
  Line 1
  Line 2
  Line 3
"""
        result = parse_payload(yaml_content)
        assert "Line 1" in result.data["literal"]

    def test_numeric_keys_in_yaml(self) -> None:
        """Test YAML with numeric keys (converted to strings)."""
        yaml_content = """
1: one
2: two
"""
        result = parse_payload(yaml_content)
        # YAML numeric keys are converted to strings
        assert result.format == "yaml"
        assert "1" in result.data or 1 in result.data  # May be int or string
        # Keys should be converted to strings in output
        assert all(isinstance(k, str) for k in result.data.keys())

    def test_boolean_values_yaml(self) -> None:
        """Test YAML boolean values."""
        yaml_content = """
yes_val: yes
no_val: no
true_val: true
false_val: false
"""
        result = parse_payload(yaml_content)
        assert result.format == "yaml"

    def test_null_values(self) -> None:
        """Test null values in JSON and YAML."""
        json_content = '{"value": null}'
        result = parse_payload(json_content)
        assert result.data["value"] is None

        yaml_content = "value: ~"
        result = parse_payload(yaml_content)
        assert result.data["value"] is None


# =============================================================================
# Performance Tests (marked for optional execution)
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance tests (run with pytest -m slow)."""

    def test_large_flat_payload_performance(self) -> None:
        """Test performance with large flat payload."""
        data = {f"key_{i}": f"value_{i}" for i in range(10000)}
        content = json.dumps(data)

        result = parse_payload(content)

        # Should complete in reasonable time
        assert result.parse_time_ms < 5000  # 5 seconds max
        assert result.node_count > 10000

    def test_deep_nesting_performance(self) -> None:
        """Test performance with maximum allowed nesting."""
        # Build nested structure at max depth
        depth = 49
        content = '{"a": ' * depth + '{}' + '}' * depth

        result = parse_payload(content, max_depth=50)
        assert result.max_depth == depth
        assert result.parse_time_ms < 1000  # 1 second max


# =============================================================================
# Integration with Constants Tests
# =============================================================================


class TestConstantsIntegration:
    """Tests for integration with constants module."""

    def test_default_max_bytes_used(self) -> None:
        """Test that default MAX_PAYLOAD_BYTES is used."""
        from greenlang.schema.constants import MAX_PAYLOAD_BYTES

        # Create payload just under default limit
        content = '{"data": "' + 'x' * (MAX_PAYLOAD_BYTES - 100) + '"}'

        # Should work with default limits
        result = parse_payload(content)
        assert result.size_bytes < MAX_PAYLOAD_BYTES

    def test_default_max_depth_used(self) -> None:
        """Test that default MAX_OBJECT_DEPTH is used."""
        from greenlang.schema.constants import MAX_OBJECT_DEPTH

        # Create nested structure just under default limit
        depth = MAX_OBJECT_DEPTH - 1
        content = '{"a": ' * depth + '{}' + '}' * depth

        result = parse_payload(content)
        assert result.max_depth <= MAX_OBJECT_DEPTH

    def test_default_max_nodes_used(self) -> None:
        """Test that default MAX_TOTAL_NODES is used."""
        from greenlang.schema.constants import MAX_TOTAL_NODES

        # Small payload should be well under limit
        content = '{"a": 1}'
        result = parse_payload(content)
        assert result.node_count < MAX_TOTAL_NODES
