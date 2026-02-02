# -*- coding: utf-8 -*-
"""
Unit tests for GL-FOUND-X-002 Schema Resolver.

This module tests the RefResolver class and related functionality including:
- JSON Pointer (RFC 6901) parsing and navigation
- Local reference resolution (#/definitions/Foo)
- External reference resolution (gl://schemas/...)
- Circular reference detection
- Expansion limit enforcement
- Reference caching
- LocalFileRegistry for development

Test Categories:
- JSON Pointer tests: Parsing and navigation
- Local ref tests: Within-document resolution
- External ref tests: Cross-document resolution
- Cycle detection tests: Circular reference handling
- Limit tests: Expansion limit enforcement
- Cache tests: Caching behavior
- Registry tests: LocalFileRegistry functionality

Author: GreenLang Team
Date: 2026-01-29
"""

import json
import pytest
from pathlib import Path
from typing import Any, Dict

from greenlang.schema.compiler.resolver import (
    RefResolver,
    LocalFileRegistry,
    SchemaSource,
    CircularRefError,
    RefResolutionError,
    MaxExpansionsExceededError,
    parse_json_pointer,
    navigate_json_pointer,
    parse_ref,
    resolve_all_refs,
    validate_ref_format,
    RefType,
    ParsedRef,
)


# ============================================================================
# JSON POINTER TESTS (RFC 6901)
# ============================================================================


class TestParseJsonPointer:
    """Tests for JSON Pointer parsing (RFC 6901)."""

    def test_empty_pointer_returns_empty_list(self):
        """Empty pointer should return empty list (refers to root)."""
        assert parse_json_pointer("") == []

    def test_single_segment_pointer(self):
        """Single segment pointer should return single element list."""
        assert parse_json_pointer("/foo") == ["foo"]

    def test_multi_segment_pointer(self):
        """Multi-segment pointer should return correct segments."""
        assert parse_json_pointer("/definitions/Foo") == ["definitions", "Foo"]
        assert parse_json_pointer("/a/b/c/d") == ["a", "b", "c", "d"]

    def test_pointer_with_escaped_tilde(self):
        """~0 should be unescaped to ~."""
        assert parse_json_pointer("/a~0b") == ["a~b"]
        assert parse_json_pointer("/~0") == ["~"]

    def test_pointer_with_escaped_slash(self):
        """~1 should be unescaped to /."""
        assert parse_json_pointer("/a~1b") == ["a/b"]
        assert parse_json_pointer("/~1") == ["/"]

    def test_pointer_with_multiple_escapes(self):
        """Multiple escape sequences should be handled correctly."""
        assert parse_json_pointer("/a~1b/c~0d") == ["a/b", "c~d"]
        assert parse_json_pointer("/~0~1") == ["~/"]

    def test_pointer_with_empty_segment(self):
        """Empty segment should be allowed."""
        assert parse_json_pointer("/a//b") == ["a", "", "b"]

    def test_pointer_with_numeric_index(self):
        """Numeric segments should be strings."""
        assert parse_json_pointer("/items/0") == ["items", "0"]
        assert parse_json_pointer("/0/1/2") == ["0", "1", "2"]

    def test_invalid_pointer_without_leading_slash(self):
        """Pointer without leading / should raise ValueError."""
        with pytest.raises(ValueError, match="must start with"):
            parse_json_pointer("definitions/Foo")

    def test_pointer_with_special_characters(self):
        """Pointers with special characters should work."""
        assert parse_json_pointer("/key with spaces") == ["key with spaces"]
        assert parse_json_pointer("/$defs") == ["$defs"]
        assert parse_json_pointer("/type") == ["type"]


class TestNavigateJsonPointer:
    """Tests for JSON Pointer navigation."""

    def test_navigate_empty_pointer(self):
        """Empty pointer should return the document itself."""
        doc = {"foo": "bar"}
        assert navigate_json_pointer(doc, "") == doc

    def test_navigate_root_slash(self):
        """Root slash should return the document."""
        doc = {"foo": "bar"}
        assert navigate_json_pointer(doc, "/") == doc

    def test_navigate_single_level(self):
        """Single level navigation."""
        doc = {"definitions": {"Foo": {"type": "string"}}}
        assert navigate_json_pointer(doc, "/definitions") == {"Foo": {"type": "string"}}

    def test_navigate_multi_level(self):
        """Multi-level navigation."""
        doc = {"definitions": {"Foo": {"type": "string"}}}
        assert navigate_json_pointer(doc, "/definitions/Foo") == {"type": "string"}
        assert navigate_json_pointer(doc, "/definitions/Foo/type") == "string"

    def test_navigate_array_index(self):
        """Navigate to array index."""
        doc = {"items": ["a", "b", "c"]}
        assert navigate_json_pointer(doc, "/items/0") == "a"
        assert navigate_json_pointer(doc, "/items/1") == "b"
        assert navigate_json_pointer(doc, "/items/2") == "c"

    def test_navigate_nested_array(self):
        """Navigate through nested arrays."""
        doc = {"matrix": [[1, 2], [3, 4]]}
        assert navigate_json_pointer(doc, "/matrix/0/0") == 1
        assert navigate_json_pointer(doc, "/matrix/1/1") == 4

    def test_navigate_not_found_raises_error(self):
        """Missing path segment should raise RefResolutionError."""
        doc = {"foo": "bar"}
        with pytest.raises(RefResolutionError, match="not found"):
            navigate_json_pointer(doc, "/missing")

    def test_navigate_array_index_out_of_bounds(self):
        """Out of bounds array index should raise RefResolutionError."""
        doc = {"items": ["a", "b"]}
        with pytest.raises(RefResolutionError, match="out of bounds"):
            navigate_json_pointer(doc, "/items/10")

    def test_navigate_invalid_array_index(self):
        """Non-numeric array index should raise RefResolutionError."""
        doc = {"items": ["a", "b"]}
        with pytest.raises(RefResolutionError, match="Invalid array index"):
            navigate_json_pointer(doc, "/items/foo")

    def test_navigate_into_primitive_raises_error(self):
        """Navigating into a primitive value should raise error."""
        doc = {"value": 42}
        with pytest.raises(RefResolutionError, match="Cannot navigate"):
            navigate_json_pointer(doc, "/value/child")


# ============================================================================
# REF PARSING TESTS
# ============================================================================


class TestParseRef:
    """Tests for $ref value parsing."""

    def test_parse_local_ref_with_hash(self):
        """Parse local refs starting with #."""
        parsed = parse_ref("#/definitions/Foo")
        assert parsed.ref_type == RefType.LOCAL
        assert parsed.fragment == "/definitions/Foo"
        assert parsed.original == "#/definitions/Foo"

    def test_parse_local_ref_empty_fragment(self):
        """Parse local ref with empty fragment (refers to root)."""
        parsed = parse_ref("#")
        assert parsed.ref_type == RefType.LOCAL
        assert parsed.fragment == ""

    def test_parse_local_ref_defs(self):
        """Parse $defs style local refs."""
        parsed = parse_ref("#/$defs/Bar")
        assert parsed.ref_type == RefType.LOCAL
        assert parsed.fragment == "/$defs/Bar"

    def test_parse_gl_uri_with_version(self):
        """Parse gl:// URIs with version."""
        parsed = parse_ref("gl://schemas/emissions/activity@1.3.0")
        assert parsed.ref_type == RefType.EXTERNAL_GL
        assert parsed.schema_id == "emissions/activity"
        assert parsed.version == "1.3.0"
        assert parsed.fragment is None

    def test_parse_gl_uri_with_fragment(self):
        """Parse gl:// URIs with fragment."""
        parsed = parse_ref("gl://schemas/emissions/activity@1.3.0#/definitions/Fuel")
        assert parsed.ref_type == RefType.EXTERNAL_GL
        assert parsed.schema_id == "emissions/activity"
        assert parsed.version == "1.3.0"
        assert parsed.fragment == "/definitions/Fuel"

    def test_parse_http_uri(self):
        """Parse HTTP URIs (not supported but valid)."""
        parsed = parse_ref("https://example.com/schema.json")
        assert parsed.ref_type == RefType.EXTERNAL_HTTP
        assert parsed.uri == "https://example.com/schema.json"

    def test_parse_http_uri_with_fragment(self):
        """Parse HTTP URIs with fragment."""
        parsed = parse_ref("https://example.com/schema.json#/definitions/Foo")
        assert parsed.ref_type == RefType.EXTERNAL_HTTP
        assert parsed.fragment == "/definitions/Foo"

    def test_parse_relative_ref(self):
        """Parse relative refs (not supported but valid)."""
        parsed = parse_ref("./other-schema.json")
        assert parsed.ref_type == RefType.RELATIVE
        assert parsed.uri == "./other-schema.json"


class TestValidateRefFormat:
    """Tests for ref format validation."""

    def test_valid_local_ref(self):
        """Valid local refs should pass."""
        is_valid, error = validate_ref_format("#/definitions/Foo")
        assert is_valid
        assert error is None

    def test_valid_gl_uri(self):
        """Valid gl:// URIs should pass."""
        is_valid, error = validate_ref_format("gl://schemas/emissions/activity@1.3.0")
        assert is_valid
        assert error is None

    def test_empty_ref_invalid(self):
        """Empty ref should be invalid."""
        is_valid, error = validate_ref_format("")
        assert not is_valid
        assert "Empty" in error


# ============================================================================
# LOCAL REF RESOLUTION TESTS
# ============================================================================


class TestLocalRefResolution:
    """Tests for local $ref resolution."""

    @pytest.fixture
    def resolver(self):
        """Create a RefResolver instance."""
        return RefResolver()

    @pytest.fixture
    def schema_with_definitions(self) -> Dict[str, Any]:
        """Create a schema with definitions."""
        return {
            "type": "object",
            "properties": {
                "activity": {"$ref": "#/definitions/Activity"}
            },
            "definitions": {
                "Activity": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "value": {"type": "number"}
                    }
                },
                "Unit": {
                    "type": "string",
                    "enum": ["kWh", "kg", "m3"]
                }
            }
        }

    @pytest.fixture
    def schema_with_defs(self) -> Dict[str, Any]:
        """Create a schema with $defs (draft 2019-09 style)."""
        return {
            "type": "object",
            "properties": {
                "item": {"$ref": "#/$defs/Item"}
            },
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }

    def test_resolve_simple_local_ref(self, resolver, schema_with_definitions):
        """Resolve a simple local ref to definitions."""
        result = resolver.resolve(
            ref="#/definitions/Activity",
            context_document=schema_with_definitions,
            context_path=""
        )
        assert result["type"] == "object"
        assert "type" in result["properties"]
        assert "value" in result["properties"]

    def test_resolve_defs_style_ref(self, resolver, schema_with_defs):
        """Resolve a $defs style local ref."""
        result = resolver.resolve(
            ref="#/$defs/Item",
            context_document=schema_with_defs,
            context_path=""
        )
        assert result["type"] == "object"
        assert "name" in result["properties"]

    def test_resolve_root_ref(self, resolver):
        """Resolve ref to root (#)."""
        doc = {"type": "object"}
        result = resolver.resolve(
            ref="#",
            context_document=doc,
            context_path=""
        )
        assert result == doc

    def test_resolve_nested_path(self, resolver, schema_with_definitions):
        """Resolve ref to nested path."""
        result = resolver.resolve(
            ref="#/definitions/Activity/properties/type",
            context_document=schema_with_definitions,
            context_path=""
        )
        assert result == {"type": "string"}

    def test_resolve_missing_ref_raises_error(self, resolver, schema_with_definitions):
        """Missing ref target should raise RefResolutionError."""
        with pytest.raises(RefResolutionError, match="not found"):
            resolver.resolve(
                ref="#/definitions/MissingDefinition",
                context_document=schema_with_definitions,
                context_path=""
            )

    def test_resolve_caches_result(self, resolver, schema_with_definitions):
        """Resolved refs should be cached."""
        # First resolution
        result1 = resolver.resolve(
            ref="#/definitions/Activity",
            context_document=schema_with_definitions,
            context_path=""
        )

        # Second resolution (should be cached)
        result2 = resolver.resolve(
            ref="#/definitions/Activity",
            context_document=schema_with_definitions,
            context_path=""
        )

        assert result1 is result2  # Same object due to caching
        assert len(resolver._cache) > 0


# ============================================================================
# NESTED REF RESOLUTION TESTS
# ============================================================================


class TestNestedRefResolution:
    """Tests for nested $ref resolution (refs pointing to refs)."""

    @pytest.fixture
    def resolver(self):
        """Create a RefResolver instance."""
        return RefResolver()

    @pytest.fixture
    def schema_with_nested_refs(self) -> Dict[str, Any]:
        """Create a schema with nested refs."""
        return {
            "type": "object",
            "definitions": {
                "A": {"$ref": "#/definitions/B"},
                "B": {"$ref": "#/definitions/C"},
                "C": {"type": "string"}
            }
        }

    def test_resolve_nested_ref_chain(self, resolver, schema_with_nested_refs):
        """Nested refs should be followed to the final definition."""
        result = resolver.resolve(
            ref="#/definitions/A",
            context_document=schema_with_nested_refs,
            context_path=""
        )
        assert result == {"type": "string"}

    def test_nested_refs_increment_counter(self, resolver, schema_with_nested_refs):
        """Each ref in a chain should increment the expansion counter."""
        resolver.resolve(
            ref="#/definitions/A",
            context_document=schema_with_nested_refs,
            context_path=""
        )
        # A -> B -> C = 3 expansions
        assert resolver._expansion_count == 3


# ============================================================================
# CIRCULAR REFERENCE DETECTION TESTS
# ============================================================================


class TestCircularReferenceDetection:
    """Tests for circular reference detection."""

    @pytest.fixture
    def resolver(self):
        """Create a RefResolver instance."""
        return RefResolver()

    @pytest.fixture
    def schema_with_direct_cycle(self) -> Dict[str, Any]:
        """Schema with direct self-reference."""
        return {
            "definitions": {
                "A": {"$ref": "#/definitions/A"}
            }
        }

    @pytest.fixture
    def schema_with_indirect_cycle(self) -> Dict[str, Any]:
        """Schema with indirect cycle (A -> B -> A)."""
        return {
            "definitions": {
                "A": {"$ref": "#/definitions/B"},
                "B": {"$ref": "#/definitions/A"}
            }
        }

    @pytest.fixture
    def schema_with_long_cycle(self) -> Dict[str, Any]:
        """Schema with longer cycle (A -> B -> C -> A)."""
        return {
            "definitions": {
                "A": {"$ref": "#/definitions/B"},
                "B": {"$ref": "#/definitions/C"},
                "C": {"$ref": "#/definitions/A"}
            }
        }

    def test_detect_direct_cycle(self, resolver, schema_with_direct_cycle):
        """Direct self-reference should be detected."""
        with pytest.raises(CircularRefError) as exc_info:
            resolver.resolve(
                ref="#/definitions/A",
                context_document=schema_with_direct_cycle,
                context_path=""
            )
        assert "#/definitions/A" in exc_info.value.cycle
        assert len(exc_info.value.cycle) == 2  # [A, A]

    def test_detect_indirect_cycle(self, resolver, schema_with_indirect_cycle):
        """Indirect cycle (A -> B -> A) should be detected."""
        with pytest.raises(CircularRefError) as exc_info:
            resolver.resolve(
                ref="#/definitions/A",
                context_document=schema_with_indirect_cycle,
                context_path=""
            )
        cycle = exc_info.value.cycle
        assert "#/definitions/A" in cycle
        assert "#/definitions/B" in cycle

    def test_detect_long_cycle(self, resolver, schema_with_long_cycle):
        """Longer cycle (A -> B -> C -> A) should be detected."""
        with pytest.raises(CircularRefError) as exc_info:
            resolver.resolve(
                ref="#/definitions/A",
                context_document=schema_with_long_cycle,
                context_path=""
            )
        cycle = exc_info.value.cycle
        assert len(cycle) == 4  # [A, B, C, A]

    def test_cycle_error_message_contains_trace(self, resolver, schema_with_indirect_cycle):
        """Cycle error should contain readable trace."""
        with pytest.raises(CircularRefError) as exc_info:
            resolver.resolve(
                ref="#/definitions/A",
                context_document=schema_with_indirect_cycle,
                context_path=""
            )
        assert " -> " in str(exc_info.value)


# ============================================================================
# EXPANSION LIMIT TESTS
# ============================================================================


class TestExpansionLimits:
    """Tests for ref expansion limit enforcement."""

    @pytest.fixture
    def deep_schema(self) -> Dict[str, Any]:
        """Create a schema with a deep ref chain."""
        # Create A -> B -> C -> ... -> Z
        schema = {"definitions": {}}
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, letter in enumerate(letters[:-1]):
            schema["definitions"][letter] = {"$ref": f"#/definitions/{letters[i+1]}"}
        schema["definitions"]["Z"] = {"type": "string"}
        return schema

    def test_default_limit_allows_normal_schemas(self, deep_schema):
        """Default limit should allow normal ref chains."""
        resolver = RefResolver(max_expansions=100)
        result = resolver.resolve(
            ref="#/definitions/A",
            context_document=deep_schema,
            context_path=""
        )
        assert result == {"type": "string"}

    def test_low_limit_blocks_deep_chains(self, deep_schema):
        """Low expansion limit should block deep chains."""
        resolver = RefResolver(max_expansions=5)
        with pytest.raises(MaxExpansionsExceededError) as exc_info:
            resolver.resolve(
                ref="#/definitions/A",
                context_document=deep_schema,
                context_path=""
            )
        assert exc_info.value.count >= 5
        assert exc_info.value.max_expansions == 5

    def test_reset_clears_expansion_count(self, deep_schema):
        """Reset should clear the expansion count."""
        resolver = RefResolver(max_expansions=100)
        resolver.resolve(
            ref="#/definitions/A",
            context_document=deep_schema,
            context_path=""
        )
        assert resolver._expansion_count > 0

        resolver.reset()
        assert resolver._expansion_count == 0


# ============================================================================
# RESOLVER RESET TESTS
# ============================================================================


class TestResolverReset:
    """Tests for resolver state reset."""

    def test_reset_clears_cache(self):
        """Reset should clear the resolution cache."""
        resolver = RefResolver()
        doc = {"definitions": {"A": {"type": "string"}}}

        resolver.resolve("#/definitions/A", doc, "")
        assert len(resolver._cache) > 0

        resolver.reset()
        assert len(resolver._cache) == 0

    def test_reset_clears_stack(self):
        """Reset should clear the resolution stack."""
        resolver = RefResolver()
        # Manually add to stack to simulate mid-resolution state
        resolver._resolution_stack.append("#/definitions/A")

        resolver.reset()
        assert len(resolver._resolution_stack) == 0

    def test_reset_clears_external_documents(self):
        """Reset should clear external document cache."""
        resolver = RefResolver()
        resolver._external_documents["test@1.0"] = {"type": "object"}

        resolver.reset()
        assert len(resolver._external_documents) == 0


# ============================================================================
# RESOLVER STATS TESTS
# ============================================================================


class TestResolverStats:
    """Tests for resolver statistics."""

    def test_get_stats_returns_metrics(self):
        """get_stats should return resolver metrics."""
        resolver = RefResolver(max_expansions=1000)
        doc = {"definitions": {"A": {"type": "string"}}}

        resolver.resolve("#/definitions/A", doc, "")

        stats = resolver.get_stats()
        assert "expansion_count" in stats
        assert "max_expansions" in stats
        assert "cache_size" in stats
        assert stats["expansion_count"] == 1
        assert stats["max_expansions"] == 1000
        assert stats["cache_size"] > 0


# ============================================================================
# LOCAL FILE REGISTRY TESTS
# ============================================================================


class TestLocalFileRegistry:
    """Tests for LocalFileRegistry."""

    @pytest.fixture
    def temp_schema_dir(self, tmp_path) -> Path:
        """Create a temporary schema directory."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        return schema_dir

    @pytest.fixture
    def registry_with_schemas(self, temp_schema_dir) -> LocalFileRegistry:
        """Create a registry with test schemas."""
        # Create test schema files
        schema1 = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema2 = {"type": "array", "items": {"type": "number"}}

        # Flat structure: schema_id@version.json
        (temp_schema_dir / "test@1.0.0.json").write_text(json.dumps(schema1))

        # Nested structure: schema_id/version.json
        (temp_schema_dir / "nested").mkdir()
        (temp_schema_dir / "nested" / "2.0.0.json").write_text(json.dumps(schema2))

        # YAML file
        (temp_schema_dir / "yaml-test@1.0.0.yaml").write_text("type: string")

        return LocalFileRegistry(str(temp_schema_dir))

    def test_resolve_flat_structure(self, registry_with_schemas):
        """Resolve schema from flat directory structure."""
        source = registry_with_schemas.resolve("test", "1.0.0")
        assert source.schema_id == "test"
        assert source.version == "1.0.0"
        assert source.content_type == "application/json"
        assert "object" in source.content

    def test_resolve_nested_structure(self, registry_with_schemas):
        """Resolve schema from nested directory structure."""
        source = registry_with_schemas.resolve("nested", "2.0.0")
        assert source.schema_id == "nested"
        assert source.version == "2.0.0"
        assert "array" in source.content

    def test_resolve_yaml_file(self, registry_with_schemas):
        """Resolve YAML schema file."""
        source = registry_with_schemas.resolve("yaml-test", "1.0.0")
        assert source.content_type == "application/yaml"
        assert "string" in source.content

    def test_resolve_missing_schema_raises_error(self, registry_with_schemas):
        """Missing schema should raise RefResolutionError."""
        with pytest.raises(RefResolutionError, match="not found"):
            registry_with_schemas.resolve("missing", "1.0.0")

    def test_list_versions(self, temp_schema_dir):
        """List available versions for a schema."""
        # Create multiple versions
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            (temp_schema_dir / f"multi@{version}.json").write_text("{}")

        registry = LocalFileRegistry(str(temp_schema_dir))
        versions = registry.list_versions("multi")

        assert "1.0.0" in versions
        assert "1.1.0" in versions
        assert "2.0.0" in versions

    def test_etag_generated(self, registry_with_schemas):
        """Schema source should have an ETag."""
        source = registry_with_schemas.resolve("test", "1.0.0")
        assert source.etag is not None
        assert len(source.etag) == 16  # Truncated SHA-256


# ============================================================================
# EXTERNAL REF RESOLUTION TESTS
# ============================================================================


class TestExternalRefResolution:
    """Tests for external $ref resolution."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry that returns predefined schemas."""
        class MockRegistry:
            def __init__(self):
                self.schemas = {
                    ("emissions/activity", "1.0.0"): SchemaSource(
                        content='{"type": "object", "properties": {"value": {"type": "number"}}}',
                        content_type="application/json",
                        schema_id="emissions/activity",
                        version="1.0.0",
                        etag="abc123"
                    ),
                    ("common/unit", "2.0.0"): SchemaSource(
                        content='{"type": "string", "enum": ["kWh", "kg"]}',
                        content_type="application/json",
                        schema_id="common/unit",
                        version="2.0.0",
                        etag="def456"
                    ),
                }

            def resolve(self, schema_id: str, version: str) -> SchemaSource:
                key = (schema_id, version)
                if key not in self.schemas:
                    raise RefResolutionError(
                        ref=f"{schema_id}@{version}",
                        reason="Schema not found"
                    )
                return self.schemas[key]

            def list_versions(self, schema_id: str):
                return [v for (s, v) in self.schemas.keys() if s == schema_id]

        return MockRegistry()

    def test_resolve_external_gl_ref(self, mock_registry):
        """Resolve an external gl:// reference."""
        resolver = RefResolver(schema_registry=mock_registry)
        result = resolver.resolve(
            ref="gl://schemas/emissions/activity@1.0.0",
            context_document={},
            context_path=""
        )
        assert result["type"] == "object"
        assert "value" in result["properties"]

    def test_resolve_external_ref_without_registry_fails(self):
        """External ref without registry should fail."""
        resolver = RefResolver(schema_registry=None)
        with pytest.raises(RefResolutionError, match="No schema registry"):
            resolver.resolve(
                ref="gl://schemas/emissions/activity@1.0.0",
                context_document={},
                context_path=""
            )

    def test_external_ref_caches_document(self, mock_registry):
        """External documents should be cached."""
        resolver = RefResolver(schema_registry=mock_registry)

        # First resolution
        resolver.resolve(
            ref="gl://schemas/emissions/activity@1.0.0",
            context_document={},
            context_path=""
        )

        assert "emissions/activity@1.0.0" in resolver._external_documents

    def test_http_ref_not_supported(self):
        """HTTP refs should raise an error."""
        resolver = RefResolver()
        with pytest.raises(RefResolutionError, match="not supported"):
            resolver.resolve(
                ref="https://example.com/schema.json",
                context_document={},
                context_path=""
            )


# ============================================================================
# RESOLVE ALL REFS TESTS
# ============================================================================


class TestResolveAllRefs:
    """Tests for resolve_all_refs utility function."""

    def test_resolve_all_refs_in_document(self):
        """All refs in a document should be resolved."""
        doc = {
            "type": "object",
            "properties": {
                "activity": {"$ref": "#/definitions/Activity"},
                "unit": {"$ref": "#/definitions/Unit"}
            },
            "definitions": {
                "Activity": {"type": "object", "properties": {"name": {"type": "string"}}},
                "Unit": {"type": "string"}
            }
        }

        resolver = RefResolver()
        result = resolve_all_refs(doc, resolver)

        # Refs should be replaced with their definitions
        assert result["properties"]["activity"]["type"] == "object"
        assert result["properties"]["unit"]["type"] == "string"

    def test_resolve_all_refs_preserves_non_ref_fields(self):
        """Non-ref fields should be preserved."""
        doc = {
            "type": "object",
            "title": "Test Schema",
            "definitions": {"A": {"type": "string"}}
        }

        resolver = RefResolver()
        result = resolve_all_refs(doc, resolver)

        assert result["title"] == "Test Schema"
        assert result["type"] == "object"

    def test_resolve_all_refs_in_arrays(self):
        """Refs in arrays should be resolved."""
        doc = {
            "oneOf": [
                {"$ref": "#/definitions/A"},
                {"$ref": "#/definitions/B"}
            ],
            "definitions": {
                "A": {"type": "string"},
                "B": {"type": "number"}
            }
        }

        resolver = RefResolver()
        result = resolve_all_refs(doc, resolver)

        assert result["oneOf"][0]["type"] == "string"
        assert result["oneOf"][1]["type"] == "number"


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_ref_to_primitive_value(self):
        """Ref pointing to primitive should raise error."""
        doc = {"value": 42}
        resolver = RefResolver()

        with pytest.raises(RefResolutionError, match="not an object"):
            resolver.resolve("#/value", doc, "")

    def test_ref_to_array(self):
        """Ref pointing to array should raise error."""
        doc = {"items": [1, 2, 3]}
        resolver = RefResolver()

        with pytest.raises(RefResolutionError, match="not an object"):
            resolver.resolve("#/items", doc, "")

    def test_ref_with_unicode_characters(self):
        """Refs with unicode characters should work."""
        doc = {"definitions": {"cafe": {"type": "string"}}}
        resolver = RefResolver()

        result = resolver.resolve("#/definitions/cafe", doc, "")
        assert result == {"type": "string"}

    def test_ref_with_special_json_pointer_chars(self):
        """Refs with special JSON Pointer characters should work."""
        doc = {"definitions": {"a/b": {"type": "string"}, "c~d": {"type": "number"}}}
        resolver = RefResolver()

        # ~ is escaped as ~0, / is escaped as ~1
        result1 = resolver.resolve("#/definitions/a~1b", doc, "")
        result2 = resolver.resolve("#/definitions/c~0d", doc, "")

        assert result1 == {"type": "string"}
        assert result2 == {"type": "number"}


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestResolverIntegration:
    """Integration tests for RefResolver with LocalFileRegistry."""

    @pytest.fixture
    def schema_registry(self, tmp_path) -> LocalFileRegistry:
        """Create a registry with interconnected schemas."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()

        # Base schema
        base_schema = {
            "definitions": {
                "BaseType": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"}
                    }
                }
            }
        }
        (schema_dir / "base@1.0.0.json").write_text(json.dumps(base_schema))

        # Schema that references base
        derived_schema = {
            "type": "object",
            "properties": {
                "data": {"$ref": "#/definitions/Data"}
            },
            "definitions": {
                "Data": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"}
                    }
                }
            }
        }
        (schema_dir / "derived@1.0.0.json").write_text(json.dumps(derived_schema))

        return LocalFileRegistry(str(schema_dir))

    def test_full_resolution_workflow(self, schema_registry):
        """Test complete resolution workflow with local refs."""
        resolver = RefResolver(schema_registry=schema_registry)

        # Fetch external schema
        external = resolver.resolve(
            ref="gl://schemas/derived@1.0.0",
            context_document={},
            context_path=""
        )

        # Schema should be loaded
        assert external["type"] == "object"

        # Now resolve a local ref within that schema
        resolver.reset()  # Reset for clean state

        # The resolved external doc becomes our context
        local_result = resolver.resolve(
            ref="#/definitions/Data",
            context_document=external,
            context_path=""
        )

        assert local_result["type"] == "object"
        assert "value" in local_result["properties"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
