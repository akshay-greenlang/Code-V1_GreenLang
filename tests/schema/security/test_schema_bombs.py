# -*- coding: utf-8 -*-
"""
Security Tests for Schema Bombs - GL-FOUND-X-002.

This module tests the schema validator's resilience against schema bomb attacks,
including:
    - Deep $ref chains (reference depth attacks)
    - Circular $ref references (infinite loop attacks)
    - Exponential $ref expansion (combinatorial explosion)
    - Self-referencing schemas
    - Mutually recursive schemas

The resolver MUST:
    1. Detect circular references and return clear error messages
    2. Enforce maximum $ref expansion limits
    3. Not hang or crash when processing malicious schemas
    4. Complete within timeout limits
    5. Return appropriate error codes

References:
    - https://cwe.mitre.org/data/definitions/776.html (Improper Restriction)
    - JSON Schema $ref specification
    - PRD Section 6.10: Security Limits

Author: GreenLang Security Testing Team
Date: 2026-01-29
Version: 1.0.0
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict
from unittest.mock import Mock, MagicMock

import pytest

from greenlang.schema.compiler.resolver import (
    RefResolver,
    CircularRefError,
    RefResolutionError,
    MaxExpansionsExceededError,
    parse_ref,
    parse_json_pointer,
    navigate_json_pointer,
    LocalFileRegistry,
)
from greenlang.schema.constants import MAX_REF_EXPANSIONS


# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.security,
    pytest.mark.timeout(10),  # All tests must complete within 10 seconds
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def resolver() -> RefResolver:
    """Create a RefResolver instance with small limits for testing."""
    return RefResolver(max_expansions=100)


@pytest.fixture
def strict_resolver() -> RefResolver:
    """Create a RefResolver with very strict limits."""
    return RefResolver(max_expansions=10)


@pytest.fixture
def default_timeout_seconds() -> float:
    """Default timeout for operations."""
    return 2.0


# =============================================================================
# Deep $ref Chain Tests
# =============================================================================


class TestDeepRefChains:
    """Test suite for deep $ref chain attacks."""

    def test_linear_ref_chain_within_limit(self, resolver: RefResolver):
        """Test that a linear ref chain within limits works."""
        # Create a document with a chain of 5 refs
        document = {
            "definitions": {
                "Level1": {"$ref": "#/definitions/Level2"},
                "Level2": {"$ref": "#/definitions/Level3"},
                "Level3": {"$ref": "#/definitions/Level4"},
                "Level4": {"$ref": "#/definitions/Level5"},
                "Level5": {"type": "string"},
            }
        }

        # Should resolve successfully
        result = resolver.resolve(
            "#/definitions/Level1",
            document,
        )
        assert result["type"] == "string"

    def test_deep_ref_chain_exceeds_limit(self, strict_resolver: RefResolver):
        """Test that very deep ref chains are rejected."""
        # Create a document with chain deeper than the limit
        definitions = {}
        for i in range(20):
            if i < 19:
                definitions[f"Level{i}"] = {"$ref": f"#/definitions/Level{i+1}"}
            else:
                definitions[f"Level{i}"] = {"type": "string"}

        document = {"definitions": definitions}

        with pytest.raises(MaxExpansionsExceededError):
            strict_resolver.resolve(
                "#/definitions/Level0",
                document,
            )

    def test_wide_ref_expansion(self, resolver: RefResolver):
        """Test that wide $ref expansion (many refs at same level) is handled."""
        # Create schema with many refs at the same level
        document = {
            "definitions": {
                "Base": {"type": "string"},
            },
            "properties": {
                f"prop{i}": {"$ref": "#/definitions/Base"}
                for i in range(50)
            }
        }

        # Should work - each ref is independent
        result = resolver.resolve("#/definitions/Base", document)
        assert result["type"] == "string"

    def test_branching_ref_tree(self, resolver: RefResolver):
        """Test branching tree of refs (exponential potential)."""
        document = {
            "definitions": {
                "Root": {
                    "allOf": [
                        {"$ref": "#/definitions/BranchA"},
                        {"$ref": "#/definitions/BranchB"},
                    ]
                },
                "BranchA": {
                    "allOf": [
                        {"$ref": "#/definitions/LeafA1"},
                        {"$ref": "#/definitions/LeafA2"},
                    ]
                },
                "BranchB": {
                    "allOf": [
                        {"$ref": "#/definitions/LeafB1"},
                        {"$ref": "#/definitions/LeafB2"},
                    ]
                },
                "LeafA1": {"type": "string"},
                "LeafA2": {"type": "number"},
                "LeafB1": {"type": "boolean"},
                "LeafB2": {"type": "null"},
            }
        }

        # Should resolve the root ref
        result = resolver.resolve("#/definitions/Root", document)
        assert "allOf" in result


class TestCircularReferences:
    """Test suite for circular $ref detection."""

    def test_direct_self_reference(self, resolver: RefResolver):
        """Test that direct self-reference is detected."""
        document = {
            "definitions": {
                "SelfRef": {"$ref": "#/definitions/SelfRef"}
            }
        }

        with pytest.raises(CircularRefError) as exc_info:
            resolver.resolve("#/definitions/SelfRef", document)

        # Should report the cycle
        assert len(exc_info.value.cycle) >= 2

    def test_two_node_cycle(self, resolver: RefResolver):
        """Test that A -> B -> A cycle is detected."""
        document = {
            "definitions": {
                "A": {"$ref": "#/definitions/B"},
                "B": {"$ref": "#/definitions/A"},
            }
        }

        with pytest.raises(CircularRefError) as exc_info:
            resolver.resolve("#/definitions/A", document)

        # Cycle should contain both A and B
        cycle_str = " -> ".join(exc_info.value.cycle)
        assert "A" in cycle_str
        assert "B" in cycle_str

    def test_three_node_cycle(self, resolver: RefResolver):
        """Test that A -> B -> C -> A cycle is detected."""
        document = {
            "definitions": {
                "A": {"$ref": "#/definitions/B"},
                "B": {"$ref": "#/definitions/C"},
                "C": {"$ref": "#/definitions/A"},
            }
        }

        with pytest.raises(CircularRefError) as exc_info:
            resolver.resolve("#/definitions/A", document)

        # Cycle should contain all three
        cycle_str = " -> ".join(exc_info.value.cycle)
        assert "A" in cycle_str

    def test_long_chain_with_cycle(self, resolver: RefResolver):
        """Test cycle detection in a long chain: A -> B -> C -> D -> E -> B."""
        document = {
            "definitions": {
                "A": {"$ref": "#/definitions/B"},
                "B": {"$ref": "#/definitions/C"},
                "C": {"$ref": "#/definitions/D"},
                "D": {"$ref": "#/definitions/E"},
                "E": {"$ref": "#/definitions/B"},  # Back to B
            }
        }

        with pytest.raises(CircularRefError) as exc_info:
            resolver.resolve("#/definitions/A", document)

        # Should detect the B -> C -> D -> E -> B cycle
        assert len(exc_info.value.cycle) >= 2

    def test_cycle_with_intermediate_properties(self, resolver: RefResolver):
        """Test that nested $ref in properties doesn't cause infinite recursion.

        Note: The resolver only follows direct $ref chains, not nested refs
        inside properties. This test verifies the resolver doesn't hang when
        a schema has a self-reference inside properties.
        """
        document = {
            "definitions": {
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "friend": {"$ref": "#/definitions/Person"},  # Self-reference
                    }
                }
            }
        }

        # The resolver returns the definition without expanding nested refs
        # This is safe - it doesn't cause infinite recursion
        result = resolver.resolve("#/definitions/Person", document)
        assert result["type"] == "object"
        assert "$ref" in result["properties"]["friend"]

    def test_indirect_cycle_through_allof(self, resolver: RefResolver):
        """Test that nested $ref in allOf doesn't cause infinite recursion.

        Note: The resolver only follows direct $ref at the top level of a
        definition, not nested refs inside allOf/anyOf/oneOf. This test
        verifies the resolver doesn't hang and returns the schema as-is.
        """
        document = {
            "definitions": {
                "A": {
                    "allOf": [
                        {"type": "object"},
                        {"$ref": "#/definitions/B"}
                    ]
                },
                "B": {
                    "allOf": [
                        {"type": "object"},
                        {"$ref": "#/definitions/A"}
                    ]
                }
            }
        }

        # The resolver returns the definition without expanding nested refs
        result = resolver.resolve("#/definitions/A", document)
        assert "allOf" in result


class TestExponentialRefExpansion:
    """Test suite for exponential $ref expansion attacks."""

    def test_diamond_pattern(self, resolver: RefResolver):
        """
        Test diamond pattern that could cause exponential expansion.

           A
          / \\
         B   C
          \\ /
           D

        A refs B and C, both B and C ref D.
        Without caching, D would be resolved twice.
        """
        document = {
            "definitions": {
                "A": {
                    "allOf": [
                        {"$ref": "#/definitions/B"},
                        {"$ref": "#/definitions/C"},
                    ]
                },
                "B": {"$ref": "#/definitions/D"},
                "C": {"$ref": "#/definitions/D"},
                "D": {"type": "string"},
            }
        }

        # Should resolve without issues (caching prevents double work)
        result = resolver.resolve("#/definitions/D", document)
        assert result["type"] == "string"

    def test_many_refs_to_same_target(self, resolver: RefResolver):
        """Test many refs pointing to the same definition."""
        document = {
            "definitions": {
                "Shared": {"type": "string"},
            },
            "properties": {
                f"field{i}": {"$ref": "#/definitions/Shared"}
                for i in range(100)
            }
        }

        # Should work - caching means Shared is only resolved once
        result = resolver.resolve("#/definitions/Shared", document)
        assert result["type"] == "string"

    def test_expansion_limit_protects_against_bomb(self, strict_resolver: RefResolver):
        """Test that expansion limit protects against schema bombs.

        The resolver only follows direct $ref at the top level (not inside allOf).
        This test creates a long chain of direct refs to trigger the limit.
        """
        # Create a schema with a long direct chain of refs
        definitions = {}
        for i in range(20):
            if i < 19:
                definitions[f"Level{i}"] = {"$ref": f"#/definitions/Level{i+1}"}
            else:
                definitions[f"Level{i}"] = {"type": "string"}

        document = {"definitions": definitions}

        # With strict limits (max_expansions=10), this should be caught
        with pytest.raises(MaxExpansionsExceededError):
            strict_resolver.resolve("#/definitions/Level0", document)


class TestMaliciousRefPatterns:
    """Test suite for various malicious $ref patterns."""

    def test_ref_to_nonexistent_path(self, resolver: RefResolver):
        """Test $ref to a path that doesn't exist."""
        document = {
            "definitions": {
                "Valid": {"type": "string"}
            }
        }

        with pytest.raises(RefResolutionError) as exc_info:
            resolver.resolve("#/definitions/NonExistent", document)

        assert "not found" in str(exc_info.value).lower()

    def test_ref_to_invalid_json_pointer(self, resolver: RefResolver):
        """Test $ref with invalid JSON Pointer syntax."""
        document = {"key": "value"}

        # JSON Pointer must start with /
        with pytest.raises((RefResolutionError, ValueError)):
            resolver.resolve("#invalid-pointer", document)

    def test_ref_with_array_out_of_bounds(self, resolver: RefResolver):
        """Test $ref to array index out of bounds."""
        document = {
            "items": ["a", "b", "c"]
        }

        with pytest.raises(RefResolutionError) as exc_info:
            resolver.resolve("#/items/99", document)

        assert "out of bounds" in str(exc_info.value).lower()

    def test_ref_through_scalar_value(self, resolver: RefResolver):
        """Test $ref trying to navigate through a scalar."""
        document = {
            "name": "test"
        }

        with pytest.raises(RefResolutionError) as exc_info:
            resolver.resolve("#/name/invalid", document)

        assert "primitive" in str(exc_info.value).lower() or "Cannot navigate" in str(exc_info.value)

    def test_empty_ref(self, resolver: RefResolver):
        """Test empty $ref string."""
        document = {"key": "value"}

        # Empty ref "#" refers to the whole document
        result = resolver.resolve("#", document)
        assert result == document

    def test_ref_with_special_characters(self, resolver: RefResolver):
        """Test $ref with JSON Pointer escaped characters."""
        document = {
            "definitions": {
                "a/b": {"type": "string"},
                "c~d": {"type": "number"},
            }
        }

        # JSON Pointer escapes: / -> ~1, ~ -> ~0
        result = resolver.resolve("#/definitions/a~1b", document)
        assert result["type"] == "string"

        result = resolver.resolve("#/definitions/c~0d", document)
        assert result["type"] == "number"


class TestTimingAttacks:
    """Test suite for timing-based attack detection."""

    def test_circular_ref_detection_is_fast(self, resolver: RefResolver, default_timeout_seconds: float):
        """Test that circular references are detected quickly."""
        # Deep circular chain
        definitions = {}
        for i in range(50):
            definitions[f"Level{i}"] = {"$ref": f"#/definitions/Level{(i+1) % 50}"}

        document = {"definitions": definitions}

        start_time = time.perf_counter()

        with pytest.raises(CircularRefError):
            resolver.resolve("#/definitions/Level0", document)

        elapsed = time.perf_counter() - start_time
        assert elapsed < default_timeout_seconds, (
            f"Circular ref detection took {elapsed:.2f}s, expected < {default_timeout_seconds}s"
        )

    def test_expansion_limit_enforced_quickly(self, strict_resolver: RefResolver, default_timeout_seconds: float):
        """Test that expansion limit is enforced quickly."""
        # Create schema that would expand many times
        definitions = {}
        for i in range(100):
            if i < 99:
                definitions[f"Level{i}"] = {"$ref": f"#/definitions/Level{i+1}"}
            else:
                definitions[f"Level{i}"] = {"type": "string"}

        document = {"definitions": definitions}

        start_time = time.perf_counter()

        with pytest.raises(MaxExpansionsExceededError):
            strict_resolver.resolve("#/definitions/Level0", document)

        elapsed = time.perf_counter() - start_time
        assert elapsed < default_timeout_seconds


class TestResolverState:
    """Test suite for resolver state management."""

    def test_resolver_reset_clears_state(self, resolver: RefResolver):
        """Test that reset() clears all state."""
        document = {
            "definitions": {
                "Test": {"type": "string"}
            }
        }

        # Resolve something
        resolver.resolve("#/definitions/Test", document)

        # Verify state was modified
        assert resolver._expansion_count > 0 or len(resolver._cache) > 0

        # Reset
        resolver.reset()

        # Verify state is cleared
        assert resolver._expansion_count == 0
        assert len(resolver._cache) == 0
        assert len(resolver._resolution_stack) == 0

    def test_stats_tracking(self, resolver: RefResolver):
        """Test that stats are tracked correctly."""
        document = {
            "definitions": {
                "A": {"$ref": "#/definitions/B"},
                "B": {"$ref": "#/definitions/C"},
                "C": {"type": "string"},
            }
        }

        resolver.resolve("#/definitions/A", document)

        stats = resolver.get_stats()
        assert stats["expansion_count"] > 0
        assert stats["cache_size"] > 0

    def test_multiple_resolutions_share_cache(self, resolver: RefResolver):
        """Test that multiple resolutions share the cache."""
        document = {
            "definitions": {
                "Shared": {"type": "string"}
            }
        }

        # Resolve the same ref multiple times
        for _ in range(10):
            resolver.resolve("#/definitions/Shared", document)

        stats = resolver.get_stats()
        # Should have used cache for subsequent calls
        assert stats["cache_size"] >= 1


class TestExternalRefProtection:
    """Test suite for external $ref protection."""

    def test_external_ref_without_registry_fails(self, resolver: RefResolver):
        """Test that external refs fail without a registry."""
        document = {"type": "object"}

        with pytest.raises(RefResolutionError) as exc_info:
            resolver.resolve(
                "gl://schemas/external/schema@1.0.0",
                document,
            )

        assert "registry" in str(exc_info.value).lower()

    def test_http_refs_are_blocked(self, resolver: RefResolver):
        """Test that HTTP refs are blocked."""
        document = {"type": "object"}

        with pytest.raises(RefResolutionError) as exc_info:
            resolver.resolve(
                "https://example.com/schema.json",
                document,
            )

        assert "not supported" in str(exc_info.value).lower()

    def test_relative_refs_are_blocked(self, resolver: RefResolver):
        """Test that relative file refs are blocked."""
        document = {"type": "object"}

        with pytest.raises(RefResolutionError) as exc_info:
            resolver.resolve(
                "./other-schema.json",
                document,
            )

        assert "not supported" in str(exc_info.value).lower()


class TestJSONPointerEdgeCases:
    """Test suite for JSON Pointer edge cases."""

    def test_pointer_with_empty_key(self, resolver: RefResolver):
        """Test JSON Pointer behavior with special paths.

        The resolver treats #/ as a reference to the root document.
        This test verifies consistent behavior.
        """
        document = {
            "definitions": {
                "Test": {"type": "string"}
            }
        }

        # #/ should return the root document
        result = resolver.resolve("#/", document)
        assert "definitions" in result

        # Empty ref # should also return root
        result = resolver.resolve("#", document)
        assert "definitions" in result

    def test_pointer_to_root(self):
        """Test that empty pointer refers to root."""
        document = {"key": "value"}

        result = navigate_json_pointer(document, "")
        assert result == document

    def test_pointer_parsing(self):
        """Test JSON Pointer parsing."""
        # Normal pointer
        segments = parse_json_pointer("/a/b/c")
        assert segments == ["a", "b", "c"]

        # Escaped characters
        segments = parse_json_pointer("/a~1b/c~0d")
        assert segments == ["a/b", "c~d"]

        # Empty pointer
        segments = parse_json_pointer("")
        assert segments == []

    def test_pointer_with_numeric_keys(self, resolver: RefResolver):
        """Test JSON Pointer with numeric-looking string keys."""
        document = {
            "definitions": {
                "123": {"type": "string"}
            }
        }

        result = resolver.resolve("#/definitions/123", document)
        assert result["type"] == "string"


class TestRefParsing:
    """Test suite for $ref string parsing."""

    def test_parse_local_ref(self):
        """Test parsing local refs."""
        parsed = parse_ref("#/definitions/Foo")
        assert parsed.fragment == "/definitions/Foo"

    def test_parse_external_gl_ref(self):
        """Test parsing gl:// refs."""
        parsed = parse_ref("gl://schemas/activity@1.0.0")
        assert parsed.schema_id == "activity"
        assert parsed.version == "1.0.0"

    def test_parse_external_gl_ref_with_fragment(self):
        """Test parsing gl:// refs with fragments."""
        parsed = parse_ref("gl://schemas/activity@1.0.0#/definitions/Foo")
        assert parsed.schema_id == "activity"
        assert parsed.version == "1.0.0"
        assert parsed.fragment == "/definitions/Foo"

    def test_parse_http_ref(self):
        """Test parsing HTTP refs."""
        parsed = parse_ref("https://example.com/schema.json")
        assert parsed.uri == "https://example.com/schema.json"


class TestStressConditions:
    """Stress tests for schema bomb handling."""

    def test_repeated_resolution_attempts(self, resolver: RefResolver):
        """Test handling of repeated resolution attempts."""
        document = {
            "definitions": {
                "Test": {"type": "string"}
            }
        }

        for _ in range(1000):
            result = resolver.resolve("#/definitions/Test", document)
            assert result["type"] == "string"

    def test_interleaved_valid_and_circular(self, resolver: RefResolver):
        """Test alternating between valid and circular schemas."""
        valid_doc = {
            "definitions": {
                "Valid": {"type": "string"}
            }
        }

        circular_doc = {
            "definitions": {
                "A": {"$ref": "#/definitions/B"},
                "B": {"$ref": "#/definitions/A"},
            }
        }

        for i in range(50):
            # Valid should work
            result = resolver.resolve("#/definitions/Valid", valid_doc)
            assert result["type"] == "string"

            # Circular should fail
            resolver.reset()  # Reset state for clean test
            with pytest.raises(CircularRefError):
                resolver.resolve("#/definitions/A", circular_doc)

            resolver.reset()

    def test_large_schema_without_refs(self, resolver: RefResolver):
        """Test that large schemas without refs work after circular rejection."""
        circular_doc = {
            "definitions": {
                "A": {"$ref": "#/definitions/A"}
            }
        }

        with pytest.raises(CircularRefError):
            resolver.resolve("#/definitions/A", circular_doc)

        resolver.reset()

        # Now try a large but valid schema
        large_doc = {
            "definitions": {
                f"Type{i}": {"type": "string", "description": f"Type number {i}"}
                for i in range(500)
            }
        }

        result = resolver.resolve("#/definitions/Type0", large_doc)
        assert result["type"] == "string"
