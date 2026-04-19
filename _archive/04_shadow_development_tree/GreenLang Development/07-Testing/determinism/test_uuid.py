"""
Tests for GreenLang Deterministic ID Generation

Tests deterministic ID generation, UUID generation, and content hashing.
"""

import pytest

from greenlang.determinism.uuid import (
    deterministic_id,
    deterministic_uuid,
    content_hash,
)


class TestDeterministicId:
    """Test deterministic_id functionality."""

    def test_string_input_deterministic(self):
        """Test that same string always produces same ID."""
        id1 = deterministic_id("hello world")
        id2 = deterministic_id("hello world")
        assert id1 == id2

    def test_bytes_input_deterministic(self):
        """Test that same bytes always produces same ID."""
        data = b"hello world"
        id1 = deterministic_id(data)
        id2 = deterministic_id(data)
        assert id1 == id2

    def test_dict_input_deterministic(self):
        """Test that same dict always produces same ID."""
        data = {"key": "value", "number": 42}
        id1 = deterministic_id(data)
        id2 = deterministic_id(data)
        assert id1 == id2

    def test_dict_key_order_irrelevant(self):
        """Test that dict key order doesn't affect ID."""
        dict1 = {"a": 1, "b": 2, "c": 3}
        dict2 = {"c": 3, "a": 1, "b": 2}
        id1 = deterministic_id(dict1)
        id2 = deterministic_id(dict2)
        assert id1 == id2

    def test_prefix_added(self):
        """Test that prefix is correctly added to ID."""
        id_with_prefix = deterministic_id("test", prefix="doc_")
        assert id_with_prefix.startswith("doc_")

    def test_different_content_different_id(self):
        """Test that different content produces different IDs."""
        id1 = deterministic_id("hello")
        id2 = deterministic_id("world")
        assert id1 != id2

    def test_id_length_consistent(self):
        """Test that ID length is consistent (prefix + 16 chars)."""
        id1 = deterministic_id("short", prefix="x_")
        id2 = deterministic_id("very long content string", prefix="x_")
        assert len(id1) == len(id2)
        assert len(id1) == 18  # "x_" + 16 hex chars


class TestDeterministicUuid:
    """Test deterministic_uuid functionality."""

    def test_namespace_name_deterministic(self):
        """Test that same namespace:name produces same UUID."""
        uuid1 = deterministic_uuid("namespace", "name")
        uuid2 = deterministic_uuid("namespace", "name")
        assert uuid1 == uuid2

    def test_different_namespace_different_uuid(self):
        """Test that different namespace produces different UUID."""
        uuid1 = deterministic_uuid("namespace1", "name")
        uuid2 = deterministic_uuid("namespace2", "name")
        assert uuid1 != uuid2

    def test_different_name_different_uuid(self):
        """Test that different name produces different UUID."""
        uuid1 = deterministic_uuid("namespace", "name1")
        uuid2 = deterministic_uuid("namespace", "name2")
        assert uuid1 != uuid2

    def test_uuid_has_prefix(self):
        """Test that UUID has uuid_ prefix."""
        uuid = deterministic_uuid("namespace", "name")
        assert uuid.startswith("uuid_")


class TestContentHash:
    """Test content_hash functionality."""

    def test_string_content_hash(self):
        """Test hashing string content."""
        hash1 = content_hash("hello world")
        hash2 = content_hash("hello world")
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_dict_content_hash(self):
        """Test hashing dict content."""
        data = {"key": "value"}
        hash1 = content_hash(data)
        hash2 = content_hash(data)
        assert hash1 == hash2

    def test_dict_key_order_irrelevant(self):
        """Test that dict key order doesn't affect hash."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 2, "a": 1}
        hash1 = content_hash(dict1)
        hash2 = content_hash(dict2)
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Test that different content produces different hashes."""
        hash1 = content_hash("hello")
        hash2 = content_hash("world")
        assert hash1 != hash2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
