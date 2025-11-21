# -*- coding: utf-8 -*-
"""
Comprehensive tests for GreenLang Provenance Hashing Module.

Tests cover:
- hash_file() function with SHA256, chunked reading, and metadata
- hash_data() function for dicts, strings, and bytes
- MerkleTree class for hierarchical hashing
- Audit trail functionality for regulatory compliance
"""

import pytest
import hashlib
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

from greenlang.provenance.hashing import (
    hash_file,
    hash_data,
    MerkleTree,
    _format_bytes
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Test content for hashing\n" * 100)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_binary_file():
    """Create a temporary binary file for testing."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
        f.write(b'\x00\x01\x02\x03\x04\x05' * 1000)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def large_temp_file():
    """Create a large temporary file to test chunked reading."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        # Write 1MB of data
        for i in range(10000):
            f.write(f"Line {i}: " + "x" * 100 + "\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_dict():
    """Sample dictionary for testing."""
    return {
        "shipment_id": "SHIP-001",
        "carbon_emissions": 125.5,
        "items": ["steel", "aluminum"],
        "verified": True
    }


# ============================================================================
# HASH_FILE TESTS
# ============================================================================

class TestHashFile:
    """Test suite for hash_file() function."""

    def test_hash_file_basic(self, temp_file):
        """Test basic file hashing with SHA256."""
        result = hash_file(temp_file)

        # Verify structure
        assert isinstance(result, dict)
        assert "file_path" in result
        assert "file_name" in result
        assert "file_size_bytes" in result
        assert "hash_algorithm" in result
        assert "hash_value" in result
        assert "hash_timestamp" in result
        assert "verification" in result
        assert "human_readable_size" in result

        # Verify hash algorithm
        assert result["hash_algorithm"] == "SHA256"
        assert len(result["hash_value"]) == 64  # SHA256 hex digest length

        # Verify file info
        assert Path(result["file_path"]).name == Path(temp_file).name
        assert result["file_size_bytes"] > 0

    def test_hash_file_sha256(self, temp_file):
        """Test SHA256 hashing produces correct hash."""
        result = hash_file(temp_file, algorithm="sha256")

        # Manually compute SHA256 for verification
        hasher = hashlib.sha256()
        with open(temp_file, 'rb') as f:
            hasher.update(f.read())
        expected_hash = hasher.hexdigest()

        assert result["hash_value"] == expected_hash
        assert result["hash_algorithm"] == "SHA256"

    def test_hash_file_sha512(self, temp_file):
        """Test SHA512 hashing."""
        result = hash_file(temp_file, algorithm="sha512")

        assert result["hash_algorithm"] == "SHA512"
        assert len(result["hash_value"]) == 128  # SHA512 hex digest length

        # Manually verify
        hasher = hashlib.sha512()
        with open(temp_file, 'rb') as f:
            hasher.update(f.read())
        expected_hash = hasher.hexdigest()

        assert result["hash_value"] == expected_hash

    def test_hash_file_md5(self, temp_file):
        """Test MD5 hashing (legacy support - now redirects to SHA256)."""
        import pytest
        # MD5 is deprecated and now uses SHA256 instead
        with pytest.warns(UserWarning, match="MD5 is cryptographically broken"):
            result = hash_file(temp_file, algorithm="md5")

        # MD5 algorithm parameter now returns SHA256 hash with warning
        assert result["hash_algorithm"] == "MD5"  # Algorithm name preserved for compatibility
        assert len(result["hash_value"]) == 64  # SHA256 hex digest length (not 32 for MD5)

    def test_hash_file_chunked_reading(self, large_temp_file):
        """Test chunked reading for large files."""
        # Hash with default chunk size
        result1 = hash_file(large_temp_file, chunk_size=65536)

        # Hash with smaller chunk size
        result2 = hash_file(large_temp_file, chunk_size=1024)

        # Hash with larger chunk size
        result3 = hash_file(large_temp_file, chunk_size=1048576)

        # All should produce the same hash
        assert result1["hash_value"] == result2["hash_value"]
        assert result1["hash_value"] == result3["hash_value"]

    def test_hash_file_binary(self, temp_binary_file):
        """Test hashing binary files."""
        result = hash_file(temp_binary_file)

        assert isinstance(result, dict)
        assert len(result["hash_value"]) == 64
        assert result["file_size_bytes"] == 6000  # 6 bytes * 1000

    def test_hash_file_deterministic(self, temp_file):
        """Test that hashing is deterministic."""
        result1 = hash_file(temp_file)
        result2 = hash_file(temp_file)

        # Hash values should be identical
        assert result1["hash_value"] == result2["hash_value"]

        # Other fields may differ (timestamp)
        assert result1["file_path"] == result2["file_path"]
        assert result1["file_size_bytes"] == result2["file_size_bytes"]

    def test_hash_file_metadata(self, temp_file):
        """Test metadata fields in hash result."""
        result = hash_file(temp_file)

        # Verify timestamp format (ISO 8601)
        assert "T" in result["hash_timestamp"]
        assert "Z" in result["hash_timestamp"] or "+" in result["hash_timestamp"]

        # Verify verification command
        assert "sha256sum" in result["verification"].lower() or result["hash_algorithm"].lower() in result["verification"].lower()

        # Verify human-readable size
        assert "B" in result["human_readable_size"]

    def test_hash_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            hash_file("/nonexistent/file/path.txt")

    def test_hash_file_unsupported_algorithm(self, temp_file):
        """Test error handling for unsupported algorithm."""
        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            hash_file(temp_file, algorithm="blake2b")

    def test_hash_file_path_types(self, temp_file):
        """Test that both string and Path objects work."""
        # String path
        result1 = hash_file(temp_file)

        # Path object
        result2 = hash_file(Path(temp_file))

        assert result1["hash_value"] == result2["hash_value"]


# ============================================================================
# HASH_DATA TESTS
# ============================================================================

class TestHashData:
    """Test suite for hash_data() function."""

    def test_hash_data_string(self):
        """Test hashing string data."""
        data = "Test string for hashing"
        result = hash_data(data)

        # Verify it's a hex string
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256
        assert all(c in '0123456789abcdef' for c in result)

    def test_hash_data_bytes(self):
        """Test hashing bytes data."""
        data = b"Test bytes for hashing"
        result = hash_data(data)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_data_dict(self, sample_dict):
        """Test hashing dictionary data."""
        result = hash_data(sample_dict)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_data_dict_deterministic(self, sample_dict):
        """Test that dict hashing is deterministic (sorted keys)."""
        result1 = hash_data(sample_dict)
        result2 = hash_data(sample_dict)

        assert result1 == result2

        # Test with different key order
        reversed_dict = dict(reversed(list(sample_dict.items())))
        result3 = hash_data(reversed_dict)

        assert result1 == result3  # Should be same due to sort_keys=True

    def test_hash_data_algorithms(self):
        """Test different hashing algorithms."""
        import pytest
        data = "Test data"

        sha256_hash = hash_data(data, algorithm="sha256")
        assert len(sha256_hash) == 64

        sha512_hash = hash_data(data, algorithm="sha512")
        assert len(sha512_hash) == 128

        # MD5 is deprecated and now uses SHA256 instead
        with pytest.warns(UserWarning, match="MD5 is cryptographically broken"):
            md5_hash = hash_data(data, algorithm="md5")
        assert len(md5_hash) == 64  # SHA256 hex digest length (not 32 for MD5)

    def test_hash_data_empty_string(self):
        """Test hashing empty string."""
        result = hash_data("")

        # Empty string should have a known SHA256 hash
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_hash_data_empty_dict(self):
        """Test hashing empty dictionary."""
        result = hash_data({})

        # Empty dict as JSON is "{}"
        expected = hashlib.sha256(b"{}").hexdigest()
        assert result == expected

    def test_hash_data_unicode(self):
        """Test hashing Unicode strings."""
        data = "Test with Unicode: 日本語 émojis 🎉"
        result = hash_data(data)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_data_complex_dict(self):
        """Test hashing complex nested dictionary."""
        complex_data = {
            "level1": {
                "level2": {
                    "level3": ["a", "b", "c"],
                    "numbers": [1, 2, 3]
                }
            },
            "metadata": {
                "version": "1.0.0",
                "verified": True
            }
        }

        result = hash_data(complex_data)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_data_unsupported_type(self):
        """Test error handling for unsupported data types."""
        with pytest.raises(ValueError, match="Unsupported data type"):
            hash_data([1, 2, 3])  # Lists not directly supported

    def test_hash_data_unsupported_algorithm(self):
        """Test error handling for unsupported algorithm."""
        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            hash_data("test", algorithm="blake2b")


# ============================================================================
# MERKLE TREE TESTS
# ============================================================================

class TestMerkleTree:
    """Test suite for MerkleTree class."""

    def test_merkle_tree_initialization(self):
        """Test MerkleTree initialization."""
        tree = MerkleTree()

        assert tree.algorithm == "sha256"
        assert tree.leaves == []
        assert tree.tree == []
        assert tree.root_hash is None

    def test_merkle_tree_custom_algorithm(self):
        """Test MerkleTree with custom algorithm."""
        tree = MerkleTree(algorithm="sha512")
        assert tree.algorithm == "sha512"

    def test_add_leaf_string(self):
        """Test adding string leaves to tree."""
        tree = MerkleTree()
        tree.add_leaf("data1")
        tree.add_leaf("data2")

        assert len(tree.leaves) == 2
        assert all(len(leaf) == 64 for leaf in tree.leaves)  # SHA256 hashes

    def test_add_leaf_dict(self, sample_dict):
        """Test adding dictionary leaves to tree."""
        tree = MerkleTree()
        tree.add_leaf(sample_dict)

        assert len(tree.leaves) == 1

    def test_add_leaf_bytes(self):
        """Test adding bytes leaves to tree."""
        tree = MerkleTree()
        tree.add_leaf(b"binary data")

        assert len(tree.leaves) == 1

    def test_build_tree_single_leaf(self):
        """Test building tree with single leaf."""
        tree = MerkleTree()
        tree.add_leaf("single")

        root = tree.build()

        assert root is not None
        assert tree.root_hash == root
        assert len(tree.tree) > 0

    def test_build_tree_multiple_leaves(self):
        """Test building tree with multiple leaves."""
        tree = MerkleTree()
        tree.add_leaf("data1")
        tree.add_leaf("data2")
        tree.add_leaf("data3")
        tree.add_leaf("data4")

        root = tree.build()

        assert root is not None
        assert len(root) == 64  # SHA256
        assert tree.root_hash == root

    def test_build_tree_odd_leaves(self):
        """Test building tree with odd number of leaves (requires duplication)."""
        tree = MerkleTree()
        tree.add_leaf("data1")
        tree.add_leaf("data2")
        tree.add_leaf("data3")  # Odd number

        root = tree.build()

        assert root is not None
        assert len(tree.tree) > 0

    def test_build_empty_tree(self):
        """Test error when building empty tree."""
        tree = MerkleTree()

        with pytest.raises(ValueError, match="Cannot build tree with no leaves"):
            tree.build()

    def test_get_proof(self):
        """Test getting Merkle proof for a leaf."""
        tree = MerkleTree()
        tree.add_leaf("data1")
        tree.add_leaf("data2")
        tree.add_leaf("data3")
        tree.add_leaf("data4")

        root = tree.build()
        proof = tree.get_proof(1)  # Proof for "data2"

        assert isinstance(proof, list)
        assert len(proof) > 0
        assert all("hash" in elem and "position" in elem for elem in proof)

    def test_get_proof_not_built(self):
        """Test error when getting proof before building tree."""
        tree = MerkleTree()
        tree.add_leaf("data1")

        with pytest.raises(ValueError, match="Tree not built yet"):
            tree.get_proof(0)

    def test_get_proof_invalid_index(self):
        """Test error for invalid leaf index."""
        tree = MerkleTree()
        tree.add_leaf("data1")
        tree.build()

        with pytest.raises(ValueError, match="Invalid leaf index"):
            tree.get_proof(5)

    def test_verify_proof_valid(self):
        """Test verifying a valid Merkle proof."""
        tree = MerkleTree()
        tree.add_leaf("data1")
        tree.add_leaf("data2")
        tree.add_leaf("data3")

        root = tree.build()
        proof = tree.get_proof(1)

        is_valid = tree.verify_proof("data2", 1, proof, root)

        assert is_valid is True

    def test_verify_proof_invalid_data(self):
        """Test verifying proof with wrong data."""
        tree = MerkleTree()
        tree.add_leaf("data1")
        tree.add_leaf("data2")
        tree.add_leaf("data3")

        root = tree.build()
        proof = tree.get_proof(1)

        # Use wrong data
        is_valid = tree.verify_proof("wrong_data", 1, proof, root)

        assert is_valid is False

    def test_verify_proof_invalid_root(self):
        """Test verifying proof with wrong root."""
        tree = MerkleTree()
        tree.add_leaf("data1")
        tree.add_leaf("data2")

        root = tree.build()
        proof = tree.get_proof(1)

        # Use wrong root
        is_valid = tree.verify_proof("data2", 1, proof, "wrong_root_hash")

        assert is_valid is False

    def test_get_stats(self):
        """Test getting tree statistics."""
        tree = MerkleTree()
        tree.add_leaf("data1")
        tree.add_leaf("data2")
        tree.add_leaf("data3")

        root = tree.build()
        stats = tree.get_stats()

        assert stats["leaves"] == 3
        assert stats["levels"] > 0
        assert stats["root_hash"] == root
        assert stats["algorithm"] == "sha256"

    def test_merkle_tree_large_dataset(self):
        """Test Merkle tree with large number of leaves."""
        tree = MerkleTree()

        # Add 100 leaves
        for i in range(100):
            tree.add_leaf(f"data_{i}")

        root = tree.build()

        assert len(tree.leaves) == 100
        assert root is not None

        # Test proof for middle element
        proof = tree.get_proof(50)
        is_valid = tree.verify_proof("data_50", 50, proof, root)

        assert is_valid is True


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestFormatBytes:
    """Test suite for _format_bytes utility function."""

    def test_format_bytes_small(self):
        """Test formatting small byte counts."""
        assert "B" in _format_bytes(100)
        assert "100" in _format_bytes(100)

    def test_format_bytes_kb(self):
        """Test formatting kilobytes."""
        assert "KB" in _format_bytes(1024)
        assert "1.00 KB" == _format_bytes(1024)

    def test_format_bytes_mb(self):
        """Test formatting megabytes."""
        assert "MB" in _format_bytes(1024 * 1024)
        assert "1.00 MB" == _format_bytes(1024 * 1024)

    def test_format_bytes_gb(self):
        """Test formatting gigabytes."""
        assert "GB" in _format_bytes(1024 * 1024 * 1024)

    def test_format_bytes_zero(self):
        """Test formatting zero bytes."""
        assert "0.00 B" == _format_bytes(0)


# ============================================================================
# INTEGRATION TESTS FOR AUDIT TRAIL
# ============================================================================

class TestAuditTrailIntegration:
    """Integration tests for audit trail functionality."""

    def test_file_integrity_audit_trail(self, temp_file):
        """Test complete audit trail for file integrity."""
        # Initial hash
        hash1 = hash_file(temp_file)

        # Verify file
        hash2 = hash_file(temp_file)
        assert hash1["hash_value"] == hash2["hash_value"]

        # Modify file
        with open(temp_file, 'a') as f:
            f.write("\nModified content")

        # Hash should change
        hash3 = hash_file(temp_file)
        assert hash1["hash_value"] != hash3["hash_value"]

    def test_merkle_tree_audit_trail(self):
        """Test Merkle tree for hierarchical audit trail."""
        # Create tree with shipment data
        tree = MerkleTree()

        shipments = [
            {"id": "SHIP-001", "carbon": 100},
            {"id": "SHIP-002", "carbon": 150},
            {"id": "SHIP-003", "carbon": 200}
        ]

        for shipment in shipments:
            tree.add_leaf(shipment)

        root = tree.build()

        # Verify individual shipment
        proof = tree.get_proof(1)
        is_valid = tree.verify_proof(shipments[1], 1, proof, root)

        assert is_valid is True

        # Create audit record
        audit_record = {
            "root_hash": root,
            "total_shipments": len(shipments),
            "algorithm": tree.algorithm,
            "stats": tree.get_stats()
        }

        assert audit_record["root_hash"] == root
        assert audit_record["total_shipments"] == 3
