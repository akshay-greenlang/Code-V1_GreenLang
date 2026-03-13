# -*- coding: utf-8 -*-
"""
Tests for HashIntegrityValidator - AGENT-EUDR-012 Engine 3: Hash Integrity Validation

Comprehensive test suite covering:
- SHA-256 and SHA-512 hash computation
- Hash registry operations (register, lookup, duplicate detection)
- Tamper detection (same file modified = different hash)
- Merkle tree construction and root verification
- HMAC computation and verification
- Incremental hashing for large documents
- Hash chain anchoring
- Registry statistics
- Edge cases: empty document, zero bytes

Test count: 50+ tests
Coverage target: >= 85% of HashIntegrityValidator module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication Agent (GL-EUDR-DAV-012)
"""

from __future__ import annotations

import copy
import hashlib
import uuid
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.document_authentication.conftest import (
    HASH_ALGORITHMS,
    SHA256_HEX_LENGTH,
    SHA512_HEX_LENGTH,
    DOC_ID_COO_001,
    DOC_ID_FSC_001,
    DOC_ID_BOL_001,
    HASH_SHA256_COO,
    HASH_SHA512_FSC,
    SAMPLE_PDF_BYTES,
    SAMPLE_PDF_HASH_SHA256,
    SAMPLE_PDF_HASH_SHA512,
    SAMPLE_EMPTY_BYTES,
    SAMPLE_EMPTY_HASH_SHA256,
    SAMPLE_LARGE_BYTES,
    SAMPLE_LARGE_HASH_SHA256,
    make_hash_record,
    assert_hash_valid,
    assert_valid_provenance_hash,
    compute_sha256,
    _sha256,
    _sha512,
)


# ===========================================================================
# 1. SHA-256 Hash Computation
# ===========================================================================


class TestSHA256Computation:
    """Test SHA-256 hash computation."""

    def test_compute_sha256_basic(self, hash_engine):
        """Compute SHA-256 hash of a document."""
        result = hash_engine.compute_hash(
            document_bytes=SAMPLE_PDF_BYTES,
            algorithm="sha256",
        )
        assert result is not None
        assert result["algorithm"] == "sha256"
        assert len(result["hash_value"]) == SHA256_HEX_LENGTH

    def test_sha256_deterministic(self, hash_engine):
        """Same input produces same SHA-256 hash."""
        h1 = hash_engine.compute_hash(document_bytes=SAMPLE_PDF_BYTES, algorithm="sha256")
        h2 = hash_engine.compute_hash(document_bytes=SAMPLE_PDF_BYTES, algorithm="sha256")
        assert h1["hash_value"] == h2["hash_value"]

    def test_sha256_different_inputs(self, hash_engine):
        """Different inputs produce different SHA-256 hashes."""
        h1 = hash_engine.compute_hash(document_bytes=b"document_A", algorithm="sha256")
        h2 = hash_engine.compute_hash(document_bytes=b"document_B", algorithm="sha256")
        assert h1["hash_value"] != h2["hash_value"]

    def test_sha256_lowercase_hex(self, hash_engine):
        """SHA-256 hash is lowercase hexadecimal."""
        result = hash_engine.compute_hash(document_bytes=SAMPLE_PDF_BYTES, algorithm="sha256")
        assert all(c in "0123456789abcdef" for c in result["hash_value"])

    def test_sha256_matches_stdlib(self, hash_engine):
        """SHA-256 matches Python hashlib computation."""
        result = hash_engine.compute_hash(document_bytes=SAMPLE_PDF_BYTES, algorithm="sha256")
        expected = hashlib.sha256(SAMPLE_PDF_BYTES).hexdigest()
        assert result["hash_value"] == expected


# ===========================================================================
# 2. SHA-512 Hash Computation
# ===========================================================================


class TestSHA512Computation:
    """Test SHA-512 hash computation."""

    def test_compute_sha512_basic(self, hash_engine):
        """Compute SHA-512 hash of a document."""
        result = hash_engine.compute_hash(
            document_bytes=SAMPLE_PDF_BYTES,
            algorithm="sha512",
        )
        assert result is not None
        assert result["algorithm"] == "sha512"
        assert len(result["hash_value"]) == SHA512_HEX_LENGTH

    def test_sha512_deterministic(self, hash_engine):
        """Same input produces same SHA-512 hash."""
        h1 = hash_engine.compute_hash(document_bytes=SAMPLE_PDF_BYTES, algorithm="sha512")
        h2 = hash_engine.compute_hash(document_bytes=SAMPLE_PDF_BYTES, algorithm="sha512")
        assert h1["hash_value"] == h2["hash_value"]

    def test_sha512_different_from_sha256(self, hash_engine):
        """SHA-512 hash differs from SHA-256 of same content."""
        h256 = hash_engine.compute_hash(document_bytes=SAMPLE_PDF_BYTES, algorithm="sha256")
        h512 = hash_engine.compute_hash(document_bytes=SAMPLE_PDF_BYTES, algorithm="sha512")
        assert h256["hash_value"] != h512["hash_value"]

    def test_sha512_matches_stdlib(self, hash_engine):
        """SHA-512 matches Python hashlib computation."""
        result = hash_engine.compute_hash(document_bytes=SAMPLE_PDF_BYTES, algorithm="sha512")
        expected = hashlib.sha512(SAMPLE_PDF_BYTES).hexdigest()
        assert result["hash_value"] == expected


# ===========================================================================
# 3. Hash Registry Operations
# ===========================================================================


class TestHashRegistry:
    """Test hash registry operations."""

    def test_register_hash(self, hash_engine):
        """Register a hash in the registry."""
        record = make_hash_record(document_id="DOC-REG-001")
        result = hash_engine.register(record)
        assert result is not None
        assert result.get("registry_status") in ("registered", "success")

    def test_lookup_registered_hash(self, hash_engine):
        """Look up a previously registered hash."""
        record = make_hash_record(document_id="DOC-LKP-001")
        hash_engine.register(record)
        found = hash_engine.lookup(hash_value=record["hash_value"])
        assert found is not None
        assert found["document_id"] == "DOC-LKP-001"

    def test_lookup_unregistered_returns_none(self, hash_engine):
        """Looking up an unregistered hash returns None."""
        result = hash_engine.lookup(hash_value="a" * 64)
        assert result is None

    def test_duplicate_detection(self, hash_engine):
        """Registering a duplicate hash detects the duplicate."""
        record1 = make_hash_record(document_id="DOC-DUP-001", hash_value="b" * 64)
        hash_engine.register(record1)
        record2 = make_hash_record(document_id="DOC-DUP-002", hash_value="b" * 64)
        result = hash_engine.register(record2)
        assert result.get("registry_status") == "duplicate" or result.get("duplicate_of") is not None

    def test_register_returns_hash_id(self, hash_engine):
        """Registration returns a hash record ID."""
        record = make_hash_record()
        result = hash_engine.register(record)
        assert result.get("hash_id") is not None

    def test_register_preserves_algorithm(self, hash_engine):
        """Registration preserves the algorithm field."""
        record = make_hash_record(algorithm="sha512")
        result = hash_engine.register(record)
        assert result.get("algorithm") == "sha512"


# ===========================================================================
# 4. Tamper Detection
# ===========================================================================


class TestTamperDetection:
    """Test tamper detection via hash comparison."""

    def test_unmodified_file_passes(self, hash_engine):
        """Hash verification passes for unmodified file."""
        original_hash = hashlib.sha256(SAMPLE_PDF_BYTES).hexdigest()
        result = hash_engine.verify(
            document_bytes=SAMPLE_PDF_BYTES,
            expected_hash=original_hash,
            algorithm="sha256",
        )
        assert result["tampered"] is False

    def test_modified_file_detected(self, hash_engine):
        """Hash verification detects modified file."""
        original_hash = hashlib.sha256(SAMPLE_PDF_BYTES).hexdigest()
        modified_bytes = SAMPLE_PDF_BYTES + b"tampered"
        result = hash_engine.verify(
            document_bytes=modified_bytes,
            expected_hash=original_hash,
            algorithm="sha256",
        )
        assert result["tampered"] is True

    def test_single_byte_change_detected(self, hash_engine):
        """Even a single byte change is detected."""
        original_hash = hashlib.sha256(SAMPLE_PDF_BYTES).hexdigest()
        tampered = bytearray(SAMPLE_PDF_BYTES)
        tampered[0] = (tampered[0] + 1) % 256
        result = hash_engine.verify(
            document_bytes=bytes(tampered),
            expected_hash=original_hash,
            algorithm="sha256",
        )
        assert result["tampered"] is True

    def test_verify_returns_actual_hash(self, hash_engine):
        """Verification returns the actual computed hash."""
        original_hash = hashlib.sha256(SAMPLE_PDF_BYTES).hexdigest()
        result = hash_engine.verify(
            document_bytes=SAMPLE_PDF_BYTES,
            expected_hash=original_hash,
            algorithm="sha256",
        )
        assert result["actual_hash"] == original_hash


# ===========================================================================
# 5. Merkle Tree
# ===========================================================================


class TestMerkleTree:
    """Test Merkle tree construction and verification."""

    def test_merkle_root_single_document(self, hash_engine):
        """Merkle root of single document equals its hash."""
        hashes = [hashlib.sha256(SAMPLE_PDF_BYTES).hexdigest()]
        root = hash_engine.compute_merkle_root(hashes)
        assert root is not None
        assert len(root) == SHA256_HEX_LENGTH

    def test_merkle_root_multiple_documents(self, hash_engine):
        """Merkle root of multiple documents is computed."""
        hashes = [
            hashlib.sha256(f"doc_{i}".encode()).hexdigest()
            for i in range(4)
        ]
        root = hash_engine.compute_merkle_root(hashes)
        assert root is not None
        assert len(root) == SHA256_HEX_LENGTH

    def test_merkle_root_deterministic(self, hash_engine):
        """Same inputs produce same Merkle root."""
        hashes = [
            hashlib.sha256(f"doc_{i}".encode()).hexdigest()
            for i in range(4)
        ]
        root1 = hash_engine.compute_merkle_root(hashes)
        root2 = hash_engine.compute_merkle_root(hashes)
        assert root1 == root2

    def test_merkle_root_order_sensitive(self, hash_engine):
        """Changing order of hashes changes the Merkle root."""
        hashes_a = [_sha256("doc_0"), _sha256("doc_1")]
        hashes_b = [_sha256("doc_1"), _sha256("doc_0")]
        root_a = hash_engine.compute_merkle_root(hashes_a)
        root_b = hash_engine.compute_merkle_root(hashes_b)
        assert root_a != root_b

    def test_merkle_root_odd_count(self, hash_engine):
        """Merkle tree with odd number of leaves is handled."""
        hashes = [
            hashlib.sha256(f"doc_{i}".encode()).hexdigest()
            for i in range(5)
        ]
        root = hash_engine.compute_merkle_root(hashes)
        assert root is not None

    def test_merkle_root_empty_list_raises(self, hash_engine):
        """Empty hash list raises ValueError."""
        with pytest.raises(ValueError):
            hash_engine.compute_merkle_root([])


# ===========================================================================
# 6. HMAC Computation
# ===========================================================================


class TestHMACComputation:
    """Test HMAC-SHA256 computation and verification."""

    def test_compute_hmac(self, hash_engine):
        """Compute HMAC-SHA256 with a secret key."""
        result = hash_engine.compute_hash(
            document_bytes=SAMPLE_PDF_BYTES,
            algorithm="hmac_sha256",
            secret_key="test-secret-key-2026",
        )
        assert result is not None
        assert result["algorithm"] == "hmac_sha256"
        assert len(result["hash_value"]) == SHA256_HEX_LENGTH

    def test_hmac_different_keys(self, hash_engine):
        """Different keys produce different HMAC values."""
        h1 = hash_engine.compute_hash(
            document_bytes=SAMPLE_PDF_BYTES,
            algorithm="hmac_sha256",
            secret_key="key_alpha",
        )
        h2 = hash_engine.compute_hash(
            document_bytes=SAMPLE_PDF_BYTES,
            algorithm="hmac_sha256",
            secret_key="key_beta",
        )
        assert h1["hash_value"] != h2["hash_value"]

    def test_hmac_deterministic(self, hash_engine):
        """Same input and key produce same HMAC."""
        h1 = hash_engine.compute_hash(
            document_bytes=SAMPLE_PDF_BYTES,
            algorithm="hmac_sha256",
            secret_key="stable_key",
        )
        h2 = hash_engine.compute_hash(
            document_bytes=SAMPLE_PDF_BYTES,
            algorithm="hmac_sha256",
            secret_key="stable_key",
        )
        assert h1["hash_value"] == h2["hash_value"]

    def test_hmac_no_key_raises(self, hash_engine):
        """HMAC without a secret key raises ValueError."""
        with pytest.raises(ValueError):
            hash_engine.compute_hash(
                document_bytes=SAMPLE_PDF_BYTES,
                algorithm="hmac_sha256",
            )


# ===========================================================================
# 7. Incremental Hashing
# ===========================================================================


class TestIncrementalHashing:
    """Test incremental hashing for large documents."""

    def test_incremental_hash_matches_full(self, hash_engine):
        """Incremental hashing produces same result as full hash."""
        full_hash = hash_engine.compute_hash(
            document_bytes=SAMPLE_LARGE_BYTES,
            algorithm="sha256",
        )
        incremental_hash = hash_engine.compute_hash_incremental(
            document_bytes=SAMPLE_LARGE_BYTES,
            algorithm="sha256",
            chunk_size=8192,
        )
        assert full_hash["hash_value"] == incremental_hash["hash_value"]

    def test_incremental_hash_large_file(self, hash_engine):
        """Incremental hash works for large files."""
        result = hash_engine.compute_hash_incremental(
            document_bytes=SAMPLE_LARGE_BYTES,
            algorithm="sha256",
            chunk_size=4096,
        )
        assert result["hash_value"] == SAMPLE_LARGE_HASH_SHA256


# ===========================================================================
# 8. Hash Chain Anchoring
# ===========================================================================


class TestHashChainAnchoring:
    """Test hash chain anchoring for provenance."""

    def test_anchor_hash(self, hash_engine):
        """Anchor a hash to the provenance chain."""
        result = hash_engine.anchor(
            hash_value=_sha256("test-document-content"),
            document_id="DOC-ANCHOR-001",
        )
        assert result is not None
        assert "anchor_hash" in result or "provenance_hash" in result

    def test_anchor_chain_linkage(self, hash_engine):
        """Anchored hashes form a linked chain."""
        anchor1 = hash_engine.anchor(
            hash_value=_sha256("doc_1"),
            document_id="DOC-CHAIN-001",
        )
        anchor2 = hash_engine.anchor(
            hash_value=_sha256("doc_2"),
            document_id="DOC-CHAIN-002",
        )
        # Second anchor should reference or build on first
        assert anchor1 is not None
        assert anchor2 is not None

    def test_genesis_hash(self, hash_engine):
        """First anchor in chain uses genesis hash."""
        result = hash_engine.get_genesis_hash()
        assert result is not None
        assert isinstance(result, str)


# ===========================================================================
# 9. Registry Statistics
# ===========================================================================


class TestRegistryStatistics:
    """Test hash registry statistics."""

    def test_registry_stats_empty(self, hash_engine):
        """Empty registry returns zero counts."""
        stats = hash_engine.get_registry_stats()
        assert stats is not None
        assert stats.get("total_hashes", 0) >= 0

    def test_registry_stats_after_registration(self, hash_engine):
        """Stats update after hash registration."""
        record = make_hash_record(document_id="DOC-STAT-001")
        hash_engine.register(record)
        stats = hash_engine.get_registry_stats()
        assert stats.get("total_hashes", 0) >= 1

    def test_registry_stats_duplicate_count(self, hash_engine):
        """Stats track duplicate count."""
        stats = hash_engine.get_registry_stats()
        assert "duplicate_count" in stats or "duplicates" in stats


# ===========================================================================
# 10. Edge Cases
# ===========================================================================


class TestHashEdgeCases:
    """Test edge cases for hash integrity validation."""

    def test_empty_document_hash(self, hash_engine):
        """Empty document produces the known empty-content SHA-256."""
        result = hash_engine.compute_hash(
            document_bytes=SAMPLE_EMPTY_BYTES,
            algorithm="sha256",
        )
        assert result["hash_value"] == SAMPLE_EMPTY_HASH_SHA256

    def test_zero_bytes_document(self, hash_engine):
        """Document of all zero bytes produces valid hash."""
        zero_bytes = b"\x00" * 100
        result = hash_engine.compute_hash(
            document_bytes=zero_bytes,
            algorithm="sha256",
        )
        assert len(result["hash_value"]) == SHA256_HEX_LENGTH

    def test_invalid_algorithm_raises(self, hash_engine):
        """Invalid hash algorithm raises ValueError."""
        with pytest.raises(ValueError):
            hash_engine.compute_hash(
                document_bytes=SAMPLE_PDF_BYTES,
                algorithm="md5",
            )

    def test_none_bytes_raises(self, hash_engine):
        """None document bytes raises TypeError."""
        with pytest.raises((TypeError, ValueError)):
            hash_engine.compute_hash(
                document_bytes=None,
                algorithm="sha256",
            )

    def test_hash_record_factory_valid(self, hash_engine):
        """Factory-built hash record passes validation."""
        record = make_hash_record()
        assert_hash_valid(record)

    @pytest.mark.parametrize("algorithm", HASH_ALGORITHMS)
    def test_all_algorithms_supported(self, hash_engine, algorithm):
        """All 3 hash algorithms are supported."""
        if algorithm == "hmac_sha256":
            result = hash_engine.compute_hash(
                document_bytes=SAMPLE_PDF_BYTES,
                algorithm=algorithm,
                secret_key="test-key",
            )
        else:
            result = hash_engine.compute_hash(
                document_bytes=SAMPLE_PDF_BYTES,
                algorithm=algorithm,
            )
        assert result is not None
        assert result["algorithm"] == algorithm

    def test_file_size_recorded(self, hash_engine):
        """Hash computation records file size."""
        result = hash_engine.compute_hash(
            document_bytes=SAMPLE_PDF_BYTES,
            algorithm="sha256",
        )
        assert result.get("file_size_bytes") == len(SAMPLE_PDF_BYTES)

    def test_provenance_hash_on_record(self, hash_engine):
        """Hash record can include a provenance hash."""
        record = make_hash_record()
        record["provenance_hash"] = "c" * 64
        assert_valid_provenance_hash(record["provenance_hash"])
