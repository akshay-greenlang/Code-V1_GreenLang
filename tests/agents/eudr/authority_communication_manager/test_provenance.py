# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-EUDR-040

Tests SHA-256 hash computation, determinism, chain building, chain
verification, entry creation, chain accumulation, reset functionality,
and edge cases.

35+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.authority_communication_manager.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


# ====================================================================
# Genesis Hash
# ====================================================================


class TestGenesisHash:
    """Test genesis hash constant."""

    def test_genesis_hash_length(self):
        assert len(GENESIS_HASH) == 64

    def test_genesis_hash_all_zeros(self):
        assert GENESIS_HASH == "0" * 64

    def test_genesis_hash_is_string(self):
        assert isinstance(GENESIS_HASH, str)


# ====================================================================
# Initialization
# ====================================================================


class TestProvenanceInit:
    """Test ProvenanceTracker initialization."""

    def test_init_default(self, provenance_tracker):
        assert provenance_tracker._algorithm == "sha256"

    def test_init_sha256_explicit(self):
        tracker = ProvenanceTracker(algorithm="sha256")
        assert tracker._algorithm == "sha256"

    def test_init_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported"):
            ProvenanceTracker(algorithm="md5")

    def test_init_invalid_sha1(self):
        with pytest.raises(ValueError, match="Unsupported"):
            ProvenanceTracker(algorithm="sha1")

    def test_init_empty_chain(self, provenance_tracker):
        assert len(provenance_tracker.get_chain()) == 0


# ====================================================================
# Hash Computation
# ====================================================================


class TestComputeHash:
    """Test deterministic hash computation."""

    def test_hash_deterministic(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash({"key": "value"})
        h2 = provenance_tracker.compute_hash({"key": "value"})
        assert h1 == h2

    def test_hash_length(self, provenance_tracker):
        h = provenance_tracker.compute_hash({"test": 123})
        assert len(h) == 64

    def test_hash_hex_string(self, provenance_tracker):
        h = provenance_tracker.compute_hash({"a": "b"})
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_different_inputs(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash({"a": 1})
        h2 = provenance_tracker.compute_hash({"a": 2})
        assert h1 != h2

    def test_hash_key_order_independent(self, provenance_tracker):
        """Canonical JSON sorts keys, so order should not matter."""
        h1 = provenance_tracker.compute_hash({"z": 1, "a": 2})
        h2 = provenance_tracker.compute_hash({"a": 2, "z": 1})
        assert h1 == h2

    def test_hash_empty_dict(self, provenance_tracker):
        h = provenance_tracker.compute_hash({})
        assert len(h) == 64

    def test_hash_nested_dict(self, provenance_tracker):
        h = provenance_tracker.compute_hash({"a": {"b": {"c": 1}}})
        assert len(h) == 64

    def test_hash_with_list(self, provenance_tracker):
        h = provenance_tracker.compute_hash({"items": [1, 2, 3]})
        assert len(h) == 64


# ====================================================================
# Hash Bytes
# ====================================================================


class TestComputeHashBytes:
    """Test hash computation from raw bytes."""

    def test_hash_bytes(self, provenance_tracker):
        h = provenance_tracker.compute_hash_bytes(b"test data")
        assert len(h) == 64

    def test_hash_bytes_deterministic(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash_bytes(b"hello")
        h2 = provenance_tracker.compute_hash_bytes(b"hello")
        assert h1 == h2

    def test_hash_bytes_different(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash_bytes(b"hello")
        h2 = provenance_tracker.compute_hash_bytes(b"world")
        assert h1 != h2

    def test_hash_empty_bytes(self, provenance_tracker):
        h = provenance_tracker.compute_hash_bytes(b"")
        assert len(h) == 64


# ====================================================================
# Create Entry
# ====================================================================


class TestCreateEntry:
    """Test provenance entry creation."""

    def test_create_entry(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="abc123",
        )
        assert entry["step"] == "test"
        assert entry["source"] == "src"
        assert entry["input_hash"] == GENESIS_HASH
        assert entry["output_hash"] == "abc123"

    def test_entry_has_timestamp(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="abc",
        )
        assert "timestamp" in entry

    def test_entry_default_actor(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="abc",
        )
        assert entry["actor"] == "AGENT-EUDR-040"

    def test_entry_custom_actor(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="abc",
            actor="custom-actor",
        )
        assert entry["actor"] == "custom-actor"

    def test_entry_added_to_chain(self, provenance_tracker):
        provenance_tracker.create_entry(
            step="s1", source="src",
            input_hash=GENESIS_HASH, output_hash="h1",
        )
        assert len(provenance_tracker.get_chain()) == 1


# ====================================================================
# Chain Operations
# ====================================================================


class TestChainOperations:
    """Test chain accumulation and retrieval."""

    def test_chain_accumulation(self, provenance_tracker):
        provenance_tracker.create_entry("s1", "src", GENESIS_HASH, "h1")
        provenance_tracker.create_entry("s2", "src", "h1", "h2")
        chain = provenance_tracker.get_chain()
        assert len(chain) == 2

    def test_get_chain_returns_copy(self, provenance_tracker):
        provenance_tracker.create_entry("s1", "src", GENESIS_HASH, "h1")
        chain1 = provenance_tracker.get_chain()
        chain2 = provenance_tracker.get_chain()
        assert chain1 == chain2
        assert chain1 is not chain2

    def test_reset_clears_chain(self, provenance_tracker):
        provenance_tracker.create_entry("s1", "src", GENESIS_HASH, "h1")
        provenance_tracker.reset()
        assert len(provenance_tracker.get_chain()) == 0


# ====================================================================
# Verify Chain
# ====================================================================


class TestVerifyChain:
    """Test chain integrity verification."""

    def test_verify_empty_chain(self, provenance_tracker):
        assert provenance_tracker.verify_chain([]) is True

    def test_verify_valid_chain(self, provenance_tracker):
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "aaa"},
            {"input_hash": "aaa", "output_hash": "bbb"},
            {"input_hash": "bbb", "output_hash": "ccc"},
        ]
        assert provenance_tracker.verify_chain(entries) is True

    def test_verify_broken_chain(self, provenance_tracker):
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "aaa"},
            {"input_hash": "WRONG", "output_hash": "ccc"},
        ]
        assert provenance_tracker.verify_chain(entries) is False

    def test_verify_single_entry(self, provenance_tracker):
        entries = [{"input_hash": GENESIS_HASH, "output_hash": "abc"}]
        assert provenance_tracker.verify_chain(entries) is True


# ====================================================================
# Build Chain
# ====================================================================


class TestBuildChain:
    """Test chain building from step definitions."""

    def test_build_chain(self, provenance_tracker):
        steps = [
            {"step": "s1", "source": "src", "data": {"k": "v1"}},
            {"step": "s2", "source": "src", "data": {"k": "v2"}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert len(chain) == 2

    def test_build_chain_links(self, provenance_tracker):
        steps = [
            {"step": "s1", "source": "src", "data": {"k": "v1"}},
            {"step": "s2", "source": "src", "data": {"k": "v2"}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert chain[0]["input_hash"] == GENESIS_HASH
        assert chain[1]["input_hash"] == chain[0]["output_hash"]

    def test_build_chain_verifiable(self, provenance_tracker):
        steps = [
            {"step": "s1", "source": "src", "data": {"a": 1}},
            {"step": "s2", "source": "src", "data": {"b": 2}},
            {"step": "s3", "source": "src", "data": {"c": 3}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert provenance_tracker.verify_chain(chain) is True

    def test_build_empty_chain(self, provenance_tracker):
        chain = provenance_tracker.build_chain([])
        assert chain == []

    def test_build_chain_custom_genesis(self, provenance_tracker):
        custom = "a" * 64
        steps = [{"step": "s1", "source": "src", "data": {"k": "v"}}]
        chain = provenance_tracker.build_chain(steps, genesis_hash=custom)
        assert chain[0]["input_hash"] == custom
