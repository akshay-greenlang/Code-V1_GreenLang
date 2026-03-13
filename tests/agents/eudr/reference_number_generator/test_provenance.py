# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-038 Reference Number Generator -- provenance.py

Tests SHA-256 provenance hash chain computation, determinism,
chain building, chain verification, reference-specific hashing,
and reset behavior. 25+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from greenlang.agents.eudr.reference_number_generator.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


# ====================================================================
# Test: Genesis Hash
# ====================================================================


class TestGenesisHash:
    """Test the genesis hash constant."""

    def test_genesis_hash_is_64_zeros(self):
        assert GENESIS_HASH == "0" * 64

    def test_genesis_hash_length(self):
        assert len(GENESIS_HASH) == 64

    def test_genesis_hash_all_zeros(self):
        assert all(c == "0" for c in GENESIS_HASH)


# ====================================================================
# Test: ProvenanceTracker Initialization
# ====================================================================


class TestProvenanceTrackerInit:
    """Test ProvenanceTracker initialization."""

    def test_default_algorithm(self, provenance_tracker):
        assert provenance_tracker._algorithm == "sha256"

    def test_empty_chain_on_init(self, provenance_tracker):
        assert provenance_tracker.get_chain() == []

    def test_unsupported_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            ProvenanceTracker(algorithm="md5")

    def test_unsupported_algorithm_sha1_raises(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            ProvenanceTracker(algorithm="sha1")


# ====================================================================
# Test: compute_hash
# ====================================================================


class TestComputeHash:
    """Test deterministic SHA-256 hash computation."""

    def test_hash_returns_string(self, provenance_tracker):
        h = provenance_tracker.compute_hash({"key": "value"})
        assert isinstance(h, str)

    def test_hash_length_is_64(self, provenance_tracker):
        h = provenance_tracker.compute_hash({"test": 123})
        assert len(h) == 64

    def test_hash_is_hex(self, provenance_tracker):
        h = provenance_tracker.compute_hash({"test": "data"})
        int(h, 16)  # Should not raise

    def test_hash_determinism(self, provenance_tracker):
        """Same input must produce same hash."""
        data = {"operator_id": "OP-001", "member_state": "DE", "sequence": 42}
        h1 = provenance_tracker.compute_hash(data)
        h2 = provenance_tracker.compute_hash(data)
        assert h1 == h2

    def test_hash_different_data_different_hash(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash({"key": "value1"})
        h2 = provenance_tracker.compute_hash({"key": "value2"})
        assert h1 != h2

    def test_hash_key_order_independent(self, provenance_tracker):
        """JSON sorted keys ensures order independence."""
        h1 = provenance_tracker.compute_hash({"a": 1, "b": 2})
        h2 = provenance_tracker.compute_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_hash_empty_dict(self, provenance_tracker):
        h = provenance_tracker.compute_hash({})
        assert len(h) == 64

    def test_hash_nested_data(self, provenance_tracker):
        data = {
            "reference": "EUDR-DE-2026-OP001-000001-7",
            "components": {"prefix": "EUDR", "ms": "DE"},
        }
        h = provenance_tracker.compute_hash(data)
        assert len(h) == 64


# ====================================================================
# Test: compute_hash_bytes
# ====================================================================


class TestComputeHashBytes:
    """Test raw bytes hashing."""

    def test_bytes_hash_length(self, provenance_tracker):
        h = provenance_tracker.compute_hash_bytes(b"hello world")
        assert len(h) == 64

    def test_bytes_hash_determinism(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash_bytes(b"test data")
        h2 = provenance_tracker.compute_hash_bytes(b"test data")
        assert h1 == h2

    def test_bytes_hash_empty(self, provenance_tracker):
        h = provenance_tracker.compute_hash_bytes(b"")
        assert len(h) == 64


# ====================================================================
# Test: compute_reference_hash
# ====================================================================


class TestComputeReferenceHash:
    """Test reference-number-specific hashing."""

    def test_reference_hash_length(self, provenance_tracker):
        h = provenance_tracker.compute_reference_hash(
            "EUDR-DE-2026-OP001-000001-7", "OP-001"
        )
        assert len(h) == 64

    def test_reference_hash_determinism(self, provenance_tracker):
        h1 = provenance_tracker.compute_reference_hash(
            "EUDR-DE-2026-OP001-000001-7", "OP-001"
        )
        h2 = provenance_tracker.compute_reference_hash(
            "EUDR-DE-2026-OP001-000001-7", "OP-001"
        )
        assert h1 == h2

    def test_different_reference_different_hash(self, provenance_tracker):
        h1 = provenance_tracker.compute_reference_hash(
            "EUDR-DE-2026-OP001-000001-7", "OP-001"
        )
        h2 = provenance_tracker.compute_reference_hash(
            "EUDR-DE-2026-OP001-000002-4", "OP-001"
        )
        assert h1 != h2

    def test_different_operator_different_hash(self, provenance_tracker):
        h1 = provenance_tracker.compute_reference_hash(
            "EUDR-DE-2026-OP001-000001-7", "OP-001"
        )
        h2 = provenance_tracker.compute_reference_hash(
            "EUDR-DE-2026-OP001-000001-7", "OP-002"
        )
        assert h1 != h2


# ====================================================================
# Test: create_entry
# ====================================================================


class TestCreateEntry:
    """Test provenance entry creation."""

    def test_entry_has_required_fields(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="generate",
            source="rng_038",
            input_hash=GENESIS_HASH,
            output_hash="a" * 64,
        )
        assert "step" in entry
        assert "source" in entry
        assert "timestamp" in entry
        assert "actor" in entry
        assert "input_hash" in entry
        assert "output_hash" in entry

    def test_entry_default_actor(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="validate",
            source="rng_038",
            input_hash=GENESIS_HASH,
            output_hash="b" * 64,
        )
        assert entry["actor"] == "AGENT-EUDR-038"

    def test_entry_custom_actor(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="transfer",
            source="rng_038",
            input_hash=GENESIS_HASH,
            output_hash="c" * 64,
            actor="admin",
        )
        assert entry["actor"] == "admin"

    def test_entry_custom_timestamp(self, provenance_tracker):
        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        entry = provenance_tracker.create_entry(
            step="revoke",
            source="rng_038",
            input_hash=GENESIS_HASH,
            output_hash="d" * 64,
            timestamp=ts,
        )
        assert entry["timestamp"] == ts.isoformat()

    def test_entry_appended_to_chain(self, provenance_tracker):
        provenance_tracker.create_entry(
            step="generate",
            source="rng_038",
            input_hash=GENESIS_HASH,
            output_hash="e" * 64,
        )
        assert len(provenance_tracker.get_chain()) == 1


# ====================================================================
# Test: verify_chain
# ====================================================================


class TestVerifyChain:
    """Test provenance chain verification."""

    def test_empty_chain_is_valid(self, provenance_tracker):
        assert provenance_tracker.verify_chain([]) is True

    def test_single_entry_chain_is_valid(self, provenance_tracker):
        entries = [
            {
                "step": "generate",
                "input_hash": GENESIS_HASH,
                "output_hash": "a" * 64,
            }
        ]
        assert provenance_tracker.verify_chain(entries) is True

    def test_valid_chain(self, provenance_tracker):
        entries = [
            {"step": "generate", "input_hash": GENESIS_HASH, "output_hash": "a" * 64},
            {"step": "validate", "input_hash": "a" * 64, "output_hash": "b" * 64},
            {"step": "activate", "input_hash": "b" * 64, "output_hash": "c" * 64},
        ]
        assert provenance_tracker.verify_chain(entries) is True

    def test_broken_chain(self, provenance_tracker):
        entries = [
            {"step": "generate", "input_hash": GENESIS_HASH, "output_hash": "a" * 64},
            {"step": "validate", "input_hash": "x" * 64, "output_hash": "b" * 64},
        ]
        assert provenance_tracker.verify_chain(entries) is False


# ====================================================================
# Test: build_chain
# ====================================================================


class TestBuildChain:
    """Test building a provenance chain from step definitions."""

    def test_build_chain_creates_entries(self, provenance_tracker):
        steps = [
            {"step": "generate", "source": "rng_038", "data": {"ref": "001"}},
            {"step": "validate", "source": "rng_038", "data": {"valid": True}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert len(chain) == 2

    def test_build_chain_first_input_is_genesis(self, provenance_tracker):
        steps = [
            {"step": "generate", "source": "rng_038", "data": {"ref": "001"}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert chain[0]["input_hash"] == GENESIS_HASH

    def test_build_chain_is_verifiable(self, provenance_tracker):
        steps = [
            {"step": "generate", "source": "rng_038", "data": {"ref": "001"}},
            {"step": "validate", "source": "rng_038", "data": {"valid": True}},
            {"step": "activate", "source": "rng_038", "data": {"status": "active"}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert provenance_tracker.verify_chain(chain) is True

    def test_build_chain_custom_genesis(self, provenance_tracker):
        custom_genesis = "f" * 64
        steps = [
            {"step": "generate", "source": "rng_038", "data": {"ref": "001"}},
        ]
        chain = provenance_tracker.build_chain(steps, genesis_hash=custom_genesis)
        assert chain[0]["input_hash"] == custom_genesis


# ====================================================================
# Test: get_chain / reset
# ====================================================================


class TestChainManagement:
    """Test chain retrieval and reset."""

    def test_get_chain_returns_copy(self, provenance_tracker):
        provenance_tracker.create_entry(
            step="generate",
            source="rng_038",
            input_hash=GENESIS_HASH,
            output_hash="a" * 64,
        )
        chain = provenance_tracker.get_chain()
        chain.clear()
        assert len(provenance_tracker.get_chain()) == 1

    def test_reset_clears_chain(self, provenance_tracker):
        provenance_tracker.create_entry(
            step="generate",
            source="rng_038",
            input_hash=GENESIS_HASH,
            output_hash="a" * 64,
        )
        provenance_tracker.reset()
        assert provenance_tracker.get_chain() == []
