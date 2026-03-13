# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-EUDR-036

Tests hash computation, chain building, verification, and state management
for the EU Information System Interface provenance tracking.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.eu_information_system_interface.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


class TestGenesisHash:
    """Test GENESIS_HASH constant."""

    def test_genesis_hash_is_64_hex_chars(self):
        assert len(GENESIS_HASH) == 64

    def test_genesis_hash_is_all_zeros(self):
        assert GENESIS_HASH == "0" * 64


class TestComputeHash:
    """Test ProvenanceTracker.compute_hash()."""

    def test_returns_64_hex_string(self, provenance_tracker):
        h = provenance_tracker.compute_hash({"key": "value"})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic_same_input(self, provenance_tracker):
        data = {"test": "data", "num": 42}
        h1 = provenance_tracker.compute_hash(data)
        h2 = provenance_tracker.compute_hash(data)
        assert h1 == h2

    def test_different_inputs_different_hashes(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash({"a": 1})
        h2 = provenance_tracker.compute_hash({"a": 2})
        assert h1 != h2

    def test_empty_dict(self, provenance_tracker):
        h = provenance_tracker.compute_hash({})
        assert len(h) == 64

    def test_nested_dict(self, provenance_tracker):
        h = provenance_tracker.compute_hash({"outer": {"inner": "val"}})
        assert len(h) == 64

    def test_order_independent(self, provenance_tracker):
        # JSON sort_keys makes this order-independent
        h1 = provenance_tracker.compute_hash({"b": 2, "a": 1})
        h2 = provenance_tracker.compute_hash({"a": 1, "b": 2})
        assert h1 == h2


class TestCreateEntry:
    """Test ProvenanceTracker.create_entry()."""

    def test_creates_entry(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="test_step",
            source="test_source",
            input_hash="a" * 64,
            output_hash="b" * 64,
        )
        assert entry["step"] == "test_step"
        assert entry["source"] == "test_source"
        assert entry["input_hash"] == "a" * 64
        assert entry["output_hash"] == "b" * 64

    def test_entry_has_timestamp(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="step1",
            source="src1",
            input_hash="a" * 64,
            output_hash="b" * 64,
        )
        assert "timestamp" in entry

    def test_entry_has_actor(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="step1",
            source="src1",
            input_hash="a" * 64,
            output_hash="b" * 64,
        )
        assert "actor" in entry
        assert "EUDR-036" in entry["actor"]

    def test_chain_grows(self, provenance_tracker):
        provenance_tracker.create_entry(
            step="s1", source="src", input_hash="a" * 64, output_hash="b" * 64,
        )
        provenance_tracker.create_entry(
            step="s2", source="src", input_hash="b" * 64, output_hash="c" * 64,
        )
        chain = provenance_tracker.get_chain()
        assert len(chain) >= 2


class TestBuildChain:
    """Test ProvenanceTracker.build_chain()."""

    def test_empty_chain(self, provenance_tracker):
        chain = provenance_tracker.build_chain(steps=[])
        assert isinstance(chain, list)
        assert len(chain) == 0

    def test_chain_after_entries(self, provenance_tracker):
        steps = [
            {"step": "s1", "source": "src", "data": {"key": "value"}},
        ]
        chain = provenance_tracker.build_chain(steps=steps)
        assert len(chain) == 1

    def test_chain_ordering(self, provenance_tracker):
        steps = [
            {"step": "first", "source": "src", "data": {"a": 1}},
            {"step": "second", "source": "src", "data": {"b": 2}},
        ]
        chain = provenance_tracker.build_chain(steps=steps)
        assert chain[0]["step"] == "first"
        assert chain[1]["step"] == "second"

    def test_chain_links_hashes(self, provenance_tracker):
        steps = [
            {"step": "s1", "source": "src", "data": {"a": 1}},
            {"step": "s2", "source": "src", "data": {"b": 2}},
        ]
        chain = provenance_tracker.build_chain(steps=steps)
        # Second entry's input_hash should match first entry's output_hash
        assert chain[1]["input_hash"] == chain[0]["output_hash"]

    def test_chain_starts_from_genesis(self, provenance_tracker):
        steps = [
            {"step": "s1", "source": "src", "data": {"key": "val"}},
        ]
        chain = provenance_tracker.build_chain(steps=steps)
        assert chain[0]["input_hash"] == GENESIS_HASH


class TestVerifyChain:
    """Test ProvenanceTracker.verify_chain()."""

    def test_empty_chain_valid(self, provenance_tracker):
        result = provenance_tracker.verify_chain(entries=[])
        assert result is True

    def test_chain_with_entries_valid(self, provenance_tracker):
        steps = [
            {"step": "s1", "source": "src", "data": {"key": "val"}},
        ]
        chain = provenance_tracker.build_chain(steps=steps)
        result = provenance_tracker.verify_chain(entries=chain)
        assert result is True

    def test_verify_linked_chain(self, provenance_tracker):
        steps = [
            {"step": "s1", "source": "src", "data": {"a": 1}},
            {"step": "s2", "source": "src", "data": {"b": 2}},
        ]
        chain = provenance_tracker.build_chain(steps=steps)
        result = provenance_tracker.verify_chain(entries=chain)
        assert result is True

    def test_verify_broken_chain(self, provenance_tracker):
        entries = [
            {
                "step": "s1", "source": "src",
                "input_hash": GENESIS_HASH,
                "output_hash": "a" * 64,
            },
            {
                "step": "s2", "source": "src",
                "input_hash": "b" * 64,  # Broken: should be "a" * 64
                "output_hash": "c" * 64,
            },
        ]
        result = provenance_tracker.verify_chain(entries=entries)
        assert result is False


class TestGetChain:
    """Test ProvenanceTracker.get_chain()."""

    def test_empty_initially(self, provenance_tracker):
        assert provenance_tracker.get_chain() == []

    def test_accumulates_entries(self, provenance_tracker):
        provenance_tracker.create_entry(
            step="s1", source="src", input_hash="a" * 64, output_hash="b" * 64,
        )
        provenance_tracker.create_entry(
            step="s2", source="src", input_hash="b" * 64, output_hash="c" * 64,
        )
        chain = provenance_tracker.get_chain()
        assert len(chain) == 2
        assert chain[0]["step"] == "s1"
        assert chain[1]["step"] == "s2"

    def test_returns_copy(self, provenance_tracker):
        provenance_tracker.create_entry(
            step="s1", source="src", input_hash="a" * 64, output_hash="b" * 64,
        )
        chain1 = provenance_tracker.get_chain()
        chain2 = provenance_tracker.get_chain()
        assert chain1 is not chain2


class TestReset:
    """Test ProvenanceTracker.reset()."""

    def test_reset_clears_chain(self, provenance_tracker):
        provenance_tracker.create_entry(
            step="s1", source="src", input_hash="a" * 64, output_hash="b" * 64,
        )
        assert len(provenance_tracker.get_chain()) == 1
        provenance_tracker.reset()
        assert len(provenance_tracker.get_chain()) == 0


class TestComputeHashBytes:
    """Test ProvenanceTracker.compute_hash_bytes()."""

    def test_returns_bytes(self, provenance_tracker):
        result = provenance_tracker.compute_hash_bytes(b"test data")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_deterministic(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash_bytes(b"hello")
        h2 = provenance_tracker.compute_hash_bytes(b"hello")
        assert h1 == h2

    def test_different_input_different_hash(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash_bytes(b"hello")
        h2 = provenance_tracker.compute_hash_bytes(b"world")
        assert h1 != h2


class TestInitialization:
    """Test ProvenanceTracker initialization."""

    def test_default_algorithm(self):
        tracker = ProvenanceTracker()
        assert tracker._algorithm == "sha256"

    def test_unsupported_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            ProvenanceTracker(algorithm="md5")
