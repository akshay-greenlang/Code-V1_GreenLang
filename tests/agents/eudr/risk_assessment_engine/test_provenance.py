# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-028 ProvenanceTracker.

Tests SHA-256 hash computation, chain building, chain verification,
entry creation, and reset functionality.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from greenlang.agents.eudr.risk_assessment_engine.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


class TestComputeHash:
    """Tests for ProvenanceTracker.compute_hash static method."""

    def test_compute_hash_deterministic(self, tracker: ProvenanceTracker):
        """Same data must always produce the same hash."""
        data = {"key": "value", "number": 42}
        h1 = tracker.compute_hash(data)
        h2 = tracker.compute_hash(data)
        assert h1 == h2
        assert len(h1) == 64

    def test_compute_hash_different_data(self, tracker: ProvenanceTracker):
        """Different data must produce different hashes."""
        h1 = tracker.compute_hash({"a": 1})
        h2 = tracker.compute_hash({"a": 2})
        assert h1 != h2

    def test_compute_hash_sorted_keys(self, tracker: ProvenanceTracker):
        """Key order must not affect the hash (sorted keys)."""
        h1 = tracker.compute_hash({"b": 2, "a": 1})
        h2 = tracker.compute_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_compute_hash_bytes(self, tracker: ProvenanceTracker):
        """compute_hash_bytes should hash raw bytes."""
        data = b"hello world"
        h = tracker.compute_hash_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert h == expected
        assert len(h) == 64


class TestCreateEntry:
    """Tests for ProvenanceTracker.create_entry method."""

    def test_create_entry_actor_default(self, tracker: ProvenanceTracker):
        """Default actor should be AGENT-EUDR-028."""
        entry = tracker.create_entry(
            step="test",
            source="unit_test",
            input_hash=GENESIS_HASH,
            output_hash="a" * 64,
        )
        assert entry["actor"] == "AGENT-EUDR-028"
        assert entry["step"] == "test"
        assert entry["source"] == "unit_test"
        assert entry["input_hash"] == GENESIS_HASH
        assert entry["output_hash"] == "a" * 64

    def test_create_entry_appends_to_chain(self, tracker: ProvenanceTracker):
        """Each create_entry call must add to the internal chain."""
        assert len(tracker.get_chain()) == 0
        tracker.create_entry("s1", "src", GENESIS_HASH, "a" * 64)
        assert len(tracker.get_chain()) == 1
        tracker.create_entry("s2", "src", "a" * 64, "b" * 64)
        assert len(tracker.get_chain()) == 2

    def test_create_entry_custom_actor(self, tracker: ProvenanceTracker):
        """Custom actor should override the default."""
        entry = tracker.create_entry(
            step="test",
            source="src",
            input_hash=GENESIS_HASH,
            output_hash="a" * 64,
            actor="CUSTOM-ACTOR",
        )
        assert entry["actor"] == "CUSTOM-ACTOR"

    def test_create_entry_timestamp(self, tracker: ProvenanceTracker):
        """Entry should include an ISO timestamp."""
        entry = tracker.create_entry("s", "src", GENESIS_HASH, "a" * 64)
        assert "timestamp" in entry
        # Should be parseable
        datetime.fromisoformat(entry["timestamp"])


class TestVerifyChain:
    """Tests for ProvenanceTracker.verify_chain method."""

    def test_verify_chain_valid(self, tracker: ProvenanceTracker):
        """A properly linked chain must verify as valid."""
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "aaa"},
            {"input_hash": "aaa", "output_hash": "bbb"},
            {"input_hash": "bbb", "output_hash": "ccc"},
        ]
        assert tracker.verify_chain(entries) is True

    def test_verify_chain_broken(self, tracker: ProvenanceTracker):
        """A broken chain (mismatched hashes) must verify as invalid."""
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "aaa"},
            {"input_hash": "WRONG", "output_hash": "bbb"},
        ]
        assert tracker.verify_chain(entries) is False

    def test_verify_chain_empty(self, tracker: ProvenanceTracker):
        """An empty chain is trivially valid."""
        assert tracker.verify_chain([]) is True

    def test_verify_chain_single_entry(self, tracker: ProvenanceTracker):
        """A single-entry chain is valid (no link to verify)."""
        entries = [{"input_hash": GENESIS_HASH, "output_hash": "aaa"}]
        assert tracker.verify_chain(entries) is True


class TestBuildChain:
    """Tests for ProvenanceTracker.build_chain method."""

    def test_build_chain(self, tracker: ProvenanceTracker):
        """build_chain should produce a linked chain from step definitions."""
        steps = [
            {"step": "aggregate", "source": "eudr_016", "data": {"x": 1}},
            {"step": "classify", "source": "eudr_028", "data": {"y": 2}},
        ]
        chain = tracker.build_chain(steps)
        assert len(chain) == 2
        # First entry input is genesis
        assert chain[0]["input_hash"] == GENESIS_HASH
        # Second entry input is first entry output
        assert chain[1]["input_hash"] == chain[0]["output_hash"]
        # Built chain should also be verifiable
        assert tracker.verify_chain(chain) is True


class TestChainManagement:
    """Tests for get_chain and reset."""

    def test_get_chain_returns_copy(self, tracker: ProvenanceTracker):
        """get_chain should return a copy, not a reference."""
        tracker.create_entry("s", "src", GENESIS_HASH, "a" * 64)
        chain_copy = tracker.get_chain()
        chain_copy.clear()
        assert len(tracker.get_chain()) == 1

    def test_reset_clears_chain(self, tracker: ProvenanceTracker):
        """reset should empty the chain."""
        tracker.create_entry("s", "src", GENESIS_HASH, "a" * 64)
        assert len(tracker.get_chain()) == 1
        tracker.reset()
        assert len(tracker.get_chain()) == 0


class TestUnsupportedAlgorithm:
    """Test unsupported algorithm raises ValueError."""

    def test_unsupported_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            ProvenanceTracker(algorithm="md5")
