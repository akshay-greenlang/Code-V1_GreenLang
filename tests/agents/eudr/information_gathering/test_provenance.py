# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-EUDR-027

Tests SHA-256 hash chain computation, entry creation, chain verification,
chain building, and reset behavior. Validates deterministic hashing,
sorted-key canonicalization, bytes hashing, broken chain detection,
and genesis hash handling.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (GL-EUDR-IGA-027)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from greenlang.agents.eudr.information_gathering.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


class TestComputeHash:
    """Test ProvenanceTracker.compute_hash determinism and correctness."""

    def test_compute_hash_deterministic(self, tracker):
        data = {"key": "value", "number": 42}
        h1 = tracker.compute_hash(data)
        h2 = tracker.compute_hash(data)
        assert h1 == h2
        assert len(h1) == 64

    def test_compute_hash_different_data(self, tracker):
        h1 = tracker.compute_hash({"a": 1})
        h2 = tracker.compute_hash({"a": 2})
        assert h1 != h2

    def test_compute_hash_sorted_keys(self, tracker):
        h1 = tracker.compute_hash({"b": 2, "a": 1})
        h2 = tracker.compute_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_compute_hash_matches_manual(self, tracker):
        data = {"key": "value"}
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert tracker.compute_hash(data) == expected

    def test_compute_hash_empty_dict(self, tracker):
        h = tracker.compute_hash({})
        assert len(h) == 64
        assert h == hashlib.sha256(b"{}").hexdigest()

    def test_compute_hash_nested_data(self, tracker):
        data = {"outer": {"inner": "value"}, "list": [1, 2, 3]}
        h = tracker.compute_hash(data)
        assert len(h) == 64


class TestComputeHashBytes:
    """Test ProvenanceTracker.compute_hash_bytes."""

    def test_compute_hash_bytes(self, tracker):
        data = b"hello world"
        h = tracker.compute_hash_bytes(data)
        assert h == hashlib.sha256(data).hexdigest()
        assert len(h) == 64

    def test_compute_hash_bytes_empty(self, tracker):
        h = tracker.compute_hash_bytes(b"")
        assert h == hashlib.sha256(b"").hexdigest()


class TestCreateEntry:
    """Test ProvenanceTracker.create_entry."""

    def test_create_entry(self, tracker):
        entry = tracker.create_entry(
            step="collect",
            source="eu_traces",
            input_hash="a" * 64,
            output_hash="b" * 64,
        )
        assert entry["step"] == "collect"
        assert entry["source"] == "eu_traces"
        assert entry["input_hash"] == "a" * 64
        assert entry["output_hash"] == "b" * 64
        assert entry["actor"] == "AGENT-EUDR-027"
        assert "timestamp" in entry

    def test_create_entry_custom_actor(self, tracker):
        entry = tracker.create_entry(
            step="normalize",
            source="internal",
            input_hash="x" * 64,
            output_hash="y" * 64,
            actor="CUSTOM-ACTOR",
        )
        assert entry["actor"] == "CUSTOM-ACTOR"

    def test_create_entry_custom_timestamp(self, tracker):
        ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        entry = tracker.create_entry(
            step="test",
            source="test",
            input_hash="a" * 64,
            output_hash="b" * 64,
            timestamp=ts,
        )
        assert entry["timestamp"] == ts.isoformat()

    def test_create_entry_appends_to_chain(self, tracker):
        assert len(tracker.get_chain()) == 0
        tracker.create_entry("step1", "src1", "a" * 64, "b" * 64)
        assert len(tracker.get_chain()) == 1
        tracker.create_entry("step2", "src2", "b" * 64, "c" * 64)
        assert len(tracker.get_chain()) == 2


class TestVerifyChain:
    """Test ProvenanceTracker.verify_chain."""

    def test_verify_chain_valid(self, tracker):
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "aaa"},
            {"input_hash": "aaa", "output_hash": "bbb"},
            {"input_hash": "bbb", "output_hash": "ccc"},
        ]
        assert tracker.verify_chain(entries) is True

    def test_verify_chain_broken(self, tracker):
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "aaa"},
            {"input_hash": "WRONG", "output_hash": "bbb"},
        ]
        assert tracker.verify_chain(entries) is False

    def test_verify_chain_empty(self, tracker):
        assert tracker.verify_chain([]) is True

    def test_verify_chain_single_entry(self, tracker):
        entries = [{"input_hash": GENESIS_HASH, "output_hash": "aaa"}]
        assert tracker.verify_chain(entries) is True


class TestBuildChain:
    """Test ProvenanceTracker.build_chain."""

    def test_build_chain(self, tracker):
        steps = [
            {"step": "collect", "source": "eu_traces", "data": {"key": "val1"}},
            {"step": "normalize", "source": "internal", "data": {"key": "val2"}},
        ]
        chain = tracker.build_chain(steps)
        assert len(chain) == 2
        assert chain[0]["step"] == "collect"
        assert chain[1]["step"] == "normalize"

    def test_build_chain_genesis_hash(self, tracker):
        steps = [
            {"step": "collect", "source": "test", "data": {"x": 1}},
        ]
        chain = tracker.build_chain(steps, genesis_hash=GENESIS_HASH)
        assert chain[0]["input_hash"] == GENESIS_HASH

    def test_build_chain_links_hashes(self, tracker):
        steps = [
            {"step": "s1", "source": "a", "data": {"k": "v1"}},
            {"step": "s2", "source": "b", "data": {"k": "v2"}},
            {"step": "s3", "source": "c", "data": {"k": "v3"}},
        ]
        chain = tracker.build_chain(steps)
        # Each step's input_hash should match previous step's output_hash
        for i in range(1, len(chain)):
            assert chain[i]["input_hash"] == chain[i - 1]["output_hash"]

    def test_build_chain_valid(self, tracker):
        steps = [
            {"step": "s1", "source": "a", "data": {"k": 1}},
            {"step": "s2", "source": "b", "data": {"k": 2}},
        ]
        chain = tracker.build_chain(steps)
        assert tracker.verify_chain(chain) is True


class TestGetChainAndReset:
    """Test ProvenanceTracker.get_chain and reset."""

    def test_get_chain_copy(self, tracker):
        tracker.create_entry("s1", "src", "a" * 64, "b" * 64)
        chain = tracker.get_chain()
        assert len(chain) == 1
        # Modifying returned list should not affect internal chain
        chain.clear()
        assert len(tracker.get_chain()) == 1

    def test_reset_clears_chain(self, tracker):
        tracker.create_entry("s1", "src", "a" * 64, "b" * 64)
        tracker.create_entry("s2", "src", "b" * 64, "c" * 64)
        assert len(tracker.get_chain()) == 2
        tracker.reset()
        assert len(tracker.get_chain()) == 0


class TestUnsupportedAlgorithm:
    """Test that unsupported hash algorithms raise ValueError."""

    def test_unsupported_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            ProvenanceTracker(algorithm="md5")

    def test_sha256_accepted(self):
        tracker = ProvenanceTracker(algorithm="sha256")
        assert tracker is not None
