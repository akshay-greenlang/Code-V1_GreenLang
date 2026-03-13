# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-EUDR-034

Tests hash computation, chain building, verification, and state management.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from greenlang.agents.eudr.annual_review_scheduler.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


class TestGenesisHash:
    """Test GENESIS_HASH constant."""

    def test_genesis_hash_is_64_zeros(self):
        assert GENESIS_HASH == "0" * 64

    def test_genesis_hash_length(self):
        assert len(GENESIS_HASH) == 64


class TestProvenanceTrackerInit:
    """Test ProvenanceTracker initialization."""

    def test_default_init(self):
        tracker = ProvenanceTracker()
        assert tracker._algorithm == "sha256"
        assert tracker.get_chain() == []

    def test_sha256_init(self):
        tracker = ProvenanceTracker(algorithm="sha256")
        assert tracker._algorithm == "sha256"

    def test_unsupported_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            ProvenanceTracker(algorithm="md5")


class TestComputeHash:
    """Test compute_hash determinism and correctness."""

    def test_hash_returns_64_char_hex(self, provenance_tracker):
        h = provenance_tracker.compute_hash({"key": "value"})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_is_deterministic(self, provenance_tracker):
        data = {"a": 1, "b": "hello", "c": [1, 2, 3]}
        h1 = provenance_tracker.compute_hash(data)
        h2 = provenance_tracker.compute_hash(data)
        assert h1 == h2

    def test_hash_key_order_independent(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash({"b": 2, "a": 1})
        h2 = provenance_tracker.compute_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_hash_different_data_different_hash(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash({"x": 1})
        h2 = provenance_tracker.compute_hash({"x": 2})
        assert h1 != h2

    def test_hash_matches_manual_computation(self, provenance_tracker):
        data = {"key": "value"}
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        result = provenance_tracker.compute_hash(data)
        assert result == expected

    def test_hash_with_decimal_values(self, provenance_tracker):
        from decimal import Decimal
        data = {"score": Decimal("95.50"), "rate": Decimal("0.01")}
        h = provenance_tracker.compute_hash(data)
        assert len(h) == 64

    def test_hash_with_datetime_values(self, provenance_tracker):
        data = {"timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc)}
        h = provenance_tracker.compute_hash(data)
        assert len(h) == 64

    def test_hash_with_nested_data(self, provenance_tracker):
        data = {"outer": {"inner": {"deep": [1, 2, 3]}}}
        h = provenance_tracker.compute_hash(data)
        assert len(h) == 64


class TestComputeHashBytes:
    """Test compute_hash_bytes for raw byte hashing."""

    def test_hash_bytes_returns_64_chars(self):
        h = ProvenanceTracker.compute_hash_bytes(b"test data")
        assert len(h) == 64

    def test_hash_bytes_deterministic(self):
        data = b"deterministic test"
        h1 = ProvenanceTracker.compute_hash_bytes(data)
        h2 = ProvenanceTracker.compute_hash_bytes(data)
        assert h1 == h2

    def test_hash_bytes_matches_hashlib(self):
        data = b"hello world"
        expected = hashlib.sha256(data).hexdigest()
        result = ProvenanceTracker.compute_hash_bytes(data)
        assert result == expected

    def test_hash_bytes_empty_input(self):
        h = ProvenanceTracker.compute_hash_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert h == expected


class TestCreateEntry:
    """Test provenance entry creation."""

    def test_create_entry_returns_dict(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="schedule_review",
            source="eudr_034",
            input_hash=GENESIS_HASH,
            output_hash="a" * 64,
        )
        assert isinstance(entry, dict)
        assert entry["step"] == "schedule_review"
        assert entry["source"] == "eudr_034"
        assert entry["input_hash"] == GENESIS_HASH
        assert entry["output_hash"] == "a" * 64
        assert entry["actor"] == "AGENT-EUDR-034"

    def test_create_entry_appends_to_chain(self, provenance_tracker):
        assert len(provenance_tracker.get_chain()) == 0
        provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="b" * 64,
        )
        assert len(provenance_tracker.get_chain()) == 1

    def test_create_entry_with_custom_actor(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="c" * 64,
            actor="custom-actor",
        )
        assert entry["actor"] == "custom-actor"

    def test_create_entry_with_custom_timestamp(self, provenance_tracker):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        entry = provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="d" * 64,
            timestamp=ts,
        )
        assert entry["timestamp"] == ts.isoformat()

    def test_create_multiple_entries(self, provenance_tracker):
        for i in range(5):
            provenance_tracker.create_entry(
                step=f"step_{i}", source="src",
                input_hash=GENESIS_HASH, output_hash=f"{i}" * 64,
            )
        assert len(provenance_tracker.get_chain()) == 5


class TestVerifyChain:
    """Test chain verification logic."""

    def test_empty_chain_is_valid(self, provenance_tracker):
        assert provenance_tracker.verify_chain([]) is True

    def test_single_entry_chain_is_valid(self, provenance_tracker):
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "a" * 64}
        ]
        assert provenance_tracker.verify_chain(entries) is True

    def test_valid_linked_chain(self, provenance_tracker):
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "aaa"},
            {"input_hash": "aaa", "output_hash": "bbb"},
            {"input_hash": "bbb", "output_hash": "ccc"},
        ]
        assert provenance_tracker.verify_chain(entries) is True

    def test_broken_chain_returns_false(self, provenance_tracker):
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "aaa"},
            {"input_hash": "WRONG", "output_hash": "bbb"},
        ]
        assert provenance_tracker.verify_chain(entries) is False

    def test_long_valid_chain(self, provenance_tracker):
        entries = [{"input_hash": GENESIS_HASH, "output_hash": "h0"}]
        for i in range(1, 20):
            entries.append({
                "input_hash": f"h{i - 1}",
                "output_hash": f"h{i}",
            })
        assert provenance_tracker.verify_chain(entries) is True

    def test_broken_chain_at_end(self, provenance_tracker):
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "aaa"},
            {"input_hash": "aaa", "output_hash": "bbb"},
            {"input_hash": "TAMPERED", "output_hash": "ccc"},
        ]
        assert provenance_tracker.verify_chain(entries) is False


class TestBuildChain:
    """Test building a chain from step definitions."""

    def test_build_chain_empty_steps(self, provenance_tracker):
        chain = provenance_tracker.build_chain(steps=[])
        assert chain == []

    def test_build_chain_single_step(self, provenance_tracker):
        steps = [
            {"step": "init", "source": "src", "data": {"key": "value"}}
        ]
        chain = provenance_tracker.build_chain(steps)
        assert len(chain) == 1
        assert chain[0]["step"] == "init"
        assert chain[0]["input_hash"] == GENESIS_HASH

    def test_build_chain_multiple_steps_linked(self, provenance_tracker):
        steps = [
            {"step": "step1", "source": "s1", "data": {"a": 1}},
            {"step": "step2", "source": "s2", "data": {"b": 2}},
            {"step": "step3", "source": "s3", "data": {"c": 3}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert len(chain) == 3
        assert chain[0]["input_hash"] == GENESIS_HASH
        assert chain[1]["input_hash"] == chain[0]["output_hash"]
        assert chain[2]["input_hash"] == chain[1]["output_hash"]

    def test_built_chain_verifies(self, provenance_tracker):
        steps = [
            {"step": "s1", "source": "a", "data": {"x": 1}},
            {"step": "s2", "source": "b", "data": {"y": 2}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert provenance_tracker.verify_chain(chain) is True


class TestGetChainAndReset:
    """Test get_chain and reset methods."""

    def test_get_chain_returns_copy(self, provenance_tracker):
        provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="e" * 64,
        )
        chain = provenance_tracker.get_chain()
        chain.append({"fake": True})
        assert len(provenance_tracker.get_chain()) == 1

    def test_reset_clears_chain(self, provenance_tracker):
        provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="f" * 64,
        )
        assert len(provenance_tracker.get_chain()) == 1
        provenance_tracker.reset()
        assert len(provenance_tracker.get_chain()) == 0

    def test_reset_allows_fresh_chain(self, provenance_tracker):
        provenance_tracker.create_entry(
            step="old", source="src",
            input_hash=GENESIS_HASH, output_hash="o" * 64,
        )
        provenance_tracker.reset()
        provenance_tracker.create_entry(
            step="new", source="src",
            input_hash=GENESIS_HASH, output_hash="n" * 64,
        )
        chain = provenance_tracker.get_chain()
        assert len(chain) == 1
        assert chain[0]["step"] == "new"
