# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-EUDR-031

Tests hash computation, chain building, verification, and state management
for the Stakeholder Engagement Tool provenance tracking.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from greenlang.agents.eudr.stakeholder_engagement.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


class TestGenesisHash:
    """Test GENESIS_HASH constant."""

    def test_genesis_hash_is_64_zeros(self):
        """Test GENESIS_HASH is a string of 64 zeros."""
        assert GENESIS_HASH == "0" * 64

    def test_genesis_hash_length(self):
        """Test GENESIS_HASH has exactly 64 characters."""
        assert len(GENESIS_HASH) == 64

    def test_genesis_hash_is_valid_hex(self):
        """Test GENESIS_HASH contains only valid hex characters."""
        assert all(c in "0123456789abcdef" for c in GENESIS_HASH)


class TestProvenanceTrackerInit:
    """Test ProvenanceTracker initialization."""

    def test_default_init(self):
        """Test ProvenanceTracker initializes with default algorithm."""
        tracker = ProvenanceTracker()
        assert tracker._algorithm == "sha256"
        assert tracker.get_chain() == []

    def test_sha256_init(self):
        """Test ProvenanceTracker initializes with sha256 algorithm."""
        tracker = ProvenanceTracker(algorithm="sha256")
        assert tracker._algorithm == "sha256"

    def test_unsupported_algorithm_raises(self):
        """Test ProvenanceTracker raises error for unsupported algorithm."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            ProvenanceTracker(algorithm="md5")

    def test_unsupported_algorithm_error_message(self):
        """Test ProvenanceTracker error message for unsupported algorithm."""
        with pytest.raises(ValueError) as exc_info:
            ProvenanceTracker(algorithm="sha1")
        assert "sha1" in str(exc_info.value)
        assert "sha256" in str(exc_info.value)


class TestComputeHash:
    """Test compute_hash determinism and correctness."""

    def test_hash_returns_64_char_hex(self, provenance_tracker):
        """Test compute_hash returns 64-character hex string."""
        h = provenance_tracker.compute_hash({"key": "value"})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_is_deterministic(self, provenance_tracker):
        """Test compute_hash produces same hash for same input."""
        data = {"stakeholder_id": "STK-001", "status": "active"}
        h1 = provenance_tracker.compute_hash(data)
        h2 = provenance_tracker.compute_hash(data)
        assert h1 == h2

    def test_hash_key_order_independent(self, provenance_tracker):
        """Test compute_hash is independent of key order."""
        h1 = provenance_tracker.compute_hash({"b": 2, "a": 1})
        h2 = provenance_tracker.compute_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_hash_different_data_different_hash(self, provenance_tracker):
        """Test compute_hash produces different hashes for different data."""
        h1 = provenance_tracker.compute_hash({"stakeholder_id": "STK-001"})
        h2 = provenance_tracker.compute_hash({"stakeholder_id": "STK-002"})
        assert h1 != h2

    def test_hash_matches_manual_computation(self, provenance_tracker):
        """Test compute_hash matches manual SHA-256 computation."""
        data = {"grievance_id": "GRV-001", "severity": "critical"}
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        result = provenance_tracker.compute_hash(data)
        assert result == expected

    def test_hash_empty_dict(self, provenance_tracker):
        """Test compute_hash handles empty dictionary."""
        h = provenance_tracker.compute_hash({})
        assert len(h) == 64
        assert isinstance(h, str)

    def test_hash_nested_dict(self, provenance_tracker):
        """Test compute_hash handles nested dictionaries."""
        data = {"stakeholder": {"rights": {"fpic_required": True}}}
        h = provenance_tracker.compute_hash(data)
        assert len(h) == 64

    def test_hash_with_datetime(self, provenance_tracker):
        """Test compute_hash handles datetime objects."""
        data = {"timestamp": datetime.now(timezone.utc)}
        h = provenance_tracker.compute_hash(data)
        assert len(h) == 64

    def test_hash_with_list(self, provenance_tracker):
        """Test compute_hash handles lists."""
        data = {"stakeholder_ids": ["STK-001", "STK-002", "STK-003"]}
        h = provenance_tracker.compute_hash(data)
        assert len(h) == 64

    def test_hash_with_mixed_types(self, provenance_tracker):
        """Test compute_hash handles mixed data types."""
        data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
        }
        h = provenance_tracker.compute_hash(data)
        assert len(h) == 64


class TestComputeHashBytes:
    """Test compute_hash_bytes for raw byte hashing."""

    def test_hash_bytes_returns_64_chars(self):
        """Test compute_hash_bytes returns 64-character hex string."""
        h = ProvenanceTracker.compute_hash_bytes(b"stakeholder engagement data")
        assert len(h) == 64

    def test_hash_bytes_deterministic(self):
        """Test compute_hash_bytes is deterministic."""
        data = b"fpic consent document"
        h1 = ProvenanceTracker.compute_hash_bytes(data)
        h2 = ProvenanceTracker.compute_hash_bytes(data)
        assert h1 == h2

    def test_hash_bytes_matches_hashlib(self):
        """Test compute_hash_bytes matches hashlib SHA-256."""
        data = b"grievance mechanism test"
        expected = hashlib.sha256(data).hexdigest()
        result = ProvenanceTracker.compute_hash_bytes(data)
        assert result == expected

    def test_hash_bytes_empty(self):
        """Test compute_hash_bytes handles empty bytes."""
        h = ProvenanceTracker.compute_hash_bytes(b"")
        assert len(h) == 64

    def test_hash_bytes_different_data(self):
        """Test compute_hash_bytes produces different hashes for different data."""
        h1 = ProvenanceTracker.compute_hash_bytes(b"consultation_001")
        h2 = ProvenanceTracker.compute_hash_bytes(b"consultation_002")
        assert h1 != h2

    def test_hash_bytes_utf8(self):
        """Test compute_hash_bytes handles UTF-8 encoded strings."""
        text = "Comunidad Wayuu - La Guajira"
        h = ProvenanceTracker.compute_hash_bytes(text.encode("utf-8"))
        assert len(h) == 64


class TestCreateEntry:
    """Test provenance entry creation."""

    def test_create_entry_returns_dict(self, provenance_tracker):
        """Test create_entry returns a dictionary."""
        entry = provenance_tracker.create_entry(
            step="map_stakeholder",
            source="eudr_031",
            input_hash=GENESIS_HASH,
            output_hash="a" * 64,
        )
        assert isinstance(entry, dict)
        assert entry["step"] == "map_stakeholder"
        assert entry["source"] == "eudr_031"
        assert entry["input_hash"] == GENESIS_HASH
        assert entry["output_hash"] == "a" * 64
        assert entry["actor"] == "AGENT-EUDR-031"

    def test_create_entry_appends_to_chain(self, provenance_tracker):
        """Test create_entry appends to internal chain."""
        assert len(provenance_tracker.get_chain()) == 0
        provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="b" * 64,
        )
        assert len(provenance_tracker.get_chain()) == 1

    def test_create_entry_with_custom_actor(self, provenance_tracker):
        """Test create_entry with custom actor."""
        entry = provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="c" * 64,
            actor="custom-actor",
        )
        assert entry["actor"] == "custom-actor"

    def test_create_entry_with_custom_timestamp(self, provenance_tracker):
        """Test create_entry with custom timestamp."""
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        entry = provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="d" * 64,
            timestamp=ts,
        )
        assert entry["timestamp"] == ts.isoformat()

    def test_create_entry_default_timestamp(self, provenance_tracker):
        """Test create_entry uses current timestamp by default."""
        entry = provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="e" * 64,
        )
        assert "timestamp" in entry
        assert isinstance(entry["timestamp"], str)

    def test_create_entry_multiple_entries(self, provenance_tracker):
        """Test creating multiple entries."""
        provenance_tracker.create_entry(
            step="step1", source="src1",
            input_hash=GENESIS_HASH, output_hash="f" * 64,
        )
        provenance_tracker.create_entry(
            step="step2", source="src2",
            input_hash="f" * 64, output_hash="g" * 64,
        )
        assert len(provenance_tracker.get_chain()) == 2

    def test_create_entry_preserves_all_fields(self, provenance_tracker):
        """Test create_entry preserves all required fields."""
        entry = provenance_tracker.create_entry(
            step="initiate_fpic",
            source="eudr_031",
            input_hash="input123",
            output_hash="output456",
            actor="test-agent",
        )
        assert "step" in entry
        assert "source" in entry
        assert "timestamp" in entry
        assert "actor" in entry
        assert "input_hash" in entry
        assert "output_hash" in entry


class TestVerifyChain:
    """Test chain verification logic."""

    def test_empty_chain_is_valid(self, provenance_tracker):
        """Test verify_chain returns True for empty chain."""
        assert provenance_tracker.verify_chain([]) is True

    def test_single_entry_chain_is_valid(self, provenance_tracker):
        """Test verify_chain returns True for single entry."""
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "a" * 64}
        ]
        assert provenance_tracker.verify_chain(entries) is True

    def test_valid_linked_chain(self, provenance_tracker):
        """Test verify_chain returns True for valid linked chain."""
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "aaa"},
            {"input_hash": "aaa", "output_hash": "bbb"},
            {"input_hash": "bbb", "output_hash": "ccc"},
        ]
        assert provenance_tracker.verify_chain(entries) is True

    def test_broken_chain_returns_false(self, provenance_tracker):
        """Test verify_chain returns False for broken chain."""
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "aaa"},
            {"input_hash": "WRONG", "output_hash": "bbb"},
        ]
        assert provenance_tracker.verify_chain(entries) is False

    def test_chain_with_missing_hash_fields(self, provenance_tracker):
        """Test verify_chain handles missing hash fields."""
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "aaa"},
            {"input_hash": "", "output_hash": "bbb"},
        ]
        result = provenance_tracker.verify_chain(entries)
        assert isinstance(result, bool)

    def test_chain_verification_multiple_breaks(self, provenance_tracker):
        """Test verify_chain detects first break in chain."""
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "aaa"},
            {"input_hash": "WRONG1", "output_hash": "bbb"},
            {"input_hash": "WRONG2", "output_hash": "ccc"},
        ]
        assert provenance_tracker.verify_chain(entries) is False

    def test_valid_stakeholder_provenance_chain(self, provenance_tracker):
        """Test valid chain for stakeholder engagement workflow."""
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "stk_mapped"},
            {"input_hash": "stk_mapped", "output_hash": "fpic_initiated"},
            {"input_hash": "fpic_initiated", "output_hash": "consultation_held"},
            {"input_hash": "consultation_held", "output_hash": "consent_recorded"},
        ]
        assert provenance_tracker.verify_chain(entries) is True

    def test_four_entry_chain_with_break(self, provenance_tracker):
        """Test chain break detection in longer chain."""
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "hash1"},
            {"input_hash": "hash1", "output_hash": "hash2"},
            {"input_hash": "TAMPERED", "output_hash": "hash3"},
            {"input_hash": "hash3", "output_hash": "hash4"},
        ]
        assert provenance_tracker.verify_chain(entries) is False

    def test_verify_chain_returns_bool(self, provenance_tracker):
        """Test verify_chain always returns a boolean."""
        assert isinstance(provenance_tracker.verify_chain([]), bool)
        assert isinstance(provenance_tracker.verify_chain([{"input_hash": "a", "output_hash": "b"}]), bool)

    def test_chain_with_only_output_hashes(self, provenance_tracker):
        """Test chain with entries missing input_hash key."""
        entries = [
            {"output_hash": "aaa"},
            {"output_hash": "bbb"},
        ]
        result = provenance_tracker.verify_chain(entries)
        assert isinstance(result, bool)


class TestBuildChain:
    """Test building a chain from step definitions."""

    def test_build_chain_empty_steps(self, provenance_tracker):
        """Test build_chain returns empty list for empty steps."""
        chain = provenance_tracker.build_chain(steps=[])
        assert chain == []

    def test_build_chain_single_step(self, provenance_tracker):
        """Test build_chain with single step."""
        steps = [
            {"step": "map_stakeholder", "source": "eudr_031", "data": {"stakeholder_id": "STK-001"}}
        ]
        chain = provenance_tracker.build_chain(steps)
        assert len(chain) == 1
        assert chain[0]["step"] == "map_stakeholder"
        assert chain[0]["input_hash"] == GENESIS_HASH

    def test_build_chain_multiple_steps_linked(self, provenance_tracker):
        """Test build_chain creates linked chain."""
        steps = [
            {"step": "map_stakeholder", "source": "s1", "data": {"a": 1}},
            {"step": "initiate_fpic", "source": "s2", "data": {"b": 2}},
            {"step": "record_consent", "source": "s3", "data": {"c": 3}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert len(chain) == 3
        assert chain[0]["input_hash"] == GENESIS_HASH
        assert chain[1]["input_hash"] == chain[0]["output_hash"]
        assert chain[2]["input_hash"] == chain[1]["output_hash"]

    def test_built_chain_verifies(self, provenance_tracker):
        """Test build_chain produces verifiable chain."""
        steps = [
            {"step": "s1", "source": "a", "data": {"x": 1}},
            {"step": "s2", "source": "b", "data": {"y": 2}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert provenance_tracker.verify_chain(chain) is True

    def test_build_chain_with_custom_genesis(self, provenance_tracker):
        """Test build_chain with custom genesis hash."""
        custom_genesis = "f" * 64
        steps = [
            {"step": "step1", "source": "src", "data": {"test": "data"}}
        ]
        chain = provenance_tracker.build_chain(steps, genesis_hash=custom_genesis)
        assert chain[0]["input_hash"] == custom_genesis

    def test_build_chain_handles_empty_data(self, provenance_tracker):
        """Test build_chain handles steps with empty data."""
        steps = [
            {"step": "step1", "source": "src", "data": {}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert len(chain) == 1

    def test_build_chain_handles_missing_data(self, provenance_tracker):
        """Test build_chain handles steps without data field."""
        steps = [
            {"step": "step1", "source": "src"},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert len(chain) == 1

    def test_build_chain_preserves_step_info(self, provenance_tracker):
        """Test build_chain preserves step and source information."""
        steps = [
            {"step": "submit_grievance", "source": "eudr_031", "data": {"grv": 1}},
            {"step": "triage_grievance", "source": "eudr_031", "data": {"grv": 2}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert chain[0]["step"] == "submit_grievance"
        assert chain[0]["source"] == "eudr_031"
        assert chain[1]["step"] == "triage_grievance"
        assert chain[1]["source"] == "eudr_031"

    def test_build_chain_stakeholder_workflow(self, provenance_tracker):
        """Test build_chain with full stakeholder engagement workflow."""
        steps = [
            {"step": "discover_stakeholders", "source": "supply_chain", "data": {"region": "CO"}},
            {"step": "classify_rights", "source": "legal_db", "data": {"category": "indigenous"}},
            {"step": "initiate_fpic", "source": "workflow", "data": {"stage": "notification"}},
            {"step": "conduct_consultation", "source": "field", "data": {"session": 1}},
            {"step": "record_consent", "source": "fpic", "data": {"consent": "granted"}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert len(chain) == 5
        assert provenance_tracker.verify_chain(chain) is True

    def test_build_chain_adds_to_internal_chain(self, provenance_tracker):
        """Test build_chain updates the internal chain state."""
        steps = [
            {"step": "step1", "source": "src", "data": {"x": 1}},
        ]
        provenance_tracker.build_chain(steps)
        internal = provenance_tracker.get_chain()
        assert len(internal) >= 1


class TestGetChainAndReset:
    """Test get_chain and reset methods."""

    def test_get_chain_returns_copy(self, provenance_tracker):
        """Test get_chain returns a copy of the internal chain."""
        provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="e" * 64,
        )
        chain = provenance_tracker.get_chain()
        chain.append({"fake": True})
        assert len(provenance_tracker.get_chain()) == 1

    def test_get_chain_returns_list(self, provenance_tracker):
        """Test get_chain returns a list."""
        chain = provenance_tracker.get_chain()
        assert isinstance(chain, list)

    def test_get_chain_empty_initially(self, provenance_tracker):
        """Test get_chain returns empty list initially."""
        assert provenance_tracker.get_chain() == []

    def test_reset_clears_chain(self, provenance_tracker):
        """Test reset clears the internal chain."""
        provenance_tracker.create_entry(
            step="test", source="src",
            input_hash=GENESIS_HASH, output_hash="f" * 64,
        )
        assert len(provenance_tracker.get_chain()) == 1
        provenance_tracker.reset()
        assert len(provenance_tracker.get_chain()) == 0

    def test_reset_allows_new_entries(self, provenance_tracker):
        """Test reset allows new entries to be added."""
        provenance_tracker.create_entry(
            step="test1", source="src1",
            input_hash=GENESIS_HASH, output_hash="g" * 64,
        )
        provenance_tracker.reset()
        provenance_tracker.create_entry(
            step="test2", source="src2",
            input_hash=GENESIS_HASH, output_hash="h" * 64,
        )
        assert len(provenance_tracker.get_chain()) == 1
        assert provenance_tracker.get_chain()[0]["step"] == "test2"

    def test_multiple_reset_calls(self, provenance_tracker):
        """Test multiple reset calls work correctly."""
        provenance_tracker.reset()
        provenance_tracker.reset()
        provenance_tracker.reset()
        assert provenance_tracker.get_chain() == []
