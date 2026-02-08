# -*- coding: utf-8 -*-
"""
Unit Tests for ProvenanceTracker (AGENT-FOUND-010)

Tests provenance entry recording, chain hashing, chain verification,
chain retrieval, JSON export, genesis hash, multiple entities,
statistics, and chain ordering.

Coverage target: 85%+ of provenance.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json

import pytest

from greenlang.observability_agent.provenance import ProvenanceTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker():
    return ProvenanceTracker()


# ==========================================================================
# Record Entry Tests
# ==========================================================================

class TestProvenanceTrackerRecord:
    """Tests for record."""

    def test_record_entry(self, tracker):
        chain_hash = tracker.record("metric", "met-001", "record", "abc123")
        assert chain_hash
        assert len(chain_hash) == 64  # SHA-256 hex

    def test_record_chain_hash_is_deterministic(self, tracker):
        # Two trackers recording the same sequence should produce same hashes
        tracker2 = ProvenanceTracker()
        h1 = tracker.record("metric", "met-001", "record", "abc123")
        h2 = tracker2.record("metric", "met-001", "record", "abc123")
        # Both should be equal since they start from same genesis and same data
        assert h1 == h2

    def test_record_increments_entry_count(self, tracker):
        assert tracker.entry_count == 0
        tracker.record("metric", "met-001", "record", "abc123")
        assert tracker.entry_count == 1
        tracker.record("metric", "met-002", "record", "def456")
        assert tracker.entry_count == 2

    def test_record_with_custom_user_id(self, tracker):
        chain_hash = tracker.record("metric", "met-001", "record", "abc", user_id="admin")
        chain = tracker.get_chain("met-001")
        assert chain[0]["user_id"] == "admin"

    def test_record_stores_all_fields(self, tracker):
        tracker.record("span", "span-001", "create", "hash123", user_id="test-user")
        chain = tracker.get_chain("span-001")
        entry = chain[0]
        assert entry["entity_type"] == "span"
        assert entry["entity_id"] == "span-001"
        assert entry["action"] == "create"
        assert entry["data_hash"] == "hash123"
        assert entry["user_id"] == "test-user"
        assert entry["timestamp"]
        assert entry["chain_hash"]

    def test_record_chain_links_to_previous(self, tracker):
        h1 = tracker.record("metric", "met-001", "record", "aaa")
        h2 = tracker.record("metric", "met-001", "update", "bbb")
        assert h1 != h2
        # The chain entries should be ordered
        chain = tracker.get_chain("met-001")
        assert len(chain) == 2
        assert chain[0]["chain_hash"] == h1
        assert chain[1]["chain_hash"] == h2


# ==========================================================================
# Verify Chain Tests
# ==========================================================================

class TestProvenanceTrackerVerify:
    """Tests for verify_chain."""

    def test_verify_chain_valid(self, tracker):
        tracker.record("metric", "met-001", "record", "abc")
        tracker.record("metric", "met-001", "update", "def")
        is_valid, chain = tracker.verify_chain("met-001")
        assert is_valid is True
        assert len(chain) == 2

    def test_verify_chain_empty_entity(self, tracker):
        is_valid, chain = tracker.verify_chain("nonexistent")
        assert is_valid is True
        assert chain == []

    def test_verify_chain_tampered(self, tracker):
        tracker.record("metric", "met-001", "record", "abc")
        # Tamper with the chain by removing chain_hash
        tracker._chain_store["met-001"][0]["chain_hash"] = ""
        is_valid, _chain = tracker.verify_chain("met-001")
        assert is_valid is False

    def test_verify_chain_missing_field(self, tracker):
        tracker.record("metric", "met-001", "record", "abc")
        tracker.record("metric", "met-001", "update", "def")
        # Remove a required field from the second entry
        del tracker._chain_store["met-001"][1]["action"]
        is_valid, _chain = tracker.verify_chain("met-001")
        assert is_valid is False


# ==========================================================================
# Get Chain Tests
# ==========================================================================

class TestProvenanceTrackerGetChain:
    """Tests for get_chain and get_global_chain."""

    def test_get_chain(self, tracker):
        tracker.record("metric", "met-001", "record", "abc")
        tracker.record("metric", "met-001", "update", "def")
        chain = tracker.get_chain("met-001")
        assert len(chain) == 2

    def test_get_chain_nonexistent(self, tracker):
        chain = tracker.get_chain("ghost")
        assert chain == []

    def test_get_chain_returns_copy(self, tracker):
        tracker.record("metric", "met-001", "record", "abc")
        chain1 = tracker.get_chain("met-001")
        chain2 = tracker.get_chain("met-001")
        assert chain1 is not chain2

    def test_get_global_chain(self, tracker):
        tracker.record("metric", "met-001", "record", "abc")
        tracker.record("span", "span-001", "create", "def")
        global_chain = tracker.get_global_chain()
        assert len(global_chain) == 2
        # Newest first
        assert global_chain[0]["entity_id"] == "span-001"

    def test_get_global_chain_limit(self, tracker):
        for i in range(10):
            tracker.record("metric", f"met-{i:03d}", "record", f"hash-{i}")
        global_chain = tracker.get_global_chain(limit=3)
        assert len(global_chain) == 3

    def test_chain_ordering(self, tracker):
        tracker.record("metric", "met-001", "record", "first")
        tracker.record("metric", "met-001", "update", "second")
        tracker.record("metric", "met-001", "delete", "third")
        chain = tracker.get_chain("met-001")
        assert chain[0]["action"] == "record"
        assert chain[1]["action"] == "update"
        assert chain[2]["action"] == "delete"


# ==========================================================================
# Export JSON Tests
# ==========================================================================

class TestProvenanceTrackerExport:
    """Tests for export_json."""

    def test_export_json(self, tracker):
        tracker.record("metric", "met-001", "record", "abc")
        exported = tracker.export_json()
        assert isinstance(exported, str)
        data = json.loads(exported)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_export_json_empty(self, tracker):
        exported = tracker.export_json()
        data = json.loads(exported)
        assert data == []


# ==========================================================================
# Genesis Hash Tests
# ==========================================================================

class TestProvenanceTrackerGenesis:
    """Tests for genesis hash."""

    def test_genesis_hash_exists(self, tracker):
        assert tracker._GENESIS_HASH
        assert len(tracker._GENESIS_HASH) == 64

    def test_genesis_hash_is_constant(self):
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        assert t1._GENESIS_HASH == t2._GENESIS_HASH

    def test_initial_last_chain_hash_is_genesis(self, tracker):
        assert tracker._last_chain_hash == tracker._GENESIS_HASH


# ==========================================================================
# Multiple Entities Tests
# ==========================================================================

class TestProvenanceTrackerMultipleEntities:
    """Tests for multiple entity tracking."""

    def test_multiple_entities(self, tracker):
        tracker.record("metric", "met-001", "record", "abc")
        tracker.record("span", "span-001", "create", "def")
        tracker.record("alert", "alert-001", "fire", "ghi")
        assert tracker.entity_count == 3

    def test_entities_are_independent(self, tracker):
        tracker.record("metric", "met-001", "record", "abc")
        tracker.record("span", "span-001", "create", "def")
        chain_met = tracker.get_chain("met-001")
        chain_span = tracker.get_chain("span-001")
        assert len(chain_met) == 1
        assert len(chain_span) == 1


# ==========================================================================
# Build Hash Tests
# ==========================================================================

class TestProvenanceTrackerBuildHash:
    """Tests for the build_hash utility."""

    def test_build_hash(self, tracker):
        h = tracker.build_hash({"key": "value"})
        assert len(h) == 64

    def test_build_hash_deterministic(self, tracker):
        h1 = tracker.build_hash({"a": 1, "b": 2})
        h2 = tracker.build_hash({"b": 2, "a": 1})
        assert h1 == h2  # sorted keys

    def test_build_hash_different_data(self, tracker):
        h1 = tracker.build_hash({"key": "value1"})
        h2 = tracker.build_hash({"key": "value2"})
        assert h1 != h2


# ==========================================================================
# Statistics Tests
# ==========================================================================

class TestProvenanceTrackerStatistics:
    """Tests for get_statistics."""

    def test_statistics_empty(self, tracker):
        stats = tracker.get_statistics()
        assert stats["entry_count"] == 0
        assert stats["entity_count"] == 0
        assert stats["genesis_hash"] == tracker._GENESIS_HASH
        assert stats["last_chain_hash"] == tracker._GENESIS_HASH

    def test_statistics_after_records(self, tracker):
        tracker.record("metric", "met-001", "record", "abc")
        tracker.record("span", "span-001", "create", "def")
        stats = tracker.get_statistics()
        assert stats["entry_count"] == 2
        assert stats["entity_count"] == 2
        assert stats["last_chain_hash"] != tracker._GENESIS_HASH
