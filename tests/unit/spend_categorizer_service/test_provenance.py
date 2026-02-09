# -*- coding: utf-8 -*-
"""
Unit tests for Spend Data Categorizer ProvenanceTracker (AGENT-DATA-009)

Tests the ProvenanceTracker class including initialization, record creation,
chain verification, chain retrieval, JSON export, build_hash utility, and
chain isolation between entities.

Target: 50+ tests for comprehensive provenance tracking coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json

import pytest

from greenlang.spend_categorizer.provenance import ProvenanceTracker


# ============================================================================
# Initialization tests
# ============================================================================


class TestProvenanceTrackerInit:
    """Test ProvenanceTracker initializes correctly."""

    def test_genesis_hash_is_sha256(self):
        expected = hashlib.sha256(b"greenlang-spend-categorizer-genesis").hexdigest()
        assert ProvenanceTracker._GENESIS_HASH == expected

    def test_genesis_hash_is_64_chars(self):
        assert len(ProvenanceTracker._GENESIS_HASH) == 64

    def test_entry_count_starts_at_zero(self):
        tracker = ProvenanceTracker()
        assert tracker.entry_count == 0

    def test_entity_count_starts_at_zero(self):
        tracker = ProvenanceTracker()
        assert tracker.entity_count == 0

    def test_last_chain_hash_is_genesis(self):
        tracker = ProvenanceTracker()
        assert tracker._last_chain_hash == ProvenanceTracker._GENESIS_HASH

    def test_chain_store_is_empty_dict(self):
        tracker = ProvenanceTracker()
        assert tracker._chain_store == {}

    def test_global_chain_is_empty_list(self):
        tracker = ProvenanceTracker()
        assert tracker._global_chain == []


# ============================================================================
# record() tests
# ============================================================================


class TestProvenanceTrackerRecord:
    """Test ProvenanceTracker.record() method."""

    def test_record_returns_string(self):
        tracker = ProvenanceTracker()
        result = tracker.record("spend_record", "rec-001", "ingest", "abc123")
        assert isinstance(result, str)

    def test_record_returns_64_char_hex(self):
        tracker = ProvenanceTracker()
        result = tracker.record("spend_record", "rec-001", "ingest", "abc123")
        assert len(result) == 64
        int(result, 16)  # Should not raise; valid hex

    def test_record_increments_entry_count(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        assert tracker.entry_count == 1
        tracker.record("spend_record", "rec-002", "ingest", "h2")
        assert tracker.entry_count == 2

    def test_record_increments_entity_count(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        assert tracker.entity_count == 1
        tracker.record("spend_record", "rec-002", "ingest", "h2")
        assert tracker.entity_count == 2

    def test_same_entity_does_not_increment_entity_count(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        tracker.record("spend_record", "rec-001", "classify", "h2")
        assert tracker.entity_count == 1
        assert tracker.entry_count == 2

    def test_record_groups_by_entity_id(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        tracker.record("spend_record", "rec-001", "classify", "h2")
        chain = tracker.get_chain("rec-001")
        assert len(chain) == 2

    def test_record_default_user_id(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        chain = tracker.get_chain("rec-001")
        assert chain[0]["user_id"] == "system"

    def test_record_custom_user_id(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1", user_id="user-42")
        chain = tracker.get_chain("rec-001")
        assert chain[0]["user_id"] == "user-42"

    def test_chain_linking_different_hashes(self):
        tracker = ProvenanceTracker()
        hash1 = tracker.record("spend_record", "rec-001", "ingest", "h1")
        hash2 = tracker.record("spend_record", "rec-001", "classify", "h2")
        assert hash1 != hash2

    def test_entry_contains_required_fields(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "abc123")
        chain = tracker.get_chain("rec-001")
        entry = chain[0]
        required = {"entity_type", "entity_id", "action", "data_hash", "user_id", "timestamp", "chain_hash"}
        assert required.issubset(entry.keys())

    def test_entry_entity_type_matches(self):
        tracker = ProvenanceTracker()
        tracker.record("classification", "cls-001", "classify", "h1")
        chain = tracker.get_chain("cls-001")
        assert chain[0]["entity_type"] == "classification"

    def test_entry_action_matches(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        chain = tracker.get_chain("rec-001")
        assert chain[0]["action"] == "ingest"

    def test_entry_data_hash_matches(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "my_data_hash")
        chain = tracker.get_chain("rec-001")
        assert chain[0]["data_hash"] == "my_data_hash"

    def test_entry_timestamp_is_isoformat(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        chain = tracker.get_chain("rec-001")
        ts = chain[0]["timestamp"]
        # Should be parseable ISO format
        assert "T" in ts


# ============================================================================
# verify_chain() tests
# ============================================================================


class TestProvenanceTrackerVerifyChain:
    """Test ProvenanceTracker.verify_chain() method."""

    def test_empty_chain_is_valid(self):
        tracker = ProvenanceTracker()
        is_valid, chain = tracker.verify_chain("nonexistent")
        assert is_valid is True
        assert chain == []

    def test_single_entry_chain_is_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        is_valid, chain = tracker.verify_chain("rec-001")
        assert is_valid is True
        assert len(chain) == 1

    def test_multiple_entry_chain_is_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        tracker.record("spend_record", "rec-001", "classify", "h2")
        tracker.record("spend_record", "rec-001", "map", "h3")
        is_valid, chain = tracker.verify_chain("rec-001")
        assert is_valid is True
        assert len(chain) == 3

    def test_tampered_chain_detected(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")

        # Tamper with the chain hash
        tracker._chain_store["rec-001"][0]["chain_hash"] = ""

        is_valid, chain = tracker.verify_chain("rec-001")
        assert is_valid is False

    def test_tampered_missing_field_detected(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")

        # Remove a required field
        del tracker._chain_store["rec-001"][0]["action"]

        is_valid, chain = tracker.verify_chain("rec-001")
        assert is_valid is False

    def test_verify_returns_chain_entries(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        tracker.record("spend_record", "rec-001", "classify", "h2")
        is_valid, chain = tracker.verify_chain("rec-001")
        assert len(chain) == 2
        assert chain[0]["action"] == "ingest"
        assert chain[1]["action"] == "classify"


# ============================================================================
# get_chain() tests
# ============================================================================


class TestProvenanceTrackerGetChain:
    """Test ProvenanceTracker.get_chain() method."""

    def test_unknown_entity_returns_empty(self):
        tracker = ProvenanceTracker()
        chain = tracker.get_chain("nonexistent")
        assert chain == []

    def test_returns_copy_not_reference(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        chain = tracker.get_chain("rec-001")
        chain.append({"fake": "entry"})
        # Internal chain should not be modified
        assert len(tracker.get_chain("rec-001")) == 1

    def test_order_oldest_first(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        tracker.record("spend_record", "rec-001", "classify", "h2")
        tracker.record("spend_record", "rec-001", "calculate", "h3")
        chain = tracker.get_chain("rec-001")
        assert chain[0]["action"] == "ingest"
        assert chain[1]["action"] == "classify"
        assert chain[2]["action"] == "calculate"


# ============================================================================
# get_global_chain() tests
# ============================================================================


class TestProvenanceTrackerGetGlobalChain:
    """Test ProvenanceTracker.get_global_chain() method."""

    def test_empty_global_chain(self):
        tracker = ProvenanceTracker()
        chain = tracker.get_global_chain()
        assert chain == []

    def test_global_chain_ordering_newest_first(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        tracker.record("spend_record", "rec-002", "ingest", "h2")
        chain = tracker.get_global_chain()
        # Newest first
        assert chain[0]["entity_id"] == "rec-002"
        assert chain[1]["entity_id"] == "rec-001"

    def test_global_chain_limit(self):
        tracker = ProvenanceTracker()
        for i in range(10):
            tracker.record("spend_record", f"rec-{i:03d}", "ingest", f"h{i}")
        chain = tracker.get_global_chain(limit=5)
        assert len(chain) == 5

    def test_global_chain_default_limit_100(self):
        tracker = ProvenanceTracker()
        for i in range(150):
            tracker.record("spend_record", f"rec-{i:03d}", "ingest", f"h{i}")
        chain = tracker.get_global_chain()
        assert len(chain) == 100

    def test_global_chain_includes_all_entities(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        tracker.record("classification", "cls-001", "classify", "h2")
        chain = tracker.get_global_chain()
        entity_types = {e["entity_type"] for e in chain}
        assert "spend_record" in entity_types
        assert "classification" in entity_types


# ============================================================================
# export_json() tests
# ============================================================================


class TestProvenanceTrackerExportJson:
    """Test ProvenanceTracker.export_json() method."""

    def test_empty_export(self):
        tracker = ProvenanceTracker()
        result = tracker.export_json()
        parsed = json.loads(result)
        assert parsed == []

    def test_export_with_entries(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        tracker.record("spend_record", "rec-002", "ingest", "h2")
        result = tracker.export_json()
        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_export_contains_required_fields(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        result = tracker.export_json()
        parsed = json.loads(result)
        entry = parsed[0]
        assert "entity_type" in entry
        assert "entity_id" in entry
        assert "action" in entry
        assert "data_hash" in entry
        assert "chain_hash" in entry

    def test_export_has_indentation(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        result = tracker.export_json()
        # JSON with indent=2 should have newlines and spaces
        assert "\n" in result
        assert "  " in result

    def test_export_is_valid_json(self):
        tracker = ProvenanceTracker()
        for i in range(5):
            tracker.record("spend_record", f"rec-{i}", "ingest", f"h{i}")
        result = tracker.export_json()
        json.loads(result)  # Should not raise


# ============================================================================
# build_hash() tests
# ============================================================================


class TestProvenanceTrackerBuildHash:
    """Test ProvenanceTracker.build_hash() method."""

    def test_returns_64_char_hex(self):
        tracker = ProvenanceTracker()
        result = tracker.build_hash({"key": "value"})
        assert len(result) == 64
        int(result, 16)

    def test_deterministic_same_data(self):
        tracker = ProvenanceTracker()
        h1 = tracker.build_hash({"a": 1, "b": 2})
        h2 = tracker.build_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_different_data_different_hash(self):
        tracker = ProvenanceTracker()
        h1 = tracker.build_hash({"a": 1})
        h2 = tracker.build_hash({"a": 2})
        assert h1 != h2

    def test_key_order_independent(self):
        tracker = ProvenanceTracker()
        h1 = tracker.build_hash({"a": 1, "b": 2})
        h2 = tracker.build_hash({"b": 2, "a": 1})
        assert h1 == h2, "build_hash should use sort_keys=True"

    def test_matches_manual_sha256(self):
        tracker = ProvenanceTracker()
        data = {"key": "value"}
        serialized = json.dumps(data, sort_keys=True, default=str)
        expected = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        actual = tracker.build_hash(data)
        assert actual == expected

    def test_hash_with_list_data(self):
        tracker = ProvenanceTracker()
        result = tracker.build_hash([1, 2, 3])
        assert len(result) == 64

    def test_hash_with_nested_data(self):
        tracker = ProvenanceTracker()
        result = tracker.build_hash({"nested": {"key": "val", "num": 42}})
        assert len(result) == 64

    def test_hash_with_string_data(self):
        tracker = ProvenanceTracker()
        result = tracker.build_hash("simple string")
        assert len(result) == 64


# ============================================================================
# Chain isolation tests
# ============================================================================


class TestProvenanceTrackerChainIsolation:
    """Test that chains for different entities are properly isolated."""

    def test_separate_entity_chains(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        tracker.record("classification", "cls-001", "classify", "h2")

        chain_rec = tracker.get_chain("rec-001")
        chain_cls = tracker.get_chain("cls-001")

        assert len(chain_rec) == 1
        assert len(chain_cls) == 1
        assert chain_rec[0]["entity_type"] == "spend_record"
        assert chain_cls[0]["entity_type"] == "classification"

    def test_tamper_one_entity_does_not_affect_other(self):
        tracker = ProvenanceTracker()
        tracker.record("spend_record", "rec-001", "ingest", "h1")
        tracker.record("spend_record", "rec-002", "ingest", "h2")

        # Tamper with rec-001 chain
        tracker._chain_store["rec-001"][0]["chain_hash"] = ""

        # rec-002 should still be valid
        is_valid_002, _ = tracker.verify_chain("rec-002")
        assert is_valid_002 is True

        # rec-001 should be invalid
        is_valid_001, _ = tracker.verify_chain("rec-001")
        assert is_valid_001 is False

    def test_many_entities_independent(self):
        tracker = ProvenanceTracker()
        for i in range(20):
            tracker.record("spend_record", f"rec-{i:03d}", "ingest", f"h{i}")

        assert tracker.entity_count == 20
        assert tracker.entry_count == 20

        for i in range(20):
            chain = tracker.get_chain(f"rec-{i:03d}")
            assert len(chain) == 1
