# -*- coding: utf-8 -*-
"""
Unit Tests for ProvenanceTracker - AGENT-DATA-018

Tests the SHA-256 chain-hashed provenance tracker including:
  - Record creation and entry fields
  - Chain verification (valid and tampered)
  - Filtered retrieval (by entity_type, entity_id, hash)
  - JSON export
  - Chain integrity across multiple records
  - Entity types specific to the data lineage tracker
  - Genesis hash behavior
  - len/count properties
  - Singleton helpers
  - Thread safety

50+ test cases.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import threading

import pytest

from greenlang.data_lineage_tracker.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    get_provenance_tracker,
    reset_provenance_tracker,
    set_provenance_tracker,
)


# ============================================================================
# TestProvenanceEntry
# ============================================================================


class TestProvenanceEntry:
    """Tests for the ProvenanceEntry dataclass."""

    def test_entry_creation(self):
        """ProvenanceEntry can be created with all required fields."""
        entry = ProvenanceEntry(
            entity_type="lineage_asset",
            entity_id="asset-001",
            action="asset_registered",
            hash_value="abc123",
            parent_hash="def456",
            timestamp="2026-02-17T00:00:00+00:00",
        )
        assert entry.entity_type == "lineage_asset"
        assert entry.entity_id == "asset-001"
        assert entry.action == "asset_registered"
        assert entry.hash_value == "abc123"
        assert entry.parent_hash == "def456"
        assert entry.timestamp == "2026-02-17T00:00:00+00:00"

    def test_entry_default_metadata(self):
        """ProvenanceEntry has empty dict metadata by default."""
        entry = ProvenanceEntry(
            entity_type="lineage_asset",
            entity_id="asset-001",
            action="asset_registered",
            hash_value="abc123",
            parent_hash="def456",
            timestamp="2026-02-17T00:00:00+00:00",
        )
        assert entry.metadata == {}

    def test_entry_with_metadata(self):
        """ProvenanceEntry stores provided metadata."""
        meta = {"data_hash": "abc123", "extra": "info"}
        entry = ProvenanceEntry(
            entity_type="transformation",
            entity_id="txn-001",
            action="transformation_captured",
            hash_value="hash1",
            parent_hash="hash0",
            timestamp="2026-02-17T12:00:00+00:00",
            metadata=meta,
        )
        assert entry.metadata == meta

    def test_entry_to_dict(self):
        """ProvenanceEntry.to_dict() returns a complete dictionary."""
        entry = ProvenanceEntry(
            entity_type="transformation",
            entity_id="txn-001",
            action="transformation_captured",
            hash_value="hash1",
            parent_hash="hash0",
            timestamp="2026-02-17T12:00:00+00:00",
            metadata={"data_hash": "dh1"},
        )
        d = entry.to_dict()
        assert d["entity_type"] == "transformation"
        assert d["entity_id"] == "txn-001"
        assert d["action"] == "transformation_captured"
        assert d["hash_value"] == "hash1"
        assert d["parent_hash"] == "hash0"
        assert d["timestamp"] == "2026-02-17T12:00:00+00:00"
        assert d["metadata"]["data_hash"] == "dh1"

    def test_entry_to_dict_keys(self):
        """to_dict() includes all 7 expected keys."""
        entry = ProvenanceEntry(
            entity_type="x", entity_id="y", action="z",
            hash_value="h1", parent_hash="h0", timestamp="t",
        )
        d = entry.to_dict()
        expected = {"entity_type", "entity_id", "action", "hash_value",
                     "parent_hash", "timestamp", "metadata"}
        assert set(d.keys()) == expected


# ============================================================================
# TestProvenanceTrackerRecord
# ============================================================================


class TestProvenanceTrackerRecord:
    """Tests for ProvenanceTracker.record()."""

    def test_record_entry(self, provenance):
        """record() returns a ProvenanceEntry with correct fields."""
        entry = provenance.record("lineage_asset", "asset-001", "asset_registered")
        assert isinstance(entry, ProvenanceEntry)
        assert entry.entity_type == "lineage_asset"
        assert entry.entity_id == "asset-001"
        assert entry.action == "asset_registered"
        assert len(entry.hash_value) == 64
        assert len(entry.parent_hash) == 64
        assert entry.timestamp is not None

    def test_record_stores_data_hash_in_metadata(self, provenance):
        """record() stores a data_hash key in the entry metadata."""
        entry = provenance.record(
            "lineage_asset", "asset-001", "asset_registered",
            metadata={"name": "test_asset"},
        )
        assert "data_hash" in entry.metadata

    def test_record_none_metadata(self, provenance):
        """record() handles None metadata without error."""
        entry = provenance.record("lineage_asset", "asset-001", "asset_registered")
        assert "data_hash" in entry.metadata

    def test_record_increments_count(self, provenance):
        """Each record() call increments the entry count."""
        assert len(provenance) == 0
        provenance.record("lineage_asset", "a1", "asset_registered")
        assert len(provenance) == 1
        provenance.record("transformation", "t1", "transformation_captured")
        assert len(provenance) == 2

    def test_record_empty_entity_type_raises(self, provenance):
        """record() raises ValueError for empty entity_type."""
        with pytest.raises(ValueError, match="entity_type must not be empty"):
            provenance.record("", "id-1", "action")

    def test_record_empty_entity_id_raises(self, provenance):
        """record() raises ValueError for empty entity_id."""
        with pytest.raises(ValueError, match="entity_id must not be empty"):
            provenance.record("lineage_asset", "", "action")

    def test_record_empty_action_raises(self, provenance):
        """record() raises ValueError for empty action."""
        with pytest.raises(ValueError, match="action must not be empty"):
            provenance.record("lineage_asset", "id-1", "")

    def test_record_chaining(self, provenance):
        """Second entry's parent_hash equals first entry's hash_value."""
        e1 = provenance.record("lineage_asset", "a1", "asset_registered")
        e2 = provenance.record("lineage_asset", "a2", "asset_registered")
        assert e2.parent_hash == e1.hash_value

    def test_record_with_complex_metadata(self, provenance):
        """record() handles complex nested metadata."""
        entry = provenance.record(
            "lineage_asset", "a1", "asset_registered",
            metadata={"nested": {"deep": {"value": 42}}, "list": [1, 2, 3]},
        )
        assert entry.hash_value is not None
        assert len(entry.hash_value) == 64


# ============================================================================
# TestProvenanceTrackerVerify
# ============================================================================


class TestProvenanceTrackerVerify:
    """Tests for ProvenanceTracker.verify_chain()."""

    def test_verify_chain_valid(self, provenance):
        """verify_chain() returns True for an intact chain."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance.record("transformation", "t1", "transformation_captured")
        provenance.record("lineage_edge", "e1", "edge_created")
        assert provenance.verify_chain() is True

    def test_verify_chain_empty_is_valid(self, provenance):
        """verify_chain() returns True for an empty chain."""
        assert provenance.verify_chain() is True

    def test_verify_chain_single_entry(self, provenance):
        """verify_chain() returns True with a single entry."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        assert provenance.verify_chain() is True

    def test_verify_chain_tampered_hash(self, provenance):
        """verify_chain() returns False when a hash is tampered."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance.record("transformation", "t1", "transformation_captured")
        provenance._global_chain[0].hash_value = "tampered_hash_value_" + "0" * 44
        assert provenance.verify_chain() is False

    def test_verify_chain_tampered_parent_hash(self, provenance):
        """verify_chain() returns False when a parent_hash is tampered."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance.record("transformation", "t1", "transformation_captured")
        provenance._global_chain[1].parent_hash = "wrong_parent_hash_" + "0" * 46
        assert provenance.verify_chain() is False

    def test_verify_chain_first_entry_wrong_genesis(self, provenance):
        """verify_chain() returns False when first entry parent does not match genesis."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance._global_chain[0].parent_hash = "not_genesis_" + "0" * 52
        assert provenance.verify_chain() is False

    def test_verify_chain_emptied_field(self, provenance):
        """verify_chain() returns False when a required field is emptied."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance._global_chain[0].entity_type = ""
        assert provenance.verify_chain() is False

    def test_verify_chain_emptied_action(self, provenance):
        """verify_chain() returns False when action is emptied."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance._global_chain[0].action = ""
        assert provenance.verify_chain() is False

    def test_verify_chain_emptied_hash_value(self, provenance):
        """verify_chain() returns False when hash_value is emptied."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance._global_chain[0].hash_value = ""
        assert provenance.verify_chain() is False


# ============================================================================
# TestProvenanceTrackerRetrieval
# ============================================================================


class TestProvenanceTrackerRetrieval:
    """Tests for get_entries(), get_entry_by_hash(), get_chain()."""

    def test_get_entries_all(self, provenance):
        """get_entries() with no filter returns all entries."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance.record("transformation", "t1", "transformation_captured")
        entries = provenance.get_entries()
        assert len(entries) == 2

    def test_get_entries_by_entity_type(self, provenance):
        """get_entries(entity_type=...) filters by entity type."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance.record("transformation", "t1", "transformation_captured")
        provenance.record("lineage_asset", "a2", "asset_registered")
        assets = provenance.get_entries(entity_type="lineage_asset")
        assert len(assets) == 2
        for e in assets:
            assert e.entity_type == "lineage_asset"

    def test_get_entries_by_entity_type_and_id(self, provenance):
        """get_entries(entity_type, entity_id) uses keyed store."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance.record("lineage_asset", "a1", "asset_updated")
        provenance.record("lineage_asset", "a2", "asset_registered")
        entries = provenance.get_entries(entity_type="lineage_asset", entity_id="a1")
        assert len(entries) == 2
        for e in entries:
            assert e.entity_id == "a1"

    def test_get_entries_empty_result(self, provenance):
        """get_entries() returns empty list when no match."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        entries = provenance.get_entries(entity_type="nonexistent")
        assert entries == []

    def test_get_entries_returns_copies(self, provenance):
        """get_entries() returns list copies, not internal state."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        list1 = provenance.get_entries()
        list2 = provenance.get_entries()
        assert list1 is not list2

    def test_get_entry_by_hash(self, provenance):
        """get_entry_by_hash() finds entry by hash_value."""
        e1 = provenance.record("lineage_asset", "a1", "asset_registered")
        found = provenance.get_entry_by_hash(e1.hash_value)
        assert found is not None
        assert found.entity_id == "a1"

    def test_get_entry_by_hash_not_found(self, provenance):
        """get_entry_by_hash() returns None when hash not found."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        assert provenance.get_entry_by_hash("nonexistent_hash") is None

    def test_get_entry_by_hash_empty_string(self, provenance):
        """get_entry_by_hash('') returns None."""
        assert provenance.get_entry_by_hash("") is None

    def test_get_chain_by_entity_id(self, provenance):
        """get_chain() returns all entries for a given entity_id."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance.record("transformation", "a1", "transformation_captured")
        provenance.record("lineage_asset", "a2", "asset_registered")
        chain = provenance.get_chain("a1")
        assert len(chain) == 2
        for e in chain:
            assert e.entity_id == "a1"

    def test_get_chain_empty_result(self, provenance):
        """get_chain() returns empty list for unknown entity_id."""
        assert provenance.get_chain("unknown") == []


# ============================================================================
# TestProvenanceTrackerExport
# ============================================================================


class TestProvenanceTrackerExport:
    """Tests for export_chain() and export_json()."""

    def test_export_chain(self, provenance):
        """export_chain() returns list of dicts."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        chain = provenance.export_chain()
        assert isinstance(chain, list)
        assert len(chain) == 1
        assert isinstance(chain[0], dict)

    def test_export_chain_empty(self, provenance):
        """export_chain() returns empty list for empty tracker."""
        assert provenance.export_chain() == []

    def test_export_chain_preserves_order(self, provenance):
        """export_chain() preserves insertion order."""
        provenance.record("lineage_asset", "a1", "registered")
        provenance.record("transformation", "t1", "captured")
        provenance.record("lineage_edge", "e1", "created")
        chain = provenance.export_chain()
        assert chain[0]["entity_id"] == "a1"
        assert chain[1]["entity_id"] == "t1"
        assert chain[2]["entity_id"] == "e1"

    def test_export_json(self, provenance):
        """export_json() returns valid JSON string."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        json_str = provenance.export_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_export_json_empty(self, provenance):
        """export_json() returns '[]' for empty tracker."""
        result = provenance.export_json()
        assert json.loads(result) == []

    def test_export_json_is_indented(self, provenance):
        """export_json() output is indented for readability."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        json_str = provenance.export_json()
        assert "\n" in json_str


# ============================================================================
# TestProvenanceTrackerChainIntegrity
# ============================================================================


class TestProvenanceTrackerChainIntegrity:
    """Tests for chain integrity across many records."""

    def test_chain_integrity_after_multiple_records(self, provenance):
        """Chain remains valid after 100 records."""
        for i in range(100):
            provenance.record(
                "lineage_asset", f"asset-{i}", "asset_registered",
                metadata={"index": i},
            )
        assert provenance.verify_chain() is True
        assert len(provenance) == 100

    def test_chain_parent_hashes_link_correctly(self, provenance):
        """Each entry's parent_hash equals the previous entry's hash_value."""
        entries = []
        for i in range(5):
            entries.append(
                provenance.record("lineage_asset", f"a-{i}", "asset_registered")
            )
        for i in range(1, len(entries)):
            assert entries[i].parent_hash == entries[i - 1].hash_value

    def test_first_entry_parent_is_genesis(self, provenance):
        """First entry's parent_hash is the genesis hash."""
        entry = provenance.record("lineage_asset", "a1", "asset_registered")
        assert entry.parent_hash == provenance._genesis_hash

    def test_different_metadata_produces_different_hashes(self, provenance):
        """Entries with different metadata produce different hash_values."""
        e1 = provenance.record(
            "lineage_asset", "a1", "asset_registered",
            metadata={"data": "version1"},
        )
        e2 = provenance.record(
            "lineage_asset", "a1", "asset_registered",
            metadata={"data": "version2"},
        )
        assert e1.hash_value != e2.hash_value

    def test_same_metadata_at_different_times_different_hashes(self, provenance):
        """Same metadata at different chain positions produces different hashes."""
        e1 = provenance.record(
            "lineage_asset", "a1", "asset_registered",
            metadata={"data": "same"},
        )
        e2 = provenance.record(
            "lineage_asset", "a1", "asset_registered",
            metadata={"data": "same"},
        )
        # Different parent_hash means different chain_hash
        assert e1.hash_value != e2.hash_value


# ============================================================================
# TestProvenanceTrackerEntityTypes
# ============================================================================


class TestProvenanceTrackerEntityTypes:
    """Tests for all entity types used in the data lineage tracker."""

    @pytest.mark.parametrize("entity_type,action", [
        ("lineage_asset", "asset_registered"),
        ("transformation", "transformation_captured"),
        ("lineage_edge", "edge_created"),
        ("impact_analysis", "impact_analyzed"),
        ("validation", "validation_completed"),
        ("report", "report_generated"),
        ("change_event", "change_detected"),
        ("quality_score", "quality_scored"),
    ])
    def test_entity_type_and_action(self, provenance, entity_type, action):
        """Each entity_type/action pair records successfully."""
        entry = provenance.record(entity_type, "id-001", action)
        assert entry.entity_type == entity_type
        assert entry.action == action
        assert len(entry.hash_value) == 64


# ============================================================================
# TestProvenanceTrackerGenesisHash
# ============================================================================


class TestProvenanceTrackerGenesisHash:
    """Tests for genesis hash behavior."""

    def test_genesis_hash_is_sha256(self):
        """Genesis hash is a valid SHA-256 hex digest (64 chars)."""
        tracker = ProvenanceTracker()
        assert len(tracker._genesis_hash) == 64

    def test_genesis_hash_deterministic(self):
        """Same genesis string produces same genesis hash."""
        t1 = ProvenanceTracker("seed-A")
        t2 = ProvenanceTracker("seed-A")
        assert t1._genesis_hash == t2._genesis_hash

    def test_custom_genesis_hash(self):
        """Custom genesis_hash string produces a different genesis."""
        default = ProvenanceTracker()
        custom = ProvenanceTracker("my-custom-genesis")
        assert default._genesis_hash != custom._genesis_hash

    def test_genesis_hash_matches_sha256(self):
        """Genesis hash matches manual SHA-256 computation."""
        seed = "test-genesis-seed"
        expected = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        tracker = ProvenanceTracker(seed)
        assert tracker._genesis_hash == expected

    def test_default_genesis_string(self):
        """Default genesis string is 'greenlang-data-lineage-genesis'."""
        expected = hashlib.sha256(
            "greenlang-data-lineage-genesis".encode("utf-8")
        ).hexdigest()
        tracker = ProvenanceTracker()
        assert tracker._genesis_hash == expected


# ============================================================================
# TestProvenanceTrackerProperties
# ============================================================================


class TestProvenanceTrackerProperties:
    """Tests for len, entry_count, entity_count, reset, build_hash."""

    def test_len_empty(self, provenance):
        """len() returns 0 for empty tracker."""
        assert len(provenance) == 0

    def test_len_after_records(self, provenance):
        """len() returns correct count after recording entries."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance.record("transformation", "t1", "transformation_captured")
        assert len(provenance) == 2

    def test_entry_count_property(self, provenance):
        """entry_count property matches len()."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        assert provenance.entry_count == 1
        assert provenance.entry_count == len(provenance)

    def test_entity_count_property(self, provenance):
        """entity_count tracks unique entity_type:entity_id keys."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance.record("lineage_asset", "a1", "asset_updated")
        provenance.record("lineage_asset", "a2", "asset_registered")
        assert provenance.entity_count == 2

    def test_reset_clears_state(self, provenance):
        """reset() clears all state."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance.record("transformation", "t1", "transformation_captured")
        provenance.reset()
        assert len(provenance) == 0
        assert provenance.entity_count == 0
        assert provenance.export_chain() == []
        assert provenance.verify_chain() is True

    def test_reset_restores_genesis_link(self, provenance):
        """After reset, new entry links to genesis hash."""
        provenance.record("lineage_asset", "a1", "asset_registered")
        provenance.reset()
        entry = provenance.record("lineage_asset", "a2", "asset_registered")
        assert entry.parent_hash == provenance._genesis_hash

    def test_build_hash_deterministic(self, provenance):
        """build_hash() returns deterministic SHA-256 hash."""
        h1 = provenance.build_hash({"key": "value"})
        h2 = provenance.build_hash({"key": "value"})
        assert h1 == h2
        assert len(h1) == 64

    def test_build_hash_different_data(self, provenance):
        """build_hash() returns different hash for different data."""
        h1 = provenance.build_hash({"key": "value1"})
        h2 = provenance.build_hash({"key": "value2"})
        assert h1 != h2

    def test_build_hash_none(self, provenance):
        """build_hash(None) uses 'null' sentinel."""
        h = provenance.build_hash(None)
        expected = hashlib.sha256(b"null").hexdigest()
        assert h == expected


# ============================================================================
# TestProvenanceTrackerSingleton
# ============================================================================


class TestProvenanceTrackerSingleton:
    """Tests for singleton helper functions."""

    def test_get_provenance_tracker_returns_instance(self):
        """get_provenance_tracker() returns a ProvenanceTracker."""
        reset_provenance_tracker()
        tracker = get_provenance_tracker()
        assert isinstance(tracker, ProvenanceTracker)

    def test_get_provenance_tracker_singleton(self):
        """get_provenance_tracker() returns the same instance."""
        reset_provenance_tracker()
        t1 = get_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is t2

    def test_set_provenance_tracker(self):
        """set_provenance_tracker() replaces the singleton."""
        reset_provenance_tracker()
        custom = ProvenanceTracker("custom-seed")
        set_provenance_tracker(custom)
        assert get_provenance_tracker() is custom
        reset_provenance_tracker()

    def test_set_provenance_tracker_invalid_type(self):
        """set_provenance_tracker() raises TypeError for non-tracker."""
        with pytest.raises(TypeError, match="must be a ProvenanceTracker"):
            set_provenance_tracker("not a tracker")  # type: ignore

    def test_reset_provenance_tracker(self):
        """reset_provenance_tracker() clears singleton for fresh creation."""
        reset_provenance_tracker()
        t1 = get_provenance_tracker()
        reset_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is not t2


# ============================================================================
# TestProvenanceTrackerThreadSafety
# ============================================================================


class TestProvenanceTrackerThreadSafety:
    """Tests for thread safety of ProvenanceTracker."""

    def test_concurrent_records(self):
        """Concurrent record() calls produce valid chain."""
        tracker = ProvenanceTracker()
        errors = []

        def worker(worker_id):
            try:
                for i in range(50):
                    tracker.record(
                        "lineage_asset",
                        f"worker-{worker_id}-asset-{i}",
                        "asset_registered",
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(tracker) == 250
        assert tracker.verify_chain() is True

    def test_concurrent_reads_and_writes(self):
        """Concurrent reads and writes do not corrupt state."""
        tracker = ProvenanceTracker()
        errors = []

        def writer(worker_id):
            try:
                for i in range(20):
                    tracker.record(
                        "lineage_asset",
                        f"w-{worker_id}-a-{i}",
                        "asset_registered",
                    )
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(20):
                    tracker.get_entries()
                    tracker.export_chain()
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=writer, args=(w,)) for w in range(3)]
            + [threading.Thread(target=reader) for _ in range(3)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.verify_chain() is True
