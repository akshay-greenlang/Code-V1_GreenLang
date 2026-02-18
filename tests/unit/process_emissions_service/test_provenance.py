# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-MRV-004 Process Emissions Agent

Tests all provenance tracking functionality including entry creation,
chain hashing, chain verification, trail filtering, JSON export, and
edge cases such as max_entries eviction and thread safety.

68 tests across 7 test classes.

Author: GreenLang QA Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from typing import List

import pytest

from greenlang.process_emissions.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
    get_provenance_tracker,
    reset_provenance_tracker,
    set_provenance_tracker,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker for each test."""
    return ProvenanceTracker(
        genesis_hash="TEST-GENESIS-HASH",
        max_entries=10000,
    )


@pytest.fixture
def populated_tracker(tracker: ProvenanceTracker) -> ProvenanceTracker:
    """Create a tracker pre-populated with entries across multiple entities."""
    tracker.record("PROCESS", "proc_001", "CREATE", data={"name": "cement"})
    tracker.record("PROCESS", "proc_001", "UPDATE", data={"name": "cement_v2"})
    tracker.record("MATERIAL", "mat_001", "CREATE", data={"type": "limestone"})
    tracker.record("CALCULATION", "calc_001", "CALCULATE", data={"co2": 500})
    tracker.record("FACTOR", "ef_001", "LOOKUP", data={"source": "IPCC"})
    tracker.record("PROCESS", "proc_002", "CREATE", data={"name": "lime"})
    tracker.record("COMPLIANCE", "comp_001", "CHECK", data={"passed": True})
    tracker.record("ABATEMENT", "abate_001", "REGISTER", data={"tech": "NSCR"})
    tracker.record("BATCH", "batch_001", "CALCULATE", data={"count": 10})
    tracker.record("UNCERTAINTY", "unc_001", "ANALYZE", data={"ci": 0.95})
    return tracker


# =========================================================================
# TestProvenanceEntry (8 tests)
# =========================================================================

class TestProvenanceEntry:
    """Tests for the ProvenanceEntry dataclass."""

    def test_entry_creation_basic(self):
        """Entry can be created with all required fields."""
        entry = ProvenanceEntry(
            entity_type="PROCESS",
            entity_id="proc_001",
            action="CREATE",
            hash_value="abc123",
            parent_hash="genesis",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert entry.entity_type == "PROCESS"
        assert entry.entity_id == "proc_001"
        assert entry.action == "CREATE"
        assert entry.hash_value == "abc123"
        assert entry.parent_hash == "genesis"
        assert entry.timestamp == "2026-01-01T00:00:00+00:00"
        assert entry.metadata == {}

    def test_entry_creation_with_metadata(self):
        """Entry correctly stores arbitrary metadata."""
        meta = {"data_hash": "deadbeef", "user": "test_user"}
        entry = ProvenanceEntry(
            entity_type="CALCULATION",
            entity_id="calc_001",
            action="CALCULATE",
            hash_value="hash1",
            parent_hash="hash0",
            timestamp="2026-02-01T12:00:00+00:00",
            metadata=meta,
        )
        assert entry.metadata == meta
        assert entry.metadata["user"] == "test_user"

    def test_entry_to_dict(self):
        """to_dict() returns a complete dictionary representation."""
        entry = ProvenanceEntry(
            entity_type="MATERIAL",
            entity_id="mat_001",
            action="LOOKUP",
            hash_value="h1",
            parent_hash="h0",
            timestamp="2026-01-15T08:30:00+00:00",
            metadata={"data_hash": "abc"},
        )
        d = entry.to_dict()
        assert d["entity_type"] == "MATERIAL"
        assert d["entity_id"] == "mat_001"
        assert d["action"] == "LOOKUP"
        assert d["hash_value"] == "h1"
        assert d["parent_hash"] == "h0"
        assert d["timestamp"] == "2026-01-15T08:30:00+00:00"
        assert d["metadata"]["data_hash"] == "abc"

    def test_entry_to_dict_round_trip(self):
        """to_dict output is JSON-serializable and reversible."""
        entry = ProvenanceEntry(
            entity_type="FACTOR",
            entity_id="ef_001",
            action="SELECT",
            hash_value="sha256value",
            parent_hash="sha256parent",
            timestamp="2026-03-01T00:00:00+00:00",
            metadata={"source": "IPCC", "value": "0.507"},
        )
        json_str = json.dumps(entry.to_dict())
        parsed = json.loads(json_str)
        assert parsed["entity_type"] == "FACTOR"
        assert parsed["metadata"]["source"] == "IPCC"

    def test_entry_hash_value_is_64_chars(self, tracker: ProvenanceTracker):
        """SHA-256 produces a 64-character hex hash."""
        entry = tracker.record("PROCESS", "proc_001", "CREATE")
        assert len(entry.hash_value) == 64

    def test_entry_parent_hash_is_64_chars(self, tracker: ProvenanceTracker):
        """First entry's parent_hash is the genesis hash (64 chars)."""
        entry = tracker.record("PROCESS", "proc_001", "CREATE")
        assert len(entry.parent_hash) == 64

    def test_entry_timestamp_is_iso_format(self, tracker: ProvenanceTracker):
        """Timestamp follows ISO 8601 format."""
        entry = tracker.record("PROCESS", "proc_001", "CREATE")
        assert "T" in entry.timestamp
        assert "+" in entry.timestamp or "Z" in entry.timestamp

    def test_entry_metadata_includes_data_hash(self, tracker: ProvenanceTracker):
        """When recording with data, metadata contains a data_hash key."""
        entry = tracker.record(
            "CALCULATION", "calc_001", "CALCULATE",
            data={"co2_tonnes": 100},
        )
        assert "data_hash" in entry.metadata
        assert len(entry.metadata["data_hash"]) == 64


# =========================================================================
# TestProvenanceTracker (15 tests)
# =========================================================================

class TestProvenanceTracker:
    """Tests for ProvenanceTracker core operations."""

    def test_initialization_creates_genesis_hash(self, tracker: ProvenanceTracker):
        """Tracker initializes with a SHA-256 genesis hash."""
        assert len(tracker.genesis_hash) == 64
        expected = hashlib.sha256(b"TEST-GENESIS-HASH").hexdigest()
        assert tracker.genesis_hash == expected

    def test_initialization_empty_chain(self, tracker: ProvenanceTracker):
        """Tracker starts with zero entries."""
        assert tracker.entry_count == 0
        assert tracker.entity_count == 0
        assert len(tracker) == 0

    def test_record_returns_entry(self, tracker: ProvenanceTracker):
        """record() returns a ProvenanceEntry with correct fields."""
        entry = tracker.record("PROCESS", "proc_001", "CREATE")
        assert isinstance(entry, ProvenanceEntry)
        assert entry.entity_type == "PROCESS"
        assert entry.entity_id == "proc_001"
        assert entry.action == "CREATE"

    def test_record_increments_count(self, tracker: ProvenanceTracker):
        """Each record() call increments the entry count."""
        tracker.record("PROCESS", "proc_001", "CREATE")
        assert tracker.entry_count == 1
        tracker.record("MATERIAL", "mat_001", "CREATE")
        assert tracker.entry_count == 2

    def test_record_with_data(self, tracker: ProvenanceTracker):
        """record() computes a data_hash from the provided data."""
        entry = tracker.record(
            "CALCULATION", "calc_001", "CALCULATE",
            data={"emissions": 507.0},
        )
        assert "data_hash" in entry.metadata
        assert len(entry.metadata["data_hash"]) == 64

    def test_record_with_none_data(self, tracker: ProvenanceTracker):
        """record() with data=None still produces a data_hash."""
        entry = tracker.record("PROCESS", "proc_001", "CREATE", data=None)
        assert "data_hash" in entry.metadata

    def test_record_with_extra_metadata(self, tracker: ProvenanceTracker):
        """Extra metadata is merged alongside the data_hash."""
        entry = tracker.record(
            "PROCESS", "proc_001", "CREATE",
            metadata={"user": "admin", "source": "test"},
        )
        assert entry.metadata["user"] == "admin"
        assert entry.metadata["source"] == "test"
        assert "data_hash" in entry.metadata

    def test_record_empty_entity_type_raises(self, tracker: ProvenanceTracker):
        """record() with empty entity_type raises ValueError."""
        with pytest.raises(ValueError, match="entity_type must not be empty"):
            tracker.record("", "proc_001", "CREATE")

    def test_record_empty_entity_id_raises(self, tracker: ProvenanceTracker):
        """record() with empty entity_id raises ValueError."""
        with pytest.raises(ValueError, match="entity_id must not be empty"):
            tracker.record("PROCESS", "", "CREATE")

    def test_record_empty_action_raises(self, tracker: ProvenanceTracker):
        """record() with empty action raises ValueError."""
        with pytest.raises(ValueError, match="action must not be empty"):
            tracker.record("PROCESS", "proc_001", "")

    def test_get_trail_returns_all(self, populated_tracker: ProvenanceTracker):
        """get_trail() without filters returns all entries."""
        trail = populated_tracker.get_trail(limit=1000)
        assert len(trail) == 10

    def test_get_chain_hash_changes_on_record(self, tracker: ProvenanceTracker):
        """Chain hash changes after each record."""
        hash_before = tracker.get_chain_hash()
        tracker.record("PROCESS", "proc_001", "CREATE")
        hash_after = tracker.get_chain_hash()
        assert hash_before != hash_after

    def test_clear_trail_resets(self, populated_tracker: ProvenanceTracker):
        """clear_trail() resets the tracker to genesis state."""
        populated_tracker.clear_trail()
        assert populated_tracker.entry_count == 0
        assert populated_tracker.entity_count == 0
        assert populated_tracker.get_chain_hash() == populated_tracker.genesis_hash

    def test_clear_alias(self, populated_tracker: ProvenanceTracker):
        """clear() is an alias for clear_trail()."""
        populated_tracker.clear()
        assert populated_tracker.entry_count == 0

    def test_repr_format(self, tracker: ProvenanceTracker):
        """__repr__ includes entry count and entity count."""
        tracker.record("PROCESS", "proc_001", "CREATE")
        r = repr(tracker)
        assert "ProvenanceTracker" in r
        assert "entries=1" in r


# =========================================================================
# TestChainHashing (10 tests)
# =========================================================================

class TestChainHashing:
    """Tests for SHA-256 chain hashing integrity."""

    def test_first_entry_parent_is_genesis(self, tracker: ProvenanceTracker):
        """First entry's parent_hash equals the genesis hash."""
        entry = tracker.record("PROCESS", "proc_001", "CREATE")
        assert entry.parent_hash == tracker.genesis_hash

    def test_second_entry_parent_is_first_hash(self, tracker: ProvenanceTracker):
        """Second entry's parent_hash equals the first entry's hash_value."""
        e1 = tracker.record("PROCESS", "proc_001", "CREATE")
        e2 = tracker.record("PROCESS", "proc_002", "CREATE")
        assert e2.parent_hash == e1.hash_value

    def test_chain_links_through_ten_entries(self, tracker: ProvenanceTracker):
        """Chain correctly links 10 entries in sequence."""
        entries: List[ProvenanceEntry] = []
        for i in range(10):
            entry = tracker.record("PROCESS", f"proc_{i:03d}", "CREATE")
            entries.append(entry)

        for i in range(1, len(entries)):
            assert entries[i].parent_hash == entries[i - 1].hash_value

    def test_chain_hash_updates_with_each_record(self, tracker: ProvenanceTracker):
        """get_chain_hash() returns a unique hash after each record."""
        hashes = set()
        hashes.add(tracker.get_chain_hash())
        for i in range(5):
            tracker.record("PROCESS", f"proc_{i:03d}", "CREATE")
            hashes.add(tracker.get_chain_hash())
        assert len(hashes) == 6  # genesis + 5 records

    def test_deterministic_hash_same_inputs(self):
        """Same inputs produce the same chain hash (deterministic)."""
        t1 = ProvenanceTracker(genesis_hash="SEED_A")
        t2 = ProvenanceTracker(genesis_hash="SEED_A")

        # We cannot guarantee identical timestamps so we verify genesis
        assert t1.genesis_hash == t2.genesis_hash
        assert t1.get_chain_hash() == t2.get_chain_hash()

    def test_different_genesis_different_chain(self):
        """Different genesis hashes produce different chain hashes."""
        t1 = ProvenanceTracker(genesis_hash="SEED_A")
        t2 = ProvenanceTracker(genesis_hash="SEED_B")
        assert t1.genesis_hash != t2.genesis_hash
        assert t1.get_chain_hash() != t2.get_chain_hash()

    def test_hash_value_is_hex(self, tracker: ProvenanceTracker):
        """All hash values are valid hexadecimal strings."""
        entry = tracker.record("PROCESS", "proc_001", "CREATE")
        int(entry.hash_value, 16)  # should not raise
        int(entry.parent_hash, 16)  # should not raise

    def test_data_changes_produce_different_hashes(self, tracker: ProvenanceTracker):
        """Different data payloads produce different data_hashes."""
        e1 = tracker.record("CALCULATION", "c1", "CALCULATE", data={"val": 100})
        e2 = tracker.record("CALCULATION", "c2", "CALCULATE", data={"val": 200})
        assert e1.metadata["data_hash"] != e2.metadata["data_hash"]

    def test_none_data_produces_consistent_hash(self, tracker: ProvenanceTracker):
        """None data produces the same hash for 'null' serialization."""
        h1 = tracker.build_hash(None)
        h2 = tracker.build_hash(None)
        assert h1 == h2

    def test_build_hash_utility(self, tracker: ProvenanceTracker):
        """build_hash() produces deterministic SHA-256 for any data."""
        h = tracker.build_hash({"key": "value"})
        assert len(h) == 64
        h2 = tracker.build_hash({"key": "value"})
        assert h == h2


# =========================================================================
# TestChainVerification (10 tests)
# =========================================================================

class TestChainVerification:
    """Tests for verify_chain() with valid, invalid, and tampered chains."""

    def test_empty_chain_is_valid(self, tracker: ProvenanceTracker):
        """An empty chain is trivially valid."""
        is_valid, msg = tracker.verify_chain()
        assert is_valid is True
        assert msg is None

    def test_single_entry_chain_is_valid(self, tracker: ProvenanceTracker):
        """A chain with a single entry is valid."""
        tracker.record("PROCESS", "proc_001", "CREATE")
        is_valid, msg = tracker.verify_chain()
        assert is_valid is True
        assert msg is None

    def test_multi_entry_chain_is_valid(self, populated_tracker: ProvenanceTracker):
        """A chain with 10 entries is valid."""
        is_valid, msg = populated_tracker.verify_chain()
        assert is_valid is True
        assert msg is None

    def test_large_chain_is_valid(self, tracker: ProvenanceTracker):
        """A chain with 100 entries verifies successfully."""
        for i in range(100):
            tracker.record("PROCESS", f"proc_{i:04d}", "CREATE")
        is_valid, msg = tracker.verify_chain()
        assert is_valid is True

    def test_tampered_hash_value_detected(self, tracker: ProvenanceTracker):
        """Tampering with an entry's hash_value is detected."""
        tracker.record("PROCESS", "proc_001", "CREATE")
        tracker.record("PROCESS", "proc_002", "CREATE")

        # Tamper with first entry's hash_value
        chain = tracker.get_chain()
        chain[0].hash_value = "0" * 64  # corrupt

        # verify_chain re-reads internal state, so tamper internal
        tracker._global_chain[0].hash_value = "0" * 64
        is_valid, msg = tracker.verify_chain()
        assert is_valid is False
        assert "chain break" in msg.lower() or "parent_hash" in msg.lower()

    def test_tampered_parent_hash_detected(self, tracker: ProvenanceTracker):
        """Tampering with parent_hash breaks chain verification."""
        tracker.record("PROCESS", "proc_001", "CREATE")
        tracker.record("PROCESS", "proc_002", "CREATE")

        tracker._global_chain[1].parent_hash = "f" * 64
        is_valid, msg = tracker.verify_chain()
        assert is_valid is False
        assert msg is not None

    def test_tampered_first_entry_genesis_link(self, tracker: ProvenanceTracker):
        """Tampering with first entry's parent_hash (genesis link) is detected."""
        tracker.record("PROCESS", "proc_001", "CREATE")

        tracker._global_chain[0].parent_hash = "bad_genesis"
        is_valid, msg = tracker.verify_chain()
        assert is_valid is False
        assert "genesis" in msg.lower()

    def test_missing_required_field_detected(self, tracker: ProvenanceTracker):
        """An entry with an empty required field fails verification."""
        tracker.record("PROCESS", "proc_001", "CREATE")

        tracker._global_chain[0].entity_type = ""
        is_valid, msg = tracker.verify_chain()
        assert is_valid is False
        assert "entity_type" in msg

    def test_verify_after_clear_is_valid(self, populated_tracker: ProvenanceTracker):
        """Chain is valid after clear and new entries."""
        populated_tracker.clear()
        populated_tracker.record("PROCESS", "proc_new", "CREATE")
        is_valid, msg = populated_tracker.verify_chain()
        assert is_valid is True

    def test_verify_chain_returns_tuple(self, tracker: ProvenanceTracker):
        """verify_chain always returns a (bool, Optional[str]) tuple."""
        result = tracker.verify_chain()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)


# =========================================================================
# TestTrailFiltering (10 tests)
# =========================================================================

class TestTrailFiltering:
    """Tests for get_trail() and get_entries() filtering."""

    def test_filter_by_entity_type(self, populated_tracker: ProvenanceTracker):
        """get_trail(entity_type=...) returns only matching entries."""
        trail = populated_tracker.get_trail(entity_type="PROCESS")
        assert all(e.entity_type == "PROCESS" for e in trail)
        assert len(trail) == 3  # proc_001 CREATE, proc_001 UPDATE, proc_002 CREATE

    def test_filter_by_entity_type_and_id(self, populated_tracker: ProvenanceTracker):
        """get_trail(entity_type, entity_id) uses keyed store."""
        trail = populated_tracker.get_trail(entity_type="PROCESS", entity_id="proc_001")
        assert len(trail) == 2
        assert all(e.entity_id == "proc_001" for e in trail)

    def test_filter_by_action(self, populated_tracker: ProvenanceTracker):
        """get_trail(action=...) filters by action type."""
        trail = populated_tracker.get_trail(action="CREATE")
        assert all(e.action == "CREATE" for e in trail)

    def test_filter_by_entity_type_and_action(
        self, populated_tracker: ProvenanceTracker
    ):
        """Combined entity_type + action filtering works."""
        trail = populated_tracker.get_trail(
            entity_type="PROCESS", action="CREATE"
        )
        assert all(e.entity_type == "PROCESS" and e.action == "CREATE" for e in trail)
        assert len(trail) == 2  # proc_001 CREATE, proc_002 CREATE

    def test_limit_returns_most_recent(self, populated_tracker: ProvenanceTracker):
        """get_trail(limit=3) returns the 3 most recent entries."""
        trail = populated_tracker.get_trail(limit=3)
        assert len(trail) == 3
        # They should be the last 3 recorded
        full = populated_tracker.get_trail(limit=1000)
        assert trail == full[-3:]

    def test_limit_larger_than_chain(self, populated_tracker: ProvenanceTracker):
        """Limit larger than chain returns all entries."""
        trail = populated_tracker.get_trail(limit=9999)
        assert len(trail) == 10

    def test_get_entries_no_filter(self, populated_tracker: ProvenanceTracker):
        """get_entries() without filters returns all entries."""
        entries = populated_tracker.get_entries()
        assert len(entries) == 10

    def test_get_entries_with_limit(self, populated_tracker: ProvenanceTracker):
        """get_entries(limit=N) returns the N most recent entries."""
        entries = populated_tracker.get_entries(limit=2)
        assert len(entries) == 2

    def test_get_entries_for_entity(self, populated_tracker: ProvenanceTracker):
        """get_entries_for_entity() returns entries for a specific entity."""
        entries = populated_tracker.get_entries_for_entity("MATERIAL", "mat_001")
        assert len(entries) == 1
        assert entries[0].entity_type == "MATERIAL"

    def test_get_entries_for_entity_missing(self, populated_tracker: ProvenanceTracker):
        """get_entries_for_entity() returns empty list for unknown entity."""
        entries = populated_tracker.get_entries_for_entity("PROCESS", "nonexistent")
        assert entries == []


# =========================================================================
# TestExport (8 tests)
# =========================================================================

class TestExport:
    """Tests for export_trail() JSON export."""

    def test_export_empty_trail(self, tracker: ProvenanceTracker):
        """Exporting an empty trail returns '[]' JSON."""
        exported = tracker.export_trail()
        parsed = json.loads(exported)
        assert parsed == []

    def test_export_single_entry(self, tracker: ProvenanceTracker):
        """Exporting a single entry produces valid JSON array with one item."""
        tracker.record("PROCESS", "proc_001", "CREATE")
        exported = tracker.export_trail()
        parsed = json.loads(exported)
        assert len(parsed) == 1
        assert parsed[0]["entity_type"] == "PROCESS"

    def test_export_preserves_all_fields(self, tracker: ProvenanceTracker):
        """Exported JSON contains all required fields."""
        tracker.record(
            "CALCULATION", "calc_001", "CALCULATE",
            data={"co2": 100},
            metadata={"user": "test"},
        )
        exported = tracker.export_trail()
        parsed = json.loads(exported)
        entry = parsed[0]
        for key in ["entity_type", "entity_id", "action", "hash_value",
                     "parent_hash", "timestamp", "metadata"]:
            assert key in entry, f"Missing key: {key}"

    def test_export_multiple_entries(self, populated_tracker: ProvenanceTracker):
        """Exporting multiple entries preserves order."""
        exported = populated_tracker.export_trail()
        parsed = json.loads(exported)
        assert len(parsed) == 10
        # First entry should be the first recorded
        assert parsed[0]["entity_type"] == "PROCESS"
        assert parsed[0]["entity_id"] == "proc_001"
        assert parsed[0]["action"] == "CREATE"

    def test_export_with_indent(self, tracker: ProvenanceTracker):
        """export_trail(indent=4) produces indented JSON."""
        tracker.record("PROCESS", "proc_001", "CREATE")
        exported = tracker.export_trail(indent=4)
        assert "    " in exported  # 4-space indent

    def test_export_unsupported_format_raises(self, tracker: ProvenanceTracker):
        """export_trail() with unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            tracker.export_trail(format="csv")

    def test_export_json_parseable(self, populated_tracker: ProvenanceTracker):
        """Exported string is valid JSON."""
        exported = populated_tracker.export_trail()
        parsed = json.loads(exported)
        assert isinstance(parsed, list)

    def test_get_audit_trail_dict_format(self, populated_tracker: ProvenanceTracker):
        """get_audit_trail() returns list of dicts."""
        trail = populated_tracker.get_audit_trail()
        assert isinstance(trail, list)
        assert len(trail) == 10
        for item in trail:
            assert isinstance(item, dict)
            assert "entity_type" in item


# =========================================================================
# TestEdgeCases (7 tests)
# =========================================================================

class TestEdgeCases:
    """Tests for edge cases: max_entries eviction, thread safety, empty trail."""

    def test_max_entries_eviction(self):
        """Entries are evicted when max_entries is exceeded."""
        tracker = ProvenanceTracker(genesis_hash="SMALL", max_entries=5)
        for i in range(10):
            tracker.record("PROCESS", f"proc_{i:03d}", "CREATE")
        assert tracker.entry_count == 5

    def test_max_entries_property(self):
        """max_entries property reflects constructor value."""
        tracker = ProvenanceTracker(genesis_hash="TEST", max_entries=42)
        assert tracker.max_entries == 42

    def test_eviction_preserves_chain_tail(self):
        """After eviction, the remaining entries are the most recent."""
        tracker = ProvenanceTracker(genesis_hash="EVICT", max_entries=3)
        ids = []
        for i in range(6):
            entry = tracker.record("PROCESS", f"proc_{i:03d}", "CREATE")
            ids.append(entry.entity_id)

        chain = tracker.get_chain()
        # Last 3 should remain
        assert len(chain) == 3
        assert chain[-1].entity_id == "proc_005"

    def test_entity_count_after_eviction(self):
        """entity_count correctly reflects entries after eviction cleanup."""
        tracker = ProvenanceTracker(genesis_hash="EVICT_ENT", max_entries=3)
        tracker.record("PROCESS", "proc_001", "CREATE")
        tracker.record("PROCESS", "proc_002", "CREATE")
        tracker.record("PROCESS", "proc_003", "CREATE")
        tracker.record("PROCESS", "proc_004", "CREATE")
        # After eviction, proc_001 should be gone from chain_store
        entries = tracker.get_entries_for_entity("PROCESS", "proc_001")
        assert len(entries) == 0

    def test_thread_safety_concurrent_records(self):
        """Concurrent record() calls do not corrupt the chain."""
        tracker = ProvenanceTracker(genesis_hash="THREAD_SAFE", max_entries=10000)
        num_threads = 10
        records_per_thread = 50

        def worker(thread_id: int):
            for i in range(records_per_thread):
                tracker.record("PROCESS", f"t{thread_id}_p{i:03d}", "CREATE")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, tid) for tid in range(num_threads)]
            for f in as_completed(futures):
                f.result()  # propagate exceptions

        assert tracker.entry_count == num_threads * records_per_thread

    def test_thread_safety_chain_integrity(self):
        """verify_chain() passes after concurrent writes."""
        tracker = ProvenanceTracker(genesis_hash="THREAD_VERIFY", max_entries=10000)

        def worker(thread_id: int):
            for i in range(20):
                tracker.record("BATCH", f"t{thread_id}_{i}", "CALCULATE")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, tid) for tid in range(5)]
            for f in as_completed(futures):
                f.result()

        is_valid, msg = tracker.verify_chain()
        assert is_valid is True

    def test_singleton_functions(self):
        """get/set/reset singleton pattern works correctly."""
        reset_provenance_tracker()

        t1 = get_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is t2

        custom = ProvenanceTracker(genesis_hash="CUSTOM_SINGLETON")
        set_provenance_tracker(custom)
        assert get_provenance_tracker() is custom

        reset_provenance_tracker()
        t3 = get_provenance_tracker()
        assert t3 is not custom
