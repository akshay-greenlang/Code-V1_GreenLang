# -*- coding: utf-8 -*-
"""
Unit tests for Mobile Combustion Provenance Tracking - AGENT-MRV-003

Tests ProvenanceEntry dataclass, ProvenanceTracker class (genesis hash,
record, chain hashing, verify_chain, get_entries, entity types, actions),
thread safety, and singleton management.

Target: 57+ tests
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import pytest

from greenlang.mobile_combustion.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    VALID_ACTIONS,
    VALID_ENTITY_TYPES,
    get_provenance_tracker,
    reset_provenance_tracker,
    set_provenance_tracker,
)


# =========================================================================
# TestProvenanceEntry - 8 tests
# =========================================================================


class TestProvenanceEntry:
    """Tests for the ProvenanceEntry dataclass."""

    def test_construction_with_all_fields(self) -> None:
        entry = ProvenanceEntry(
            entity_type="vehicle",
            entity_id="veh_001",
            action="register",
            hash_value="abc123",
            parent_hash="genesis",
            timestamp="2026-02-18T12:00:00+00:00",
            metadata={"data_hash": "def456"},
        )
        assert entry.entity_type == "vehicle"
        assert entry.entity_id == "veh_001"
        assert entry.action == "register"
        assert entry.hash_value == "abc123"
        assert entry.parent_hash == "genesis"

    def test_default_metadata_empty_dict(self) -> None:
        entry = ProvenanceEntry(
            entity_type="trip",
            entity_id="trip_001",
            action="create",
            hash_value="h1",
            parent_hash="h0",
            timestamp="2026-02-18T12:00:00+00:00",
        )
        assert entry.metadata == {}

    def test_to_dict_returns_all_fields(self) -> None:
        entry = ProvenanceEntry(
            entity_type="factor",
            entity_id="ef_001",
            action="read",
            hash_value="hash_val",
            parent_hash="parent_val",
            timestamp="2026-02-18T12:00:00+00:00",
            metadata={"source": "EPA"},
        )
        d = entry.to_dict()
        assert d["entity_type"] == "factor"
        assert d["entity_id"] == "ef_001"
        assert d["action"] == "read"
        assert d["hash_value"] == "hash_val"
        assert d["parent_hash"] == "parent_val"
        assert d["timestamp"] == "2026-02-18T12:00:00+00:00"
        assert d["metadata"]["source"] == "EPA"

    def test_to_dict_has_exactly_seven_keys(self) -> None:
        entry = ProvenanceEntry(
            entity_type="vehicle",
            entity_id="v1",
            action="create",
            hash_value="h",
            parent_hash="p",
            timestamp="ts",
        )
        d = entry.to_dict()
        assert set(d.keys()) == {
            "entity_type", "entity_id", "action",
            "hash_value", "parent_hash", "timestamp", "metadata",
        }

    def test_metadata_is_mutable(self) -> None:
        entry = ProvenanceEntry(
            entity_type="vehicle",
            entity_id="v1",
            action="create",
            hash_value="h",
            parent_hash="p",
            timestamp="ts",
        )
        entry.metadata["extra"] = "value"
        assert entry.metadata["extra"] == "value"

    def test_metadata_preserved_in_to_dict(self) -> None:
        meta = {"key1": "val1", "key2": 42}
        entry = ProvenanceEntry(
            entity_type="batch",
            entity_id="b1",
            action="create",
            hash_value="h",
            parent_hash="p",
            timestamp="ts",
            metadata=meta,
        )
        d = entry.to_dict()
        assert d["metadata"]["key1"] == "val1"
        assert d["metadata"]["key2"] == 42

    def test_timestamp_stored_as_string(self) -> None:
        entry = ProvenanceEntry(
            entity_type="vehicle",
            entity_id="v1",
            action="create",
            hash_value="h",
            parent_hash="p",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert isinstance(entry.timestamp, str)

    def test_entry_equality_by_values(self) -> None:
        kwargs = dict(
            entity_type="vehicle",
            entity_id="v1",
            action="create",
            hash_value="h",
            parent_hash="p",
            timestamp="ts",
        )
        e1 = ProvenanceEntry(**kwargs)
        e2 = ProvenanceEntry(**kwargs)
        assert e1 == e2


# =========================================================================
# TestGenesisHash - 6 tests
# =========================================================================


class TestGenesisHash:
    """Tests for genesis hash computation and properties."""

    def test_genesis_hash_is_sha256(self, tracker: ProvenanceTracker) -> None:
        assert len(tracker.genesis_hash) == 64
        # Verify only hex characters
        int(tracker.genesis_hash, 16)

    def test_genesis_hash_deterministic(self) -> None:
        t1 = ProvenanceTracker(genesis_hash="SAME-SEED")
        t2 = ProvenanceTracker(genesis_hash="SAME-SEED")
        assert t1.genesis_hash == t2.genesis_hash

    def test_different_seeds_different_hashes(self) -> None:
        t1 = ProvenanceTracker(genesis_hash="SEED-A")
        t2 = ProvenanceTracker(genesis_hash="SEED-B")
        assert t1.genesis_hash != t2.genesis_hash

    def test_genesis_matches_manual_sha256(self) -> None:
        seed = "GL-MRV-X-003-MOBILE-COMBUSTION-GENESIS"
        expected = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        t = ProvenanceTracker(genesis_hash=seed)
        assert t.genesis_hash == expected

    def test_custom_genesis_hash(self, custom_tracker: ProvenanceTracker) -> None:
        expected = hashlib.sha256(
            "CUSTOM-MOBILE-COMBUSTION-GENESIS-TEST".encode("utf-8")
        ).hexdigest()
        assert custom_tracker.genesis_hash == expected

    def test_last_chain_hash_initially_equals_genesis(self, tracker: ProvenanceTracker) -> None:
        assert tracker.last_chain_hash == tracker.genesis_hash


# =========================================================================
# TestRecord - 11 tests
# =========================================================================


class TestRecord:
    """Tests for the record() method."""

    def test_record_returns_provenance_entry(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("vehicle", "register", "veh_001")
        assert isinstance(entry, ProvenanceEntry)

    def test_record_sets_entity_type(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("trip", "create", "trip_001")
        assert entry.entity_type == "trip"

    def test_record_sets_entity_id(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("factor", "read", "ef_001")
        assert entry.entity_id == "ef_001"

    def test_record_sets_action(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("calculation", "calculate", "calc_001")
        assert entry.action == "calculate"

    def test_record_sets_hash_value(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("vehicle", "register", "veh_001")
        assert len(entry.hash_value) == 64

    def test_record_increments_entry_count(self, tracker: ProvenanceTracker) -> None:
        assert tracker.entry_count == 0
        tracker.record("vehicle", "register", "veh_001")
        assert tracker.entry_count == 1
        tracker.record("trip", "create", "trip_001")
        assert tracker.entry_count == 2

    def test_record_with_data(self, tracker: ProvenanceTracker) -> None:
        data = {"make": "Toyota", "model": "Corolla"}
        entry = tracker.record("vehicle", "register", "veh_001", data=data)
        assert "data_hash" in entry.metadata
        assert len(entry.metadata["data_hash"]) == 64

    def test_record_with_metadata(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record(
            "vehicle", "register", "veh_001",
            metadata={"user": "admin"},
        )
        assert entry.metadata["user"] == "admin"

    def test_record_empty_entity_type_raises(self, tracker: ProvenanceTracker) -> None:
        with pytest.raises(ValueError, match="entity_type"):
            tracker.record("", "create", "id_001")

    def test_record_empty_action_raises(self, tracker: ProvenanceTracker) -> None:
        with pytest.raises(ValueError, match="action"):
            tracker.record("vehicle", "", "veh_001")

    def test_record_empty_entity_id_raises(self, tracker: ProvenanceTracker) -> None:
        with pytest.raises(ValueError, match="entity_id"):
            tracker.record("vehicle", "register", "")


# =========================================================================
# TestChainHashing - 5 tests
# =========================================================================


class TestChainHashing:
    """Tests for chain hash linking integrity."""

    def test_first_entry_parent_is_genesis(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("vehicle", "register", "veh_001")
        assert entry.parent_hash == tracker.genesis_hash

    def test_second_entry_parent_is_first_hash(self, tracker: ProvenanceTracker) -> None:
        e1 = tracker.record("vehicle", "register", "veh_001")
        e2 = tracker.record("trip", "create", "trip_001")
        assert e2.parent_hash == e1.hash_value

    def test_chain_hashes_are_unique(self, tracker: ProvenanceTracker) -> None:
        entries = []
        for i in range(10):
            e = tracker.record("vehicle", "register", f"veh_{i:03d}")
            entries.append(e)
        hashes = [e.hash_value for e in entries]
        assert len(set(hashes)) == 10

    def test_last_chain_hash_updates_after_record(self, tracker: ProvenanceTracker) -> None:
        initial = tracker.last_chain_hash
        e = tracker.record("vehicle", "register", "veh_001")
        assert tracker.last_chain_hash == e.hash_value
        assert tracker.last_chain_hash != initial

    def test_chain_is_sequential(self, tracker: ProvenanceTracker) -> None:
        e1 = tracker.record("vehicle", "register", "veh_001")
        e2 = tracker.record("trip", "create", "trip_001")
        e3 = tracker.record("factor", "read", "ef_001")
        assert e1.parent_hash == tracker.genesis_hash
        assert e2.parent_hash == e1.hash_value
        assert e3.parent_hash == e2.hash_value


# =========================================================================
# TestVerifyChain - 7 tests
# =========================================================================


class TestVerifyChain:
    """Tests for verify_chain() integrity checking."""

    def test_empty_chain_is_valid(self, tracker: ProvenanceTracker) -> None:
        assert tracker.verify_chain() is True

    def test_single_entry_chain_valid(self, tracker: ProvenanceTracker) -> None:
        tracker.record("vehicle", "register", "veh_001")
        assert tracker.verify_chain() is True

    def test_multi_entry_chain_valid(self, tracker: ProvenanceTracker) -> None:
        for i in range(5):
            tracker.record("vehicle", "register", f"veh_{i:03d}")
        assert tracker.verify_chain() is True

    def test_populated_tracker_valid(self, populated_tracker: ProvenanceTracker) -> None:
        assert populated_tracker.verify_chain() is True

    def test_tampered_hash_detected(self, tracker: ProvenanceTracker) -> None:
        tracker.record("vehicle", "register", "veh_001")
        tracker.record("trip", "create", "trip_001")
        # Tamper with the first entry's hash
        with tracker._lock:
            tracker._global_chain[0].hash_value = "tampered_hash_value_000000000000000"
        assert tracker.verify_chain() is False

    def test_tampered_parent_hash_detected(self, tracker: ProvenanceTracker) -> None:
        tracker.record("vehicle", "register", "veh_001")
        tracker.record("trip", "create", "trip_001")
        # Tamper with the second entry's parent_hash
        with tracker._lock:
            tracker._global_chain[1].parent_hash = "wrong_parent_hash_0000000000000000"
        assert tracker.verify_chain() is False

    def test_cleared_and_re_recorded_chain_valid(self, tracker: ProvenanceTracker) -> None:
        tracker.record("vehicle", "register", "veh_001")
        tracker.clear()
        assert tracker.verify_chain() is True
        tracker.record("trip", "create", "trip_002")
        assert tracker.verify_chain() is True


# =========================================================================
# TestEntityTypes - 6 tests
# =========================================================================


class TestEntityTypes:
    """Tests for VALID_ENTITY_TYPES and VALID_ACTIONS constants."""

    def test_ten_entity_types(self) -> None:
        assert len(VALID_ENTITY_TYPES) == 10

    @pytest.mark.parametrize("entity_type", [
        "vehicle", "trip", "fuel", "factor", "calculation",
        "batch", "fleet", "compliance", "uncertainty", "audit",
    ])
    def test_entity_type_present(self, entity_type: str) -> None:
        assert entity_type in VALID_ENTITY_TYPES

    def test_fifteen_actions(self) -> None:
        assert len(VALID_ACTIONS) == 15

    @pytest.mark.parametrize("action", [
        "create", "read", "update", "delete",
        "calculate", "aggregate", "validate", "estimate",
        "analyze", "check",
        "export", "import",
        "register", "deregister",
        "pipeline",
    ])
    def test_action_present(self, action: str) -> None:
        assert action in VALID_ACTIONS

    def test_entity_types_frozenset(self) -> None:
        assert isinstance(VALID_ENTITY_TYPES, frozenset)

    def test_actions_frozenset(self) -> None:
        assert isinstance(VALID_ACTIONS, frozenset)


# =========================================================================
# TestGetEntries - 8 tests
# =========================================================================


class TestGetEntries:
    """Tests for get_entries(), get_entries_for_entity(), get_chain(), get_audit_trail()."""

    def test_get_chain_returns_all_entries(self, populated_tracker: ProvenanceTracker) -> None:
        chain = populated_tracker.get_chain()
        assert len(chain) == 9  # 9 entries in populated_tracker

    def test_get_entries_filter_by_entity_type(self, populated_tracker: ProvenanceTracker) -> None:
        entries = populated_tracker.get_entries(entity_type="vehicle")
        assert len(entries) == 2
        for e in entries:
            assert e.entity_type == "vehicle"

    def test_get_entries_filter_by_action(self, populated_tracker: ProvenanceTracker) -> None:
        entries = populated_tracker.get_entries(action="register")
        assert len(entries) == 2
        for e in entries:
            assert e.action == "register"

    def test_get_entries_filter_by_both(self, populated_tracker: ProvenanceTracker) -> None:
        entries = populated_tracker.get_entries(entity_type="vehicle", action="register")
        assert len(entries) == 2

    def test_get_entries_with_limit(self, populated_tracker: ProvenanceTracker) -> None:
        entries = populated_tracker.get_entries(limit=3)
        assert len(entries) == 3

    def test_get_entries_for_entity(self, populated_tracker: ProvenanceTracker) -> None:
        entries = populated_tracker.get_entries_for_entity("vehicle", "veh_001")
        assert len(entries) == 1
        assert entries[0].entity_id == "veh_001"

    def test_get_entries_for_missing_entity(self, populated_tracker: ProvenanceTracker) -> None:
        entries = populated_tracker.get_entries_for_entity("vehicle", "nonexistent")
        assert entries == []

    def test_get_audit_trail_returns_dicts(self, populated_tracker: ProvenanceTracker) -> None:
        trail = populated_tracker.get_audit_trail()
        assert len(trail) == 9
        for item in trail:
            assert isinstance(item, dict)
            assert "entity_type" in item
            assert "hash_value" in item

    def test_get_audit_trail_filtered_by_type(self, populated_tracker: ProvenanceTracker) -> None:
        trail = populated_tracker.get_audit_trail(entity_type="calculation")
        assert len(trail) == 1
        assert trail[0]["entity_type"] == "calculation"

    def test_get_audit_trail_filtered_by_type_and_id(self, populated_tracker: ProvenanceTracker) -> None:
        trail = populated_tracker.get_audit_trail(entity_type="trip", entity_id="trip_001")
        assert len(trail) == 1
        assert trail[0]["entity_id"] == "trip_001"

    def test_export_trail_is_valid_json(self, populated_tracker: ProvenanceTracker) -> None:
        json_str = populated_tracker.export_trail()
        parsed = json.loads(json_str)
        assert len(parsed) == 9

    def test_entry_count_property(self, populated_tracker: ProvenanceTracker) -> None:
        assert populated_tracker.entry_count == 9

    def test_entity_count_property(self, populated_tracker: ProvenanceTracker) -> None:
        assert populated_tracker.entity_count == 9  # 9 unique entity_type:entity_id pairs

    def test_len_matches_entry_count(self, populated_tracker: ProvenanceTracker) -> None:
        assert len(populated_tracker) == populated_tracker.entry_count

    def test_repr_contains_entries(self, populated_tracker: ProvenanceTracker) -> None:
        r = repr(populated_tracker)
        assert "entries=9" in r

    def test_clear_resets_to_empty(self, populated_tracker: ProvenanceTracker) -> None:
        populated_tracker.clear()
        assert populated_tracker.entry_count == 0
        assert populated_tracker.entity_count == 0
        assert populated_tracker.last_chain_hash == populated_tracker.genesis_hash

    def test_build_hash_utility(self, tracker: ProvenanceTracker) -> None:
        h1 = tracker.build_hash({"key": "value"})
        h2 = tracker.build_hash({"key": "value"})
        assert h1 == h2
        assert len(h1) == 64

    def test_build_hash_none(self, tracker: ProvenanceTracker) -> None:
        h = tracker.build_hash(None)
        expected = hashlib.sha256(b"null").hexdigest()
        assert h == expected


# =========================================================================
# TestThreadSafety - 2 tests
# =========================================================================


class TestThreadSafety:
    """Tests for concurrent access to ProvenanceTracker."""

    def test_concurrent_record_no_data_loss(self) -> None:
        tracker = ProvenanceTracker()
        num_threads = 8
        records_per_thread = 50
        barrier = threading.Barrier(num_threads)

        def _worker(thread_id: int) -> None:
            barrier.wait()
            for i in range(records_per_thread):
                tracker.record(
                    "vehicle",
                    "register",
                    f"veh_t{thread_id}_{i:03d}",
                )

        threads = []
        for tid in range(num_threads):
            t = threading.Thread(target=_worker, args=(tid,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        assert tracker.entry_count == num_threads * records_per_thread

    def test_concurrent_record_and_verify(self) -> None:
        tracker = ProvenanceTracker()
        errors: List[Exception] = []

        def _recorder() -> None:
            try:
                for i in range(100):
                    tracker.record("trip", "create", f"trip_{i:04d}")
            except Exception as exc:
                errors.append(exc)

        def _verifier() -> None:
            try:
                for _ in range(20):
                    tracker.verify_chain()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_recorder),
            threading.Thread(target=_verifier),
            threading.Thread(target=_recorder),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Thread errors: {errors}"


# =========================================================================
# TestSingleton - 4 tests
# =========================================================================


class TestSingleton:
    """Tests for get/set/reset provenance tracker singleton."""

    def test_get_provenance_tracker_returns_instance(self) -> None:
        t = get_provenance_tracker()
        assert isinstance(t, ProvenanceTracker)

    def test_get_provenance_tracker_returns_same_instance(self) -> None:
        t1 = get_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is t2

    def test_set_provenance_tracker_replaces(self) -> None:
        custom = ProvenanceTracker(genesis_hash="CUSTOM-SINGLETON-TEST")
        set_provenance_tracker(custom)
        assert get_provenance_tracker() is custom

    def test_reset_provenance_tracker_clears(self) -> None:
        t1 = get_provenance_tracker()
        reset_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is not t2

    def test_set_provenance_tracker_type_error(self) -> None:
        with pytest.raises(TypeError):
            set_provenance_tracker("not_a_tracker")  # type: ignore[arg-type]
