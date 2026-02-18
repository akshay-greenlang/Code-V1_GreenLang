# -*- coding: utf-8 -*-
"""
Unit tests for Stationary Combustion Agent provenance tracking - AGENT-MRV-001.

Tests ProvenanceEntry, ProvenanceTracker, chain verification, tamper
detection, singleton helpers, and thread safety. 35+ tests total.

AGENT-MRV-001: Stationary Combustion Agent (GL-MRV-SCOPE1-001)
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import List

import pytest

from greenlang.stationary_combustion.provenance import (
    VALID_ACTIONS,
    VALID_ENTITY_TYPES,
    ProvenanceEntry,
    ProvenanceTracker,
    get_provenance_tracker,
    reset_provenance_tracker,
    set_provenance_tracker,
)


# =============================================================================
# TestProvenanceEntry
# =============================================================================


class TestProvenanceEntry:
    """Test ProvenanceEntry dataclass creation and serialization."""

    def test_create_entry(self):
        entry = ProvenanceEntry(
            entity_type="fuel_type",
            entity_id="natural_gas",
            action="register",
            hash_value="a" * 64,
            parent_hash="b" * 64,
            timestamp="2026-02-18T00:00:00+00:00",
        )
        assert entry.entity_type == "fuel_type"
        assert entry.entity_id == "natural_gas"
        assert entry.action == "register"
        assert entry.hash_value == "a" * 64
        assert entry.parent_hash == "b" * 64

    def test_entry_hash_is_64_hex_chars(self):
        entry = ProvenanceEntry(
            entity_type="calculation",
            entity_id="calc_001",
            action="calculate",
            hash_value="abcdef1234567890" * 4,
            parent_hash="0" * 64,
            timestamp="2026-02-18T00:00:00+00:00",
        )
        assert len(entry.hash_value) == 64

    def test_entry_timestamp_stored(self):
        ts = "2026-02-18T12:30:45+00:00"
        entry = ProvenanceEntry(
            entity_type="batch",
            entity_id="batch_001",
            action="calculate_batch",
            hash_value="a" * 64,
            parent_hash="b" * 64,
            timestamp=ts,
        )
        assert entry.timestamp == ts

    def test_entry_metadata_default_empty(self):
        entry = ProvenanceEntry(
            entity_type="fuel_type",
            entity_id="diesel",
            action="register",
            hash_value="a" * 64,
            parent_hash="b" * 64,
            timestamp="2026-02-18T00:00:00+00:00",
        )
        assert entry.metadata == {}

    def test_entry_metadata_custom(self):
        entry = ProvenanceEntry(
            entity_type="fuel_type",
            entity_id="diesel",
            action="register",
            hash_value="a" * 64,
            parent_hash="b" * 64,
            timestamp="2026-02-18T00:00:00+00:00",
            metadata={"data_hash": "c" * 64, "source": "EPA"},
        )
        assert entry.metadata["source"] == "EPA"
        assert entry.metadata["data_hash"] == "c" * 64

    def test_entry_to_dict(self):
        entry = ProvenanceEntry(
            entity_type="equipment",
            entity_id="eq_001",
            action="register",
            hash_value="a" * 64,
            parent_hash="b" * 64,
            timestamp="2026-02-18T00:00:00+00:00",
            metadata={"data_hash": "d" * 64},
        )
        d = entry.to_dict()
        assert isinstance(d, dict)
        assert d["entity_type"] == "equipment"
        assert d["entity_id"] == "eq_001"
        assert d["action"] == "register"
        assert d["hash_value"] == "a" * 64
        assert d["parent_hash"] == "b" * 64
        assert d["timestamp"] == "2026-02-18T00:00:00+00:00"
        assert d["metadata"]["data_hash"] == "d" * 64


# =============================================================================
# TestProvenanceConstants
# =============================================================================


class TestProvenanceConstants:
    """Verify valid entity types and actions sets."""

    def test_valid_entity_types_count(self):
        assert len(VALID_ENTITY_TYPES) == 9

    def test_valid_entity_types_members(self):
        expected = {
            "fuel_type", "emission_factor", "equipment", "calculation",
            "batch", "aggregation", "uncertainty", "audit", "pipeline",
        }
        assert VALID_ENTITY_TYPES == expected

    def test_valid_actions_count(self):
        assert len(VALID_ACTIONS) == 14

    def test_valid_actions_members(self):
        expected = {
            "register", "update", "delete",
            "calculate", "calculate_batch", "aggregate",
            "quantify_uncertainty", "generate_audit",
            "validate", "select_factor",
            "convert_units", "apply_gwp", "decompose_gas",
            "run_pipeline",
        }
        assert VALID_ACTIONS == expected


# =============================================================================
# TestProvenanceTracker - Core operations
# =============================================================================


class TestProvenanceTracker:
    """Test ProvenanceTracker record, verify, get, export, clear, len."""

    def test_init_default_genesis(self):
        tracker = ProvenanceTracker()
        assert tracker.genesis_hash is not None
        assert len(tracker.genesis_hash) == 64

    def test_init_custom_genesis(self):
        tracker = ProvenanceTracker(genesis_hash="custom-genesis")
        expected = hashlib.sha256(b"custom-genesis").hexdigest()
        assert tracker.genesis_hash == expected

    def test_genesis_hash_is_sha256_of_input(self):
        input_str = "GL-MRV-X-001-STATIONARY-COMBUSTION-GENESIS"
        tracker = ProvenanceTracker(genesis_hash=input_str)
        expected = hashlib.sha256(input_str.encode("utf-8")).hexdigest()
        assert tracker.genesis_hash == expected

    def test_record_returns_entry(self):
        tracker = ProvenanceTracker()
        entry = tracker.record("fuel_type", "register", "natural_gas")
        assert isinstance(entry, ProvenanceEntry)
        assert entry.entity_type == "fuel_type"
        assert entry.action == "register"
        assert entry.entity_id == "natural_gas"

    def test_record_hash_is_64_hex(self):
        tracker = ProvenanceTracker()
        entry = tracker.record("fuel_type", "register", "natural_gas")
        assert len(entry.hash_value) == 64
        int(entry.hash_value, 16)  # Must be valid hex

    def test_record_first_entry_chains_from_genesis(self):
        tracker = ProvenanceTracker()
        entry = tracker.record("fuel_type", "register", "natural_gas")
        assert entry.parent_hash == tracker.genesis_hash

    def test_record_with_data(self):
        tracker = ProvenanceTracker()
        data = {"hhv": 1.028, "ncv": 0.926}
        entry = tracker.record("fuel_type", "register", "natural_gas", data=data)
        assert "data_hash" in entry.metadata
        assert len(entry.metadata["data_hash"]) == 64

    def test_record_with_metadata(self):
        tracker = ProvenanceTracker()
        entry = tracker.record(
            "calculation", "calculate", "calc_001",
            metadata={"fuel_type": "natural_gas", "quantity": 1000},
        )
        assert entry.metadata["fuel_type"] == "natural_gas"
        assert entry.metadata["quantity"] == 1000

    def test_record_empty_entity_type_raises(self):
        tracker = ProvenanceTracker()
        with pytest.raises(ValueError, match="entity_type must not be empty"):
            tracker.record("", "register", "id")

    def test_record_empty_action_raises(self):
        tracker = ProvenanceTracker()
        with pytest.raises(ValueError, match="action must not be empty"):
            tracker.record("fuel_type", "", "id")

    def test_record_empty_entity_id_raises(self):
        tracker = ProvenanceTracker()
        with pytest.raises(ValueError, match="entity_id must not be empty"):
            tracker.record("fuel_type", "register", "")

    def test_verify_chain_empty_is_valid(self):
        tracker = ProvenanceTracker()
        assert tracker.verify_chain() is True

    def test_verify_chain_single_entry(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        assert tracker.verify_chain() is True

    def test_verify_chain_multiple_entries(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        tracker.record("fuel_type", "register", "diesel")
        tracker.record("emission_factor", "register", "ef_001")
        tracker.record("calculation", "calculate", "calc_001")
        assert tracker.verify_chain() is True

    def test_get_entries_all(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        tracker.record("fuel_type", "register", "diesel")
        tracker.record("emission_factor", "register", "ef_001")
        entries = tracker.get_entries()
        assert len(entries) == 3

    def test_get_entries_by_entity_type(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        tracker.record("fuel_type", "register", "diesel")
        tracker.record("emission_factor", "register", "ef_001")
        entries = tracker.get_entries(entity_type="fuel_type")
        assert len(entries) == 2

    def test_get_entries_by_action(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        tracker.record("fuel_type", "update", "natural_gas")
        tracker.record("fuel_type", "register", "diesel")
        entries = tracker.get_entries(action="register")
        assert len(entries) == 2

    def test_get_entries_with_limit(self):
        tracker = ProvenanceTracker()
        for i in range(10):
            tracker.record("fuel_type", "register", f"fuel_{i}")
        entries = tracker.get_entries(limit=3)
        assert len(entries) == 3

    def test_get_entries_for_entity(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        tracker.record("fuel_type", "update", "natural_gas")
        tracker.record("fuel_type", "register", "diesel")
        entries = tracker.get_entries_for_entity("fuel_type", "natural_gas")
        assert len(entries) == 2

    def test_get_entries_for_entity_not_found(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        entries = tracker.get_entries_for_entity("fuel_type", "nonexistent")
        assert entries == []

    def test_export_json(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        tracker.record("emission_factor", "register", "ef_001")
        json_str = tracker.export_json()
        data = json.loads(json_str)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["entity_type"] == "fuel_type"

    def test_export_json_empty(self):
        tracker = ProvenanceTracker()
        json_str = tracker.export_json()
        data = json.loads(json_str)
        assert data == []

    def test_clear(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        tracker.record("fuel_type", "register", "diesel")
        assert len(tracker) == 2
        tracker.clear()
        assert len(tracker) == 0
        assert tracker.entry_count == 0
        assert tracker.entity_count == 0

    def test_clear_resets_chain_to_genesis(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        original_genesis = tracker.genesis_hash
        tracker.clear()
        assert tracker.last_chain_hash == original_genesis

    def test_len(self):
        tracker = ProvenanceTracker()
        assert len(tracker) == 0
        tracker.record("fuel_type", "register", "natural_gas")
        assert len(tracker) == 1
        tracker.record("fuel_type", "register", "diesel")
        assert len(tracker) == 2

    def test_entry_count_property(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        assert tracker.entry_count == 1

    def test_entity_count_property(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        tracker.record("fuel_type", "update", "natural_gas")
        tracker.record("fuel_type", "register", "diesel")
        assert tracker.entity_count == 2  # 2 unique entity keys

    def test_repr(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        r = repr(tracker)
        assert "ProvenanceTracker(" in r
        assert "entries=1" in r

    def test_build_hash(self):
        tracker = ProvenanceTracker()
        h1 = tracker.build_hash({"key": "value"})
        h2 = tracker.build_hash({"key": "value"})
        assert h1 == h2
        assert len(h1) == 64

    def test_build_hash_none(self):
        tracker = ProvenanceTracker()
        h = tracker.build_hash(None)
        expected = hashlib.sha256(b"null").hexdigest()
        assert h == expected


# =============================================================================
# TestProvenanceChaining - Chain hash linking
# =============================================================================


class TestProvenanceChaining:
    """Verify that multiple entries chain correctly via parent_hash links."""

    def test_second_entry_parent_is_first_hash(self):
        tracker = ProvenanceTracker()
        e1 = tracker.record("fuel_type", "register", "natural_gas")
        e2 = tracker.record("fuel_type", "register", "diesel")
        assert e2.parent_hash == e1.hash_value

    def test_third_entry_parent_is_second_hash(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        e2 = tracker.record("fuel_type", "register", "diesel")
        e3 = tracker.record("emission_factor", "register", "ef_001")
        assert e3.parent_hash == e2.hash_value

    def test_five_entry_chain(self):
        tracker = ProvenanceTracker()
        entries = []
        for i in range(5):
            entry = tracker.record("fuel_type", "register", f"fuel_{i}")
            entries.append(entry)

        # Verify chain integrity manually
        assert entries[0].parent_hash == tracker.genesis_hash
        for i in range(1, 5):
            assert entries[i].parent_hash == entries[i - 1].hash_value

    def test_chain_hashes_are_all_unique(self):
        tracker = ProvenanceTracker()
        hashes = set()
        for i in range(20):
            entry = tracker.record("calculation", "calculate", f"calc_{i}")
            hashes.add(entry.hash_value)
        assert len(hashes) == 20

    def test_last_chain_hash_property(self):
        tracker = ProvenanceTracker()
        assert tracker.last_chain_hash == tracker.genesis_hash
        e1 = tracker.record("fuel_type", "register", "natural_gas")
        assert tracker.last_chain_hash == e1.hash_value


# =============================================================================
# TestProvenanceTamperDetection
# =============================================================================


class TestProvenanceTamperDetection:
    """Verify that tampering with entries breaks chain verification."""

    def test_modified_hash_breaks_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        tracker.record("fuel_type", "register", "diesel")
        tracker.record("emission_factor", "register", "ef_001")

        # Tamper with the first entry's hash_value
        tracker._global_chain[0].hash_value = "0" * 64

        assert tracker.verify_chain() is False

    def test_modified_parent_hash_breaks_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")
        tracker.record("fuel_type", "register", "diesel")

        # Tamper with the second entry's parent_hash
        tracker._global_chain[1].parent_hash = "f" * 64

        assert tracker.verify_chain() is False

    def test_empty_field_breaks_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")

        # Clear a required field
        tracker._global_chain[0].entity_type = ""

        assert tracker.verify_chain() is False

    def test_genesis_mismatch_breaks_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("fuel_type", "register", "natural_gas")

        # Tamper with first entry's parent_hash (should match genesis)
        tracker._global_chain[0].parent_hash = "0" * 64

        assert tracker.verify_chain() is False


# =============================================================================
# TestProvenanceSingleton
# =============================================================================


class TestProvenanceSingleton:
    """Test singleton helpers: get, set, reset."""

    def test_get_provenance_tracker_returns_instance(self):
        tracker = get_provenance_tracker()
        assert isinstance(tracker, ProvenanceTracker)

    def test_get_provenance_tracker_same_instance(self):
        t1 = get_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is t2

    def test_set_provenance_tracker(self):
        custom = ProvenanceTracker(genesis_hash="custom-singleton-genesis")
        set_provenance_tracker(custom)
        retrieved = get_provenance_tracker()
        assert retrieved is custom

    def test_set_provenance_tracker_invalid_type_raises(self):
        with pytest.raises(TypeError, match="must be a ProvenanceTracker"):
            set_provenance_tracker("not_a_tracker")  # type: ignore[arg-type]

    def test_reset_provenance_tracker(self):
        t1 = get_provenance_tracker()
        reset_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is not t2

    def test_reset_creates_fresh_instance(self):
        t1 = get_provenance_tracker()
        t1.record("fuel_type", "register", "natural_gas")
        assert t1.entry_count >= 1

        reset_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t2.entry_count == 0


# =============================================================================
# TestProvenanceThreadSafety
# =============================================================================


class TestProvenanceThreadSafety:
    """Verify concurrent record operations maintain chain integrity."""

    def test_concurrent_records_from_multiple_threads(self):
        """10 threads each recording 20 entries; chain must remain valid."""
        tracker = ProvenanceTracker(genesis_hash="thread-safety-test")
        barrier = threading.Barrier(10)
        errors: List[str] = []

        def worker(thread_id: int):
            try:
                barrier.wait(timeout=5.0)
                for i in range(20):
                    tracker.record(
                        "calculation",
                        "calculate",
                        f"thread_{thread_id}_calc_{i}",
                        data={"thread": thread_id, "index": i},
                    )
            except Exception as exc:
                errors.append(f"Thread {thread_id} error: {exc}")

        threads = [
            threading.Thread(target=worker, args=(tid,))
            for tid in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert errors == [], f"Thread errors: {errors}"
        assert len(tracker) == 200  # 10 threads * 20 entries
        assert tracker.verify_chain() is True

    def test_concurrent_singleton_access(self):
        """Multiple threads accessing singleton get the same tracker."""
        trackers = []
        barrier = threading.Barrier(5)

        def worker():
            barrier.wait(timeout=5.0)
            t = get_provenance_tracker()
            trackers.append(id(t))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(set(trackers)) == 1
