# -*- coding: utf-8 -*-
"""
Unit tests for Refrigerants & F-Gas Agent Provenance Tracking - AGENT-MRV-002

Tests the ProvenanceTracker, ProvenanceEntry, chain hashing, verification,
filtering, JSON export, singleton management, thread safety, and all
entity types and actions.

Target: 60+ tests.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import pytest

from greenlang.refrigerants_fgas.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
    get_provenance_tracker,
    set_provenance_tracker,
    reset_provenance_tracker,
)


# ===================================================================
# ProvenanceEntry tests
# ===================================================================


class TestProvenanceEntry:
    """Test ProvenanceEntry dataclass."""

    def test_construction(self, sample_provenance_entry: ProvenanceEntry):
        assert sample_provenance_entry.entity_type == "refrigerant"
        assert sample_provenance_entry.entity_id == "R_410A"
        assert sample_provenance_entry.action == "register"

    def test_hash_value_length(self, sample_provenance_entry: ProvenanceEntry):
        assert len(sample_provenance_entry.hash_value) == 64

    def test_parent_hash_length(self, sample_provenance_entry: ProvenanceEntry):
        assert len(sample_provenance_entry.parent_hash) == 64

    def test_timestamp_is_string(self, sample_provenance_entry: ProvenanceEntry):
        assert isinstance(sample_provenance_entry.timestamp, str)

    def test_metadata_is_dict(self, sample_provenance_entry: ProvenanceEntry):
        assert isinstance(sample_provenance_entry.metadata, dict)

    def test_to_dict_keys(self, sample_provenance_entry: ProvenanceEntry):
        d = sample_provenance_entry.to_dict()
        expected_keys = {
            "entity_type", "entity_id", "action",
            "hash_value", "parent_hash", "timestamp", "metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self, sample_provenance_entry: ProvenanceEntry):
        d = sample_provenance_entry.to_dict()
        assert d["entity_type"] == "refrigerant"
        assert d["entity_id"] == "R_410A"
        assert d["action"] == "register"

    def test_default_metadata_empty_dict(self):
        entry = ProvenanceEntry(
            entity_type="calculation",
            entity_id="calc_001",
            action="calculate",
            hash_value="x" * 64,
            parent_hash="y" * 64,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert entry.metadata == {}


# ===================================================================
# Genesis hash tests
# ===================================================================


class TestGenesisHash:
    """Test genesis hash computation."""

    def test_genesis_hash_is_sha256(self, tracker: ProvenanceTracker):
        assert len(tracker.genesis_hash) == 64

    def test_genesis_hash_deterministic(self):
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        assert t1.genesis_hash == t2.genesis_hash

    def test_custom_genesis_hash_differs(self, tracker: ProvenanceTracker, custom_tracker: ProvenanceTracker):
        assert tracker.genesis_hash != custom_tracker.genesis_hash

    def test_genesis_hash_matches_manual_sha256(self):
        expected = hashlib.sha256(
            "GL-MRV-X-002-REFRIGERANTS-FGAS-GENESIS".encode("utf-8")
        ).hexdigest()
        t = ProvenanceTracker()
        assert t.genesis_hash == expected

    def test_custom_genesis_hash_matches_manual(self):
        custom_text = "custom-test-genesis-hash"
        expected = hashlib.sha256(custom_text.encode("utf-8")).hexdigest()
        t = ProvenanceTracker(genesis_hash=custom_text)
        assert t.genesis_hash == expected

    def test_last_chain_hash_equals_genesis_initially(self, tracker: ProvenanceTracker):
        assert tracker.last_chain_hash == tracker.genesis_hash


# ===================================================================
# Record creation tests
# ===================================================================


class TestRecord:
    """Test recording provenance entries."""

    def test_record_creates_entry(self, tracker: ProvenanceTracker):
        entry = tracker.record("refrigerant", "register", "R_410A")
        assert isinstance(entry, ProvenanceEntry)

    def test_record_entry_fields(self, tracker: ProvenanceTracker):
        entry = tracker.record("refrigerant", "register", "R_410A")
        assert entry.entity_type == "refrigerant"
        assert entry.entity_id == "R_410A"
        assert entry.action == "register"
        assert len(entry.hash_value) == 64
        assert len(entry.parent_hash) == 64
        assert entry.timestamp != ""

    def test_record_with_data(self, tracker: ProvenanceTracker):
        entry = tracker.record(
            "refrigerant", "register", "R_134A",
            data={"gwp": 1430, "category": "hfc"},
        )
        assert "data_hash" in entry.metadata

    def test_record_with_metadata(self, tracker: ProvenanceTracker):
        entry = tracker.record(
            "equipment", "register", "eq_001",
            metadata={"location": "Building A"},
        )
        assert entry.metadata["location"] == "Building A"
        assert "data_hash" in entry.metadata

    def test_record_without_data_uses_null_hash(self, tracker: ProvenanceTracker):
        entry = tracker.record("refrigerant", "register", "R_32")
        # data=None should produce hash of "null"
        expected_data_hash = hashlib.sha256("null".encode("utf-8")).hexdigest()
        assert entry.metadata["data_hash"] == expected_data_hash

    def test_record_increments_entry_count(self, tracker: ProvenanceTracker):
        assert tracker.entry_count == 0
        tracker.record("refrigerant", "register", "R_410A")
        assert tracker.entry_count == 1
        tracker.record("equipment", "register", "eq_001")
        assert tracker.entry_count == 2

    def test_record_increments_entity_count(self, tracker: ProvenanceTracker):
        assert tracker.entity_count == 0
        tracker.record("refrigerant", "register", "R_410A")
        assert tracker.entity_count == 1
        tracker.record("refrigerant", "lookup", "R_410A")
        # Same entity key, so count stays 1
        assert tracker.entity_count == 1
        tracker.record("equipment", "register", "eq_001")
        assert tracker.entity_count == 2

    def test_empty_entity_type_raises(self, tracker: ProvenanceTracker):
        with pytest.raises(ValueError, match="entity_type"):
            tracker.record("", "register", "R_410A")

    def test_empty_action_raises(self, tracker: ProvenanceTracker):
        with pytest.raises(ValueError, match="action"):
            tracker.record("refrigerant", "", "R_410A")

    def test_empty_entity_id_raises(self, tracker: ProvenanceTracker):
        with pytest.raises(ValueError, match="entity_id"):
            tracker.record("refrigerant", "register", "")


# ===================================================================
# Chain hashing tests
# ===================================================================


class TestChainHashing:
    """Test chain hash linking between entries."""

    def test_first_entry_parent_is_genesis(self, tracker: ProvenanceTracker):
        entry = tracker.record("refrigerant", "register", "R_410A")
        assert entry.parent_hash == tracker.genesis_hash

    def test_second_entry_parent_is_first_hash(self, tracker: ProvenanceTracker):
        e1 = tracker.record("refrigerant", "register", "R_410A")
        e2 = tracker.record("refrigerant", "register", "R_134A")
        assert e2.parent_hash == e1.hash_value

    def test_third_entry_parent_is_second_hash(self, tracker: ProvenanceTracker):
        e1 = tracker.record("refrigerant", "register", "R_410A")
        e2 = tracker.record("equipment", "register", "eq_001")
        e3 = tracker.record("calculation", "calculate", "calc_001")
        assert e3.parent_hash == e2.hash_value

    def test_last_chain_hash_updated(self, tracker: ProvenanceTracker):
        e1 = tracker.record("refrigerant", "register", "R_410A")
        assert tracker.last_chain_hash == e1.hash_value
        e2 = tracker.record("equipment", "register", "eq_001")
        assert tracker.last_chain_hash == e2.hash_value

    def test_hashes_are_unique(self, tracker: ProvenanceTracker):
        hashes = set()
        for i in range(20):
            entry = tracker.record("refrigerant", "register", f"R_{i}")
            hashes.add(entry.hash_value)
        assert len(hashes) == 20


# ===================================================================
# Chain verification tests
# ===================================================================


class TestVerifyChain:
    """Test chain integrity verification."""

    def test_empty_chain_is_valid(self, tracker: ProvenanceTracker):
        assert tracker.verify_chain() is True

    def test_single_entry_chain_is_valid(self, tracker: ProvenanceTracker):
        tracker.record("refrigerant", "register", "R_410A")
        assert tracker.verify_chain() is True

    def test_multi_entry_chain_is_valid(self, populated_tracker: ProvenanceTracker):
        assert populated_tracker.verify_chain() is True

    def test_tampered_hash_detected(self, tracker: ProvenanceTracker):
        tracker.record("refrigerant", "register", "R_410A")
        tracker.record("equipment", "register", "eq_001")

        # Tamper with the first entry's hash
        tracker._global_chain[0].hash_value = "0" * 64

        assert tracker.verify_chain() is False

    def test_tampered_parent_hash_detected(self, tracker: ProvenanceTracker):
        tracker.record("refrigerant", "register", "R_410A")
        tracker.record("equipment", "register", "eq_001")

        # Tamper with the second entry's parent_hash
        tracker._global_chain[1].parent_hash = "0" * 64

        assert tracker.verify_chain() is False

    def test_missing_genesis_link_detected(self, tracker: ProvenanceTracker):
        tracker.record("refrigerant", "register", "R_410A")

        # Break genesis link
        tracker._global_chain[0].parent_hash = "0" * 64

        assert tracker.verify_chain() is False

    def test_empty_field_detected(self, tracker: ProvenanceTracker):
        tracker.record("refrigerant", "register", "R_410A")

        # Clear a required field
        tracker._global_chain[0].entity_type = ""

        assert tracker.verify_chain() is False


# ===================================================================
# Entity types and actions tests
# ===================================================================


class TestEntityTypesAndActions:
    """Test all valid entity types and actions."""

    def test_valid_entity_types_count(self):
        assert len(VALID_ENTITY_TYPES) == 9

    @pytest.mark.parametrize("entity_type", [
        "refrigerant", "equipment", "service_event", "leak_rate",
        "calculation", "blend", "compliance", "audit", "pipeline",
    ])
    def test_entity_type_in_valid_set(self, entity_type: str):
        assert entity_type in VALID_ENTITY_TYPES

    def test_valid_actions_count(self):
        assert len(VALID_ACTIONS) == 14

    @pytest.mark.parametrize("action", [
        "register", "lookup", "service",
        "estimate", "calculate", "decompose", "aggregate",
        "validate", "check_compliance",
        "batch", "export",
        "pipeline_start", "pipeline_end", "pipeline_fail",
    ])
    def test_action_in_valid_set(self, action: str):
        assert action in VALID_ACTIONS

    @pytest.mark.parametrize("entity_type", list(VALID_ENTITY_TYPES))
    def test_record_all_entity_types(self, tracker: ProvenanceTracker, entity_type: str):
        entry = tracker.record(entity_type, "register", f"test_{entity_type}")
        assert entry.entity_type == entity_type

    @pytest.mark.parametrize("action", list(VALID_ACTIONS))
    def test_record_all_actions(self, tracker: ProvenanceTracker, action: str):
        entry = tracker.record("refrigerant", action, "test_entity")
        assert entry.action == action


# ===================================================================
# get_entries tests
# ===================================================================


class TestGetEntries:
    """Test entry retrieval and filtering."""

    def test_get_all_entries(self, populated_tracker: ProvenanceTracker):
        entries = populated_tracker.get_entries()
        assert len(entries) == 7

    def test_filter_by_entity_type(self, populated_tracker: ProvenanceTracker):
        entries = populated_tracker.get_entries(entity_type="refrigerant")
        assert len(entries) == 2
        assert all(e.entity_type == "refrigerant" for e in entries)

    def test_filter_by_action(self, populated_tracker: ProvenanceTracker):
        entries = populated_tracker.get_entries(action="register")
        assert len(entries) == 3  # R_410A, R_134A, eq_chiller_01

    def test_filter_by_entity_type_and_action(self, populated_tracker: ProvenanceTracker):
        entries = populated_tracker.get_entries(
            entity_type="refrigerant",
            action="register",
        )
        assert len(entries) == 2

    def test_limit(self, populated_tracker: ProvenanceTracker):
        entries = populated_tracker.get_entries(limit=3)
        assert len(entries) == 3

    def test_limit_returns_most_recent(self, populated_tracker: ProvenanceTracker):
        entries = populated_tracker.get_entries(limit=1)
        assert len(entries) == 1
        # Should be the last recorded entry (compliance check)
        assert entries[0].entity_type == "compliance"

    def test_empty_result(self, populated_tracker: ProvenanceTracker):
        entries = populated_tracker.get_entries(entity_type="nonexistent")
        assert len(entries) == 0

    def test_get_entries_for_entity(self, populated_tracker: ProvenanceTracker):
        entries = populated_tracker.get_entries_for_entity("refrigerant", "R_410A")
        assert len(entries) == 1
        assert entries[0].entity_id == "R_410A"

    def test_get_entries_for_unknown_entity(self, populated_tracker: ProvenanceTracker):
        entries = populated_tracker.get_entries_for_entity("refrigerant", "R_999")
        assert len(entries) == 0


# ===================================================================
# Export and serialization tests
# ===================================================================


class TestExportAndSerialization:
    """Test JSON export and dict conversion."""

    def test_export_json_empty_tracker(self, tracker: ProvenanceTracker):
        result = tracker.export_json()
        assert result == "[]"

    def test_export_json_returns_valid_json(self, populated_tracker: ProvenanceTracker):
        result = populated_tracker.export_json()
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 7

    def test_export_json_entry_keys(self, tracker: ProvenanceTracker):
        tracker.record("refrigerant", "register", "R_410A")
        result = json.loads(tracker.export_json())
        entry = result[0]
        expected_keys = {
            "entity_type", "entity_id", "action",
            "hash_value", "parent_hash", "timestamp", "metadata",
        }
        assert set(entry.keys()) == expected_keys

    def test_metadata_serialization(self, tracker: ProvenanceTracker):
        tracker.record(
            "equipment", "register", "eq_001",
            data={"charge_kg": 50.0, "type": "chiller"},
            metadata={"facility": "Building A"},
        )
        result = json.loads(tracker.export_json())
        meta = result[0]["metadata"]
        assert "data_hash" in meta
        assert meta["facility"] == "Building A"


# ===================================================================
# Clear / reset tests
# ===================================================================


class TestClear:
    """Test tracker clear/reset behavior."""

    def test_clear_empties_chain(self, populated_tracker: ProvenanceTracker):
        assert populated_tracker.entry_count > 0
        populated_tracker.clear()
        assert populated_tracker.entry_count == 0

    def test_clear_empties_entity_store(self, populated_tracker: ProvenanceTracker):
        assert populated_tracker.entity_count > 0
        populated_tracker.clear()
        assert populated_tracker.entity_count == 0

    def test_clear_resets_last_chain_hash_to_genesis(self, populated_tracker: ProvenanceTracker):
        populated_tracker.clear()
        assert populated_tracker.last_chain_hash == populated_tracker.genesis_hash

    def test_clear_then_record(self, populated_tracker: ProvenanceTracker):
        populated_tracker.clear()
        entry = populated_tracker.record("refrigerant", "register", "R_32")
        assert entry.parent_hash == populated_tracker.genesis_hash
        assert populated_tracker.verify_chain() is True


# ===================================================================
# Property tests
# ===================================================================


class TestProperties:
    """Test tracker properties."""

    def test_entry_count_zero(self, tracker: ProvenanceTracker):
        assert tracker.entry_count == 0

    def test_entity_count_zero(self, tracker: ProvenanceTracker):
        assert tracker.entity_count == 0

    def test_len_dunder(self, tracker: ProvenanceTracker):
        assert len(tracker) == 0
        tracker.record("refrigerant", "register", "R_410A")
        assert len(tracker) == 1

    def test_repr(self, tracker: ProvenanceTracker):
        rep = repr(tracker)
        assert "ProvenanceTracker(" in rep
        assert "entries=0" in rep
        assert "entities=0" in rep
        assert "genesis_prefix=" in rep

    def test_repr_after_records(self, populated_tracker: ProvenanceTracker):
        rep = repr(populated_tracker)
        assert "entries=7" in rep


# ===================================================================
# build_hash utility tests
# ===================================================================


class TestBuildHash:
    """Test the public build_hash utility method."""

    def test_build_hash_returns_64_char_hex(self, tracker: ProvenanceTracker):
        h = tracker.build_hash({"key": "value"})
        assert len(h) == 64

    def test_build_hash_deterministic(self, tracker: ProvenanceTracker):
        h1 = tracker.build_hash({"a": 1, "b": 2})
        h2 = tracker.build_hash({"b": 2, "a": 1})
        # Sort keys ensures same hash regardless of dict order
        assert h1 == h2

    def test_build_hash_none(self, tracker: ProvenanceTracker):
        h = tracker.build_hash(None)
        expected = hashlib.sha256("null".encode("utf-8")).hexdigest()
        assert h == expected

    def test_build_hash_different_data(self, tracker: ProvenanceTracker):
        h1 = tracker.build_hash({"x": 1})
        h2 = tracker.build_hash({"x": 2})
        assert h1 != h2


# ===================================================================
# Large chain tests
# ===================================================================


class TestLargeChain:
    """Test performance with a large chain."""

    def test_large_chain_100_entries(self, tracker: ProvenanceTracker):
        for i in range(100):
            tracker.record("calculation", "calculate", f"calc_{i:04d}")
        assert tracker.entry_count == 100
        assert tracker.verify_chain() is True

    def test_large_chain_entry_uniqueness(self, tracker: ProvenanceTracker):
        hashes = set()
        for i in range(150):
            entry = tracker.record("refrigerant", "register", f"R_{i}")
            hashes.add(entry.hash_value)
        assert len(hashes) == 150


# ===================================================================
# Thread safety tests
# ===================================================================


class TestThreadSafety:
    """Test thread-safe access to ProvenanceTracker."""

    def test_concurrent_records(self, tracker: ProvenanceTracker):
        """Concurrent record calls should not corrupt the chain."""
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(10):
                    tracker.record(
                        "calculation",
                        "calculate",
                        f"t{thread_id}_calc_{i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert tracker.entry_count == 80
        assert tracker.verify_chain() is True

    def test_concurrent_record_and_verify(self, tracker: ProvenanceTracker):
        """Recording and verifying concurrently should not crash."""
        errors = []

        def recorder():
            try:
                for i in range(20):
                    tracker.record("refrigerant", "register", f"R_concurrent_{i}")
            except Exception as e:
                errors.append(e)

        def verifier():
            try:
                for _ in range(20):
                    tracker.verify_chain()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=recorder)
        t2 = threading.Thread(target=verifier)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert len(errors) == 0


# ===================================================================
# Singleton management tests
# ===================================================================


class TestSingletonManagement:
    """Test module-level singleton accessor functions."""

    def test_get_provenance_tracker_returns_instance(self):
        tracker = get_provenance_tracker()
        assert isinstance(tracker, ProvenanceTracker)

    def test_get_provenance_tracker_returns_same_instance(self):
        t1 = get_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is t2

    def test_set_provenance_tracker_replaces(self):
        custom = ProvenanceTracker(genesis_hash="singleton-test")
        set_provenance_tracker(custom)
        retrieved = get_provenance_tracker()
        assert retrieved is custom

    def test_set_provenance_tracker_invalid_type_raises(self):
        with pytest.raises(TypeError):
            set_provenance_tracker("not a tracker")  # type: ignore[arg-type]

    def test_set_provenance_tracker_none_raises(self):
        with pytest.raises(TypeError):
            set_provenance_tracker(None)  # type: ignore[arg-type]

    def test_reset_provenance_tracker(self):
        t1 = get_provenance_tracker()
        reset_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is not t2

    def test_reset_then_get_creates_fresh(self):
        tracker = get_provenance_tracker()
        tracker.record("refrigerant", "register", "R_410A")
        assert tracker.entry_count >= 1

        reset_provenance_tracker()
        fresh = get_provenance_tracker()
        assert fresh.entry_count == 0

    def test_concurrent_get_singleton(self):
        """Multiple threads getting the singleton should all get the same object."""
        results = []

        def worker():
            return id(get_provenance_tracker())

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            for f in as_completed(futures):
                results.append(f.result())

        assert len(set(results)) == 1
