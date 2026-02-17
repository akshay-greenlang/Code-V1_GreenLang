# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-DATA-017

Tests SHA-256 chain hashing, deterministic hashing, chain verification,
entity-scoped and global chains, export, reset, clear, thread safety,
singleton helpers, edge cases, and determinism guarantees.
Target: 80+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
"""

from __future__ import annotations

import hashlib
import json
import threading
import time

import pytest

from greenlang.schema_migration.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    get_provenance_tracker,
    reset_provenance_tracker,
    set_provenance_tracker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the singleton ProvenanceTracker before and after each test."""
    reset_provenance_tracker()
    yield
    reset_provenance_tracker()


@pytest.fixture
def tracker():
    """Create a fresh ProvenanceTracker instance for testing."""
    return ProvenanceTracker()


@pytest.fixture
def tracker_custom_genesis():
    """Create a ProvenanceTracker with a custom genesis string."""
    return ProvenanceTracker(genesis_hash="custom-genesis-string-for-testing")


@pytest.fixture
def populated_tracker():
    """Create a tracker pre-populated with several entries across entity types."""
    t = ProvenanceTracker()
    t.record("schema", "schema_001", "registered", {"name": "emissions_v1"})
    t.record("version", "schema_001_v1", "version_created", {"bump": "minor"})
    t.record("change", "change_001", "change_detected", {"field": "co2e"})
    t.record("compatibility", "compat_001", "check_passed")
    t.record("plan", "plan_001", "plan_created", {"steps": 3})
    t.record("execution", "exec_001", "migration_executed", {"records": 1000})
    t.record("rollback", "rollback_001", "rollback_initiated")
    t.record("drift", "drift_001", "drift_detected", {"severity": "high"})
    return t


# =============================================================================
# TestProvenanceTrackerInit
# =============================================================================


class TestProvenanceTrackerInit:
    """Tests for ProvenanceTracker initialization."""

    def test_default_genesis_string(self, tracker):
        """Default genesis should hash 'greenlang-schema-migration-genesis'."""
        expected = hashlib.sha256(
            b"greenlang-schema-migration-genesis"
        ).hexdigest()
        assert tracker._genesis_hash == expected

    def test_default_genesis_hash_length(self, tracker):
        """Genesis hash must be a 64-char hex string (SHA-256)."""
        assert len(tracker._genesis_hash) == 64

    def test_default_genesis_hash_is_hex(self, tracker):
        """Genesis hash must be valid hexadecimal."""
        int(tracker._genesis_hash, 16)  # Should not raise

    def test_custom_genesis_string(self, tracker_custom_genesis):
        """Custom genesis string should produce a different genesis hash."""
        expected = hashlib.sha256(
            b"custom-genesis-string-for-testing"
        ).hexdigest()
        assert tracker_custom_genesis._genesis_hash == expected

    def test_custom_genesis_differs_from_default(self, tracker, tracker_custom_genesis):
        """Custom and default genesis hashes must differ."""
        assert tracker._genesis_hash != tracker_custom_genesis._genesis_hash

    def test_chain_starts_at_length_zero(self, tracker):
        """A new tracker should have entry_count == 0."""
        assert tracker.entry_count == 0

    def test_entity_count_starts_at_zero(self, tracker):
        """A new tracker should have entity_count == 0."""
        assert tracker.entity_count == 0

    def test_initial_global_chain_empty(self, tracker):
        """Internal global chain list should be empty at start."""
        assert tracker._global_chain == []

    def test_initial_chain_store_empty(self, tracker):
        """Internal chain store dict should be empty at start."""
        assert tracker._chain_store == {}

    def test_initial_last_chain_hash_equals_genesis(self, tracker):
        """_last_chain_hash should equal genesis_hash at initialization."""
        assert tracker._last_chain_hash == tracker._genesis_hash

    def test_lock_is_reentrant(self, tracker):
        """Tracker should use a reentrant lock for thread safety."""
        assert isinstance(tracker._lock, type(threading.RLock()))


# =============================================================================
# TestProvenanceTrackerRecord
# =============================================================================


class TestProvenanceTrackerRecord:
    """Tests for the record() method."""

    def test_record_returns_provenance_entry(self, tracker):
        """record() should return a ProvenanceEntry instance."""
        entry = tracker.record("schema", "s1", "registered")
        assert isinstance(entry, ProvenanceEntry)

    def test_record_hash_value_is_sha256(self, tracker):
        """Returned entry hash_value should be a 64-char hex string."""
        entry = tracker.record("schema", "s1", "registered")
        assert len(entry.hash_value) == 64
        int(entry.hash_value, 16)  # Must be valid hex

    def test_record_parent_hash_is_genesis_for_first(self, tracker):
        """First entry's parent_hash should be the genesis hash."""
        entry = tracker.record("schema", "s1", "registered")
        assert entry.parent_hash == tracker._genesis_hash

    def test_record_entity_type_stored(self, tracker):
        """Entity type should be stored in the entry."""
        entry = tracker.record("schema", "s1", "registered")
        assert entry.entity_type == "schema"

    def test_record_entity_id_stored(self, tracker):
        """Entity ID should be stored in the entry."""
        entry = tracker.record("schema", "s1", "registered")
        assert entry.entity_id == "s1"

    def test_record_action_stored(self, tracker):
        """Action should be stored in the entry."""
        entry = tracker.record("schema", "s1", "registered")
        assert entry.action == "registered"

    def test_record_timestamp_present(self, tracker):
        """Entry should have a non-empty timestamp."""
        entry = tracker.record("schema", "s1", "registered")
        assert entry.timestamp
        assert "T" in entry.timestamp  # ISO format

    def test_record_with_data(self, tracker):
        """record() should accept optional data payload."""
        entry = tracker.record("schema", "s1", "registered", {"name": "test"})
        assert entry.metadata.get("data_hash")
        assert len(entry.metadata["data_hash"]) == 64

    def test_record_with_none_data(self, tracker):
        """record() with data=None should hash the string 'null'."""
        entry = tracker.record("schema", "s1", "registered", None)
        expected_data_hash = hashlib.sha256(b"null").hexdigest()
        assert entry.metadata["data_hash"] == expected_data_hash

    def test_record_increments_entry_count(self, tracker):
        """Each call to record() should increment entry_count by 1."""
        assert tracker.entry_count == 0
        tracker.record("schema", "s1", "registered")
        assert tracker.entry_count == 1
        tracker.record("version", "v1", "created")
        assert tracker.entry_count == 2

    def test_record_increments_entity_count(self, tracker):
        """Unique entity_type:entity_id pairs should increment entity_count."""
        tracker.record("schema", "s1", "registered")
        assert tracker.entity_count == 1
        tracker.record("schema", "s2", "registered")
        assert tracker.entity_count == 2

    def test_record_same_entity_does_not_increment_entity_count(self, tracker):
        """Multiple records for the same entity should not increase entity_count."""
        tracker.record("schema", "s1", "registered")
        tracker.record("schema", "s1", "updated")
        assert tracker.entity_count == 1

    def test_multiple_records_form_chain(self, tracker):
        """Sequential records should chain parent_hash -> previous hash_value."""
        e1 = tracker.record("schema", "s1", "registered")
        e2 = tracker.record("version", "v1", "created")
        assert e2.parent_hash == e1.hash_value

    def test_three_records_chain_correctly(self, tracker):
        """Three sequential records should link in order."""
        e1 = tracker.record("schema", "s1", "registered")
        e2 = tracker.record("version", "v1", "created")
        e3 = tracker.record("change", "c1", "detected")
        assert e1.parent_hash == tracker._genesis_hash
        assert e2.parent_hash == e1.hash_value
        assert e3.parent_hash == e2.hash_value

    def test_hash_uniqueness_different_data(self, tracker):
        """Different data should produce different hash values."""
        e1 = tracker.record("schema", "s1", "registered", {"name": "a"})
        e2 = tracker.record("schema", "s2", "registered", {"name": "b"})
        assert e1.hash_value != e2.hash_value

    def test_record_all_entity_types(self, populated_tracker):
        """All 8 entity types (schema, version, change, compatibility, plan, execution, rollback, drift) should be recorded."""
        assert populated_tracker.entry_count == 8
        assert populated_tracker.entity_count == 8

    def test_record_empty_entity_type_raises(self, tracker):
        """record() with empty entity_type should raise ValueError."""
        with pytest.raises(ValueError, match="entity_type must not be empty"):
            tracker.record("", "s1", "registered")

    def test_record_empty_entity_id_raises(self, tracker):
        """record() with empty entity_id should raise ValueError."""
        with pytest.raises(ValueError, match="entity_id must not be empty"):
            tracker.record("schema", "", "registered")

    def test_record_empty_action_raises(self, tracker):
        """record() with empty action should raise ValueError."""
        with pytest.raises(ValueError, match="action must not be empty"):
            tracker.record("schema", "s1", "")

    def test_record_complex_data_payload(self, tracker):
        """record() should handle complex nested data payloads."""
        data = {
            "schema": {
                "fields": [
                    {"name": "co2e", "type": "number"},
                    {"name": "date", "type": "string"},
                ],
                "metadata": {"version": "2.0.0"},
            },
            "tags": ["emissions", "scope3"],
        }
        entry = tracker.record("schema", "s1", "registered", data)
        assert len(entry.hash_value) == 64
        assert len(entry.metadata["data_hash"]) == 64


# =============================================================================
# TestProvenanceTrackerVerify
# =============================================================================


class TestProvenanceTrackerVerify:
    """Tests for verify_chain() method."""

    def test_empty_chain_is_valid(self, tracker):
        """An empty chain should verify as True."""
        assert tracker.verify_chain() is True

    def test_single_record_chain_valid(self, tracker):
        """A chain with one entry should verify as True."""
        tracker.record("schema", "s1", "registered")
        assert tracker.verify_chain() is True

    def test_chain_with_many_entries_valid(self, populated_tracker):
        """A chain with 8 properly chained entries should verify as True."""
        assert populated_tracker.verify_chain() is True

    def test_tampered_hash_value_invalid(self, tracker):
        """Modifying an entry's hash_value should cause verification to fail."""
        tracker.record("schema", "s1", "registered")
        tracker.record("version", "v1", "created")
        # Tamper with the first entry's hash_value
        with tracker._lock:
            tracker._global_chain[0].hash_value = "0" * 64
        assert tracker.verify_chain() is False

    def test_tampered_parent_hash_invalid(self, tracker):
        """Modifying a later entry's parent_hash should cause verification to fail."""
        tracker.record("schema", "s1", "registered")
        tracker.record("version", "v1", "created")
        # Tamper with the second entry's parent_hash
        with tracker._lock:
            tracker._global_chain[1].parent_hash = "f" * 64
        assert tracker.verify_chain() is False

    def test_removed_required_field_invalid(self, tracker):
        """Clearing a required field should cause verification to fail."""
        tracker.record("schema", "s1", "registered")
        # Tamper: clear entity_type
        with tracker._lock:
            tracker._global_chain[0].entity_type = ""
        assert tracker.verify_chain() is False

    def test_first_entry_wrong_genesis_invalid(self, tracker):
        """First entry with wrong parent_hash (not genesis) should fail."""
        tracker.record("schema", "s1", "registered")
        with tracker._lock:
            tracker._global_chain[0].parent_hash = "a" * 64
        assert tracker.verify_chain() is False

    def test_verify_after_clear(self, tracker):
        """After reset(), verify_chain() should return True (empty chain)."""
        tracker.record("schema", "s1", "registered")
        tracker.reset()
        assert tracker.verify_chain() is True

    def test_verify_large_chain(self, tracker):
        """A chain with 100 entries should verify successfully."""
        for i in range(100):
            tracker.record("schema", f"s_{i}", f"action_{i}")
        assert tracker.verify_chain() is True

    def test_tampered_middle_entry_invalid(self, tracker):
        """Tampering with a middle entry in the chain should fail verification."""
        for i in range(10):
            tracker.record("schema", f"s_{i}", f"action_{i}")
        # Tamper with entry 5
        with tracker._lock:
            tracker._global_chain[5].hash_value = "b" * 64
        assert tracker.verify_chain() is False


# =============================================================================
# TestProvenanceTrackerGetChain
# =============================================================================


class TestProvenanceTrackerGetChain:
    """Tests for get_chain() method."""

    def test_get_chain_empty(self, tracker):
        """get_chain() with no matching entity should return empty list."""
        result = tracker.get_chain("nonexistent_id")
        assert result == []

    def test_get_chain_returns_matching_entries(self, tracker):
        """get_chain() should return entries matching entity_id."""
        tracker.record("schema", "s1", "registered")
        tracker.record("version", "s1", "version_created")
        tracker.record("schema", "s2", "registered")
        result = tracker.get_chain("s1")
        assert len(result) == 2

    def test_get_chain_preserves_order(self, tracker):
        """get_chain() should preserve insertion order."""
        tracker.record("schema", "s1", "registered")
        tracker.record("schema", "s1", "updated")
        tracker.record("schema", "s1", "deprecated")
        result = tracker.get_chain("s1")
        assert len(result) == 3
        assert result[0].action == "registered"
        assert result[1].action == "updated"
        assert result[2].action == "deprecated"

    def test_get_chain_returns_provenance_entries(self, tracker):
        """get_chain() should return ProvenanceEntry instances."""
        tracker.record("schema", "s1", "registered")
        result = tracker.get_chain("s1")
        assert all(isinstance(e, ProvenanceEntry) for e in result)

    def test_get_chain_isolation(self, tracker):
        """get_chain() for one entity should not include other entities."""
        tracker.record("schema", "s1", "registered")
        tracker.record("schema", "s2", "registered")
        result_s1 = tracker.get_chain("s1")
        result_s2 = tracker.get_chain("s2")
        assert len(result_s1) == 1
        assert len(result_s2) == 1
        assert result_s1[0].entity_id == "s1"
        assert result_s2[0].entity_id == "s2"


# =============================================================================
# TestProvenanceTrackerGetEntry
# =============================================================================


class TestProvenanceTrackerGetEntry:
    """Tests for get_entries() filtering by entity_type and entity_id."""

    def test_get_entries_all(self, populated_tracker):
        """get_entries() without filters returns entire global chain."""
        entries = populated_tracker.get_entries()
        assert len(entries) == 8

    def test_get_entries_by_entity_type(self, populated_tracker):
        """get_entries(entity_type=...) filters by type."""
        entries = populated_tracker.get_entries(entity_type="schema")
        assert len(entries) == 1
        assert all(e.entity_type == "schema" for e in entries)

    def test_get_entries_by_entity_type_and_id(self, populated_tracker):
        """get_entries(entity_type, entity_id) uses keyed store for O(1) lookup."""
        entries = populated_tracker.get_entries(
            entity_type="schema", entity_id="schema_001"
        )
        assert len(entries) == 1
        assert entries[0].entity_id == "schema_001"

    def test_get_entries_nonexistent_type(self, populated_tracker):
        """get_entries() for a nonexistent entity_type returns empty."""
        entries = populated_tracker.get_entries(entity_type="nonexistent")
        assert entries == []

    def test_get_entries_nonexistent_id(self, populated_tracker):
        """get_entries() for a nonexistent entity_id returns empty."""
        entries = populated_tracker.get_entries(
            entity_type="schema", entity_id="nonexistent"
        )
        assert entries == []


# =============================================================================
# TestProvenanceTrackerExport
# =============================================================================


class TestProvenanceTrackerExport:
    """Tests for export_chain() and export_json() methods."""

    def test_export_chain_empty(self, tracker):
        """export_chain() on empty tracker should return empty list."""
        result = tracker.export_chain()
        assert result == []

    def test_export_chain_returns_dicts(self, tracker):
        """export_chain() should return list of dictionaries."""
        tracker.record("schema", "s1", "registered")
        result = tracker.export_chain()
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_export_chain_dict_keys(self, tracker):
        """Exported dict should contain all ProvenanceEntry fields."""
        tracker.record("schema", "s1", "registered")
        result = tracker.export_chain()
        expected_keys = {
            "entity_type", "entity_id", "action",
            "hash_value", "parent_hash", "timestamp", "metadata",
        }
        assert set(result[0].keys()) == expected_keys

    def test_export_json_empty(self, tracker):
        """export_json() on empty tracker should return '[]'."""
        result = tracker.export_json()
        assert json.loads(result) == []

    def test_export_json_parseable(self, populated_tracker):
        """export_json() should return valid JSON."""
        result = populated_tracker.export_json()
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 8

    def test_export_json_contains_all_entries(self, populated_tracker):
        """export_json() should contain all recorded entries."""
        result = json.loads(populated_tracker.export_json())
        entity_types = {entry["entity_type"] for entry in result}
        assert "schema" in entity_types
        assert "version" in entity_types
        assert "change" in entity_types
        assert "rollback" in entity_types
        assert "drift" in entity_types

    def test_export_json_indented(self, tracker):
        """export_json() should produce indented JSON (2-space)."""
        tracker.record("schema", "s1", "registered")
        result = tracker.export_json()
        # Indented JSON has newlines
        assert "\n" in result


# =============================================================================
# TestProvenanceTrackerClear
# =============================================================================


class TestProvenanceTrackerClear:
    """Tests for the reset() method (clear)."""

    def test_reset_clears_entry_count(self, tracker):
        """reset() should set entry_count back to 0."""
        tracker.record("schema", "s1", "registered")
        tracker.record("version", "v1", "created")
        assert tracker.entry_count == 2
        tracker.reset()
        assert tracker.entry_count == 0

    def test_reset_clears_entity_count(self, tracker):
        """reset() should set entity_count back to 0."""
        tracker.record("schema", "s1", "registered")
        tracker.record("version", "v1", "created")
        assert tracker.entity_count == 2
        tracker.reset()
        assert tracker.entity_count == 0

    def test_reset_empties_global_chain(self, tracker):
        """reset() should empty the global chain list."""
        tracker.record("schema", "s1", "registered")
        tracker.reset()
        assert tracker._global_chain == []

    def test_reset_empties_chain_store(self, tracker):
        """reset() should empty the chain store dict."""
        tracker.record("schema", "s1", "registered")
        tracker.reset()
        assert tracker._chain_store == {}

    def test_reset_restores_genesis_hash(self, tracker):
        """reset() should restore _last_chain_hash to genesis."""
        tracker.record("schema", "s1", "registered")
        tracker.reset()
        assert tracker._last_chain_hash == tracker._genesis_hash

    def test_verify_after_reset(self, tracker):
        """verify_chain() after reset() should return True."""
        tracker.record("schema", "s1", "registered")
        tracker.reset()
        assert tracker.verify_chain() is True

    def test_reset_then_record_works(self, tracker):
        """After reset(), new records should work normally."""
        tracker.record("schema", "s1", "registered")
        tracker.reset()
        entry = tracker.record("schema", "s2", "registered")
        assert tracker.entry_count == 1
        assert entry.parent_hash == tracker._genesis_hash


# =============================================================================
# TestProvenanceTrackerSingleton
# =============================================================================


class TestProvenanceTrackerSingleton:
    """Tests for get_provenance_tracker(), set_provenance_tracker(), reset_provenance_tracker()."""

    def test_get_returns_provenance_tracker(self):
        """get_provenance_tracker() should return a ProvenanceTracker instance."""
        t = get_provenance_tracker()
        assert isinstance(t, ProvenanceTracker)

    def test_get_returns_same_instance(self):
        """Multiple calls to get_provenance_tracker() should return the same instance."""
        t1 = get_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is t2

    def test_set_replaces_singleton(self):
        """set_provenance_tracker() should replace the singleton."""
        custom = ProvenanceTracker(genesis_hash="custom-test")
        set_provenance_tracker(custom)
        result = get_provenance_tracker()
        assert result is custom

    def test_set_raises_on_wrong_type(self):
        """set_provenance_tracker() should raise TypeError for non-ProvenanceTracker."""
        with pytest.raises(TypeError, match="ProvenanceTracker instance"):
            set_provenance_tracker("not a tracker")  # type: ignore[arg-type]

    def test_set_raises_on_none(self):
        """set_provenance_tracker(None) should raise TypeError."""
        with pytest.raises(TypeError):
            set_provenance_tracker(None)  # type: ignore[arg-type]

    def test_reset_destroys_singleton(self):
        """reset_provenance_tracker() should set the singleton to None."""
        _ = get_provenance_tracker()
        reset_provenance_tracker()
        # Next call should create a new instance
        t_new = get_provenance_tracker()
        assert t_new is not None

    def test_reset_creates_fresh_instance(self):
        """After reset, get_provenance_tracker() should return a new object."""
        t_old = get_provenance_tracker()
        t_old.record("schema", "s1", "registered")
        reset_provenance_tracker()
        t_new = get_provenance_tracker()
        assert t_new is not t_old
        assert t_new.entry_count == 0

    def test_concurrent_get_returns_same_instance(self):
        """get_provenance_tracker() should be thread-safe and return the same instance."""
        results = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            results.append(get_provenance_tracker())

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be the same instance
        assert len(results) == 10
        assert all(r is results[0] for r in results)

    def test_set_then_get_preserves_state(self):
        """State recorded before set_provenance_tracker() should persist."""
        custom = ProvenanceTracker()
        custom.record("schema", "s1", "registered")
        set_provenance_tracker(custom)
        retrieved = get_provenance_tracker()
        assert retrieved.entry_count == 1

    def test_multiple_reset_cycles(self):
        """Multiple reset+get cycles should produce independent instances."""
        instances = []
        for _ in range(5):
            reset_provenance_tracker()
            t = get_provenance_tracker()
            t.record("schema", "s1", "registered")
            instances.append(t)

        # Each should be a different object
        for i in range(len(instances) - 1):
            assert instances[i] is not instances[i + 1]


# =============================================================================
# TestProvenanceTrackerEdgeCases
# =============================================================================


class TestProvenanceTrackerEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_very_long_details(self, tracker):
        """record() should handle very long data payloads."""
        long_data = {"description": "x" * 100_000}
        entry = tracker.record("schema", "s1", "registered", long_data)
        assert len(entry.hash_value) == 64

    def test_special_characters_in_entity_id(self, tracker):
        """record() should handle special characters in entity_id."""
        entry = tracker.record("schema", "s1/v2@latest#test", "registered")
        assert entry.entity_id == "s1/v2@latest#test"
        assert len(entry.hash_value) == 64

    def test_unicode_in_data(self, tracker):
        """record() should handle unicode strings in data payload."""
        data = {
            "name": "Schema de donnees",
            "description": "Schema japonais",
            "tags": ["measurement unit", "emissions factor"],
        }
        entry = tracker.record("schema", "s1", "registered", data)
        assert len(entry.hash_value) == 64

    def test_unicode_in_entity_type(self, tracker):
        """record() should handle unicode in entity_type."""
        entry = tracker.record("schema_type", "s1", "registered")
        assert entry.entity_type == "schema_type"

    def test_numeric_data_values(self, tracker):
        """record() should handle numeric data payloads."""
        data = {"emissions": 1234.5678, "count": 42, "ratio": 0.001}
        entry = tracker.record("schema", "s1", "registered", data)
        assert len(entry.hash_value) == 64

    def test_boolean_data_values(self, tracker):
        """record() should handle boolean data payloads."""
        data = {"active": True, "deprecated": False}
        entry = tracker.record("schema", "s1", "registered", data)
        assert len(entry.hash_value) == 64

    def test_nested_list_data(self, tracker):
        """record() should handle nested lists in data."""
        data = {"steps": [{"order": 1}, {"order": 2}]}
        entry = tracker.record("plan", "p1", "plan_created", data)
        assert len(entry.hash_value) == 64

    def test_empty_dict_data(self, tracker):
        """record() should handle empty dict data."""
        entry = tracker.record("schema", "s1", "registered", {})
        assert len(entry.hash_value) == 64

    def test_whitespace_entity_id(self, tracker):
        """record() should handle whitespace-only entity_id (non-empty string)."""
        entry = tracker.record("schema", "   ", "registered")
        assert entry.entity_id == "   "

    def test_to_dict_roundtrip(self, tracker):
        """ProvenanceEntry.to_dict() should produce JSON-serializable output."""
        entry = tracker.record("schema", "s1", "registered", {"key": "value"})
        d = entry.to_dict()
        # Should be serializable to JSON without error
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["entity_type"] == "schema"
        assert parsed["entity_id"] == "s1"
        assert parsed["action"] == "registered"


# =============================================================================
# TestProvenanceTrackerDeterminism
# =============================================================================


class TestProvenanceTrackerDeterminism:
    """Tests for deterministic hashing behavior."""

    def test_same_data_same_hash(self):
        """Same data payload should produce the same data_hash across instances."""
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        data = {"name": "emissions_v1", "version": "1.0.0"}
        h1 = t1._hash_data(data)
        h2 = t2._hash_data(data)
        assert h1 == h2

    def test_key_order_independent(self):
        """Dict key order should not affect the hash (sorted keys)."""
        t = ProvenanceTracker()
        h1 = t._hash_data({"a": 1, "b": 2, "c": 3})
        h2 = t._hash_data({"c": 3, "a": 1, "b": 2})
        assert h1 == h2

    def test_different_data_different_hash(self):
        """Different data should produce different hashes."""
        t = ProvenanceTracker()
        h1 = t._hash_data({"name": "schema_a"})
        h2 = t._hash_data({"name": "schema_b"})
        assert h1 != h2

    def test_build_hash_matches_internal_hash_data(self):
        """build_hash() should produce the same result as _hash_data()."""
        t = ProvenanceTracker()
        data = {"key": "value", "num": 42}
        assert t.build_hash(data) == t._hash_data(data)

    def test_build_hash_deterministic(self):
        """build_hash() should produce identical results on repeated calls."""
        t = ProvenanceTracker()
        data = [1, 2, 3, "test"]
        h1 = t.build_hash(data)
        h2 = t.build_hash(data)
        assert h1 == h2

    def test_hash_data_matches_manual_sha256(self):
        """_hash_data() should match a manually computed SHA-256."""
        t = ProvenanceTracker()
        data = {"key": "value"}
        expected = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        assert t._hash_data(data) == expected

    def test_null_data_hash_is_deterministic(self):
        """_hash_data(None) should always produce the same hash."""
        t = ProvenanceTracker()
        expected = hashlib.sha256(b"null").hexdigest()
        assert t._hash_data(None) == expected
        assert t._hash_data(None) == expected  # Second call

    def test_chain_hash_deterministic(self):
        """_compute_chain_hash() should be deterministic for the same inputs."""
        t = ProvenanceTracker()
        h1 = t._compute_chain_hash(
            parent_hash="a" * 64,
            data_hash="b" * 64,
            action="test_action",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        h2 = t._compute_chain_hash(
            parent_hash="a" * 64,
            data_hash="b" * 64,
            action="test_action",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert h1 == h2


# =============================================================================
# TestProvenanceTrackerThreadSafety
# =============================================================================


class TestProvenanceTrackerThreadSafety:
    """Verify thread safety of ProvenanceTracker operations."""

    def test_concurrent_record(self, tracker):
        """Concurrent record() calls should not lose entries."""
        errors = []
        barrier = threading.Barrier(10)

        def worker(idx):
            try:
                barrier.wait()
                for i in range(50):
                    tracker.record("schema", f"s_{idx}_{i}", f"action_{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.entry_count == 500  # 10 threads * 50 records

    def test_concurrent_read_and_write(self, tracker):
        """Concurrent reads and writes should not raise exceptions."""
        errors = []

        def writer():
            try:
                for i in range(50):
                    tracker.record("schema", f"s_{i}", f"action_{i}")
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(50):
                    tracker.get_entries()
                    tracker.verify_chain()
                    tracker.export_chain()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_record_and_reset(self, tracker):
        """Concurrent record() and reset() should not raise."""
        errors = []

        def writer():
            try:
                for i in range(30):
                    try:
                        tracker.record("schema", f"s_{i}", "action")
                    except Exception:
                        pass  # reset may clear mid-operation
            except Exception as exc:
                errors.append(exc)

        def resetter():
            try:
                for _ in range(5):
                    time.sleep(0.001)
                    tracker.reset()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=resetter),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# TestProvenanceEntry
# =============================================================================


class TestProvenanceEntry:
    """Tests for the ProvenanceEntry dataclass."""

    def test_to_dict(self):
        """to_dict() should return a dict with all fields."""
        entry = ProvenanceEntry(
            entity_type="schema",
            entity_id="s1",
            action="registered",
            hash_value="a" * 64,
            parent_hash="b" * 64,
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"data_hash": "c" * 64},
        )
        d = entry.to_dict()
        assert d["entity_type"] == "schema"
        assert d["entity_id"] == "s1"
        assert d["action"] == "registered"
        assert d["hash_value"] == "a" * 64
        assert d["parent_hash"] == "b" * 64
        assert d["timestamp"] == "2026-01-01T00:00:00+00:00"
        assert d["metadata"]["data_hash"] == "c" * 64

    def test_default_metadata(self):
        """ProvenanceEntry with no metadata arg should default to empty dict."""
        entry = ProvenanceEntry(
            entity_type="schema",
            entity_id="s1",
            action="registered",
            hash_value="a" * 64,
            parent_hash="b" * 64,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert entry.metadata == {}

    def test_metadata_isolation(self):
        """Default metadata dicts should not be shared between instances."""
        e1 = ProvenanceEntry(
            entity_type="schema", entity_id="s1", action="registered",
            hash_value="a" * 64, parent_hash="b" * 64,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        e2 = ProvenanceEntry(
            entity_type="schema", entity_id="s2", action="registered",
            hash_value="c" * 64, parent_hash="d" * 64,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        e1.metadata["key"] = "value"
        assert "key" not in e2.metadata
