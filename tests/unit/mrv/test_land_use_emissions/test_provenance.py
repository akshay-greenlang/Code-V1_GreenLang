# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-006 Land Use Emissions Agent Provenance Tracking.

Tests ProvenanceEntry creation, SHA-256 chain hashing, all 12 entity types,
all 16 actions, hash chain integrity, entry serialization, and edge cases.

Target: 45 tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import threading

import pytest

from greenlang.land_use_emissions.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
    get_provenance_tracker,
    set_provenance_tracker,
    reset_provenance_tracker,
)


# ===========================================================================
# ProvenanceEntry Tests
# ===========================================================================


class TestProvenanceEntry:
    """Tests for the ProvenanceEntry dataclass."""

    def test_creation(self):
        """ProvenanceEntry can be created with required fields."""
        entry = ProvenanceEntry(
            entity_type="PARCEL",
            entity_id="parcel_001",
            action="CREATE",
            hash_value="a" * 64,
            parent_hash="0" * 64,
            timestamp="2023-01-01T00:00:00+00:00",
        )
        assert entry.entity_type == "PARCEL"
        assert entry.entity_id == "parcel_001"
        assert entry.action == "CREATE"

    def test_default_metadata_is_empty_dict(self):
        """Default metadata is an empty dictionary."""
        entry = ProvenanceEntry(
            entity_type="PARCEL",
            entity_id="p1",
            action="CREATE",
            hash_value="a" * 64,
            parent_hash="0" * 64,
            timestamp="2023-01-01T00:00:00+00:00",
        )
        assert entry.metadata == {}

    def test_metadata_with_values(self):
        """Metadata can store arbitrary key-value pairs."""
        entry = ProvenanceEntry(
            entity_type="CALCULATION",
            entity_id="calc_001",
            action="CALCULATE",
            hash_value="b" * 64,
            parent_hash="a" * 64,
            timestamp="2023-06-15T12:00:00+00:00",
            metadata={"data_hash": "c" * 64, "tier": "tier_1"},
        )
        assert entry.metadata["tier"] == "tier_1"

    def test_to_dict_returns_all_fields(self):
        """to_dict includes all ProvenanceEntry fields."""
        entry = ProvenanceEntry(
            entity_type="TRANSITION",
            entity_id="trans_001",
            action="TRANSITION",
            hash_value="d" * 64,
            parent_hash="c" * 64,
            timestamp="2023-06-15T12:00:00+00:00",
            metadata={"area_ha": "50"},
        )
        d = entry.to_dict()
        assert d["entity_type"] == "TRANSITION"
        assert d["entity_id"] == "trans_001"
        assert d["action"] == "TRANSITION"
        assert d["hash_value"] == "d" * 64
        assert d["parent_hash"] == "c" * 64
        assert d["timestamp"] == "2023-06-15T12:00:00+00:00"
        assert d["metadata"]["area_ha"] == "50"

    def test_to_dict_is_json_serializable(self):
        """to_dict output is JSON-serializable."""
        entry = ProvenanceEntry(
            entity_type="PARCEL",
            entity_id="p1",
            action="CREATE",
            hash_value="a" * 64,
            parent_hash="0" * 64,
            timestamp="2023-01-01T00:00:00+00:00",
        )
        json_str = json.dumps(entry.to_dict())
        assert isinstance(json_str, str)


# ===========================================================================
# Valid Entity Types and Actions
# ===========================================================================


class TestValidConstants:
    """Tests for VALID_ENTITY_TYPES and VALID_ACTIONS constants."""

    def test_entity_types_has_12_entries(self):
        """VALID_ENTITY_TYPES has exactly 12 entries."""
        assert len(VALID_ENTITY_TYPES) == 12

    @pytest.mark.parametrize("entity_type", [
        "PARCEL", "CARBON_STOCK", "TRANSITION", "CALCULATION",
        "SOC_ASSESSMENT", "EMISSION_FACTOR", "COMPLIANCE_CHECK",
        "UNCERTAINTY_RUN", "AGGREGATION", "BATCH", "CONFIG", "SYSTEM",
    ])
    def test_entity_type_membership(self, entity_type):
        """Each expected entity type is in VALID_ENTITY_TYPES."""
        assert entity_type in VALID_ENTITY_TYPES

    def test_actions_has_16_entries(self):
        """VALID_ACTIONS has exactly 16 entries."""
        assert len(VALID_ACTIONS) == 16

    @pytest.mark.parametrize("action", [
        "CREATE", "UPDATE", "DELETE", "CALCULATE", "ASSESS", "CHECK",
        "VALIDATE", "AGGREGATE", "EXPORT", "IMPORT", "SNAPSHOT",
        "TRANSITION", "FIRE", "HARVEST", "REWET", "DRAIN",
    ])
    def test_action_membership(self, action):
        """Each expected action is in VALID_ACTIONS."""
        assert action in VALID_ACTIONS


# ===========================================================================
# ProvenanceTracker Tests
# ===========================================================================


class TestProvenanceTracker:
    """Tests for the ProvenanceTracker class."""

    def test_initialization(self, provenance_tracker):
        """Tracker initializes with genesis hash and empty chain."""
        assert provenance_tracker.entry_count == 0
        assert provenance_tracker.entity_count == 0
        assert provenance_tracker.genesis_hash == "0" * 64

    def test_record_returns_entry(self, provenance_tracker):
        """record() returns a ProvenanceEntry."""
        entry = provenance_tracker.record("PARCEL", "p1", "CREATE")
        assert isinstance(entry, ProvenanceEntry)

    def test_first_entry_parent_is_genesis(self, provenance_tracker):
        """First entry parent_hash equals genesis hash (64 zeros)."""
        entry = provenance_tracker.record("PARCEL", "p1", "CREATE")
        assert entry.parent_hash == "0" * 64

    def test_hash_is_64_hex_chars(self, provenance_tracker):
        """Hash value is a 64-character hexadecimal string."""
        entry = provenance_tracker.record("PARCEL", "p1", "CREATE")
        assert len(entry.hash_value) == 64
        int(entry.hash_value, 16)

    def test_chain_hashing_links_entries(self, provenance_tracker):
        """Second entry parent_hash equals first entry hash_value."""
        e1 = provenance_tracker.record("PARCEL", "p1", "CREATE")
        e2 = provenance_tracker.record("PARCEL", "p1", "UPDATE")
        assert e2.parent_hash == e1.hash_value

    def test_three_entry_chain(self, provenance_tracker):
        """Three entries form a proper chain."""
        e1 = provenance_tracker.record("PARCEL", "p1", "CREATE")
        e2 = provenance_tracker.record("CALCULATION", "c1", "CALCULATE")
        e3 = provenance_tracker.record("TRANSITION", "t1", "TRANSITION")
        assert e1.parent_hash == "0" * 64
        assert e2.parent_hash == e1.hash_value
        assert e3.parent_hash == e2.hash_value

    def test_verify_chain_empty(self, provenance_tracker):
        """Empty chain verifies as valid."""
        valid, msg = provenance_tracker.verify_chain()
        assert valid is True
        assert msg is None

    def test_verify_chain_valid(self, provenance_tracker):
        """Valid chain passes verification."""
        provenance_tracker.record("PARCEL", "p1", "CREATE")
        provenance_tracker.record("PARCEL", "p1", "UPDATE")
        provenance_tracker.record("CALCULATION", "c1", "CALCULATE")
        valid, msg = provenance_tracker.verify_chain()
        assert valid is True
        assert msg is None

    def test_entry_count_increments(self, provenance_tracker):
        """entry_count increments with each recorded entry."""
        provenance_tracker.record("PARCEL", "p1", "CREATE")
        assert provenance_tracker.entry_count == 1
        provenance_tracker.record("PARCEL", "p2", "CREATE")
        assert provenance_tracker.entry_count == 2

    def test_entity_count_tracks_unique_keys(self, provenance_tracker):
        """entity_count counts unique entity_type:entity_id pairs."""
        provenance_tracker.record("PARCEL", "p1", "CREATE")
        provenance_tracker.record("PARCEL", "p1", "UPDATE")
        provenance_tracker.record("PARCEL", "p2", "CREATE")
        assert provenance_tracker.entity_count == 2

    def test_get_entries_for_entity(self, provenance_tracker):
        """get_entries_for_entity returns entries for a specific entity."""
        provenance_tracker.record("PARCEL", "p1", "CREATE")
        provenance_tracker.record("CALCULATION", "c1", "CALCULATE")
        provenance_tracker.record("PARCEL", "p1", "UPDATE")
        entries = provenance_tracker.get_entries_for_entity("PARCEL", "p1")
        assert len(entries) == 2
        assert all(e.entity_id == "p1" for e in entries)

    def test_get_chain_returns_all_entries(self, provenance_tracker):
        """get_chain returns all entries in insertion order."""
        provenance_tracker.record("PARCEL", "p1", "CREATE")
        provenance_tracker.record("CALCULATION", "c1", "CALCULATE")
        chain = provenance_tracker.get_chain()
        assert len(chain) == 2
        assert chain[0].entity_type == "PARCEL"
        assert chain[1].entity_type == "CALCULATION"

    def test_get_chain_hash_updates(self, provenance_tracker):
        """get_chain_hash changes after each record."""
        h0 = provenance_tracker.get_chain_hash()
        provenance_tracker.record("PARCEL", "p1", "CREATE")
        h1 = provenance_tracker.get_chain_hash()
        assert h0 != h1
        provenance_tracker.record("PARCEL", "p1", "UPDATE")
        h2 = provenance_tracker.get_chain_hash()
        assert h1 != h2

    def test_clear_trail_resets_state(self, provenance_tracker):
        """clear_trail resets all state to genesis."""
        provenance_tracker.record("PARCEL", "p1", "CREATE")
        provenance_tracker.record("CALCULATION", "c1", "CALCULATE")
        provenance_tracker.clear_trail()
        assert provenance_tracker.entry_count == 0
        assert provenance_tracker.entity_count == 0
        assert provenance_tracker.get_chain_hash() == "0" * 64

    def test_len_returns_entry_count(self, provenance_tracker):
        """len(tracker) returns the entry count."""
        assert len(provenance_tracker) == 0
        provenance_tracker.record("PARCEL", "p1", "CREATE")
        assert len(provenance_tracker) == 1

    def test_repr_contains_entry_count(self, provenance_tracker):
        """repr includes entry and entity counts."""
        provenance_tracker.record("PARCEL", "p1", "CREATE")
        text = repr(provenance_tracker)
        assert "entries=1" in text
        assert "entities=1" in text

    def test_record_with_data(self, provenance_tracker):
        """record() accepts arbitrary data payload."""
        entry = provenance_tracker.record(
            "CALCULATION", "c1", "CALCULATE",
            data={"total_co2e": 150.5, "method": "stock_difference"},
        )
        assert "data_hash" in entry.metadata
        assert len(entry.metadata["data_hash"]) == 64

    def test_record_with_metadata(self, provenance_tracker):
        """record() accepts metadata that is stored in the entry."""
        entry = provenance_tracker.record(
            "PARCEL", "p1", "CREATE",
            metadata={"tenant_id": "t1", "region": "US"},
        )
        assert entry.metadata["tenant_id"] == "t1"
        assert entry.metadata["region"] == "US"

    def test_empty_entity_type_raises(self, provenance_tracker):
        """Empty entity_type raises ValueError."""
        with pytest.raises(ValueError, match="entity_type"):
            provenance_tracker.record("", "p1", "CREATE")

    def test_empty_entity_id_raises(self, provenance_tracker):
        """Empty entity_id raises ValueError."""
        with pytest.raises(ValueError, match="entity_id"):
            provenance_tracker.record("PARCEL", "", "CREATE")

    def test_empty_action_raises(self, provenance_tracker):
        """Empty action raises ValueError."""
        with pytest.raises(ValueError, match="action"):
            provenance_tracker.record("PARCEL", "p1", "")

    def test_export_trail_json(self, provenance_tracker):
        """export_trail returns valid JSON."""
        provenance_tracker.record("PARCEL", "p1", "CREATE")
        json_str = provenance_tracker.export_trail(format="json")
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_export_trail_invalid_format_raises(self, provenance_tracker):
        """export_trail raises ValueError for unsupported formats."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            provenance_tracker.export_trail(format="xml")

    def test_get_audit_trail_returns_dicts(self, provenance_tracker):
        """get_audit_trail returns a list of dictionaries."""
        provenance_tracker.record("PARCEL", "p1", "CREATE")
        trail = provenance_tracker.get_audit_trail()
        assert isinstance(trail, list)
        assert isinstance(trail[0], dict)
        assert trail[0]["entity_type"] == "PARCEL"

    def test_get_trail_with_action_filter(self, provenance_tracker):
        """get_trail filters by action."""
        provenance_tracker.record("PARCEL", "p1", "CREATE")
        provenance_tracker.record("PARCEL", "p1", "UPDATE")
        provenance_tracker.record("PARCEL", "p1", "DELETE")
        trail = provenance_tracker.get_trail(action="UPDATE")
        assert len(trail) == 1
        assert trail[0].action == "UPDATE"

    def test_get_entries_with_entity_type_filter(self, provenance_tracker):
        """get_entries filters by entity_type."""
        provenance_tracker.record("PARCEL", "p1", "CREATE")
        provenance_tracker.record("CALCULATION", "c1", "CALCULATE")
        provenance_tracker.record("PARCEL", "p2", "CREATE")
        entries = provenance_tracker.get_entries(entity_type="PARCEL")
        assert len(entries) == 2

    def test_eviction_when_max_exceeded(self):
        """Entries are evicted when max_entries is exceeded."""
        tracker = ProvenanceTracker(max_entries=5)
        for i in range(10):
            tracker.record("PARCEL", f"p{i}", "CREATE")
        assert tracker.entry_count == 5

    def test_deterministic_hashing(self, provenance_tracker):
        """Same data produces the same hash."""
        h1 = provenance_tracker.build_hash({"key": "value", "num": 42})
        h2 = provenance_tracker.build_hash({"num": 42, "key": "value"})
        assert h1 == h2

    def test_none_data_hashing(self, provenance_tracker):
        """None data produces a deterministic hash."""
        h1 = provenance_tracker.build_hash(None)
        h2 = provenance_tracker.build_hash(None)
        assert h1 == h2
        assert len(h1) == 64

    def test_long_data_hashing(self, provenance_tracker):
        """Very long data payloads produce valid hashes."""
        long_data = {"key": "x" * 100_000}
        entry = provenance_tracker.record(
            "SYSTEM", "sys1", "CREATE", data=long_data
        )
        assert len(entry.hash_value) == 64

    def test_thread_safety(self, provenance_tracker):
        """Concurrent record calls do not cause data corruption."""
        errors = []

        def worker(prefix):
            try:
                for i in range(50):
                    provenance_tracker.record("PARCEL", f"{prefix}_{i}", "CREATE")
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=worker, args=(f"thread_{t}",))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert provenance_tracker.entry_count == 200


# ===========================================================================
# Singleton Helper Tests
# ===========================================================================


class TestSingletonHelpers:
    """Tests for singleton provenance tracker helpers."""

    def test_get_provenance_tracker_returns_instance(self):
        """get_provenance_tracker returns a ProvenanceTracker."""
        reset_provenance_tracker()
        tracker = get_provenance_tracker()
        assert isinstance(tracker, ProvenanceTracker)

    def test_get_returns_same_instance(self):
        """get_provenance_tracker returns the same instance."""
        reset_provenance_tracker()
        t1 = get_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is t2

    def test_set_replaces_singleton(self):
        """set_provenance_tracker replaces the singleton."""
        reset_provenance_tracker()
        custom = ProvenanceTracker(max_entries=50)
        set_provenance_tracker(custom)
        assert get_provenance_tracker() is custom
        assert get_provenance_tracker().max_entries == 50
        reset_provenance_tracker()

    def test_reset_clears_singleton(self):
        """reset_provenance_tracker clears the singleton."""
        reset_provenance_tracker()
        t1 = get_provenance_tracker()
        reset_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is not t2

    def test_set_with_non_tracker_raises(self):
        """set_provenance_tracker raises TypeError for non-tracker."""
        with pytest.raises(TypeError, match="ProvenanceTracker"):
            set_provenance_tracker("not_a_tracker")
