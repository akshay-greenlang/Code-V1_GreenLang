# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Waste Treatment Emissions Agent - ProvenanceTracker.

Tests ProvenanceEntry creation, record operations for all 12 entity types and
16 actions, SHA-256 hash chain verification, tampered chain detection,
filtering (by entity_type, action, time range), JSON/CSV export, thread safety,
and edge cases.

Target: 45+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import threading

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.waste_treatment_emissions.provenance import (
        ProvenanceEntry,
        ProvenanceTracker,
        VALID_ENTITY_TYPES,
        VALID_ACTIONS,
    )
    PROVENANCE_AVAILABLE = True
except ImportError:
    PROVENANCE_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not PROVENANCE_AVAILABLE,
    reason="ProvenanceTracker not available",
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def tracker():
    """Create a fresh ProvenanceTracker."""
    return ProvenanceTracker()


# ===========================================================================
# Test Class: ProvenanceEntry Dataclass
# ===========================================================================


@_SKIP
class TestProvenanceEntry:
    """Tests for the ProvenanceEntry dataclass."""

    def test_creation_with_required_fields(self):
        """ProvenanceEntry can be created with required fields."""
        entry = ProvenanceEntry(
            entity_type="FACILITY",
            entity_id="fac_001",
            action="CREATE",
            hash_value="a" * 64,
            parent_hash="0" * 64,
            timestamp="2025-01-01T00:00:00+00:00",
        )
        assert entry.entity_type == "FACILITY"
        assert entry.entity_id == "fac_001"
        assert entry.action == "CREATE"

    def test_default_actor_is_system(self):
        """Default actor is 'system'."""
        entry = ProvenanceEntry(
            entity_type="CALCULATION",
            entity_id="calc_001",
            action="CALCULATE",
            hash_value="b" * 64,
            parent_hash="a" * 64,
            timestamp="2025-06-15T12:00:00+00:00",
        )
        assert entry.actor == "system"

    def test_default_metadata_is_empty_dict(self):
        """Default metadata is an empty dictionary."""
        entry = ProvenanceEntry(
            entity_type="FACILITY",
            entity_id="fac_002",
            action="CREATE",
            hash_value="c" * 64,
            parent_hash="0" * 64,
            timestamp="2025-01-01T00:00:00+00:00",
        )
        assert entry.metadata == {}

    def test_metadata_with_values(self):
        """Metadata can store arbitrary key-value pairs."""
        entry = ProvenanceEntry(
            entity_type="TREATMENT_EVENT",
            entity_id="te_001",
            action="TREAT",
            hash_value="d" * 64,
            parent_hash="c" * 64,
            timestamp="2025-06-15T12:00:00+00:00",
            metadata={"data_hash": "e" * 64, "method": "incineration"},
        )
        assert entry.metadata["method"] == "incineration"

    def test_to_dict_returns_all_fields(self):
        """to_dict includes all ProvenanceEntry fields."""
        entry = ProvenanceEntry(
            entity_type="WASTE_STREAM",
            entity_id="ws_001",
            action="UPDATE",
            hash_value="f" * 64,
            parent_hash="e" * 64,
            timestamp="2025-06-15T12:00:00+00:00",
            actor="user_1",
            metadata={"category": "food_waste"},
        )
        d = entry.to_dict()
        assert d["entity_type"] == "WASTE_STREAM"
        assert d["entity_id"] == "ws_001"
        assert d["action"] == "UPDATE"
        assert d["hash_value"] == "f" * 64
        assert d["parent_hash"] == "e" * 64
        assert d["actor"] == "user_1"
        assert d["metadata"]["category"] == "food_waste"

    def test_to_dict_is_json_serializable(self):
        """to_dict output is JSON-serializable."""
        entry = ProvenanceEntry(
            entity_type="CALCULATION",
            entity_id="calc_002",
            action="CALCULATE",
            hash_value="a" * 64,
            parent_hash="0" * 64,
            timestamp="2025-01-01T00:00:00+00:00",
        )
        serialized = json.dumps(entry.to_dict())
        assert isinstance(serialized, str)

    def test_to_flat_dict_metadata_as_string(self):
        """to_flat_dict serializes metadata as a JSON string."""
        entry = ProvenanceEntry(
            entity_type="FACILITY",
            entity_id="fac_003",
            action="CREATE",
            hash_value="a" * 64,
            parent_hash="0" * 64,
            timestamp="2025-01-01T00:00:00+00:00",
            metadata={"key": "value"},
        )
        flat = entry.to_flat_dict()
        assert isinstance(flat["metadata"], str)
        parsed = json.loads(flat["metadata"])
        assert parsed["key"] == "value"


# ===========================================================================
# Test Class: Entity Types and Actions Constants
# ===========================================================================


@_SKIP
class TestEntityTypesAndActions:
    """Test valid entity types and action constants."""

    def test_twelve_entity_types(self):
        """There are 12 valid entity types plus SYSTEM = 13."""
        assert len(VALID_ENTITY_TYPES) >= 12

    @pytest.mark.parametrize("entity_type", [
        "FACILITY",
        "WASTE_STREAM",
        "TREATMENT_EVENT",
        "CALCULATION",
        "METHANE_RECOVERY",
        "ENERGY_RECOVERY",
        "EMISSION_FACTOR",
        "COMPLIANCE_CHECK",
        "UNCERTAINTY_RUN",
        "AGGREGATION",
        "BATCH",
        "CONFIG",
    ])
    def test_all_entity_types_present(self, entity_type):
        """Each expected entity type is in the valid set."""
        assert entity_type in VALID_ENTITY_TYPES

    def test_sixteen_actions(self):
        """There are 16 valid actions."""
        assert len(VALID_ACTIONS) >= 16

    @pytest.mark.parametrize("action", [
        "CREATE", "UPDATE", "DELETE",
        "CALCULATE", "ASSESS", "CHECK", "VALIDATE", "AGGREGATE",
        "TREAT", "COMBUST", "COMPOST", "DIGEST", "GASIFY",
        "CAPTURE", "FLARE", "UTILIZE",
    ])
    def test_all_actions_present(self, action):
        """Each expected action is in the valid set."""
        assert action in VALID_ACTIONS


# ===========================================================================
# Test Class: ProvenanceTracker Record
# ===========================================================================


@_SKIP
class TestProvenanceRecord:
    """Test ProvenanceTracker.record() method."""

    def test_record_returns_entry(self, tracker):
        """record() returns a ProvenanceEntry."""
        entry = tracker.record("FACILITY", "fac_001", "CREATE")
        assert isinstance(entry, ProvenanceEntry)

    def test_record_hash_is_64_hex(self, tracker):
        """Recorded entry has a 64-char hex hash."""
        entry = tracker.record("FACILITY", "fac_001", "CREATE")
        assert len(entry.hash_value) == 64
        assert all(c in "0123456789abcdef" for c in entry.hash_value)

    def test_first_entry_parent_is_genesis(self, tracker):
        """First entry's parent_hash is the genesis hash (64 zeros)."""
        entry = tracker.record("FACILITY", "fac_001", "CREATE")
        assert entry.parent_hash == "0" * 64

    def test_second_entry_chains_to_first(self, tracker):
        """Second entry's parent_hash equals first entry's hash."""
        e1 = tracker.record("FACILITY", "fac_001", "CREATE")
        e2 = tracker.record("WASTE_STREAM", "ws_001", "CREATE")
        assert e2.parent_hash == e1.hash_value

    def test_entry_count_increments(self, tracker):
        """Entry count increments with each record."""
        assert tracker.entry_count == 0
        tracker.record("FACILITY", "fac_001", "CREATE")
        assert tracker.entry_count == 1
        tracker.record("CALCULATION", "calc_001", "CALCULATE")
        assert tracker.entry_count == 2

    def test_record_with_data(self, tracker):
        """record() with data includes data_hash in metadata."""
        entry = tracker.record(
            "CALCULATION", "calc_001", "CALCULATE",
            data={"total_co2e": 100.0},
        )
        assert "data_hash" in entry.metadata
        assert len(entry.metadata["data_hash"]) == 64

    def test_record_with_actor(self, tracker):
        """record() with custom actor stores the actor."""
        entry = tracker.record(
            "FACILITY", "fac_002", "CREATE",
            actor="admin_user",
        )
        assert entry.actor == "admin_user"

    def test_record_with_metadata(self, tracker):
        """record() with additional metadata merges it in."""
        entry = tracker.record(
            "WASTE_STREAM", "ws_002", "UPDATE",
            metadata={"reason": "composition_change"},
        )
        assert entry.metadata["reason"] == "composition_change"

    @pytest.mark.parametrize("entity_type", list(VALID_ENTITY_TYPES)[:12])
    def test_record_all_entity_types(self, tracker, entity_type):
        """All entity types can be recorded."""
        entry = tracker.record(entity_type, f"id_{entity_type}", "CREATE")
        assert entry.entity_type == entity_type

    @pytest.mark.parametrize("action", list(VALID_ACTIONS)[:16])
    def test_record_all_actions(self, tracker, action):
        """All actions can be recorded."""
        entry = tracker.record("FACILITY", f"fac_{action}", action)
        assert entry.action == action

    def test_empty_entity_type_raises(self, tracker):
        """Empty entity_type raises ValueError."""
        with pytest.raises(ValueError, match="entity_type"):
            tracker.record("", "fac_001", "CREATE")

    def test_empty_entity_id_raises(self, tracker):
        """Empty entity_id raises ValueError."""
        with pytest.raises(ValueError, match="entity_id"):
            tracker.record("FACILITY", "", "CREATE")

    def test_empty_action_raises(self, tracker):
        """Empty action raises ValueError."""
        with pytest.raises(ValueError, match="action"):
            tracker.record("FACILITY", "fac_001", "")


# ===========================================================================
# Test Class: Chain Verification
# ===========================================================================


@_SKIP
class TestChainVerification:
    """Test provenance chain integrity verification."""

    def test_empty_chain_is_valid(self, tracker):
        """Empty chain is trivially valid."""
        is_valid, error = tracker.verify_chain()
        assert is_valid is True
        assert error is None

    def test_single_entry_chain_valid(self, tracker):
        """Single-entry chain is valid."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        is_valid, error = tracker.verify_chain()
        assert is_valid is True

    def test_multi_entry_chain_valid(self, tracker):
        """Multi-entry chain is valid."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        tracker.record("WASTE_STREAM", "ws_001", "CREATE")
        tracker.record("CALCULATION", "calc_001", "CALCULATE")
        is_valid, error = tracker.verify_chain()
        assert is_valid is True

    def test_tampered_chain_detected(self, tracker):
        """Tampering with a chain entry hash is detected."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        tracker.record("CALCULATION", "calc_001", "CALCULATE")

        # Tamper with the first entry's hash
        with tracker._lock:
            if tracker._global_chain:
                tracker._global_chain[0].hash_value = "x" * 64

        is_valid, error = tracker.verify_chain()
        assert is_valid is False
        assert error is not None

    def test_verify_entry(self, tracker):
        """verify_entry validates a single entry's hash."""
        entry = tracker.record("FACILITY", "fac_001", "CREATE")
        is_valid = tracker.verify_entry(entry)
        assert is_valid is True


# ===========================================================================
# Test Class: Filtering
# ===========================================================================


@_SKIP
class TestProvenanceFiltering:
    """Test provenance entry filtering."""

    def test_get_entries_by_entity_type(self, tracker):
        """Filter entries by entity_type."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        tracker.record("CALCULATION", "calc_001", "CALCULATE")
        tracker.record("FACILITY", "fac_002", "CREATE")

        entries = tracker.get_entries(entity_type="FACILITY")
        assert len(entries) == 2
        assert all(e.entity_type == "FACILITY" for e in entries)

    def test_get_entries_by_action(self, tracker):
        """Filter entries by action."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        tracker.record("FACILITY", "fac_001", "UPDATE")
        tracker.record("CALCULATION", "calc_001", "CALCULATE")

        entries = tracker.get_entries(action="CREATE")
        assert len(entries) == 1

    def test_get_entries_for_entity(self, tracker):
        """Get entries for a specific entity_type:entity_id pair."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        tracker.record("FACILITY", "fac_001", "UPDATE")
        tracker.record("FACILITY", "fac_002", "CREATE")

        entries = tracker.get_entries_for_entity("FACILITY", "fac_001")
        assert len(entries) == 2
        assert all(e.entity_id == "fac_001" for e in entries)

    def test_get_trail_with_limit(self, tracker):
        """get_trail respects limit parameter."""
        for i in range(20):
            tracker.record("CALCULATION", f"calc_{i:03d}", "CALCULATE")

        entries = tracker.get_trail(limit=5)
        assert len(entries) <= 5

    def test_get_trail_by_actor(self, tracker):
        """get_trail filters by actor."""
        tracker.record("FACILITY", "fac_001", "CREATE", actor="user_a")
        tracker.record("FACILITY", "fac_002", "CREATE", actor="user_b")

        entries = tracker.get_trail(actor="user_a")
        assert len(entries) == 1
        assert entries[0].actor == "user_a"

    def test_get_audit_trail(self, tracker):
        """get_audit_trail returns list of dicts."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        trail = tracker.get_audit_trail()
        assert len(trail) == 1
        assert isinstance(trail[0], dict)

    def test_get_audit_trail_by_entity(self, tracker):
        """get_audit_trail filters by entity_type and entity_id."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        tracker.record("WASTE_STREAM", "ws_001", "CREATE")
        trail = tracker.get_audit_trail(entity_type="FACILITY", entity_id="fac_001")
        assert len(trail) == 1

    def test_entity_summary(self, tracker):
        """get_entity_summary returns counts per entity type."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        tracker.record("FACILITY", "fac_002", "CREATE")
        tracker.record("CALCULATION", "calc_001", "CALCULATE")

        summary = tracker.get_entity_summary()
        assert summary["FACILITY"] == 2
        assert summary["CALCULATION"] == 1

    def test_action_summary(self, tracker):
        """get_action_summary returns counts per action."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        tracker.record("FACILITY", "fac_001", "UPDATE")
        tracker.record("FACILITY", "fac_002", "CREATE")

        summary = tracker.get_action_summary()
        assert summary["CREATE"] == 2
        assert summary["UPDATE"] == 1


# ===========================================================================
# Test Class: Export
# ===========================================================================


@_SKIP
class TestProvenanceExport:
    """Test JSON and CSV export functionality."""

    def test_export_json(self, tracker):
        """export_json returns valid JSON string."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        json_str = tracker.export_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_export_json_empty(self, tracker):
        """export_json on empty chain returns empty list."""
        json_str = tracker.export_json()
        parsed = json.loads(json_str)
        assert parsed == []

    def test_export_csv(self, tracker):
        """export_csv returns a CSV string with headers."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        csv_str = tracker.export_csv()
        assert "timestamp" in csv_str
        assert "entity_type" in csv_str
        assert "FACILITY" in csv_str

    def test_export_trail_json(self, tracker):
        """export_trail with format='json' works."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        output = tracker.export_trail(format="json")
        parsed = json.loads(output)
        assert len(parsed) == 1

    def test_export_trail_csv(self, tracker):
        """export_trail with format='csv' works."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        output = tracker.export_trail(format="csv")
        assert "FACILITY" in output

    def test_export_trail_invalid_format_raises(self, tracker):
        """export_trail with invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            tracker.export_trail(format="xml")


# ===========================================================================
# Test Class: Properties and Clear
# ===========================================================================


@_SKIP
class TestProvenanceProperties:
    """Test properties and clear functionality."""

    def test_genesis_hash_is_64_zeros(self, tracker):
        """Genesis hash is 64 zeros."""
        assert tracker.genesis_hash == "0" * 64

    def test_last_chain_hash_after_record(self, tracker):
        """last_chain_hash updates after each record."""
        initial = tracker.last_chain_hash
        tracker.record("FACILITY", "fac_001", "CREATE")
        assert tracker.last_chain_hash != initial

    def test_entity_count(self, tracker):
        """entity_count reflects unique entity_type:entity_id pairs."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        tracker.record("FACILITY", "fac_001", "UPDATE")
        tracker.record("FACILITY", "fac_002", "CREATE")
        assert tracker.entity_count == 2

    def test_max_entries_property(self, tracker):
        """max_entries property returns configured limit."""
        assert tracker.max_entries == 10000

    def test_custom_max_entries(self):
        """Custom max_entries is respected."""
        t = ProvenanceTracker(max_entries=500)
        assert t.max_entries == 500

    def test_clear_trail(self, tracker):
        """clear_trail resets the tracker to genesis state."""
        tracker.record("FACILITY", "fac_001", "CREATE")
        tracker.record("CALCULATION", "calc_001", "CALCULATE")
        assert tracker.entry_count == 2

        tracker.clear_trail()
        assert tracker.entry_count == 0
        assert tracker.last_chain_hash == "0" * 64

    def test_len_returns_entry_count(self, tracker):
        """len(tracker) returns entry_count."""
        assert len(tracker) == 0
        tracker.record("FACILITY", "fac_001", "CREATE")
        assert len(tracker) == 1

    def test_repr(self, tracker):
        """repr returns a meaningful string."""
        r = repr(tracker)
        assert "ProvenanceTracker" in r
        assert "entries=" in r


# ===========================================================================
# Test Class: Thread Safety
# ===========================================================================


@_SKIP
class TestProvenanceThreadSafety:
    """Test thread-safe operation of ProvenanceTracker."""

    def test_concurrent_records(self, tracker):
        """Concurrent record operations maintain chain integrity."""
        errors = []

        def worker(i):
            try:
                for j in range(10):
                    tracker.record(
                        "CALCULATION",
                        f"calc_{i}_{j}",
                        "CALCULATE",
                        data={"thread": i, "iteration": j},
                    )
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.entry_count == 50

    def test_chain_valid_after_concurrent(self, tracker):
        """Chain remains valid after concurrent operations."""
        def worker(i):
            for j in range(5):
                tracker.record("FACILITY", f"fac_{i}_{j}", "CREATE")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        is_valid, error = tracker.verify_chain()
        assert is_valid is True
