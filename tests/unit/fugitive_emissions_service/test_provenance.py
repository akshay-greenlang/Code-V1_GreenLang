# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-MRV-005 Fugitive Emissions Agent

Tests SHA-256 chain-hashed audit trail for tamper-evident provenance
across all fugitive emission operations. Validates entry creation,
chain linking, verification, trail filtering, export, eviction,
singleton management, and edge cases.

Test Classes:
    - TestProvenanceEntry            (10 tests)
    - TestProvenanceTracker          (12 tests)
    - TestChainHashing               (10 tests)
    - TestChainVerification          (10 tests)
    - TestTrailFiltering             (10 tests)
    - TestExport                      (6 tests)
    - TestEdgeCases                  (10 tests)

Total: 68 tests, ~570 lines.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
"""

from __future__ import annotations

import json
import threading
from typing import List

import pytest

from greenlang.fugitive_emissions.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
    get_provenance_tracker,
    set_provenance_tracker,
    reset_provenance_tracker,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GENESIS_HASH = "0" * 64


# ===========================================================================
# TestProvenanceEntry - 10 tests
# ===========================================================================


class TestProvenanceEntry:
    """Test ProvenanceEntry dataclass creation and serialization."""

    def test_create_entry_with_all_fields(self):
        entry = ProvenanceEntry(
            entity_type="CALCULATION", entity_id="calc_001",
            action="CREATE", hash_value="a" * 64,
            parent_hash=GENESIS_HASH,
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"data_hash": "b" * 64},
        )
        assert entry.entity_type == "CALCULATION"
        assert entry.entity_id == "calc_001"
        assert entry.action == "CREATE"
        assert len(entry.hash_value) == 64
        assert entry.parent_hash == GENESIS_HASH
        assert entry.metadata["data_hash"] == "b" * 64

    def test_entry_hash_value_length(self):
        entry = ProvenanceEntry(
            entity_type="SOURCE", entity_id="src_001", action="REGISTER",
            hash_value="b" * 64, parent_hash=GENESIS_HASH,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert len(entry.hash_value) == 64

    def test_entry_parent_hash_genesis(self):
        entry = ProvenanceEntry(
            entity_type="COMPONENT", entity_id="comp_001", action="CREATE",
            hash_value="c" * 64, parent_hash=GENESIS_HASH,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert entry.parent_hash == GENESIS_HASH

    def test_entry_default_metadata_is_empty_dict(self):
        entry = ProvenanceEntry(
            entity_type="FACTOR", entity_id="fac_001", action="SELECT",
            hash_value="d" * 64, parent_hash=GENESIS_HASH,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert entry.metadata == {}

    def test_entry_with_custom_metadata(self):
        entry = ProvenanceEntry(
            entity_type="SURVEY", entity_id="srv_001", action="CREATE",
            hash_value="e" * 64, parent_hash=GENESIS_HASH,
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"survey_type": "OGI", "facility": "FAC-001"},
        )
        assert entry.metadata["survey_type"] == "OGI"
        assert entry.metadata["facility"] == "FAC-001"

    def test_to_dict_returns_all_keys(self):
        entry = ProvenanceEntry(
            entity_type="REPAIR", entity_id="rep_001", action="UPDATE",
            hash_value="f" * 64, parent_hash=GENESIS_HASH,
            timestamp="2026-02-18T00:00:00+00:00",
        )
        d = entry.to_dict()
        expected_keys = {
            "entity_type", "entity_id", "action",
            "hash_value", "parent_hash", "timestamp", "metadata",
        }
        assert set(d.keys()) == expected_keys
        assert d["entity_type"] == "REPAIR"
        assert d["action"] == "UPDATE"

    def test_to_dict_includes_metadata(self):
        entry = ProvenanceEntry(
            entity_type="BATCH", entity_id="bat_001", action="CALCULATE",
            hash_value="1" * 64, parent_hash=GENESIS_HASH,
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"batch_size": 100},
        )
        d = entry.to_dict()
        assert d["metadata"]["batch_size"] == 100

    def test_to_dict_is_json_serializable(self):
        entry = ProvenanceEntry(
            entity_type="UNCERTAINTY", entity_id="unc_001", action="ANALYZE",
            hash_value="2" * 64, parent_hash=GENESIS_HASH,
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"method": "monte_carlo"},
        )
        serialized = json.dumps(entry.to_dict())
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed["entity_type"] == "UNCERTAINTY"

    def test_metadata_isolation_between_instances(self):
        """Metadata dict should not leak between default-constructed instances."""
        entry_a = ProvenanceEntry(
            entity_type="SOURCE", entity_id="a", action="CREATE",
            hash_value="a" * 64, parent_hash=GENESIS_HASH,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        entry_b = ProvenanceEntry(
            entity_type="SOURCE", entity_id="b", action="CREATE",
            hash_value="b" * 64, parent_hash=GENESIS_HASH,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        entry_a.metadata["only_a"] = True
        assert "only_a" not in entry_b.metadata

    def test_valid_constants(self):
        """Validate the published VALID_ENTITY_TYPES and VALID_ACTIONS sets."""
        assert len(VALID_ENTITY_TYPES) == 10
        for etype in ["CALCULATION", "SOURCE", "COMPONENT", "FACTOR",
                       "SURVEY", "REPAIR", "COMPLIANCE", "BATCH",
                       "UNCERTAINTY", "PIPELINE"]:
            assert etype in VALID_ENTITY_TYPES
        assert len(VALID_ACTIONS) == 15
        for action in ["CREATE", "UPDATE", "DELETE", "CALCULATE",
                        "VALIDATE", "LOOKUP", "SELECT", "REGISTER",
                        "AGGREGATE", "ANALYZE", "CHECK", "EXPORT",
                        "IMPORT", "MIGRATE", "AUDIT"]:
            assert action in VALID_ACTIONS


# ===========================================================================
# TestProvenanceTracker - 12 tests
# ===========================================================================


class TestProvenanceTracker:
    """Tests for ProvenanceTracker initialization and basic recording."""

    def test_init_defaults(self, tracker):
        assert tracker.genesis_hash == GENESIS_HASH
        assert tracker.entry_count == 0
        assert tracker.entity_count == 0
        assert len(tracker) == 0
        assert tracker.max_entries == 10000

    def test_init_custom_max_entries(self, tracker_small):
        assert tracker_small.max_entries == 5

    def test_repr_format(self, tracker):
        r = repr(tracker)
        assert "ProvenanceTracker" in r
        assert "entries=0" in r
        assert "entities=0" in r

    def test_record_returns_provenance_entry(self, tracker):
        entry = tracker.record("CALCULATION", "calc_001", "CREATE")
        assert isinstance(entry, ProvenanceEntry)
        assert entry.entity_type == "CALCULATION"
        assert entry.entity_id == "calc_001"
        assert entry.action == "CREATE"

    def test_first_record_parent_is_genesis(self, tracker):
        entry = tracker.record("COMPONENT", "comp_1", "REGISTER")
        assert entry.parent_hash == GENESIS_HASH

    def test_record_increments_entry_count(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        tracker.record("SOURCE", "s2", "CREATE")
        assert tracker.entry_count == 2
        assert len(tracker) == 2

    def test_record_increments_entity_count(self, tracker):
        tracker.record("CALCULATION", "c1", "CREATE")
        tracker.record("SOURCE", "s1", "REGISTER")
        assert tracker.entity_count == 2

    def test_record_same_entity_single_entity_count(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        tracker.record("SOURCE", "s1", "UPDATE")
        assert tracker.entity_count == 1
        assert tracker.entry_count == 2

    def test_record_with_data_payload(self, tracker):
        data = {"emissions_kg": 100.5, "method": "AVERAGE_EMISSION_FACTOR"}
        entry = tracker.record("CALCULATION", "c1", "CALCULATE", data=data)
        assert "data_hash" in entry.metadata
        assert len(entry.metadata["data_hash"]) == 64

    def test_record_with_extra_metadata(self, tracker):
        entry = tracker.record(
            "FACTOR", "f1", "LOOKUP",
            metadata={"unit": "kg/hr", "source": "EPA"},
        )
        assert entry.metadata["unit"] == "kg/hr"
        assert entry.metadata["source"] == "EPA"
        assert "data_hash" in entry.metadata  # always present

    def test_record_empty_entity_type_raises(self, tracker):
        with pytest.raises(ValueError, match="entity_type must not be empty"):
            tracker.record("", "id", "CREATE")

    def test_record_empty_entity_id_raises(self, tracker):
        with pytest.raises(ValueError, match="entity_id must not be empty"):
            tracker.record("SOURCE", "", "CREATE")

    def test_record_empty_action_raises(self, tracker):
        with pytest.raises(ValueError, match="action must not be empty"):
            tracker.record("SOURCE", "id", "")


# ===========================================================================
# TestChainHashing - 10 tests
# ===========================================================================


class TestChainHashing:
    """Tests for SHA-256 chain hash computation and linking."""

    def test_first_entry_hash_is_valid_hex(self, tracker):
        entry = tracker.record("SOURCE", "s1", "CREATE")
        assert len(entry.hash_value) == 64
        int(entry.hash_value, 16)  # raises if not valid hex

    def test_chain_hash_changes_with_each_record(self, tracker):
        initial = tracker.get_chain_hash()
        e1 = tracker.record("SOURCE", "s1", "CREATE")
        assert tracker.get_chain_hash() != initial
        e2 = tracker.record("SOURCE", "s2", "CREATE")
        assert e1.hash_value != e2.hash_value

    def test_second_entry_parent_links_to_first(self, tracker):
        e1 = tracker.record("SOURCE", "s1", "CREATE")
        e2 = tracker.record("SOURCE", "s2", "CREATE")
        assert e2.parent_hash == e1.hash_value

    def test_three_entry_chain_links(self, tracker):
        e1 = tracker.record("CALCULATION", "c1", "CREATE")
        e2 = tracker.record("SOURCE", "s1", "REGISTER")
        e3 = tracker.record("COMPONENT", "comp1", "CREATE")
        assert e2.parent_hash == e1.hash_value
        assert e3.parent_hash == e2.hash_value

    def test_chain_hash_property_tracks_latest(self, tracker):
        assert tracker.get_chain_hash() == GENESIS_HASH
        e1 = tracker.record("SOURCE", "s1", "CREATE")
        assert tracker.get_chain_hash() == e1.hash_value

    def test_last_chain_hash_property(self, tracker):
        assert tracker.last_chain_hash == GENESIS_HASH
        e = tracker.record("SOURCE", "s1", "CREATE")
        assert tracker.last_chain_hash == e.hash_value

    def test_build_hash_deterministic(self, tracker):
        data = {"fuel_type": "natural_gas", "amount": 100}
        h1 = tracker.build_hash(data)
        h2 = tracker.build_hash(data)
        assert h1 == h2

    def test_build_hash_different_data(self, tracker):
        h1 = tracker.build_hash({"a": 1})
        h2 = tracker.build_hash({"a": 2})
        assert h1 != h2

    def test_build_hash_none_returns_64_hex(self, tracker):
        h = tracker.build_hash(None)
        assert len(h) == 64
        int(h, 16)

    def test_build_hash_key_order_independent(self, tracker):
        h1 = tracker.build_hash({"z": 1, "a": 2})
        h2 = tracker.build_hash({"a": 2, "z": 1})
        assert h1 == h2

    def test_chain_integrity_across_many_entries(self, tracker):
        entries = []
        for i in range(20):
            e = tracker.record("SOURCE", f"s{i}", "CREATE", data={"idx": i})
            entries.append(e)
        for i in range(1, 20):
            assert entries[i].parent_hash == entries[i - 1].hash_value
        assert entries[0].parent_hash == GENESIS_HASH


# ===========================================================================
# TestChainVerification - 10 tests
# ===========================================================================


class TestChainVerification:
    """Tests for verify_chain() tamper detection."""

    def test_empty_chain_is_valid(self, tracker):
        valid, msg = tracker.verify_chain()
        assert valid is True
        assert msg is None

    def test_single_entry_chain_valid(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        valid, msg = tracker.verify_chain()
        assert valid is True
        assert msg is None

    def test_multi_entry_chain_valid(self, tracker):
        for i in range(20):
            tracker.record("CALCULATION", f"calc_{i:03d}", "CREATE")
        valid, msg = tracker.verify_chain()
        assert valid is True
        assert msg is None

    def test_mixed_entity_types_chain_valid(self, tracker):
        tracker.record("CALCULATION", "c1", "CREATE")
        tracker.record("SOURCE", "s1", "REGISTER")
        tracker.record("COMPONENT", "comp1", "CREATE")
        tracker.record("SURVEY", "sv1", "CREATE")
        tracker.record("COMPLIANCE", "chk1", "CHECK")
        valid, msg = tracker.verify_chain()
        assert valid is True

    def test_all_entity_types_in_chain(self, tracker):
        for et in VALID_ENTITY_TYPES:
            tracker.record(et, f"entity_{et}", "CREATE")
        valid, msg = tracker.verify_chain()
        assert valid is True

    def test_tampered_first_entry_parent_detected(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        # Tamper with internal state directly
        tracker._global_chain[0].parent_hash = "f" * 64
        valid, msg = tracker.verify_chain()
        assert valid is False
        assert "entry[0]" in msg

    def test_tampered_middle_entry_detected(self, tracker):
        for i in range(5):
            tracker.record("SOURCE", f"s{i}", "CREATE")
        # Break link between entry 2 and 3
        tracker._global_chain[3].parent_hash = "bad_hash_" + "0" * 55
        valid, msg = tracker.verify_chain()
        assert valid is False
        assert "entry[" in msg

    def test_empty_hash_value_detected(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        tracker._global_chain[0].hash_value = ""
        valid, msg = tracker.verify_chain()
        assert valid is False
        assert "hash_value" in msg

    def test_verify_after_clear_is_valid(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        tracker.clear_trail()
        valid, msg = tracker.verify_chain()
        assert valid is True
        assert msg is None

    def test_large_chain_verification(self, tracker):
        for i in range(100):
            tracker.record("CALCULATION", f"calc_{i:04d}", "CALCULATE")
        valid, msg = tracker.verify_chain()
        assert valid is True
        assert msg is None


# ===========================================================================
# TestTrailFiltering - 10 tests
# ===========================================================================


class TestTrailFiltering:
    """Tests for get_trail, get_entries, get_entries_for_entity, get_audit_trail."""

    def test_get_trail_all(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        tracker.record("COMPONENT", "c1", "REGISTER")
        trail = tracker.get_trail()
        assert len(trail) == 2

    def test_get_trail_by_entity_type(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        tracker.record("COMPONENT", "c1", "REGISTER")
        tracker.record("SOURCE", "s2", "CREATE")
        trail = tracker.get_trail(entity_type="SOURCE")
        assert len(trail) == 2
        assert all(e.entity_type == "SOURCE" for e in trail)

    def test_get_trail_by_entity_type_and_id(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        tracker.record("SOURCE", "s1", "UPDATE")
        tracker.record("SOURCE", "s2", "CREATE")
        trail = tracker.get_trail(entity_type="SOURCE", entity_id="s1")
        assert len(trail) == 2

    def test_get_trail_by_action(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        tracker.record("SOURCE", "s2", "UPDATE")
        tracker.record("SOURCE", "s3", "CREATE")
        trail = tracker.get_trail(action="CREATE")
        assert len(trail) == 2

    def test_get_trail_with_limit(self, tracker):
        for i in range(10):
            tracker.record("SOURCE", f"s{i}", "CREATE")
        trail = tracker.get_trail(limit=3)
        assert len(trail) == 3
        # Limit returns most recent
        assert trail[-1].entity_id == "s9"

    def test_get_entries_no_filter(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        tracker.record("COMPONENT", "c1", "REGISTER")
        entries = tracker.get_entries()
        assert len(entries) == 2

    def test_get_entries_with_entity_type_and_action(self, tracker):
        tracker.record("SURVEY", "sv1", "CREATE")
        tracker.record("SURVEY", "sv1", "UPDATE")
        tracker.record("REPAIR", "r1", "CREATE")
        entries = tracker.get_entries(entity_type="SURVEY", action="CREATE")
        assert len(entries) == 1

    def test_get_entries_for_entity_returns_ordered_list(self, tracker):
        tracker.record("COMPONENT", "comp_001", "CREATE")
        tracker.record("COMPONENT", "comp_001", "UPDATE")
        tracker.record("COMPONENT", "comp_001", "DELETE")
        entries = tracker.get_entries_for_entity("COMPONENT", "comp_001")
        assert len(entries) == 3
        assert entries[0].action == "CREATE"
        assert entries[2].action == "DELETE"

    def test_get_entries_for_entity_unknown_returns_empty(self, tracker):
        entries = tracker.get_entries_for_entity("COMPONENT", "nonexistent")
        assert entries == []

    def test_get_audit_trail_returns_dicts(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        trail = tracker.get_audit_trail()
        assert isinstance(trail, list)
        assert isinstance(trail[0], dict)
        assert "entity_type" in trail[0]
        assert trail[0]["entity_type"] == "SOURCE"


# ===========================================================================
# TestExport - 6 tests
# ===========================================================================


class TestExport:
    """Tests for export_trail, clear_trail, and get_chain."""

    def test_export_json_format(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        exported = tracker.export_trail(format="json")
        parsed = json.loads(exported)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["entity_type"] == "SOURCE"

    def test_export_json_with_indent(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        exported = tracker.export_trail(format="json", indent=4)
        assert "\n" in exported
        assert "    " in exported

    def test_export_unsupported_format_raises(self, tracker):
        with pytest.raises(ValueError, match="Unsupported export format"):
            tracker.export_trail(format="csv")

    def test_export_empty_chain(self, tracker):
        exported = tracker.export_trail()
        parsed = json.loads(exported)
        assert parsed == []

    def test_clear_trail_resets_state(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        tracker.record("SOURCE", "s2", "CREATE")
        assert tracker.entry_count == 2
        tracker.clear_trail()
        assert tracker.entry_count == 0
        assert tracker.entity_count == 0
        assert tracker.get_chain_hash() == GENESIS_HASH

    def test_clear_alias_works(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        tracker.clear()
        assert tracker.entry_count == 0


# ===========================================================================
# TestEdgeCases - 10 tests
# ===========================================================================


class TestEdgeCases:
    """Tests for eviction, thread safety, singleton helpers, and edge conditions."""

    def test_eviction_at_max_entries(self, tracker_small):
        for i in range(10):
            tracker_small.record("CALCULATION", f"calc_{i}", "CREATE")
        assert tracker_small.entry_count == 5
        chain = tracker_small.get_chain()
        assert chain[0].entity_id == "calc_5"

    def test_eviction_cleans_entity_store(self, tracker_small):
        tracker_small.record("SOURCE", "s_unique", "CREATE")
        for i in range(5):
            tracker_small.record("SOURCE", f"other_{i}", "CREATE")
        entries = tracker_small.get_entries_for_entity("SOURCE", "s_unique")
        assert len(entries) == 0

    def test_singleton_get_returns_same_instance(self):
        reset_provenance_tracker()
        try:
            t1 = get_provenance_tracker()
            t2 = get_provenance_tracker()
            assert t1 is t2
        finally:
            reset_provenance_tracker()

    def test_singleton_set_replaces(self):
        reset_provenance_tracker()
        try:
            custom = ProvenanceTracker(max_entries=42)
            set_provenance_tracker(custom)
            assert get_provenance_tracker() is custom
            assert get_provenance_tracker().max_entries == 42
        finally:
            reset_provenance_tracker()

    def test_singleton_set_invalid_type_raises(self):
        with pytest.raises(TypeError):
            set_provenance_tracker("not a tracker")  # type: ignore

    def test_singleton_reset_creates_new(self):
        reset_provenance_tracker()
        try:
            t1 = get_provenance_tracker()
            reset_provenance_tracker()
            t2 = get_provenance_tracker()
            assert t1 is not t2
        finally:
            reset_provenance_tracker()

    def test_thread_safe_recording(self, tracker):
        errors: List[Exception] = []

        def record_n(prefix: str, n: int):
            try:
                for i in range(n):
                    tracker.record("SOURCE", f"{prefix}_{i}", "CREATE")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=record_n, args=(f"t{t}", 50))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.entry_count == 200

    def test_record_with_none_data(self, tracker):
        entry = tracker.record("SOURCE", "s1", "CREATE", data=None)
        assert "data_hash" in entry.metadata
        assert len(entry.metadata["data_hash"]) == 64

    def test_get_chain_returns_independent_copy(self, tracker):
        tracker.record("CALCULATION", "c1", "CREATE")
        chain1 = tracker.get_chain()
        chain2 = tracker.get_chain()
        assert chain1 is not chain2
        assert len(chain1) == len(chain2)

    def test_get_audit_trail_with_entity_type_filter(self, tracker):
        tracker.record("SOURCE", "s1", "CREATE")
        tracker.record("COMPONENT", "c1", "REGISTER")
        tracker.record("SOURCE", "s2", "UPDATE")
        trail = tracker.get_audit_trail(entity_type="COMPONENT")
        assert len(trail) == 1
        assert trail[0]["entity_type"] == "COMPONENT"
