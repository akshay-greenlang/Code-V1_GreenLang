# -*- coding: utf-8 -*-
"""
Unit Tests for ProvenanceTracker - AGENT-DATA-019

Tests the ProvenanceTracker class, ProvenanceEntry dataclass, chain
integrity verification, entity-scoped and global lookups, export helpers,
build_hash utility, singleton management, and thread safety.

Target: 40-50 tests, 85%+ coverage of greenlang.validation_rule_engine.provenance

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from greenlang.validation_rule_engine.provenance import (
    VALID_ACTIONS,
    VALID_ENTITY_TYPES,
    ProvenanceEntry,
    ProvenanceTracker,
    get_provenance_tracker,
    reset_provenance_tracker,
    set_provenance_tracker,
)


# ============================================================================
# TestProvenanceEntry - dataclass tests
# ============================================================================


class TestProvenanceEntry:
    """ProvenanceEntry dataclass construction and serialisation."""

    def test_create_entry(self):
        entry = ProvenanceEntry(
            entity_type="validation_rule",
            entity_id="rule_001",
            action="rule_registered",
            hash_value="abc123",
            parent_hash="genesis",
            timestamp="2026-02-17T00:00:00+00:00",
        )
        assert entry.entity_type == "validation_rule"
        assert entry.entity_id == "rule_001"
        assert entry.action == "rule_registered"
        assert entry.hash_value == "abc123"
        assert entry.parent_hash == "genesis"

    def test_entry_default_metadata(self):
        entry = ProvenanceEntry(
            entity_type="rule_set",
            entity_id="set_001",
            action="rule_set_created",
            hash_value="def456",
            parent_hash="abc123",
            timestamp="2026-02-17T00:00:00+00:00",
        )
        assert entry.metadata == {}

    def test_entry_custom_metadata(self):
        entry = ProvenanceEntry(
            entity_type="evaluation",
            entity_id="eval_001",
            action="evaluation_completed",
            hash_value="ghi789",
            parent_hash="def456",
            timestamp="2026-02-17T00:00:00+00:00",
            metadata={"pass_rate": 0.95, "total_rules": 10},
        )
        assert entry.metadata["pass_rate"] == 0.95
        assert entry.metadata["total_rules"] == 10

    def test_entry_to_dict(self):
        entry = ProvenanceEntry(
            entity_type="conflict",
            entity_id="conf_001",
            action="conflict_detected",
            hash_value="jkl012",
            parent_hash="ghi789",
            timestamp="2026-02-17T00:00:00+00:00",
            metadata={"conflict_type": "range_overlap"},
        )
        d = entry.to_dict()
        assert isinstance(d, dict)
        assert d["entity_type"] == "conflict"
        assert d["entity_id"] == "conf_001"
        assert d["action"] == "conflict_detected"
        assert d["hash_value"] == "jkl012"
        assert d["parent_hash"] == "ghi789"
        assert d["metadata"]["conflict_type"] == "range_overlap"

    def test_entry_to_dict_keys(self):
        entry = ProvenanceEntry(
            entity_type="report",
            entity_id="rpt_001",
            action="report_generated",
            hash_value="mno345",
            parent_hash="jkl012",
            timestamp="2026-02-17T00:00:00+00:00",
        )
        expected_keys = {
            "entity_type", "entity_id", "action",
            "hash_value", "parent_hash", "timestamp", "metadata",
        }
        assert set(entry.to_dict().keys()) == expected_keys


# ============================================================================
# TestConstants - entity types and actions
# ============================================================================


class TestConstants:
    """VALID_ENTITY_TYPES and VALID_ACTIONS must contain expected values."""

    def test_valid_entity_types_count(self):
        assert len(VALID_ENTITY_TYPES) == 8

    def test_valid_entity_types_contains_validation_rule(self):
        assert "validation_rule" in VALID_ENTITY_TYPES

    def test_valid_entity_types_contains_rule_set(self):
        assert "rule_set" in VALID_ENTITY_TYPES

    def test_valid_entity_types_contains_compound_rule(self):
        assert "compound_rule" in VALID_ENTITY_TYPES

    def test_valid_entity_types_contains_rule_pack(self):
        assert "rule_pack" in VALID_ENTITY_TYPES

    def test_valid_entity_types_contains_evaluation(self):
        assert "evaluation" in VALID_ENTITY_TYPES

    def test_valid_entity_types_contains_conflict(self):
        assert "conflict" in VALID_ENTITY_TYPES

    def test_valid_entity_types_contains_report(self):
        assert "report" in VALID_ENTITY_TYPES

    def test_valid_entity_types_contains_audit(self):
        assert "audit" in VALID_ENTITY_TYPES

    def test_valid_entity_types_is_frozenset(self):
        assert isinstance(VALID_ENTITY_TYPES, frozenset)

    def test_valid_actions_is_frozenset(self):
        assert isinstance(VALID_ACTIONS, frozenset)

    def test_valid_actions_count(self):
        # 6 + 6 + 3 + 4 + 5 + 4 + 4 + 4 = 36 actions
        assert len(VALID_ACTIONS) >= 30

    def test_valid_actions_contains_rule_registered(self):
        assert "rule_registered" in VALID_ACTIONS

    def test_valid_actions_contains_evaluation_completed(self):
        assert "evaluation_completed" in VALID_ACTIONS

    def test_valid_actions_contains_conflict_detected(self):
        assert "conflict_detected" in VALID_ACTIONS


# ============================================================================
# TestProvenanceTrackerRecord - recording entries
# ============================================================================


class TestProvenanceTrackerRecord:
    """ProvenanceTracker.record() creates chain-linked entries."""

    def test_record_returns_entry(self, provenance_tracker):
        entry = provenance_tracker.record(
            "validation_rule", "rule_001", "rule_registered"
        )
        assert isinstance(entry, ProvenanceEntry)

    def test_record_sets_hash_value(self, provenance_tracker):
        entry = provenance_tracker.record(
            "validation_rule", "rule_001", "rule_registered"
        )
        assert entry.hash_value
        assert len(entry.hash_value) == 64  # SHA-256 hex

    def test_record_sets_parent_hash(self, provenance_tracker):
        entry = provenance_tracker.record(
            "validation_rule", "rule_001", "rule_registered"
        )
        # Parent should be genesis hash
        assert entry.parent_hash
        assert len(entry.parent_hash) == 64

    def test_record_chain_links(self, provenance_tracker):
        e1 = provenance_tracker.record(
            "validation_rule", "rule_001", "rule_registered"
        )
        e2 = provenance_tracker.record(
            "rule_set", "set_001", "rule_set_created"
        )
        assert e2.parent_hash == e1.hash_value

    def test_record_three_entry_chain(self, provenance_tracker):
        e1 = provenance_tracker.record("validation_rule", "r1", "rule_registered")
        e2 = provenance_tracker.record("rule_set", "s1", "rule_set_created")
        e3 = provenance_tracker.record("evaluation", "ev1", "evaluation_completed")
        assert e2.parent_hash == e1.hash_value
        assert e3.parent_hash == e2.hash_value

    def test_record_with_metadata(self, provenance_tracker):
        entry = provenance_tracker.record(
            "evaluation",
            "eval_001",
            "evaluation_completed",
            metadata={"pass_rate": 0.95, "total_records": 1000},
        )
        assert entry.metadata.get("data_hash")

    def test_record_with_none_metadata(self, provenance_tracker):
        entry = provenance_tracker.record(
            "validation_rule", "rule_002", "rule_registered", metadata=None
        )
        assert entry.metadata.get("data_hash")

    def test_record_increments_count(self, provenance_tracker):
        assert provenance_tracker.entry_count == 0
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        assert provenance_tracker.entry_count == 1
        provenance_tracker.record("rule_set", "s1", "rule_set_created")
        assert provenance_tracker.entry_count == 2

    def test_record_all_eight_entity_types(self, provenance_tracker):
        for etype in [
            "validation_rule", "rule_set", "compound_rule", "rule_pack",
            "evaluation", "conflict", "report", "audit",
        ]:
            entry = provenance_tracker.record(etype, f"{etype}_001", "rule_registered")
            assert entry.entity_type == etype
        assert provenance_tracker.entry_count == 8

    def test_record_empty_entity_type_raises(self, provenance_tracker):
        with pytest.raises(ValueError, match="entity_type"):
            provenance_tracker.record("", "rule_001", "rule_registered")

    def test_record_empty_entity_id_raises(self, provenance_tracker):
        with pytest.raises(ValueError, match="entity_id"):
            provenance_tracker.record("validation_rule", "", "rule_registered")

    def test_record_empty_action_raises(self, provenance_tracker):
        with pytest.raises(ValueError, match="action"):
            provenance_tracker.record("validation_rule", "rule_001", "")

    def test_record_sets_timestamp(self, provenance_tracker):
        entry = provenance_tracker.record("validation_rule", "r1", "rule_registered")
        assert entry.timestamp
        assert "T" in entry.timestamp


# ============================================================================
# TestVerifyChain - chain integrity
# ============================================================================


class TestVerifyChain:
    """verify_chain() validates the entire chain integrity."""

    def test_empty_chain_is_valid(self, provenance_tracker):
        assert provenance_tracker.verify_chain() is True

    def test_single_entry_chain_valid(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        assert provenance_tracker.verify_chain() is True

    def test_multi_entry_chain_valid(self, provenance_tracker):
        for i in range(10):
            provenance_tracker.record(
                "validation_rule", f"rule_{i}", "rule_registered"
            )
        assert provenance_tracker.verify_chain() is True

    def test_tampered_chain_detectable(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        provenance_tracker.record("rule_set", "s1", "rule_set_created")
        # Tamper with first entry's hash
        provenance_tracker._global_chain[0].hash_value = "tampered"
        assert provenance_tracker.verify_chain() is False


# ============================================================================
# TestGetEntries - filtering
# ============================================================================


class TestGetEntries:
    """get_entries() filters by entity_type and/or entity_id."""

    def test_get_all_entries(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        provenance_tracker.record("rule_set", "s1", "rule_set_created")
        entries = provenance_tracker.get_entries()
        assert len(entries) == 2

    def test_get_by_entity_type(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        provenance_tracker.record("validation_rule", "r2", "rule_registered")
        provenance_tracker.record("rule_set", "s1", "rule_set_created")
        entries = provenance_tracker.get_entries(entity_type="validation_rule")
        assert len(entries) == 2

    def test_get_by_entity_type_and_id(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        provenance_tracker.record("validation_rule", "r1", "rule_updated")
        provenance_tracker.record("validation_rule", "r2", "rule_registered")
        entries = provenance_tracker.get_entries(
            entity_type="validation_rule", entity_id="r1"
        )
        assert len(entries) == 2

    def test_get_empty_when_no_match(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        entries = provenance_tracker.get_entries(entity_type="report")
        assert len(entries) == 0

    def test_get_chain_by_entity_id(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "shared_id", "rule_registered")
        provenance_tracker.record("rule_set", "shared_id", "rule_set_created")
        entries = provenance_tracker.get_chain("shared_id")
        assert len(entries) == 2


# ============================================================================
# TestGetEntryByHash - hash lookup
# ============================================================================


class TestGetEntryByHash:
    """get_entry_by_hash() retrieves a single entry by hash."""

    def test_find_existing_entry(self, provenance_tracker):
        e = provenance_tracker.record("validation_rule", "r1", "rule_registered")
        found = provenance_tracker.get_entry_by_hash(e.hash_value)
        assert found is not None
        assert found.entity_id == "r1"

    def test_not_found_returns_none(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        assert provenance_tracker.get_entry_by_hash("nonexistent") is None

    def test_empty_hash_returns_none(self, provenance_tracker):
        assert provenance_tracker.get_entry_by_hash("") is None


# ============================================================================
# TestExport - chain export
# ============================================================================


class TestExport:
    """export_chain() and export_json() serialise the global chain."""

    def test_export_chain_returns_list(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        chain = provenance_tracker.export_chain()
        assert isinstance(chain, list)
        assert len(chain) == 1
        assert isinstance(chain[0], dict)

    def test_export_chain_empty(self, provenance_tracker):
        chain = provenance_tracker.export_chain()
        assert chain == []

    def test_export_json_returns_string(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        json_str = provenance_tracker.export_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_export_json_empty(self, provenance_tracker):
        json_str = provenance_tracker.export_json()
        parsed = json.loads(json_str)
        assert parsed == []


# ============================================================================
# TestReset - state reset
# ============================================================================


class TestReset:
    """reset() clears all provenance state."""

    def test_reset_clears_entries(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        provenance_tracker.record("rule_set", "s1", "rule_set_created")
        assert provenance_tracker.entry_count == 2
        provenance_tracker.reset()
        assert provenance_tracker.entry_count == 0

    def test_reset_clears_entity_count(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        provenance_tracker.reset()
        assert provenance_tracker.entity_count == 0

    def test_chain_valid_after_reset(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        provenance_tracker.reset()
        assert provenance_tracker.verify_chain() is True

    def test_record_after_reset_works(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        provenance_tracker.reset()
        e = provenance_tracker.record("rule_set", "s1", "rule_set_created")
        assert provenance_tracker.entry_count == 1
        # Parent should be genesis again
        assert e.parent_hash == provenance_tracker._genesis_hash


# ============================================================================
# TestBuildHash - utility method
# ============================================================================


class TestBuildHash:
    """build_hash() computes deterministic SHA-256 hashes."""

    def test_build_hash_string(self, provenance_tracker):
        h = provenance_tracker.build_hash("test_data")
        assert len(h) == 64
        assert isinstance(h, str)

    def test_build_hash_dict(self, provenance_tracker):
        h = provenance_tracker.build_hash({"key": "value"})
        assert len(h) == 64

    def test_build_hash_none(self, provenance_tracker):
        h = provenance_tracker.build_hash(None)
        assert len(h) == 64

    def test_build_hash_deterministic(self, provenance_tracker):
        h1 = provenance_tracker.build_hash({"a": 1, "b": 2})
        h2 = provenance_tracker.build_hash({"b": 2, "a": 1})
        assert h1 == h2  # sorted keys

    def test_build_hash_different_data(self, provenance_tracker):
        h1 = provenance_tracker.build_hash("data_a")
        h2 = provenance_tracker.build_hash("data_b")
        assert h1 != h2


# ============================================================================
# TestProperties - entry_count, entity_count, __len__
# ============================================================================


class TestProperties:
    """Properties and dunder methods on ProvenanceTracker."""

    def test_entry_count_zero(self, provenance_tracker):
        assert provenance_tracker.entry_count == 0

    def test_entity_count_zero(self, provenance_tracker):
        assert provenance_tracker.entity_count == 0

    def test_len_equals_entry_count(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        provenance_tracker.record("rule_set", "s1", "rule_set_created")
        assert len(provenance_tracker) == provenance_tracker.entry_count
        assert len(provenance_tracker) == 2

    def test_entity_count_after_records(self, provenance_tracker):
        provenance_tracker.record("validation_rule", "r1", "rule_registered")
        provenance_tracker.record("validation_rule", "r1", "rule_updated")
        provenance_tracker.record("rule_set", "s1", "rule_set_created")
        # 2 unique keys: "validation_rule:r1" and "rule_set:s1"
        assert provenance_tracker.entity_count == 2


# ============================================================================
# TestThreadSafety - concurrent access
# ============================================================================


class TestThreadSafety:
    """ProvenanceTracker must handle concurrent access safely."""

    def test_concurrent_record(self, provenance_tracker):
        """Multiple threads recording simultaneously must not corrupt the chain."""
        def _record(idx):
            provenance_tracker.record(
                "validation_rule", f"rule_{idx}", "rule_registered"
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_record, i) for i in range(50)]
            for f in as_completed(futures):
                f.result()

        assert provenance_tracker.entry_count == 50
        assert provenance_tracker.verify_chain() is True


# ============================================================================
# TestSingleton - get/set/reset singleton
# ============================================================================


class TestSingleton:
    """Singleton accessors for ProvenanceTracker."""

    def test_get_provenance_tracker_returns_instance(self):
        reset_provenance_tracker()
        tracker = get_provenance_tracker()
        assert isinstance(tracker, ProvenanceTracker)

    def test_get_provenance_tracker_same_instance(self):
        reset_provenance_tracker()
        a = get_provenance_tracker()
        b = get_provenance_tracker()
        assert a is b

    def test_set_provenance_tracker(self):
        reset_provenance_tracker()
        custom = ProvenanceTracker(genesis_hash="custom")
        set_provenance_tracker(custom)
        assert get_provenance_tracker() is custom

    def test_set_provenance_tracker_type_error(self):
        with pytest.raises(TypeError):
            set_provenance_tracker("not_a_tracker")  # type: ignore[arg-type]

    def test_reset_provenance_tracker(self):
        reset_provenance_tracker()
        a = get_provenance_tracker()
        reset_provenance_tracker()
        b = get_provenance_tracker()
        assert a is not b
