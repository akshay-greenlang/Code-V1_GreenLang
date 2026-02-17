# -*- coding: utf-8 -*-
"""
Unit tests for Climate Hazard Connector provenance tracking module.

Tests ProvenanceTracker class including record(), verify_chain(),
get_chain(), get_entries(), get_entries_by_entity(), get_entry_by_hash(),
get_entity_chain(), export_chain(), export_json(), reset(), build_hash(),
properties, singleton accessors, chain integrity, and thread safety.

AGENT-DATA-020: Climate Hazard Connector
Target: 85%+ coverage of greenlang.climate_hazard.provenance
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from greenlang.climate_hazard.provenance import (
    VALID_ACTIONS,
    VALID_ENTITY_TYPES,
    ProvenanceEntry,
    ProvenanceTracker,
    get_provenance_tracker,
    reset_provenance_tracker,
    set_provenance_tracker,
)


# =============================================================================
# ProvenanceEntry
# =============================================================================


class TestProvenanceEntry:
    """Test ProvenanceEntry dataclass."""

    def test_entry_creation(self, sample_provenance_entry: ProvenanceEntry) -> None:
        assert sample_provenance_entry.entity_type == "hazard_source"
        assert sample_provenance_entry.entity_id == "src_test_001"
        assert sample_provenance_entry.action == "register_source"
        assert len(sample_provenance_entry.hash_value) == 64

    def test_entry_to_dict(self, sample_provenance_entry: ProvenanceEntry) -> None:
        d = sample_provenance_entry.to_dict()
        assert isinstance(d, dict)
        assert d["entity_type"] == "hazard_source"
        assert d["entity_id"] == "src_test_001"
        assert d["action"] == "register_source"
        assert d["hash_value"] == "a" * 64
        assert d["parent_hash"] == "b" * 64
        assert d["timestamp"] == "2026-02-17T00:00:00+00:00"
        assert "data_hash" in d["metadata"]

    def test_entry_to_dict_keys(self, sample_provenance_entry: ProvenanceEntry) -> None:
        d = sample_provenance_entry.to_dict()
        expected_keys = {
            "entity_type", "entity_id", "action",
            "hash_value", "parent_hash", "timestamp", "metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_entry_default_metadata(self) -> None:
        entry = ProvenanceEntry(
            entity_type="asset",
            entity_id="a1",
            action="register_asset",
            hash_value="x" * 64,
            parent_hash="y" * 64,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert entry.metadata == {}

    def test_entry_with_custom_metadata(self) -> None:
        entry = ProvenanceEntry(
            entity_type="risk_index",
            entity_id="r1",
            action="calculate_risk",
            hash_value="x" * 64,
            parent_hash="y" * 64,
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"score": 75.0, "scenario": "SSP2-4.5"},
        )
        assert entry.metadata["score"] == 75.0
        assert entry.metadata["scenario"] == "SSP2-4.5"


# =============================================================================
# Constants
# =============================================================================


class TestProvenanceConstants:
    """Test VALID_ENTITY_TYPES and VALID_ACTIONS constants."""

    def test_valid_entity_types_is_frozenset(self) -> None:
        assert isinstance(VALID_ENTITY_TYPES, frozenset)

    def test_valid_entity_types_count(self) -> None:
        assert len(VALID_ENTITY_TYPES) == 8

    def test_valid_entity_types_contents(self) -> None:
        expected = {
            "hazard_source", "hazard_data", "risk_index",
            "scenario_projection", "asset", "exposure",
            "vulnerability", "compliance_report",
        }
        assert VALID_ENTITY_TYPES == expected

    def test_valid_actions_is_frozenset(self) -> None:
        assert isinstance(VALID_ACTIONS, frozenset)

    def test_valid_actions_count(self) -> None:
        assert len(VALID_ACTIONS) == 36

    def test_valid_actions_contains_register_source(self) -> None:
        assert "register_source" in VALID_ACTIONS

    def test_valid_actions_contains_calculate_risk(self) -> None:
        assert "calculate_risk" in VALID_ACTIONS

    def test_valid_actions_contains_run_pipeline(self) -> None:
        assert "run_pipeline" in VALID_ACTIONS

    def test_valid_actions_contains_generate_tcfd(self) -> None:
        assert "generate_tcfd" in VALID_ACTIONS

    def test_valid_actions_contains_all_source_actions(self) -> None:
        source_actions = {"register_source", "update_source", "delete_source"}
        assert source_actions.issubset(VALID_ACTIONS)

    def test_valid_actions_contains_all_data_actions(self) -> None:
        data_actions = {"ingest_data", "query_data", "aggregate_data"}
        assert data_actions.issubset(VALID_ACTIONS)

    def test_valid_actions_contains_all_risk_actions(self) -> None:
        risk_actions = {
            "calculate_risk", "calculate_multi_hazard",
            "calculate_compound", "rank_hazards",
        }
        assert risk_actions.issubset(VALID_ACTIONS)

    def test_valid_actions_contains_all_pipeline_actions(self) -> None:
        pipeline_actions = {"run_pipeline", "run_batch"}
        assert pipeline_actions.issubset(VALID_ACTIONS)


# =============================================================================
# ProvenanceTracker Initialization
# =============================================================================


class TestProvenanceTrackerInit:
    """Test ProvenanceTracker initialization."""

    def test_default_genesis_hash(self, tracker: ProvenanceTracker) -> None:
        expected = hashlib.sha256(
            "greenlang-climate-hazard-connector-genesis".encode("utf-8")
        ).hexdigest()
        assert tracker.genesis_hash == expected

    def test_custom_genesis_hash(self, custom_tracker: ProvenanceTracker) -> None:
        expected = hashlib.sha256(
            "test-custom-genesis".encode("utf-8")
        ).hexdigest()
        assert custom_tracker.genesis_hash == expected

    def test_empty_chain_on_init(self, tracker: ProvenanceTracker) -> None:
        assert tracker.entry_count == 0

    def test_empty_entity_count_on_init(self, tracker: ProvenanceTracker) -> None:
        assert tracker.entity_count == 0

    def test_last_chain_hash_equals_genesis(self, tracker: ProvenanceTracker) -> None:
        assert tracker.last_chain_hash == tracker.genesis_hash

    def test_len_zero_on_init(self, tracker: ProvenanceTracker) -> None:
        assert len(tracker) == 0

    def test_repr_on_init(self, tracker: ProvenanceTracker) -> None:
        r = repr(tracker)
        assert "ProvenanceTracker(" in r
        assert "entries=0" in r
        assert "entities=0" in r


# =============================================================================
# record()
# =============================================================================


class TestProvenanceTrackerRecord:
    """Test ProvenanceTracker.record() method."""

    def test_record_returns_entry(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("hazard_source", "register_source", "s1")
        assert isinstance(entry, ProvenanceEntry)

    def test_record_sets_entity_type(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("hazard_data", "ingest_data", "d1")
        assert entry.entity_type == "hazard_data"

    def test_record_sets_entity_id(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("asset", "register_asset", "asset_123")
        assert entry.entity_id == "asset_123"

    def test_record_sets_action(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("risk_index", "calculate_risk", "r1")
        assert entry.action == "calculate_risk"

    def test_record_hash_value_is_sha256(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("hazard_source", "register_source", "s1")
        assert len(entry.hash_value) == 64
        # Verify it is hex
        int(entry.hash_value, 16)

    def test_record_parent_hash_is_genesis_for_first(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("hazard_source", "register_source", "s1")
        assert entry.parent_hash == tracker.genesis_hash

    def test_record_parent_hash_chains(self, tracker: ProvenanceTracker) -> None:
        e1 = tracker.record("hazard_source", "register_source", "s1")
        e2 = tracker.record("hazard_data", "ingest_data", "d1")
        assert e2.parent_hash == e1.hash_value

    def test_record_timestamp_is_iso_format(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("asset", "register_asset", "a1")
        # Should be parseable as ISO format
        dt = datetime.fromisoformat(entry.timestamp)
        assert dt.tzinfo is not None

    def test_record_increments_entry_count(self, tracker: ProvenanceTracker) -> None:
        assert tracker.entry_count == 0
        tracker.record("hazard_source", "register_source", "s1")
        assert tracker.entry_count == 1
        tracker.record("hazard_data", "ingest_data", "d1")
        assert tracker.entry_count == 2

    def test_record_increments_entity_count(self, tracker: ProvenanceTracker) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        assert tracker.entity_count == 1
        tracker.record("hazard_data", "ingest_data", "d1")
        assert tracker.entity_count == 2

    def test_record_same_entity_does_not_increase_entity_count(
        self, tracker: ProvenanceTracker
    ) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        tracker.record("hazard_source", "update_source", "s1")
        assert tracker.entity_count == 1
        assert tracker.entry_count == 2

    def test_record_with_data(self, tracker: ProvenanceTracker) -> None:
        data = {"name": "NOAA", "type": "global_database"}
        entry = tracker.record("hazard_source", "register_source", "s1", data=data)
        assert "data_hash" in entry.metadata

    def test_record_with_none_data(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("hazard_source", "register_source", "s1", data=None)
        assert "data_hash" in entry.metadata

    def test_record_with_metadata(self, tracker: ProvenanceTracker) -> None:
        meta = {"user": "admin", "reason": "initial setup"}
        entry = tracker.record(
            "hazard_source", "register_source", "s1", metadata=meta
        )
        assert entry.metadata["user"] == "admin"
        assert entry.metadata["reason"] == "initial setup"
        assert "data_hash" in entry.metadata

    def test_record_updates_last_chain_hash(self, tracker: ProvenanceTracker) -> None:
        genesis = tracker.last_chain_hash
        entry = tracker.record("hazard_source", "register_source", "s1")
        assert tracker.last_chain_hash != genesis
        assert tracker.last_chain_hash == entry.hash_value

    def test_record_empty_entity_type_raises(self, tracker: ProvenanceTracker) -> None:
        with pytest.raises(ValueError, match="entity_type must not be empty"):
            tracker.record("", "register_source", "s1")

    def test_record_empty_action_raises(self, tracker: ProvenanceTracker) -> None:
        with pytest.raises(ValueError, match="action must not be empty"):
            tracker.record("hazard_source", "", "s1")

    def test_record_empty_entity_id_raises(self, tracker: ProvenanceTracker) -> None:
        with pytest.raises(ValueError, match="entity_id must not be empty"):
            tracker.record("hazard_source", "register_source", "")

    def test_record_data_hash_deterministic(self, tracker: ProvenanceTracker) -> None:
        """Same data produces same data_hash in metadata."""
        data = {"value": 42}
        e1 = tracker.record("hazard_source", "register_source", "s1", data=data)
        tracker2 = ProvenanceTracker()
        e2 = tracker2.record("hazard_source", "register_source", "s1", data=data)
        assert e1.metadata["data_hash"] == e2.metadata["data_hash"]


# =============================================================================
# verify_chain()
# =============================================================================


class TestProvenanceTrackerVerifyChain:
    """Test ProvenanceTracker.verify_chain() method."""

    def test_empty_chain_is_valid(self, tracker: ProvenanceTracker) -> None:
        assert tracker.verify_chain() is True

    def test_single_entry_chain_is_valid(self, tracker: ProvenanceTracker) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        assert tracker.verify_chain() is True

    def test_multi_entry_chain_is_valid(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        assert populated_tracker.verify_chain() is True

    def test_chain_detects_broken_parent_hash(
        self, tracker: ProvenanceTracker
    ) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        tracker.record("hazard_data", "ingest_data", "d1")
        # Tamper with second entry's parent hash
        tracker._global_chain[1].parent_hash = "tampered_hash"
        assert tracker.verify_chain() is False

    def test_chain_detects_missing_hash_value(
        self, tracker: ProvenanceTracker
    ) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        tracker._global_chain[0].hash_value = ""
        assert tracker.verify_chain() is False

    def test_chain_detects_missing_entity_type(
        self, tracker: ProvenanceTracker
    ) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        tracker._global_chain[0].entity_type = ""
        assert tracker.verify_chain() is False

    def test_chain_detects_missing_entity_id(
        self, tracker: ProvenanceTracker
    ) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        tracker._global_chain[0].entity_id = ""
        assert tracker.verify_chain() is False

    def test_chain_detects_missing_action(
        self, tracker: ProvenanceTracker
    ) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        tracker._global_chain[0].action = ""
        assert tracker.verify_chain() is False

    def test_chain_detects_missing_timestamp(
        self, tracker: ProvenanceTracker
    ) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        tracker._global_chain[0].timestamp = ""
        assert tracker.verify_chain() is False

    def test_chain_detects_wrong_genesis_link(
        self, tracker: ProvenanceTracker
    ) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        tracker._global_chain[0].parent_hash = "wrong_genesis"
        assert tracker.verify_chain() is False

    def test_chain_valid_after_many_entries(self, tracker: ProvenanceTracker) -> None:
        for i in range(50):
            tracker.record("hazard_data", "ingest_data", f"d_{i}")
        assert tracker.verify_chain() is True


# =============================================================================
# get_chain()
# =============================================================================


class TestProvenanceTrackerGetChain:
    """Test ProvenanceTracker.get_chain() method."""

    def test_get_chain_empty(self, tracker: ProvenanceTracker) -> None:
        assert tracker.get_chain() == []

    def test_get_chain_returns_list(self, tracker: ProvenanceTracker) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        chain = tracker.get_chain()
        assert isinstance(chain, list)

    def test_get_chain_returns_copy(self, tracker: ProvenanceTracker) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        chain1 = tracker.get_chain()
        chain2 = tracker.get_chain()
        assert chain1 is not chain2

    def test_get_chain_preserves_order(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        chain = populated_tracker.get_chain()
        assert chain[0].entity_id == "src_001"
        assert chain[1].entity_id == "data_001"
        assert chain[2].entity_id == "risk_001"

    def test_get_chain_length(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        assert len(populated_tracker.get_chain()) == 5


# =============================================================================
# get_entries()
# =============================================================================


class TestProvenanceTrackerGetEntries:
    """Test ProvenanceTracker.get_entries() method."""

    def test_get_entries_no_filter(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        entries = populated_tracker.get_entries()
        assert len(entries) == 5

    def test_get_entries_by_entity_type(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        entries = populated_tracker.get_entries(entity_type="hazard_source")
        assert len(entries) == 1
        assert entries[0].entity_type == "hazard_source"

    def test_get_entries_by_action(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        entries = populated_tracker.get_entries(action="register_source")
        assert len(entries) == 1
        assert entries[0].action == "register_source"

    def test_get_entries_by_entity_type_and_action(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        entries = populated_tracker.get_entries(
            entity_type="risk_index", action="calculate_risk"
        )
        assert len(entries) == 1

    def test_get_entries_with_limit(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        entries = populated_tracker.get_entries(limit=2)
        assert len(entries) == 2
        # Should be the most recent 2
        assert entries[0].entity_id == "asset_001"
        assert entries[1].entity_id == "exp_001"

    def test_get_entries_with_limit_larger_than_chain(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        entries = populated_tracker.get_entries(limit=100)
        assert len(entries) == 5

    def test_get_entries_with_limit_zero(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        entries = populated_tracker.get_entries(limit=0)
        assert len(entries) == 5  # limit=0 is not applied

    def test_get_entries_no_match(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        entries = populated_tracker.get_entries(entity_type="nonexistent")
        assert entries == []


# =============================================================================
# get_entries_by_entity()
# =============================================================================


class TestProvenanceTrackerGetEntriesByEntity:
    """Test ProvenanceTracker.get_entries_by_entity() method."""

    def test_get_entries_by_entity_found(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        entries = populated_tracker.get_entries_by_entity("hazard_source", "src_001")
        assert len(entries) == 1
        assert entries[0].entity_id == "src_001"

    def test_get_entries_by_entity_not_found(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        entries = populated_tracker.get_entries_by_entity("hazard_source", "nonexistent")
        assert entries == []

    def test_get_entries_by_entity_multiple(self, tracker: ProvenanceTracker) -> None:
        tracker.record("asset", "register_asset", "a1")
        tracker.record("asset", "update_asset", "a1")
        tracker.record("asset", "delete_asset", "a1")
        entries = tracker.get_entries_by_entity("asset", "a1")
        assert len(entries) == 3

    def test_get_entries_by_entity_returns_copy(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        e1 = populated_tracker.get_entries_by_entity("hazard_source", "src_001")
        e2 = populated_tracker.get_entries_by_entity("hazard_source", "src_001")
        assert e1 is not e2


# =============================================================================
# get_entry_by_hash()
# =============================================================================


class TestProvenanceTrackerGetEntryByHash:
    """Test ProvenanceTracker.get_entry_by_hash() method."""

    def test_get_entry_by_hash_found(self, tracker: ProvenanceTracker) -> None:
        entry = tracker.record("hazard_source", "register_source", "s1")
        found = tracker.get_entry_by_hash(entry.hash_value)
        assert found is not None
        assert found.entity_id == "s1"

    def test_get_entry_by_hash_not_found(self, tracker: ProvenanceTracker) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        found = tracker.get_entry_by_hash("nonexistent_hash")
        assert found is None

    def test_get_entry_by_hash_empty_string(self, tracker: ProvenanceTracker) -> None:
        found = tracker.get_entry_by_hash("")
        assert found is None

    def test_get_entry_by_hash_from_populated(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        chain = populated_tracker.get_chain()
        for entry in chain:
            found = populated_tracker.get_entry_by_hash(entry.hash_value)
            assert found is not None
            assert found.entity_id == entry.entity_id


# =============================================================================
# get_entity_chain()
# =============================================================================


class TestProvenanceTrackerGetEntityChain:
    """Test ProvenanceTracker.get_entity_chain() method."""

    def test_get_entity_chain_single(self, tracker: ProvenanceTracker) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        chain = tracker.get_entity_chain("s1")
        assert len(chain) == 1

    def test_get_entity_chain_multiple_types(self, tracker: ProvenanceTracker) -> None:
        tracker.record("hazard_source", "register_source", "shared_id")
        tracker.record("asset", "register_asset", "shared_id")
        chain = tracker.get_entity_chain("shared_id")
        assert len(chain) == 2

    def test_get_entity_chain_not_found(self, tracker: ProvenanceTracker) -> None:
        chain = tracker.get_entity_chain("nonexistent")
        assert chain == []

    def test_get_entity_chain_preserves_order(
        self, tracker: ProvenanceTracker
    ) -> None:
        tracker.record("hazard_source", "register_source", "x1")
        tracker.record("hazard_data", "ingest_data", "other")
        tracker.record("hazard_source", "update_source", "x1")
        chain = tracker.get_entity_chain("x1")
        assert len(chain) == 2
        assert chain[0].action == "register_source"
        assert chain[1].action == "update_source"


# =============================================================================
# export_chain() and export_json()
# =============================================================================


class TestProvenanceTrackerExport:
    """Test export_chain and export_json methods."""

    def test_export_chain_empty(self, tracker: ProvenanceTracker) -> None:
        assert tracker.export_chain() == []

    def test_export_chain_returns_dicts(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        exported = populated_tracker.export_chain()
        assert isinstance(exported, list)
        assert all(isinstance(e, dict) for e in exported)

    def test_export_chain_length(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        assert len(populated_tracker.export_chain()) == 5

    def test_export_chain_dict_keys(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        exported = populated_tracker.export_chain()
        expected_keys = {
            "entity_type", "entity_id", "action",
            "hash_value", "parent_hash", "timestamp", "metadata",
        }
        for d in exported:
            assert set(d.keys()) == expected_keys

    def test_export_json_returns_string(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        json_str = populated_tracker.export_json()
        assert isinstance(json_str, str)

    def test_export_json_is_valid_json(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        json_str = populated_tracker.export_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 5

    def test_export_json_is_indented(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        json_str = populated_tracker.export_json()
        assert "\n" in json_str
        assert "  " in json_str


# =============================================================================
# reset()
# =============================================================================


class TestProvenanceTrackerReset:
    """Test ProvenanceTracker.reset() method."""

    def test_reset_clears_entries(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        assert populated_tracker.entry_count == 5
        populated_tracker.reset()
        assert populated_tracker.entry_count == 0

    def test_reset_clears_entities(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        assert populated_tracker.entity_count > 0
        populated_tracker.reset()
        assert populated_tracker.entity_count == 0

    def test_reset_restores_genesis_hash(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        genesis = populated_tracker.genesis_hash
        populated_tracker.reset()
        assert populated_tracker.last_chain_hash == genesis

    def test_reset_chain_is_empty(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        populated_tracker.reset()
        assert populated_tracker.get_chain() == []

    def test_reset_verify_chain_valid(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        populated_tracker.reset()
        assert populated_tracker.verify_chain() is True

    def test_record_after_reset(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        populated_tracker.reset()
        entry = populated_tracker.record("hazard_source", "register_source", "new_s1")
        assert entry.parent_hash == populated_tracker.genesis_hash
        assert populated_tracker.entry_count == 1


# =============================================================================
# build_hash()
# =============================================================================


class TestProvenanceTrackerBuildHash:
    """Test ProvenanceTracker.build_hash() utility method."""

    def test_build_hash_returns_64_char_hex(self, tracker: ProvenanceTracker) -> None:
        h = tracker.build_hash({"key": "value"})
        assert len(h) == 64
        int(h, 16)  # Verify hex

    def test_build_hash_deterministic(self, tracker: ProvenanceTracker) -> None:
        data = {"a": 1, "b": "hello"}
        h1 = tracker.build_hash(data)
        h2 = tracker.build_hash(data)
        assert h1 == h2

    def test_build_hash_different_data_different_hash(
        self, tracker: ProvenanceTracker
    ) -> None:
        h1 = tracker.build_hash({"x": 1})
        h2 = tracker.build_hash({"x": 2})
        assert h1 != h2

    def test_build_hash_none(self, tracker: ProvenanceTracker) -> None:
        h = tracker.build_hash(None)
        expected = hashlib.sha256("null".encode("utf-8")).hexdigest()
        assert h == expected

    def test_build_hash_string(self, tracker: ProvenanceTracker) -> None:
        h = tracker.build_hash("hello")
        expected = hashlib.sha256(
            json.dumps("hello", sort_keys=True).encode("utf-8")
        ).hexdigest()
        assert h == expected

    def test_build_hash_list(self, tracker: ProvenanceTracker) -> None:
        h = tracker.build_hash([1, 2, 3])
        assert len(h) == 64

    def test_build_hash_sorted_keys(self, tracker: ProvenanceTracker) -> None:
        """Key order should not affect hash (sorted keys)."""
        h1 = tracker.build_hash({"b": 2, "a": 1})
        h2 = tracker.build_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_build_hash_nested_dict(self, tracker: ProvenanceTracker) -> None:
        data = {"outer": {"inner": [1, 2, 3]}}
        h = tracker.build_hash(data)
        assert len(h) == 64


# =============================================================================
# Properties
# =============================================================================


class TestProvenanceTrackerProperties:
    """Test ProvenanceTracker properties."""

    def test_entry_count_property(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        assert populated_tracker.entry_count == 5

    def test_entity_count_property(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        assert populated_tracker.entity_count == 5

    def test_genesis_hash_property_is_string(self, tracker: ProvenanceTracker) -> None:
        assert isinstance(tracker.genesis_hash, str)

    def test_genesis_hash_property_is_64_chars(self, tracker: ProvenanceTracker) -> None:
        assert len(tracker.genesis_hash) == 64

    def test_last_chain_hash_property(self, tracker: ProvenanceTracker) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        assert tracker.last_chain_hash != tracker.genesis_hash

    def test_len_dunder(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        assert len(populated_tracker) == 5

    def test_repr_populated(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        r = repr(populated_tracker)
        assert "entries=5" in r
        assert "entities=5" in r
        assert "genesis_prefix=" in r


# =============================================================================
# Chain integrity (determinism)
# =============================================================================


class TestProvenanceChainIntegrity:
    """Test that chain hashing is deterministic and tamper-evident."""

    def test_chain_hashes_differ_between_entries(
        self, tracker: ProvenanceTracker
    ) -> None:
        e1 = tracker.record("hazard_source", "register_source", "s1")
        e2 = tracker.record("hazard_data", "ingest_data", "d1")
        assert e1.hash_value != e2.hash_value

    def test_chain_links_are_sequential(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        chain = populated_tracker.get_chain()
        for i in range(1, len(chain)):
            assert chain[i].parent_hash == chain[i - 1].hash_value

    def test_first_entry_links_to_genesis(
        self, populated_tracker: ProvenanceTracker
    ) -> None:
        chain = populated_tracker.get_chain()
        assert chain[0].parent_hash == populated_tracker.genesis_hash

    def test_tampered_entry_detected(self, tracker: ProvenanceTracker) -> None:
        tracker.record("hazard_source", "register_source", "s1")
        tracker.record("hazard_data", "ingest_data", "d1")
        tracker.record("risk_index", "calculate_risk", "r1")
        # Tamper with middle entry
        tracker._global_chain[1].hash_value = "tampered"
        assert tracker.verify_chain() is False


# =============================================================================
# Singleton helpers
# =============================================================================


class TestProvenanceSingletonHelpers:
    """Test get_provenance_tracker, set_provenance_tracker, reset_provenance_tracker."""

    def test_get_returns_tracker(self) -> None:
        t = get_provenance_tracker()
        assert isinstance(t, ProvenanceTracker)

    def test_get_returns_same_instance(self) -> None:
        t1 = get_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is t2

    def test_set_replaces_singleton(self) -> None:
        custom = ProvenanceTracker(genesis_hash="set-test")
        set_provenance_tracker(custom)
        assert get_provenance_tracker() is custom

    def test_set_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="ProvenanceTracker instance"):
            set_provenance_tracker("not a tracker")  # type: ignore[arg-type]

    def test_reset_clears_singleton(self) -> None:
        t1 = get_provenance_tracker()
        reset_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is not t2

    def test_singleton_thread_safety(self) -> None:
        results = []

        def worker():
            results.append(id(get_provenance_tracker()))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(results)) == 1


# =============================================================================
# Thread safety
# =============================================================================


class TestProvenanceTrackerThreadSafety:
    """Test thread safety of record and verify operations."""

    def test_concurrent_records(self, tracker: ProvenanceTracker) -> None:
        """Multiple threads recording simultaneously should not corrupt chain."""
        errors: List[str] = []

        def worker(thread_id: int):
            try:
                for i in range(20):
                    tracker.record(
                        "hazard_data", "ingest_data", f"t{thread_id}_d{i}"
                    )
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert tracker.entry_count == 100  # 5 threads x 20 records
        assert tracker.verify_chain() is True
