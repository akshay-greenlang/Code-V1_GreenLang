# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker (AGENT-DATA-008)

Tests SHA-256 chain hashing, record/verify/get operations, tamper detection,
export, statistics, and determinism of the provenance tracker.

Target: 50+ tests.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json

import pytest

from greenlang.supplier_questionnaire.provenance import ProvenanceTracker


# ============================================================================
# Initialization tests
# ============================================================================


class TestProvenanceTrackerInit:
    def test_initial_entry_count_zero(self):
        tracker = ProvenanceTracker()
        assert tracker.entry_count == 0

    def test_initial_entity_count_zero(self):
        tracker = ProvenanceTracker()
        assert tracker.entity_count == 0

    def test_genesis_hash_is_deterministic(self):
        expected = hashlib.sha256(
            b"greenlang-supplier-questionnaire-genesis"
        ).hexdigest()
        assert ProvenanceTracker._GENESIS_HASH == expected

    def test_genesis_hash_is_64_hex_chars(self):
        assert len(ProvenanceTracker._GENESIS_HASH) == 64

    def test_last_chain_hash_starts_at_genesis(self):
        tracker = ProvenanceTracker()
        assert tracker._last_chain_hash == ProvenanceTracker._GENESIS_HASH


# ============================================================================
# record() tests
# ============================================================================


class TestProvenanceTrackerRecord:
    def test_record_returns_chain_hash(self):
        tracker = ProvenanceTracker()
        result = tracker.record("template", "tpl-001", "create", "abc123")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_record_increments_entry_count(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "abc123")
        assert tracker.entry_count == 1

    def test_record_increments_entity_count(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "abc123")
        assert tracker.entity_count == 1

    def test_record_same_entity_twice_no_new_entity(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "abc123")
        tracker.record("template", "tpl-001", "update", "def456")
        assert tracker.entity_count == 1
        assert tracker.entry_count == 2

    def test_record_different_entities_counted(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "abc")
        tracker.record("distribution", "dist-001", "send", "def")
        assert tracker.entity_count == 2

    def test_record_default_user_is_system(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "abc")
        entry = tracker.get_chain("tpl-001")[0]
        assert entry["user_id"] == "system"

    def test_record_custom_user_id(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "abc", user_id="user@co.com")
        entry = tracker.get_chain("tpl-001")[0]
        assert entry["user_id"] == "user@co.com"

    def test_record_stores_entity_type(self):
        tracker = ProvenanceTracker()
        tracker.record("score", "score-001", "calculate", "xyz")
        entry = tracker.get_chain("score-001")[0]
        assert entry["entity_type"] == "score"

    def test_record_stores_action(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "clone", "abc")
        entry = tracker.get_chain("tpl-001")[0]
        assert entry["action"] == "clone"

    def test_record_stores_data_hash(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "myhash")
        entry = tracker.get_chain("tpl-001")[0]
        assert entry["data_hash"] == "myhash"

    def test_record_stores_timestamp(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "abc")
        entry = tracker.get_chain("tpl-001")[0]
        assert "timestamp" in entry
        assert entry["timestamp"] is not None

    def test_record_stores_chain_hash(self):
        tracker = ProvenanceTracker()
        chain_hash = tracker.record("template", "tpl-001", "create", "abc")
        entry = tracker.get_chain("tpl-001")[0]
        assert entry["chain_hash"] == chain_hash

    def test_chain_hash_changes_with_each_record(self):
        tracker = ProvenanceTracker()
        h1 = tracker.record("template", "tpl-001", "create", "a")
        h2 = tracker.record("template", "tpl-001", "update", "b")
        assert h1 != h2

    def test_chain_hash_links_to_previous(self):
        """The second chain hash should depend on the first."""
        tracker = ProvenanceTracker()
        h1 = tracker.record("template", "tpl-001", "create", "a")
        # The tracker._last_chain_hash should now be h1
        assert tracker._last_chain_hash == tracker.get_chain("tpl-001")[-1]["chain_hash"]

    def test_record_updates_last_chain_hash(self):
        tracker = ProvenanceTracker()
        h1 = tracker.record("template", "tpl-001", "create", "a")
        assert tracker._last_chain_hash == h1
        h2 = tracker.record("template", "tpl-001", "update", "b")
        assert tracker._last_chain_hash == h2


# ============================================================================
# verify_chain() tests
# ============================================================================


class TestProvenanceTrackerVerifyChain:
    def test_verify_empty_chain_returns_valid(self):
        tracker = ProvenanceTracker()
        valid, chain = tracker.verify_chain("nonexistent")
        assert valid is True
        assert chain == []

    def test_verify_single_entry_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "abc")
        valid, chain = tracker.verify_chain("tpl-001")
        assert valid is True
        assert len(chain) == 1

    def test_verify_multiple_entries_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "abc")
        tracker.record("template", "tpl-001", "update", "def")
        tracker.record("template", "tpl-001", "publish", "ghi")
        valid, chain = tracker.verify_chain("tpl-001")
        assert valid is True
        assert len(chain) == 3

    def test_verify_detects_missing_chain_hash(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "abc")
        # Tamper: clear the chain_hash of the first entry
        tracker._chain_store["tpl-001"][0]["chain_hash"] = ""
        valid, chain = tracker.verify_chain("tpl-001")
        assert valid is False

    def test_verify_detects_missing_required_field(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "abc")
        # Tamper: remove a required field
        del tracker._chain_store["tpl-001"][0]["action"]
        valid, chain = tracker.verify_chain("tpl-001")
        assert valid is False

    def test_verify_returns_chain_entries_in_order(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "a")
        tracker.record("template", "tpl-001", "update", "b")
        valid, chain = tracker.verify_chain("tpl-001")
        assert chain[0]["action"] == "create"
        assert chain[1]["action"] == "update"

    def test_verify_different_entity_independent(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "a")
        tracker.record("distribution", "dist-001", "send", "b")
        valid_tpl, _ = tracker.verify_chain("tpl-001")
        valid_dist, _ = tracker.verify_chain("dist-001")
        assert valid_tpl is True
        assert valid_dist is True


# ============================================================================
# get_chain() tests
# ============================================================================


class TestProvenanceTrackerGetChain:
    def test_get_chain_unknown_entity_returns_empty(self):
        tracker = ProvenanceTracker()
        assert tracker.get_chain("unknown") == []

    def test_get_chain_returns_list_copy(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "a")
        chain1 = tracker.get_chain("tpl-001")
        chain2 = tracker.get_chain("tpl-001")
        assert chain1 is not chain2  # different list objects
        assert chain1 == chain2  # same content

    def test_get_chain_preserves_order(self):
        tracker = ProvenanceTracker()
        for i, action in enumerate(["create", "validate", "score"]):
            tracker.record("response", "resp-001", action, f"hash{i}")
        chain = tracker.get_chain("resp-001")
        assert [e["action"] for e in chain] == ["create", "validate", "score"]


# ============================================================================
# get_global_chain() tests
# ============================================================================


class TestProvenanceTrackerGetGlobalChain:
    def test_global_chain_empty_initially(self):
        tracker = ProvenanceTracker()
        assert tracker.get_global_chain() == []

    def test_global_chain_returns_newest_first(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "a")
        tracker.record("distribution", "dist-001", "send", "b")
        global_chain = tracker.get_global_chain()
        assert global_chain[0]["entity_id"] == "dist-001"
        assert global_chain[1]["entity_id"] == "tpl-001"

    def test_global_chain_respects_limit(self):
        tracker = ProvenanceTracker()
        for i in range(10):
            tracker.record("entity", f"e-{i}", "action", f"h{i}")
        chain = tracker.get_global_chain(limit=5)
        assert len(chain) == 5

    def test_global_chain_default_limit_100(self):
        tracker = ProvenanceTracker()
        for i in range(5):
            tracker.record("entity", f"e-{i}", "action", f"h{i}")
        chain = tracker.get_global_chain()
        assert len(chain) == 5  # less than 100, returns all


# ============================================================================
# export_json() tests
# ============================================================================


class TestProvenanceTrackerExportJson:
    def test_export_json_empty_returns_valid_json(self):
        tracker = ProvenanceTracker()
        exported = tracker.export_json()
        parsed = json.loads(exported)
        assert parsed == []

    def test_export_json_with_entries_returns_valid_json(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "abc")
        tracker.record("template", "tpl-001", "update", "def")
        exported = tracker.export_json()
        parsed = json.loads(exported)
        assert len(parsed) == 2

    def test_export_json_contains_all_fields(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "abc")
        exported = tracker.export_json()
        parsed = json.loads(exported)
        entry = parsed[0]
        required_keys = {"entity_type", "entity_id", "action", "data_hash", "user_id", "timestamp", "chain_hash"}
        assert required_keys.issubset(set(entry.keys()))

    def test_export_json_indented(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "abc")
        exported = tracker.export_json()
        # Indented JSON has newlines
        assert "\n" in exported


# ============================================================================
# build_hash() tests
# ============================================================================


class TestProvenanceTrackerBuildHash:
    def test_build_hash_returns_64_char_hex(self):
        tracker = ProvenanceTracker()
        h = tracker.build_hash({"key": "value"})
        assert len(h) == 64

    def test_build_hash_deterministic(self):
        tracker = ProvenanceTracker()
        h1 = tracker.build_hash({"a": 1, "b": 2})
        h2 = tracker.build_hash({"b": 2, "a": 1})
        assert h1 == h2  # sort_keys=True ensures order-independent

    def test_build_hash_different_data_different_hash(self):
        tracker = ProvenanceTracker()
        h1 = tracker.build_hash({"x": 1})
        h2 = tracker.build_hash({"x": 2})
        assert h1 != h2

    def test_build_hash_handles_nested_data(self):
        tracker = ProvenanceTracker()
        data = {"outer": {"inner": [1, 2, 3]}}
        h = tracker.build_hash(data)
        assert len(h) == 64

    def test_build_hash_matches_manual_sha256(self):
        tracker = ProvenanceTracker()
        data = {"test": "value"}
        expected = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        assert tracker.build_hash(data) == expected


# ============================================================================
# _compute_chain_hash() tests
# ============================================================================


class TestComputeChainHash:
    def test_returns_64_char_hex(self):
        tracker = ProvenanceTracker()
        h = tracker._compute_chain_hash("prev", "data", "action", "2026-01-01T00:00:00")
        assert len(h) == 64

    def test_deterministic_for_same_inputs(self):
        tracker = ProvenanceTracker()
        h1 = tracker._compute_chain_hash("p", "d", "a", "t")
        h2 = tracker._compute_chain_hash("p", "d", "a", "t")
        assert h1 == h2

    def test_different_previous_hash_produces_different_result(self):
        tracker = ProvenanceTracker()
        h1 = tracker._compute_chain_hash("prev1", "d", "a", "t")
        h2 = tracker._compute_chain_hash("prev2", "d", "a", "t")
        assert h1 != h2

    def test_different_data_hash_produces_different_result(self):
        tracker = ProvenanceTracker()
        h1 = tracker._compute_chain_hash("p", "d1", "a", "t")
        h2 = tracker._compute_chain_hash("p", "d2", "a", "t")
        assert h1 != h2

    def test_different_action_produces_different_result(self):
        tracker = ProvenanceTracker()
        h1 = tracker._compute_chain_hash("p", "d", "create", "t")
        h2 = tracker._compute_chain_hash("p", "d", "update", "t")
        assert h1 != h2

    def test_matches_manual_computation(self):
        tracker = ProvenanceTracker()
        prev, data, action, ts = "prev", "data", "act", "2026-01-01"
        combined = json.dumps({
            "previous": prev,
            "data": data,
            "action": action,
            "timestamp": ts,
        }, sort_keys=True)
        expected = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        result = tracker._compute_chain_hash(prev, data, action, ts)
        assert result == expected


# ============================================================================
# Multi-entity chain isolation tests
# ============================================================================


class TestProvenanceTrackerChainIsolation:
    def test_separate_entities_have_separate_chains(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "a")
        tracker.record("response", "resp-001", "submit", "b")
        assert len(tracker.get_chain("tpl-001")) == 1
        assert len(tracker.get_chain("resp-001")) == 1

    def test_tampering_one_entity_does_not_affect_another(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "a")
        tracker.record("response", "resp-001", "submit", "b")
        # Tamper with tpl-001
        tracker._chain_store["tpl-001"][0]["chain_hash"] = ""
        valid_tpl, _ = tracker.verify_chain("tpl-001")
        valid_resp, _ = tracker.verify_chain("resp-001")
        assert valid_tpl is False
        assert valid_resp is True

    def test_global_chain_tracks_all_entities(self):
        tracker = ProvenanceTracker()
        tracker.record("template", "tpl-001", "create", "a")
        tracker.record("response", "resp-001", "submit", "b")
        tracker.record("score", "score-001", "calculate", "c")
        assert tracker.entry_count == 3
        assert tracker.entity_count == 3
