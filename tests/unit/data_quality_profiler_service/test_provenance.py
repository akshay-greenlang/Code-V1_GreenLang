# -*- coding: utf-8 -*-
"""
Unit Tests for ProvenanceTracker - AGENT-DATA-010

Tests the SHA-256 chain-hashed provenance tracker used for audit trail
tracking in the Data Quality Profiler service.

Target: 50+ tests, 85%+ coverage of greenlang.data_quality_profiler.provenance

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import re

import pytest

from greenlang.data_quality_profiler.provenance import ProvenanceTracker


# ============================================================================
# TestInit - initialization behaviour
# ============================================================================


class TestInit:
    """ProvenanceTracker initialization tests."""

    def test_genesis_hash_exists(self):
        tracker = ProvenanceTracker()
        assert tracker._last_chain_hash is not None

    def test_genesis_hash_is_64_char_hex(self):
        tracker = ProvenanceTracker()
        assert len(tracker._last_chain_hash) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", tracker._last_chain_hash)

    def test_genesis_hash_is_deterministic(self):
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        assert t1._last_chain_hash == t2._last_chain_hash

    def test_initial_entry_count_zero(self):
        tracker = ProvenanceTracker()
        assert tracker.entry_count == 0

    def test_initial_entity_count_zero(self):
        tracker = ProvenanceTracker()
        assert tracker.entity_count == 0

    def test_chain_store_empty(self):
        tracker = ProvenanceTracker()
        assert len(tracker._chain_store) == 0

    def test_global_chain_empty(self):
        tracker = ProvenanceTracker()
        assert len(tracker._global_chain) == 0


# ============================================================================
# TestRecord - recording provenance entries
# ============================================================================


class TestRecord:
    """ProvenanceTracker.record() tests."""

    def test_returns_string(self):
        tracker = ProvenanceTracker()
        result = tracker.record("profile", "p1", "create", "abc123")
        assert isinstance(result, str)

    def test_returns_64_char_hex(self):
        tracker = ProvenanceTracker()
        result = tracker.record("profile", "p1", "create", "abc123")
        assert len(result) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", result)

    def test_increments_entry_count(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "abc123")
        assert tracker.entry_count == 1

    def test_increments_entry_count_multiple(self):
        tracker = ProvenanceTracker()
        for i in range(5):
            tracker.record("profile", f"p{i}", "create", f"hash{i}")
        assert tracker.entry_count == 5

    def test_increments_entity_count(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "abc123")
        assert tracker.entity_count == 1

    def test_same_entity_does_not_increase_entity_count(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "hash1")
        tracker.record("profile", "p1", "update", "hash2")
        assert tracker.entity_count == 1
        assert tracker.entry_count == 2

    def test_different_entities_increase_entity_count(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        tracker.record("assessment", "a1", "create", "h2")
        assert tracker.entity_count == 2

    def test_chain_linking(self):
        tracker = ProvenanceTracker()
        h1 = tracker.record("profile", "p1", "create", "d1")
        h2 = tracker.record("profile", "p1", "update", "d2")
        assert h1 != h2  # Different chain hashes

    def test_entry_has_required_fields(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "abc123")
        chain = tracker.get_chain("profile", "p1")
        assert len(chain) == 1
        entry = chain[0]
        assert "entity_type" in entry
        assert "entity_id" in entry
        assert "action" in entry
        assert "data_hash" in entry
        assert "timestamp" in entry
        assert "chain_hash" in entry
        assert "user_id" in entry

    def test_entry_entity_type_correct(self):
        tracker = ProvenanceTracker()
        tracker.record("assessment", "a1", "assess", "xyz")
        chain = tracker.get_chain("assessment", "a1")
        assert chain[0]["entity_type"] == "assessment"

    def test_entry_entity_id_correct(self):
        tracker = ProvenanceTracker()
        tracker.record("rule", "r42", "evaluate", "xyz")
        chain = tracker.get_chain("rule", "r42")
        assert chain[0]["entity_id"] == "r42"

    def test_default_user_id_system(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "d1")
        chain = tracker.get_chain("profile", "p1")
        assert chain[0]["user_id"] == "system"

    def test_custom_user_id(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "d1", user_id="admin")
        chain = tracker.get_chain("profile", "p1")
        assert chain[0]["user_id"] == "admin"

    def test_updates_last_chain_hash(self):
        tracker = ProvenanceTracker()
        initial = tracker._last_chain_hash
        tracker.record("profile", "p1", "create", "d1")
        assert tracker._last_chain_hash != initial


# ============================================================================
# TestVerifyChain - chain integrity verification
# ============================================================================


class TestVerifyChain:
    """ProvenanceTracker.verify_chain() tests."""

    def test_empty_chain_is_valid(self):
        tracker = ProvenanceTracker()
        valid, chain = tracker.verify_chain("profile", "nonexistent")
        assert valid is True
        assert chain == []

    def test_single_entry_chain_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "abc")
        valid, chain = tracker.verify_chain("profile", "p1")
        assert valid is True
        assert len(chain) == 1

    def test_multiple_entries_chain_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        tracker.record("profile", "p1", "update", "h2")
        tracker.record("profile", "p1", "finalize", "h3")
        valid, chain = tracker.verify_chain("profile", "p1")
        assert valid is True
        assert len(chain) == 3

    def test_tampered_chain_detected(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        # Tamper with the chain hash directly
        tracker._chain_store["profile:p1"][0]["chain_hash"] = ""
        valid, chain = tracker.verify_chain("profile", "p1")
        assert valid is False

    def test_missing_field_detected(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        # Remove a required field
        del tracker._chain_store["profile:p1"][0]["action"]
        valid, chain = tracker.verify_chain("profile", "p1")
        assert valid is False

    def test_returns_chain_entries(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        tracker.record("profile", "p1", "update", "h2")
        valid, chain = tracker.verify_chain("profile", "p1")
        assert len(chain) == 2
        assert chain[0]["action"] == "create"
        assert chain[1]["action"] == "update"


# ============================================================================
# TestGetChain - retrieving entity chains
# ============================================================================


class TestGetChain:
    """ProvenanceTracker.get_chain() tests."""

    def test_unknown_entity_returns_empty(self):
        tracker = ProvenanceTracker()
        result = tracker.get_chain("unknown_type", "unknown_id")
        assert result == []

    def test_returns_copy_not_reference(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        chain1 = tracker.get_chain("profile", "p1")
        chain2 = tracker.get_chain("profile", "p1")
        assert chain1 is not chain2
        assert chain1 == chain2

    def test_entries_in_order(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        tracker.record("profile", "p1", "update", "h2")
        tracker.record("profile", "p1", "finalize", "h3")
        chain = tracker.get_chain("profile", "p1")
        assert chain[0]["action"] == "create"
        assert chain[1]["action"] == "update"
        assert chain[2]["action"] == "finalize"


# ============================================================================
# TestGetGlobalChain - global chain retrieval
# ============================================================================


class TestGetGlobalChain:
    """ProvenanceTracker.get_global_chain() tests."""

    def test_empty_returns_empty(self):
        tracker = ProvenanceTracker()
        result = tracker.get_global_chain()
        assert result == []

    def test_returns_entries_newest_first(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        tracker.record("assessment", "a1", "assess", "h2")
        result = tracker.get_global_chain()
        assert len(result) == 2
        # Newest first
        assert result[0]["entity_type"] == "assessment"
        assert result[1]["entity_type"] == "profile"

    def test_respects_limit(self):
        tracker = ProvenanceTracker()
        for i in range(10):
            tracker.record("profile", f"p{i}", "create", f"h{i}")
        result = tracker.get_global_chain(limit=3)
        assert len(result) == 3

    def test_limit_greater_than_total(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        result = tracker.get_global_chain(limit=100)
        assert len(result) == 1

    def test_cross_entity_ordering(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        tracker.record("rule", "r1", "evaluate", "h2")
        tracker.record("profile", "p1", "update", "h3")
        result = tracker.get_global_chain()
        assert len(result) == 3
        # Newest first
        assert result[0]["action"] == "update"
        assert result[1]["action"] == "evaluate"
        assert result[2]["action"] == "create"


# ============================================================================
# TestExportJson - JSON export
# ============================================================================


class TestExportJson:
    """ProvenanceTracker.export_json() tests."""

    def test_empty_export(self):
        tracker = ProvenanceTracker()
        result = tracker.export_json()
        parsed = json.loads(result)
        assert parsed == []

    def test_export_has_entries(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        result = tracker.export_json()
        parsed = json.loads(result)
        assert len(parsed) == 1

    def test_export_entry_has_required_fields(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        result = tracker.export_json()
        parsed = json.loads(result)
        entry = parsed[0]
        for field in ["entity_type", "entity_id", "action", "data_hash",
                       "timestamp", "chain_hash"]:
            assert field in entry

    def test_export_is_valid_json(self):
        tracker = ProvenanceTracker()
        for i in range(5):
            tracker.record("profile", f"p{i}", "create", f"h{i}")
        result = tracker.export_json()
        parsed = json.loads(result)
        assert len(parsed) == 5

    def test_export_uses_indentation(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        result = tracker.export_json()
        # Indented JSON has newlines
        assert "\n" in result


# ============================================================================
# TestBuildHash - arbitrary data hashing
# ============================================================================


class TestBuildHash:
    """ProvenanceTracker.build_hash() tests."""

    def test_returns_64_char_hex(self):
        tracker = ProvenanceTracker()
        h = tracker.build_hash({"key": "value"})
        assert len(h) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", h)

    def test_deterministic(self):
        tracker = ProvenanceTracker()
        h1 = tracker.build_hash({"a": 1, "b": 2})
        h2 = tracker.build_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_different_data_different_hash(self):
        tracker = ProvenanceTracker()
        h1 = tracker.build_hash({"a": 1})
        h2 = tracker.build_hash({"a": 2})
        assert h1 != h2

    def test_key_order_independence(self):
        tracker = ProvenanceTracker()
        h1 = tracker.build_hash({"a": 1, "b": 2})
        h2 = tracker.build_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_hashes_list_data(self):
        tracker = ProvenanceTracker()
        h = tracker.build_hash([1, 2, 3])
        assert len(h) == 64

    def test_hashes_string_data(self):
        tracker = ProvenanceTracker()
        h = tracker.build_hash("hello")
        assert len(h) == 64

    def test_hashes_numeric_data(self):
        tracker = ProvenanceTracker()
        h = tracker.build_hash(42)
        assert len(h) == 64

    def test_hashes_nested_dict(self):
        tracker = ProvenanceTracker()
        data = {"outer": {"inner": {"deep": [1, 2, 3]}}}
        h = tracker.build_hash(data)
        assert len(h) == 64


# ============================================================================
# TestChainIsolation - separate entities have separate chains
# ============================================================================


class TestChainIsolation:
    """Different entities maintain separate provenance chains."""

    def test_separate_entity_chains(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        tracker.record("assessment", "a1", "assess", "h2")
        p_chain = tracker.get_chain("profile", "p1")
        a_chain = tracker.get_chain("assessment", "a1")
        assert len(p_chain) == 1
        assert len(a_chain) == 1
        assert p_chain[0]["entity_type"] == "profile"
        assert a_chain[0]["entity_type"] == "assessment"

    def test_same_type_different_id(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        tracker.record("profile", "p2", "create", "h2")
        p1 = tracker.get_chain("profile", "p1")
        p2 = tracker.get_chain("profile", "p2")
        assert len(p1) == 1
        assert len(p2) == 1
        assert p1[0]["data_hash"] == "h1"
        assert p2[0]["data_hash"] == "h2"

    def test_modification_of_one_chain_does_not_affect_other(self):
        tracker = ProvenanceTracker()
        tracker.record("profile", "p1", "create", "h1")
        tracker.record("assessment", "a1", "assess", "h2")
        tracker.record("profile", "p1", "update", "h3")
        p_chain = tracker.get_chain("profile", "p1")
        a_chain = tracker.get_chain("assessment", "a1")
        assert len(p_chain) == 2
        assert len(a_chain) == 1


# ============================================================================
# TestModuleExports
# ============================================================================


class TestModuleExports:
    """Verify provenance module exports."""

    def test_all_list_exists(self):
        from greenlang.data_quality_profiler import provenance as mod
        assert hasattr(mod, "__all__")

    def test_all_contains_provenance_tracker(self):
        from greenlang.data_quality_profiler import provenance as mod
        assert "ProvenanceTracker" in mod.__all__

    def test_all_has_one_entry(self):
        from greenlang.data_quality_profiler import provenance as mod
        assert len(mod.__all__) == 1
