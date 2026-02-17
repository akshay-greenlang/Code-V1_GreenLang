# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-DATA-014

Tests SHA-256 chain hashing, add_entry/record/verify/get operations, reset,
export, threading safety, singleton, and genesis hash determinism.
Target: 15+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

from __future__ import annotations

import hashlib
import json
import threading
from unittest.mock import patch

import pytest

from greenlang.time_series_gap_filler.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    get_provenance_tracker,
)


# -----------------------------------------------------------------------
# 1. Genesis hash
# -----------------------------------------------------------------------


class TestGenesisHash:
    """Test the GENESIS_HASH class constant."""

    def test_genesis_hash_is_sha256_of_expected_seed(self):
        expected = hashlib.sha256(
            b"greenlang-time-series-gap-filler-genesis"
        ).hexdigest()
        assert ProvenanceTracker.GENESIS_HASH == expected

    def test_genesis_hash_length(self):
        assert len(ProvenanceTracker.GENESIS_HASH) == 64

    def test_genesis_hash_is_valid_hex(self):
        int(ProvenanceTracker.GENESIS_HASH, 16)  # should not raise

    def test_initial_last_chain_hash_equals_genesis(self):
        tracker = ProvenanceTracker()
        assert tracker._last_chain_hash == ProvenanceTracker.GENESIS_HASH


# -----------------------------------------------------------------------
# 2. ProvenanceTracker creation
# -----------------------------------------------------------------------


class TestProvenanceTrackerCreation:
    """Test ProvenanceTracker instantiation."""

    def test_creates_successfully(self):
        tracker = ProvenanceTracker()
        assert tracker is not None

    def test_initial_chain_empty(self):
        tracker = ProvenanceTracker()
        assert tracker.entry_count == 0
        assert tracker.entity_count == 0

    def test_initial_get_chain_returns_empty(self):
        tracker = ProvenanceTracker()
        assert tracker.get_chain() == []


# -----------------------------------------------------------------------
# 3. add_entry() method
# -----------------------------------------------------------------------


class TestAddEntry:
    """Test the add_entry method."""

    def test_returns_provenance_entry(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("detect_gaps", "in_hash", "out_hash")
        assert isinstance(entry, ProvenanceEntry)

    def test_entry_has_sha256_chain_hash(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("fill_gaps", "aaa", "bbb")
        assert len(entry.chain_hash) == 64
        int(entry.chain_hash, 16)  # valid hex

    def test_entry_has_unique_id(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("validate", "in1", "out1")
        assert len(entry.entry_id) > 0

    def test_entry_has_correct_operation(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("detect_frequency", "x", "y")
        assert entry.operation == "detect_frequency"

    def test_entry_has_correct_hashes(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("fill_gaps", "input_abc", "output_def")
        assert entry.input_hash == "input_abc"
        assert entry.output_hash == "output_def"

    def test_first_entry_parent_hash_is_genesis(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("detect_gaps", "a", "b")
        assert entry.parent_hash == ProvenanceTracker.GENESIS_HASH

    def test_entry_has_timestamp(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("pipeline", "in", "out")
        assert len(entry.timestamp) > 0
        assert "T" in entry.timestamp  # ISO format

    def test_entry_with_metadata(self):
        tracker = ProvenanceTracker()
        meta = {"method": "linear", "confidence": 0.95}
        entry = tracker.add_entry("fill_gaps", "a", "b", metadata=meta)
        assert entry.metadata["method"] == "linear"
        assert entry.metadata["confidence"] == 0.95

    def test_entry_without_metadata_has_empty_dict(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("detect_gaps", "a", "b")
        assert entry.metadata == {}


# -----------------------------------------------------------------------
# 4. Chain integrity (parent_hash linking)
# -----------------------------------------------------------------------


class TestChainIntegrity:
    """Test that each entry's parent_hash matches the previous entry's chain_hash."""

    def test_second_entry_parent_hash_matches_first_chain_hash(self):
        tracker = ProvenanceTracker()
        entry1 = tracker.add_entry("detect_gaps", "in1", "out1")
        entry2 = tracker.add_entry("fill_gaps", "in2", "out2")
        assert entry2.parent_hash == entry1.chain_hash

    def test_three_entry_chain_links(self):
        tracker = ProvenanceTracker()
        e1 = tracker.add_entry("detect_gaps", "a", "b")
        e2 = tracker.add_entry("detect_frequency", "c", "d")
        e3 = tracker.add_entry("fill_gaps", "e", "f")
        assert e1.parent_hash == ProvenanceTracker.GENESIS_HASH
        assert e2.parent_hash == e1.chain_hash
        assert e3.parent_hash == e2.chain_hash

    def test_chain_hashes_are_all_unique(self):
        tracker = ProvenanceTracker()
        entries = [
            tracker.add_entry(f"op{i}", f"in{i}", f"out{i}")
            for i in range(5)
        ]
        hashes = [e.chain_hash for e in entries]
        assert len(set(hashes)) == 5

    def test_chain_hash_includes_operation_and_hashes(self):
        """Verify the chain hash is computed from operation, input, output, and parent."""
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("fill_gaps", "input_h", "output_h")
        # Recompute manually using the private method
        expected = tracker._compute_chain_hash(
            ProvenanceTracker.GENESIS_HASH,
            "input_h",
            "output_h",
            "fill_gaps",
            entry.timestamp,
        )
        assert entry.chain_hash == expected


# -----------------------------------------------------------------------
# 5. get_chain()
# -----------------------------------------------------------------------


class TestGetChain:
    """Test the get_chain method."""

    def test_get_chain_returns_all_entries_in_order(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "a", "b")
        tracker.add_entry("op2", "c", "d")
        tracker.add_entry("op3", "e", "f")
        chain = tracker.get_chain()
        assert len(chain) == 3
        assert chain[0]["operation"] == "op1"
        assert chain[1]["operation"] == "op2"
        assert chain[2]["operation"] == "op3"

    def test_get_chain_by_entity(self):
        tracker = ProvenanceTracker()
        tracker.record("gap_fill_job", "j1", "fill", "h1")
        tracker.record("gap_fill_job", "j1", "validate", "h2")
        tracker.record("gap_fill_job", "j2", "fill", "h3")
        chain_j1 = tracker.get_chain("gap_fill_job", "j1")
        assert len(chain_j1) == 2
        chain_j2 = tracker.get_chain("gap_fill_job", "j2")
        assert len(chain_j2) == 1

    def test_get_chain_nonexistent_entity_returns_empty(self):
        tracker = ProvenanceTracker()
        chain = tracker.get_chain("gap_fill_job", "nonexistent")
        assert chain == []


# -----------------------------------------------------------------------
# 6. verify_chain()
# -----------------------------------------------------------------------


class TestVerifyChain:
    """Test the verify_chain method."""

    def test_empty_chain_is_valid(self):
        tracker = ProvenanceTracker()
        is_valid, chain = tracker.verify_chain("gap_fill_job", "nonexist")
        assert is_valid is True
        assert chain == []

    def test_single_entry_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("gap_fill_job", "j1", "create", "h1")
        is_valid, chain = tracker.verify_chain("gap_fill_job", "j1")
        assert is_valid is True
        assert len(chain) == 1

    def test_multi_entry_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("gap_fill_job", "j1", "detect", "h1")
        tracker.record("gap_fill_job", "j1", "fill", "h2")
        tracker.record("gap_fill_job", "j1", "validate", "h3")
        is_valid, chain = tracker.verify_chain("gap_fill_job", "j1")
        assert is_valid is True
        assert len(chain) == 3

    def test_tampered_chain_detected(self):
        tracker = ProvenanceTracker()
        tracker.record("gap_fill_job", "j1", "detect", "h1")
        tracker.record("gap_fill_job", "j1", "fill", "h2")

        # Tamper with the entity chain by removing a required field
        store_key = "gap_fill_job:j1"
        with tracker._lock:
            tracker._chain_store[store_key][0].pop("chain_hash")

        is_valid, chain = tracker.verify_chain("gap_fill_job", "j1")
        assert is_valid is False

    def test_global_chain_verify(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "in1", "out1")
        tracker.record("gap_fill_job", "j1", "create", "h1")
        is_valid, chain = tracker.verify_chain()
        assert is_valid is True
        assert len(chain) == 2

    def test_tampered_global_chain_detected(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "in1", "out1")
        # Tamper with the global chain entry
        with tracker._lock:
            tracker._global_chain[0].pop("chain_hash")
        is_valid, chain = tracker.verify_chain()
        assert is_valid is False


# -----------------------------------------------------------------------
# 7. get_entry()
# -----------------------------------------------------------------------


class TestGetEntry:
    """Test the get_entry method."""

    def test_get_entry_returns_correct_entry(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op_alpha", "a", "b")
        tracker.add_entry("op_beta", "c", "d")
        entry = tracker.get_entry(0)
        assert entry is not None
        assert entry["operation"] == "op_alpha"
        entry2 = tracker.get_entry(1)
        assert entry2 is not None
        assert entry2["operation"] == "op_beta"

    def test_get_entry_out_of_bounds_returns_none(self):
        tracker = ProvenanceTracker()
        assert tracker.get_entry(0) is None
        assert tracker.get_entry(-1) is None
        assert tracker.get_entry(999) is None

    def test_get_entry_returns_copy(self):
        """Modifying returned entry should not affect internal chain."""
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "a", "b")
        entry = tracker.get_entry(0)
        entry["operation"] = "TAMPERED"
        original = tracker.get_entry(0)
        assert original["operation"] == "op1"


# -----------------------------------------------------------------------
# 8. clear() and reset()
# -----------------------------------------------------------------------


class TestClearAndReset:
    """Test the clear and reset methods."""

    def test_clear_removes_all_entries(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "a", "b")
        tracker.record("gap_fill_job", "j1", "fill", "h1")
        assert tracker.entry_count == 2
        tracker.clear()
        assert tracker.entry_count == 0
        assert tracker.entity_count == 0

    def test_clear_restores_genesis_hash(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "a", "b")
        tracker.clear()
        assert tracker._last_chain_hash == ProvenanceTracker.GENESIS_HASH

    def test_reset_is_alias_for_clear(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "a", "b")
        tracker.reset()
        assert tracker.entry_count == 0
        assert tracker._last_chain_hash == ProvenanceTracker.GENESIS_HASH

    def test_operations_after_clear(self):
        """Chain should function normally after clear."""
        tracker = ProvenanceTracker()
        tracker.add_entry("old_op", "x", "y")
        tracker.clear()
        entry = tracker.add_entry("new_op", "a", "b")
        assert entry.parent_hash == ProvenanceTracker.GENESIS_HASH
        assert tracker.entry_count == 1


# -----------------------------------------------------------------------
# 9. Thread-safe singleton via get_provenance_tracker()
# -----------------------------------------------------------------------


class TestSingleton:
    """Test the thread-safe singleton get_provenance_tracker."""

    def test_returns_same_instance(self):
        t1 = get_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is t2

    def test_concurrent_get_returns_same_instance(self):
        instances = []
        errors = []

        def worker():
            try:
                instances.append(get_provenance_tracker())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(instances) == 10
        assert all(inst is instances[0] for inst in instances)

    def test_singleton_is_provenance_tracker(self):
        tracker = get_provenance_tracker()
        assert isinstance(tracker, ProvenanceTracker)


# -----------------------------------------------------------------------
# 10. Multiple operations produce unique hashes
# -----------------------------------------------------------------------


class TestUniqueHashes:
    """Test that different operations produce unique chain hashes."""

    def test_different_operations_produce_different_hashes(self):
        tracker = ProvenanceTracker()
        e1 = tracker.add_entry("detect_gaps", "same_in", "same_out")
        tracker2 = ProvenanceTracker()
        e2 = tracker2.add_entry("fill_gaps", "same_in", "same_out")
        assert e1.chain_hash != e2.chain_hash

    def test_different_inputs_produce_different_hashes(self):
        tracker = ProvenanceTracker()
        e1 = tracker.add_entry("detect_gaps", "input_A", "output_X")
        tracker2 = ProvenanceTracker()
        e2 = tracker2.add_entry("detect_gaps", "input_B", "output_X")
        assert e1.chain_hash != e2.chain_hash

    def test_different_outputs_produce_different_hashes(self):
        tracker = ProvenanceTracker()
        e1 = tracker.add_entry("detect_gaps", "input_A", "output_X")
        tracker2 = ProvenanceTracker()
        e2 = tracker2.add_entry("detect_gaps", "input_A", "output_Y")
        assert e1.chain_hash != e2.chain_hash


# -----------------------------------------------------------------------
# 11. hash_record and build_hash
# -----------------------------------------------------------------------


class TestHashingMethods:
    """Test hash_record and build_hash methods."""

    def test_hash_record_deterministic(self):
        tracker = ProvenanceTracker()
        data = {"fuel": "diesel", "quantity": 100}
        h1 = tracker.hash_record(data)
        h2 = tracker.hash_record(data)
        assert h1 == h2

    def test_hash_record_key_order_independent(self):
        tracker = ProvenanceTracker()
        h1 = tracker.hash_record({"b": 2, "a": 1})
        h2 = tracker.hash_record({"a": 1, "b": 2})
        assert h1 == h2

    def test_hash_record_different_data_different_hash(self):
        tracker = ProvenanceTracker()
        h1 = tracker.hash_record({"x": 1})
        h2 = tracker.hash_record({"x": 2})
        assert h1 != h2

    def test_hash_record_is_64_hex_chars(self):
        tracker = ProvenanceTracker()
        h = tracker.hash_record({"key": "val"})
        assert len(h) == 64
        int(h, 16)  # valid hex

    def test_build_hash_with_list(self):
        tracker = ProvenanceTracker()
        h = tracker.build_hash([1, 2, 3])
        assert len(h) == 64

    def test_build_hash_deterministic(self):
        tracker = ProvenanceTracker()
        h1 = tracker.build_hash({"x": 42})
        h2 = tracker.build_hash({"x": 42})
        assert h1 == h2


# -----------------------------------------------------------------------
# 12. ProvenanceEntry.to_dict()
# -----------------------------------------------------------------------


class TestProvenanceEntryToDict:
    """Test the ProvenanceEntry dataclass to_dict method."""

    def test_to_dict_returns_dict(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("op", "in_h", "out_h")
        d = entry.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_all_fields(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("op", "in_h", "out_h", metadata={"k": "v"})
        d = entry.to_dict()
        assert "entry_id" in d
        assert "operation" in d
        assert "input_hash" in d
        assert "output_hash" in d
        assert "timestamp" in d
        assert "parent_hash" in d
        assert "chain_hash" in d
        assert "metadata" in d
        assert d["metadata"]["k"] == "v"


# -----------------------------------------------------------------------
# 13. export_json
# -----------------------------------------------------------------------


class TestExportJson:
    """Test the export_json method."""

    def test_export_json_empty(self):
        tracker = ProvenanceTracker()
        exported = tracker.export_json()
        data = json.loads(exported)
        assert data == []

    def test_export_json_with_entries(self):
        tracker = ProvenanceTracker()
        tracker.record("gap_fill_job", "j1", "create", "h1")
        tracker.add_entry("detect_gaps", "a", "b")
        exported = tracker.export_json()
        data = json.loads(exported)
        assert len(data) == 2
        assert data[0]["entity_type"] == "gap_fill_job"
        assert data[1]["operation"] == "detect_gaps"


# -----------------------------------------------------------------------
# 14. Thread safety under concurrent writes
# -----------------------------------------------------------------------


class TestThreadSafety:
    """Test thread-safe operations on ProvenanceTracker."""

    def test_concurrent_add_entry(self):
        tracker = ProvenanceTracker()
        errors = []

        def worker():
            try:
                for i in range(10):
                    tracker.add_entry(f"op{i}", f"in{i}", f"out{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.get_chain_length() == 50  # 5 threads * 10 ops

    def test_concurrent_records(self):
        tracker = ProvenanceTracker()
        errors = []

        def worker(entity_id):
            try:
                for i in range(10):
                    tracker.record(
                        "gap_fill_job", entity_id, f"op{i}", f"hash{i}"
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"j{t}",))
            for t in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.entry_count == 50  # 5 threads * 10 ops

    def test_concurrent_mixed_operations(self):
        """Test add_entry and record concurrently without errors."""
        tracker = ProvenanceTracker()
        errors = []

        def add_entry_worker():
            try:
                for i in range(10):
                    tracker.add_entry(f"ae_op{i}", f"ae_in{i}", f"ae_out{i}")
            except Exception as e:
                errors.append(e)

        def record_worker(eid):
            try:
                for i in range(10):
                    tracker.record("gap_fill_job", eid, f"rec_op{i}", f"rh{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=add_entry_worker))
        for t_idx in range(3):
            threads.append(
                threading.Thread(target=record_worker, args=(f"ent{t_idx}",))
            )

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.entry_count == 60  # 3*10 add_entry + 3*10 record


# -----------------------------------------------------------------------
# 15. get_global_chain and get_chain_length
# -----------------------------------------------------------------------


class TestGlobalChainAndLength:
    """Test get_global_chain and get_chain_length methods."""

    def test_get_global_chain_newest_first(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("first", "a", "b")
        tracker.add_entry("second", "c", "d")
        global_chain = tracker.get_global_chain(limit=10)
        assert global_chain[0]["operation"] == "second"
        assert global_chain[1]["operation"] == "first"

    def test_get_global_chain_respects_limit(self):
        tracker = ProvenanceTracker()
        for i in range(20):
            tracker.add_entry(f"op{i}", "a", "b")
        global_chain = tracker.get_global_chain(limit=5)
        assert len(global_chain) == 5

    def test_get_chain_length_zero(self):
        tracker = ProvenanceTracker()
        assert tracker.get_chain_length() == 0

    def test_get_chain_length_increments(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "a", "b")
        assert tracker.get_chain_length() == 1
        tracker.record("job", "j1", "fill", "h")
        assert tracker.get_chain_length() == 2
