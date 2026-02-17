# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-DATA-015 Cross-Source Reconciliation

Tests SHA-256 chain hashing, add_entry/add_to_chain/record/verify/get
operations, reset, export, threading safety, singleton, float normalization,
and genesis hash determinism.
Target: 45+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from unittest.mock import patch

import pytest

from greenlang.cross_source_reconciliation.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    get_provenance_tracker,
    _normalize_value,
)


# -----------------------------------------------------------------------
# 1. Genesis hash
# -----------------------------------------------------------------------


class TestGenesisHash:
    """Test the GENESIS_HASH class constant."""

    def test_genesis_hash_is_sha256_of_expected_seed(self):
        expected = hashlib.sha256(
            b"greenlang-cross-source-reconciliation-genesis"
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
# 2. ProvenanceTracker construction sets genesis hash
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

    def test_initial_get_current_hash_is_genesis(self):
        tracker = ProvenanceTracker()
        assert tracker.get_current_hash() == ProvenanceTracker.GENESIS_HASH


# -----------------------------------------------------------------------
# 3. add_to_chain returns non-empty string
# -----------------------------------------------------------------------


class TestAddToChain:
    """Test the add_to_chain convenience method."""

    def test_returns_non_empty_string(self):
        tracker = ProvenanceTracker()
        chain_hash = tracker.add_to_chain("reconcile", "in_hash", "out_hash")
        assert isinstance(chain_hash, str)
        assert len(chain_hash) > 0

    def test_returns_64_char_hex(self):
        tracker = ProvenanceTracker()
        chain_hash = tracker.add_to_chain("match_records", "aaa", "bbb")
        assert len(chain_hash) == 64
        int(chain_hash, 16)  # valid hex

    def test_chain_hash_differs_from_genesis(self):
        tracker = ProvenanceTracker()
        chain_hash = tracker.add_to_chain("compare_fields", "x", "y")
        assert chain_hash != ProvenanceTracker.GENESIS_HASH

    def test_increments_chain_length(self):
        tracker = ProvenanceTracker()
        assert tracker.get_chain_length() == 0
        tracker.add_to_chain("reconcile", "a", "b")
        assert tracker.get_chain_length() == 1
        tracker.add_to_chain("validate", "c", "d")
        assert tracker.get_chain_length() == 2


# -----------------------------------------------------------------------
# 4. Chain grows with each addition
# -----------------------------------------------------------------------


class TestChainGrowth:
    """Test that the chain grows as entries are added."""

    def test_chain_grows_by_one_per_add_entry(self):
        tracker = ProvenanceTracker()
        for i in range(5):
            tracker.add_entry(f"op{i}", f"in{i}", f"out{i}")
            assert tracker.get_chain_length() == i + 1

    def test_chain_grows_by_one_per_add_to_chain(self):
        tracker = ProvenanceTracker()
        for i in range(5):
            tracker.add_to_chain(f"op{i}", f"in{i}", f"out{i}")
            assert tracker.entry_count == i + 1

    def test_chain_grows_by_one_per_record(self):
        tracker = ProvenanceTracker()
        for i in range(5):
            tracker.record("reconciliation_job", f"j{i}", "match", f"h{i}")
            assert tracker.get_chain_length() == i + 1

    def test_mixed_operations_grow_chain(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "a", "b")
        tracker.add_to_chain("op2", "c", "d")
        tracker.record("reconciliation_job", "j1", "match", "h1")
        assert tracker.get_chain_length() == 3


# -----------------------------------------------------------------------
# 5. Same inputs produce same hash (deterministic)
# -----------------------------------------------------------------------


class TestDeterministicHashing:
    """Test that same inputs produce the same hash."""

    def test_hash_record_deterministic(self):
        tracker = ProvenanceTracker()
        data = {"source_a": "erp", "source_b": "invoice", "amount": 1000.50}
        h1 = tracker.hash_record(data)
        h2 = tracker.hash_record(data)
        assert h1 == h2

    def test_hash_record_key_order_independent(self):
        tracker = ProvenanceTracker()
        h1 = tracker.hash_record({"b": 2, "a": 1})
        h2 = tracker.hash_record({"a": 1, "b": 2})
        assert h1 == h2

    def test_build_hash_deterministic(self):
        tracker = ProvenanceTracker()
        h1 = tracker.build_hash({"x": 42, "y": [1, 2, 3]})
        h2 = tracker.build_hash({"x": 42, "y": [1, 2, 3]})
        assert h1 == h2

    def test_build_hash_with_list_deterministic(self):
        tracker = ProvenanceTracker()
        h1 = tracker.build_hash([1, 2, 3])
        h2 = tracker.build_hash([1, 2, 3])
        assert h1 == h2

    def test_compute_hash_deterministic_for_strings(self):
        tracker = ProvenanceTracker()
        h1 = tracker._compute_hash("hello world")
        h2 = tracker._compute_hash("hello world")
        assert h1 == h2

    def test_compute_hash_deterministic_for_nested_dict(self):
        tracker = ProvenanceTracker()
        data = {"outer": {"inner": [1.0, 2.5]}}
        h1 = tracker._compute_hash(data)
        h2 = tracker._compute_hash(data)
        assert h1 == h2


# -----------------------------------------------------------------------
# 6. Different inputs produce different hash
# -----------------------------------------------------------------------


class TestDifferentInputsDifferentHash:
    """Test that different inputs produce different hashes."""

    def test_different_data_different_hash(self):
        tracker = ProvenanceTracker()
        h1 = tracker.hash_record({"x": 1})
        h2 = tracker.hash_record({"x": 2})
        assert h1 != h2

    def test_different_operations_different_chain_hash(self):
        t1 = ProvenanceTracker()
        e1 = t1.add_entry("match_records", "same_in", "same_out")
        t2 = ProvenanceTracker()
        e2 = t2.add_entry("compare_fields", "same_in", "same_out")
        assert e1.chain_hash != e2.chain_hash

    def test_different_inputs_different_chain_hash(self):
        t1 = ProvenanceTracker()
        e1 = t1.add_entry("reconcile", "input_A", "output_X")
        t2 = ProvenanceTracker()
        e2 = t2.add_entry("reconcile", "input_B", "output_X")
        assert e1.chain_hash != e2.chain_hash

    def test_different_outputs_different_chain_hash(self):
        t1 = ProvenanceTracker()
        e1 = t1.add_entry("reconcile", "input_A", "output_X")
        t2 = ProvenanceTracker()
        e2 = t2.add_entry("reconcile", "input_A", "output_Y")
        assert e1.chain_hash != e2.chain_hash

    def test_sequential_entries_produce_unique_hashes(self):
        tracker = ProvenanceTracker()
        entries = [
            tracker.add_entry(f"op{i}", f"in{i}", f"out{i}")
            for i in range(10)
        ]
        hashes = [e.chain_hash for e in entries]
        assert len(set(hashes)) == 10


# -----------------------------------------------------------------------
# 7. verify_chain returns True for untampered chain
# -----------------------------------------------------------------------


class TestVerifyChain:
    """Test the verify_chain method."""

    def test_empty_chain_is_valid(self):
        tracker = ProvenanceTracker()
        is_valid, chain = tracker.verify_chain("reconciliation_job", "nonexist")
        assert is_valid is True
        assert chain == []

    def test_single_entry_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("reconciliation_job", "j1", "match", "h1")
        is_valid, chain = tracker.verify_chain("reconciliation_job", "j1")
        assert is_valid is True
        assert len(chain) == 1

    def test_multi_entry_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("reconciliation_job", "j1", "match", "h1")
        tracker.record("reconciliation_job", "j1", "compare", "h2")
        tracker.record("reconciliation_job", "j1", "resolve", "h3")
        is_valid, chain = tracker.verify_chain("reconciliation_job", "j1")
        assert is_valid is True
        assert len(chain) == 3

    def test_global_chain_verify(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "in1", "out1")
        tracker.record("reconciliation_job", "j1", "match", "h1")
        is_valid, chain = tracker.verify_chain()
        assert is_valid is True
        assert len(chain) == 2

    def test_tampered_entity_chain_detected(self):
        tracker = ProvenanceTracker()
        tracker.record("reconciliation_job", "j1", "match", "h1")
        tracker.record("reconciliation_job", "j1", "compare", "h2")

        # Tamper by removing a required field
        store_key = "reconciliation_job:j1"
        with tracker._lock:
            tracker._chain_store[store_key][0].pop("chain_hash")

        is_valid, chain = tracker.verify_chain("reconciliation_job", "j1")
        assert is_valid is False

    def test_tampered_global_chain_detected(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "in1", "out1")
        with tracker._lock:
            tracker._global_chain[0].pop("chain_hash")
        is_valid, chain = tracker.verify_chain()
        assert is_valid is False


# -----------------------------------------------------------------------
# 8. reset() restores to genesis state
# -----------------------------------------------------------------------


class TestResetAndClear:
    """Test the reset and clear methods."""

    def test_reset_clears_all_entries(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "a", "b")
        tracker.record("reconciliation_job", "j1", "match", "h1")
        assert tracker.entry_count == 2
        tracker.reset()
        assert tracker.entry_count == 0
        assert tracker.entity_count == 0

    def test_reset_restores_genesis_hash(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "a", "b")
        assert tracker.get_current_hash() != ProvenanceTracker.GENESIS_HASH
        tracker.reset()
        assert tracker.get_current_hash() == ProvenanceTracker.GENESIS_HASH
        assert tracker._last_chain_hash == ProvenanceTracker.GENESIS_HASH

    def test_clear_is_alias_for_reset(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "a", "b")
        tracker.clear()
        assert tracker.entry_count == 0
        assert tracker._last_chain_hash == ProvenanceTracker.GENESIS_HASH

    def test_operations_after_reset(self):
        """Chain should function normally after reset."""
        tracker = ProvenanceTracker()
        tracker.add_entry("old_op", "x", "y")
        tracker.reset()
        entry = tracker.add_entry("new_op", "a", "b")
        assert entry.parent_hash == ProvenanceTracker.GENESIS_HASH
        assert tracker.entry_count == 1

    def test_reset_clears_entity_chains(self):
        tracker = ProvenanceTracker()
        tracker.record("reconciliation_job", "j1", "match", "h1")
        tracker.record("reconciliation_job", "j2", "compare", "h2")
        assert tracker.entity_count == 2
        tracker.reset()
        assert tracker.entity_count == 0
        assert tracker.get_chain("reconciliation_job", "j1") == []


# -----------------------------------------------------------------------
# 9. get_chain() returns list of entries
# -----------------------------------------------------------------------


class TestGetChain:
    """Test the get_chain method."""

    def test_get_chain_returns_list(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("op1", "a", "b")
        chain = tracker.get_chain()
        assert isinstance(chain, list)

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
        tracker.record("reconciliation_job", "j1", "match", "h1")
        tracker.record("reconciliation_job", "j1", "compare", "h2")
        tracker.record("reconciliation_job", "j2", "match", "h3")
        chain_j1 = tracker.get_chain("reconciliation_job", "j1")
        assert len(chain_j1) == 2
        chain_j2 = tracker.get_chain("reconciliation_job", "j2")
        assert len(chain_j2) == 1

    def test_get_chain_nonexistent_entity_returns_empty(self):
        tracker = ProvenanceTracker()
        chain = tracker.get_chain("reconciliation_job", "nonexistent")
        assert chain == []

    def test_get_chain_entries_have_expected_fields(self):
        tracker = ProvenanceTracker()
        tracker.add_entry("reconcile", "in_h", "out_h", metadata={"k": "v"})
        chain = tracker.get_chain()
        assert len(chain) == 1
        entry = chain[0]
        assert "entry_id" in entry
        assert "operation" in entry
        assert "input_hash" in entry
        assert "output_hash" in entry
        assert "timestamp" in entry
        assert "parent_hash" in entry
        assert "chain_hash" in entry
        assert "metadata" in entry


# -----------------------------------------------------------------------
# 10. Float normalization handles NaN, Inf
# -----------------------------------------------------------------------


class TestFloatNormalization:
    """Test float normalization for deterministic hashing."""

    def test_nan_normalized_to_string(self):
        result = _normalize_value(float("nan"))
        assert result == "__NaN__"

    def test_positive_inf_normalized_to_string(self):
        result = _normalize_value(float("inf"))
        assert result == "__Inf__"

    def test_negative_inf_normalized_to_string(self):
        result = _normalize_value(float("-inf"))
        assert result == "__-Inf__"

    def test_float_rounded_to_10_decimals(self):
        result = _normalize_value(1.123456789012345)
        assert result == round(1.123456789012345, 10)

    def test_integer_passthrough(self):
        result = _normalize_value(42)
        assert result == 42

    def test_string_passthrough(self):
        result = _normalize_value("hello")
        assert result == "hello"

    def test_none_passthrough(self):
        result = _normalize_value(None)
        assert result is None

    def test_nested_dict_normalization(self):
        data = {"b": float("nan"), "a": 1.123456789012345}
        result = _normalize_value(data)
        assert isinstance(result, dict)
        assert result["b"] == "__NaN__"
        assert result["a"] == round(1.123456789012345, 10)

    def test_nested_list_normalization(self):
        data = [float("inf"), float("-inf"), 3.14]
        result = _normalize_value(data)
        assert result[0] == "__Inf__"
        assert result[1] == "__-Inf__"
        assert result[2] == round(3.14, 10)

    def test_nested_tuple_normalization(self):
        data = (1.5, float("nan"))
        result = _normalize_value(data)
        assert isinstance(result, list)
        assert result[0] == round(1.5, 10)
        assert result[1] == "__NaN__"

    def test_hash_with_nan_is_deterministic(self):
        tracker = ProvenanceTracker()
        h1 = tracker.hash_record({"value": float("nan")})
        h2 = tracker.hash_record({"value": float("nan")})
        assert h1 == h2

    def test_hash_with_inf_is_deterministic(self):
        tracker = ProvenanceTracker()
        h1 = tracker.hash_record({"value": float("inf")})
        h2 = tracker.hash_record({"value": float("inf")})
        assert h1 == h2

    def test_hash_nan_differs_from_inf(self):
        tracker = ProvenanceTracker()
        h_nan = tracker.hash_record({"value": float("nan")})
        h_inf = tracker.hash_record({"value": float("inf")})
        assert h_nan != h_inf

    def test_deeply_nested_normalization(self):
        data = {"level1": {"level2": [{"level3": float("nan")}]}}
        result = _normalize_value(data)
        assert result["level1"]["level2"][0]["level3"] == "__NaN__"

    def test_dict_keys_sorted(self):
        """_normalize_value sorts dictionary keys for determinism."""
        data = {"z": 1, "a": 2, "m": 3}
        result = _normalize_value(data)
        keys = list(result.keys())
        assert keys == sorted(keys)


# -----------------------------------------------------------------------
# 11. Thread safety (concurrent add_to_chain)
# -----------------------------------------------------------------------


class TestThreadSafety:
    """Test thread-safe operations on ProvenanceTracker."""

    def test_concurrent_add_to_chain(self):
        tracker = ProvenanceTracker()
        errors = []

        def worker():
            try:
                for i in range(10):
                    tracker.add_to_chain(f"op{i}", f"in{i}", f"out{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.get_chain_length() == 50  # 5 threads * 10 ops

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
        assert tracker.get_chain_length() == 50

    def test_concurrent_records(self):
        tracker = ProvenanceTracker()
        errors = []

        def worker(entity_id):
            try:
                for i in range(10):
                    tracker.record(
                        "reconciliation_job",
                        entity_id,
                        f"op{i}",
                        f"hash{i}",
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
        """Test add_entry, add_to_chain, and record concurrently without errors."""
        tracker = ProvenanceTracker()
        errors = []

        def add_entry_worker():
            try:
                for i in range(10):
                    tracker.add_entry(f"ae_op{i}", f"ae_in{i}", f"ae_out{i}")
            except Exception as e:
                errors.append(e)

        def add_to_chain_worker():
            try:
                for i in range(10):
                    tracker.add_to_chain(
                        f"atc_op{i}", f"atc_in{i}", f"atc_out{i}"
                    )
            except Exception as e:
                errors.append(e)

        def record_worker(eid):
            try:
                for i in range(10):
                    tracker.record(
                        "reconciliation_job", eid, f"rec_op{i}", f"rh{i}"
                    )
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=add_entry_worker))
            threads.append(threading.Thread(target=add_to_chain_worker))
        for t_idx in range(2):
            threads.append(
                threading.Thread(target=record_worker, args=(f"ent{t_idx}",))
            )

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # 2*10 add_entry + 2*10 add_to_chain + 2*10 record = 60
        assert tracker.entry_count == 60

    def test_all_chain_hashes_unique_under_concurrency(self):
        """After concurrent writes, all chain hashes in global chain are unique."""
        tracker = ProvenanceTracker()
        errors = []

        def worker(prefix):
            try:
                for i in range(20):
                    tracker.add_to_chain(
                        f"{prefix}_op{i}", f"{prefix}_in{i}", f"{prefix}_out{i}"
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"w{t}",))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        chain = tracker.get_chain()
        hashes = [entry["chain_hash"] for entry in chain]
        assert len(set(hashes)) == len(hashes)  # All unique


# -----------------------------------------------------------------------
# 12. add_entry returns ProvenanceEntry
# -----------------------------------------------------------------------


class TestAddEntry:
    """Test the add_entry method."""

    def test_returns_provenance_entry(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("match_records", "in_hash", "out_hash")
        assert isinstance(entry, ProvenanceEntry)

    def test_entry_has_sha256_chain_hash(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("compare_fields", "aaa", "bbb")
        assert len(entry.chain_hash) == 64
        int(entry.chain_hash, 16)  # valid hex

    def test_entry_has_unique_id(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("detect_discrepancy", "in1", "out1")
        assert len(entry.entry_id) > 0

    def test_entry_has_correct_operation(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("resolve_conflict", "x", "y")
        assert entry.operation == "resolve_conflict"

    def test_entry_has_correct_hashes(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry(
            "create_golden_record", "input_abc", "output_def"
        )
        assert entry.input_hash == "input_abc"
        assert entry.output_hash == "output_def"

    def test_first_entry_parent_hash_is_genesis(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("reconcile", "a", "b")
        assert entry.parent_hash == ProvenanceTracker.GENESIS_HASH

    def test_entry_has_timestamp(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("validate", "in", "out")
        assert len(entry.timestamp) > 0
        assert "T" in entry.timestamp  # ISO format

    def test_entry_with_metadata(self):
        tracker = ProvenanceTracker()
        meta = {"strategy": "source_priority", "confidence": 0.95}
        entry = tracker.add_entry("resolve_conflict", "a", "b", metadata=meta)
        assert entry.metadata["strategy"] == "source_priority"
        assert entry.metadata["confidence"] == 0.95

    def test_entry_without_metadata_has_empty_dict(self):
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("reconcile", "a", "b")
        assert entry.metadata == {}


# -----------------------------------------------------------------------
# 13. Chain integrity (parent_hash linking)
# -----------------------------------------------------------------------


class TestChainIntegrity:
    """Test that each entry's parent_hash matches the previous chain_hash."""

    def test_second_entry_parent_hash_matches_first_chain_hash(self):
        tracker = ProvenanceTracker()
        entry1 = tracker.add_entry("match_records", "in1", "out1")
        entry2 = tracker.add_entry("compare_fields", "in2", "out2")
        assert entry2.parent_hash == entry1.chain_hash

    def test_three_entry_chain_links(self):
        tracker = ProvenanceTracker()
        e1 = tracker.add_entry("match_records", "a", "b")
        e2 = tracker.add_entry("compare_fields", "c", "d")
        e3 = tracker.add_entry("resolve_conflict", "e", "f")
        assert e1.parent_hash == ProvenanceTracker.GENESIS_HASH
        assert e2.parent_hash == e1.chain_hash
        assert e3.parent_hash == e2.chain_hash

    def test_chain_hash_includes_operation_and_hashes(self):
        """Verify the chain hash is computed from operation, input, output, and parent."""
        tracker = ProvenanceTracker()
        entry = tracker.add_entry("reconcile", "input_h", "output_h")
        expected = tracker._compute_chain_hash(
            ProvenanceTracker.GENESIS_HASH,
            "input_h",
            "output_h",
            "reconcile",
            entry.timestamp,
        )
        assert entry.chain_hash == expected


# -----------------------------------------------------------------------
# 14. ProvenanceEntry.to_dict()
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
# 15. get_entry()
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
# 16. get_global_chain and get_chain_length
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
        tracker.record("reconciliation_job", "j1", "match", "h")
        assert tracker.get_chain_length() == 2


# -----------------------------------------------------------------------
# 17. export_json
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
        tracker.record("reconciliation_job", "j1", "match", "h1")
        tracker.add_entry("compare_fields", "a", "b")
        exported = tracker.export_json()
        data = json.loads(exported)
        assert len(data) == 2
        assert data[0]["entity_type"] == "reconciliation_job"
        assert data[1]["operation"] == "compare_fields"

    def test_export_json_is_valid_json(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("reconcile", "in", "out")
        exported = tracker.export_json()
        # Should not raise
        parsed = json.loads(exported)
        assert isinstance(parsed, list)


# -----------------------------------------------------------------------
# 18. Thread-safe singleton via get_provenance_tracker()
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
# 19. record() entity-scoped provenance
# -----------------------------------------------------------------------


class TestRecordMethod:
    """Test the record method for entity-scoped provenance."""

    def test_record_returns_chain_hash_string(self):
        tracker = ProvenanceTracker()
        chain_hash = tracker.record(
            "reconciliation_job", "j1", "match", "abc123"
        )
        assert isinstance(chain_hash, str)
        assert len(chain_hash) == 64

    def test_record_stores_in_entity_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("reconciliation_job", "j1", "match", "h1")
        chain = tracker.get_chain("reconciliation_job", "j1")
        assert len(chain) == 1
        assert chain[0]["entity_type"] == "reconciliation_job"
        assert chain[0]["entity_id"] == "j1"
        assert chain[0]["action"] == "match"

    def test_record_with_custom_user_id(self):
        tracker = ProvenanceTracker()
        tracker.record(
            "reconciliation_job", "j1", "match", "h1", user_id="admin"
        )
        chain = tracker.get_chain("reconciliation_job", "j1")
        assert chain[0]["user_id"] == "admin"

    def test_record_default_user_id_is_system(self):
        tracker = ProvenanceTracker()
        tracker.record("reconciliation_job", "j1", "match", "h1")
        chain = tracker.get_chain("reconciliation_job", "j1")
        assert chain[0]["user_id"] == "system"

    def test_record_multiple_entities_isolated(self):
        tracker = ProvenanceTracker()
        tracker.record("reconciliation_job", "j1", "match", "h1")
        tracker.record("golden_record", "g1", "create", "h2")
        j1_chain = tracker.get_chain("reconciliation_job", "j1")
        g1_chain = tracker.get_chain("golden_record", "g1")
        assert len(j1_chain) == 1
        assert len(g1_chain) == 1
        assert j1_chain[0]["entity_type"] == "reconciliation_job"
        assert g1_chain[0]["entity_type"] == "golden_record"
