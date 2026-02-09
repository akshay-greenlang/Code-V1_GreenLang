# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-DATA-011

Tests SHA-256 chain hashing, deterministic hashing, chain verification,
entity-scoped and global chains, reset, export, thread safety,
and edge cases.
Target: 55+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
"""

from __future__ import annotations

import hashlib
import json
import threading

import pytest

from greenlang.duplicate_detector.provenance import ProvenanceTracker


# =============================================================================
# Test genesis hash
# =============================================================================


class TestProvenanceGenesis:
    """Tests for the genesis hash initialization."""

    def test_genesis_hash_is_sha256(self):
        expected = hashlib.sha256(
            b"greenlang-duplicate-detector-genesis"
        ).hexdigest()
        assert ProvenanceTracker.GENESIS_HASH == expected

    def test_genesis_hash_length(self):
        assert len(ProvenanceTracker.GENESIS_HASH) == 64

    def test_genesis_hash_is_hex(self):
        int(ProvenanceTracker.GENESIS_HASH, 16)  # Should not raise

    def test_initial_last_chain_hash_equals_genesis(self):
        tracker = ProvenanceTracker()
        assert tracker._last_chain_hash == ProvenanceTracker.GENESIS_HASH

    def test_initial_chain_store_empty(self):
        tracker = ProvenanceTracker()
        assert tracker._chain_store == {}

    def test_initial_global_chain_empty(self):
        tracker = ProvenanceTracker()
        assert tracker._global_chain == []

    def test_initial_entry_count_zero(self):
        tracker = ProvenanceTracker()
        assert tracker.entry_count == 0

    def test_initial_entity_count_zero(self):
        tracker = ProvenanceTracker()
        assert tracker.entity_count == 0


# =============================================================================
# Test hash_record
# =============================================================================


class TestHashRecord:
    """Tests for the hash_record method."""

    def test_hash_record_returns_sha256(self):
        tracker = ProvenanceTracker()
        result = tracker.hash_record({"name": "Alice", "age": 30})
        assert len(result) == 64

    def test_hash_record_deterministic(self):
        tracker = ProvenanceTracker()
        data = {"name": "Bob", "email": "bob@example.com"}
        h1 = tracker.hash_record(data)
        h2 = tracker.hash_record(data)
        assert h1 == h2

    def test_hash_record_key_order_independent(self):
        tracker = ProvenanceTracker()
        h1 = tracker.hash_record({"a": 1, "b": 2})
        h2 = tracker.hash_record({"b": 2, "a": 1})
        assert h1 == h2

    def test_hash_record_different_data_different_hash(self):
        tracker = ProvenanceTracker()
        h1 = tracker.hash_record({"name": "Alice"})
        h2 = tracker.hash_record({"name": "Bob"})
        assert h1 != h2

    def test_hash_record_empty_dict(self):
        tracker = ProvenanceTracker()
        result = tracker.hash_record({})
        assert len(result) == 64

    def test_hash_record_nested_dict(self):
        tracker = ProvenanceTracker()
        data = {"outer": {"inner": "value"}}
        result = tracker.hash_record(data)
        assert len(result) == 64

    def test_hash_record_with_list(self):
        tracker = ProvenanceTracker()
        data = {"items": [1, 2, 3]}
        result = tracker.hash_record(data)
        assert len(result) == 64

    def test_hash_record_with_none(self):
        tracker = ProvenanceTracker()
        data = {"field": None}
        result = tracker.hash_record(data)
        assert len(result) == 64

    def test_hash_record_with_numeric_types(self):
        tracker = ProvenanceTracker()
        data = {"int_field": 42, "float_field": 3.14, "bool_field": True}
        result = tracker.hash_record(data)
        assert len(result) == 64

    def test_hash_record_matches_manual_sha256(self):
        tracker = ProvenanceTracker()
        data = {"key": "value"}
        expected = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        assert tracker.hash_record(data) == expected


# =============================================================================
# Test add_to_chain
# =============================================================================


class TestAddToChain:
    """Tests for the add_to_chain method."""

    def test_add_to_chain_returns_hash(self):
        tracker = ProvenanceTracker()
        chain_hash = tracker.add_to_chain("fingerprint", "in_hash", "out_hash")
        assert len(chain_hash) == 64

    def test_add_to_chain_increments_global(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("fingerprint", "ih", "oh")
        assert tracker.entry_count == 1

    def test_add_to_chain_stores_entry(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("block", "ih", "oh", {"extra": "meta"})
        chain = tracker.get_chain()
        assert len(chain) == 1
        assert chain[0]["operation"] == "block"
        assert chain[0]["input_hash"] == "ih"
        assert chain[0]["output_hash"] == "oh"
        assert chain[0]["metadata"] == {"extra": "meta"}

    def test_add_to_chain_links_hashes(self):
        tracker = ProvenanceTracker()
        h1 = tracker.add_to_chain("fingerprint", "ih1", "oh1")
        h2 = tracker.add_to_chain("block", "ih2", "oh2")
        assert h1 != h2
        # The second chain hash should have been computed using h1 as previous
        assert tracker._last_chain_hash == h2

    def test_add_to_chain_with_metadata_none(self):
        tracker = ProvenanceTracker()
        chain_hash = tracker.add_to_chain("compare", "ih", "oh", None)
        assert len(chain_hash) == 64
        chain = tracker.get_chain()
        assert chain[0]["metadata"] == {}

    def test_add_to_chain_multiple_operations(self):
        tracker = ProvenanceTracker()
        for op in ["fingerprint", "block", "compare", "classify", "cluster", "merge"]:
            tracker.add_to_chain(op, f"ih_{op}", f"oh_{op}")
        assert tracker.entry_count == 6


# =============================================================================
# Test record (entity-scoped)
# =============================================================================


class TestRecord:
    """Tests for the record method (entity-scoped provenance)."""

    def test_record_returns_hash(self):
        tracker = ProvenanceTracker()
        h = tracker.record("dedup_job", "job_001", "fingerprint", "abc123")
        assert len(h) == 64

    def test_record_stores_in_entity_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("dedup_job", "job_001", "fingerprint", "abc123")
        chain = tracker.get_chain("dedup_job", "job_001")
        assert len(chain) == 1
        assert chain[0]["entity_type"] == "dedup_job"
        assert chain[0]["entity_id"] == "job_001"
        assert chain[0]["action"] == "fingerprint"
        assert chain[0]["data_hash"] == "abc123"

    def test_record_stores_in_global_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("dedup_job", "job_001", "fingerprint", "abc123")
        assert tracker.entry_count == 1

    def test_record_default_user_id(self):
        tracker = ProvenanceTracker()
        tracker.record("dedup_job", "job_001", "fingerprint", "abc123")
        chain = tracker.get_chain("dedup_job", "job_001")
        assert chain[0]["user_id"] == "system"

    def test_record_custom_user_id(self):
        tracker = ProvenanceTracker()
        tracker.record("dedup_job", "job_001", "create", "h", user_id="alice")
        chain = tracker.get_chain("dedup_job", "job_001")
        assert chain[0]["user_id"] == "alice"

    def test_record_multiple_entities(self):
        tracker = ProvenanceTracker()
        tracker.record("dedup_job", "job_001", "create", "h1")
        tracker.record("dedup_job", "job_002", "create", "h2")
        tracker.record("match", "match_001", "classify", "h3")

        assert tracker.entity_count == 3
        assert tracker.entry_count == 3

        assert len(tracker.get_chain("dedup_job", "job_001")) == 1
        assert len(tracker.get_chain("dedup_job", "job_002")) == 1
        assert len(tracker.get_chain("match", "match_001")) == 1

    def test_record_same_entity_multiple_actions(self):
        tracker = ProvenanceTracker()
        tracker.record("dedup_job", "job_001", "fingerprint", "h1")
        tracker.record("dedup_job", "job_001", "block", "h2")
        tracker.record("dedup_job", "job_001", "compare", "h3")

        chain = tracker.get_chain("dedup_job", "job_001")
        assert len(chain) == 3
        assert chain[0]["action"] == "fingerprint"
        assert chain[1]["action"] == "block"
        assert chain[2]["action"] == "compare"


# =============================================================================
# Test verify_chain
# =============================================================================


class TestVerifyChain:
    """Tests for chain verification."""

    def test_verify_empty_chain_returns_true(self):
        tracker = ProvenanceTracker()
        valid, chain = tracker.verify_chain()
        assert valid is True
        assert chain == []

    def test_verify_entity_empty_chain_returns_true(self):
        tracker = ProvenanceTracker()
        valid, chain = tracker.verify_chain("nonexistent", "id")
        assert valid is True
        assert chain == []

    def test_verify_global_chain_after_add(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("fingerprint", "ih", "oh")
        valid, chain = tracker.verify_chain()
        assert valid is True
        assert len(chain) == 1

    def test_verify_entity_chain_after_record(self):
        tracker = ProvenanceTracker()
        tracker.record("dedup_job", "j1", "fingerprint", "h1")
        valid, chain = tracker.verify_chain("dedup_job", "j1")
        assert valid is True
        assert len(chain) == 1

    def test_verify_chain_multiple_entries(self):
        tracker = ProvenanceTracker()
        for i in range(10):
            tracker.record("dedup_job", "j1", f"op_{i}", f"h_{i}")
        valid, chain = tracker.verify_chain("dedup_job", "j1")
        assert valid is True
        assert len(chain) == 10

    def test_verify_chain_detects_missing_chain_hash(self):
        tracker = ProvenanceTracker()
        tracker.record("dedup_job", "j1", "fingerprint", "h1")
        # Tamper: remove chain_hash from first entry
        with tracker._lock:
            tracker._chain_store["dedup_job:j1"][0]["chain_hash"] = ""
        valid, chain = tracker.verify_chain("dedup_job", "j1")
        assert valid is False

    def test_verify_global_chain_detects_missing_field(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("op", "ih", "oh")
        # Tamper: remove a required field
        with tracker._lock:
            del tracker._global_chain[0]["input_hash"]
        valid, chain = tracker.verify_chain()
        assert valid is False


# =============================================================================
# Test get_chain / get_global_chain
# =============================================================================


class TestGetChain:
    """Tests for chain retrieval methods."""

    def test_get_chain_global(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("op1", "ih1", "oh1")
        tracker.record("job", "j1", "op2", "h2")
        chain = tracker.get_chain()
        assert len(chain) == 2

    def test_get_chain_entity_scoped(self):
        tracker = ProvenanceTracker()
        tracker.record("job", "j1", "create", "h1")
        tracker.record("job", "j2", "create", "h2")
        chain = tracker.get_chain("job", "j1")
        assert len(chain) == 1

    def test_get_chain_nonexistent_entity(self):
        tracker = ProvenanceTracker()
        chain = tracker.get_chain("nonexistent", "id")
        assert chain == []

    def test_get_global_chain_limit(self):
        tracker = ProvenanceTracker()
        for i in range(20):
            tracker.add_to_chain(f"op_{i}", f"ih_{i}", f"oh_{i}")
        result = tracker.get_global_chain(limit=5)
        assert len(result) == 5

    def test_get_global_chain_newest_first(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("first_op", "ih1", "oh1")
        tracker.add_to_chain("second_op", "ih2", "oh2")
        result = tracker.get_global_chain(limit=10)
        assert result[0]["operation"] == "second_op"
        assert result[1]["operation"] == "first_op"

    def test_get_global_chain_default_limit(self):
        tracker = ProvenanceTracker()
        for i in range(150):
            tracker.add_to_chain(f"op_{i}", f"ih_{i}", f"oh_{i}")
        result = tracker.get_global_chain()
        assert len(result) == 100  # default limit


# =============================================================================
# Test chain length tracking
# =============================================================================


class TestChainLength:
    """Tests for chain length via get_chain_length and entry_count."""

    def test_chain_length_initial(self):
        tracker = ProvenanceTracker()
        assert tracker.get_chain_length() == 0

    def test_chain_length_after_add(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("op", "ih", "oh")
        assert tracker.get_chain_length() == 1

    def test_chain_length_after_record(self):
        tracker = ProvenanceTracker()
        tracker.record("job", "j1", "create", "h")
        assert tracker.get_chain_length() == 1

    def test_chain_length_after_multiple(self):
        tracker = ProvenanceTracker()
        for i in range(25):
            tracker.add_to_chain(f"op_{i}", "ih", "oh")
        assert tracker.get_chain_length() == 25

    def test_entry_count_matches_chain_length(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("op", "ih", "oh")
        tracker.record("job", "j1", "create", "h")
        assert tracker.entry_count == tracker.get_chain_length()


# =============================================================================
# Test reset
# =============================================================================


class TestReset:
    """Tests for the reset method."""

    def test_reset_clears_global_chain(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("op", "ih", "oh")
        tracker.reset()
        assert tracker.get_chain_length() == 0

    def test_reset_clears_entity_chains(self):
        tracker = ProvenanceTracker()
        tracker.record("job", "j1", "create", "h")
        tracker.reset()
        assert tracker.entity_count == 0
        assert tracker.get_chain("job", "j1") == []

    def test_reset_restores_genesis_hash(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("op", "ih", "oh")
        tracker.reset()
        assert tracker._last_chain_hash == ProvenanceTracker.GENESIS_HASH

    def test_reset_then_add(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("op1", "ih1", "oh1")
        tracker.reset()
        tracker.add_to_chain("op2", "ih2", "oh2")
        assert tracker.get_chain_length() == 1
        chain = tracker.get_chain()
        assert chain[0]["operation"] == "op2"


# =============================================================================
# Test build_hash
# =============================================================================


class TestBuildHash:
    """Tests for the build_hash method."""

    def test_build_hash_dict(self):
        tracker = ProvenanceTracker()
        result = tracker.build_hash({"key": "value"})
        assert len(result) == 64

    def test_build_hash_list(self):
        tracker = ProvenanceTracker()
        result = tracker.build_hash([1, 2, 3])
        assert len(result) == 64

    def test_build_hash_string(self):
        tracker = ProvenanceTracker()
        result = tracker.build_hash("hello")
        assert len(result) == 64

    def test_build_hash_deterministic(self):
        tracker = ProvenanceTracker()
        h1 = tracker.build_hash({"x": 1})
        h2 = tracker.build_hash({"x": 1})
        assert h1 == h2


# =============================================================================
# Test export_json
# =============================================================================


class TestExportJson:
    """Tests for the export_json method."""

    def test_export_json_empty(self):
        tracker = ProvenanceTracker()
        result = tracker.export_json()
        assert json.loads(result) == []

    def test_export_json_with_entries(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("op", "ih", "oh")
        tracker.record("job", "j1", "create", "h")
        result = json.loads(tracker.export_json())
        assert len(result) == 2

    def test_export_json_valid_json(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("op", "ih", "oh")
        # Should not raise
        json.loads(tracker.export_json())


# =============================================================================
# Test thread safety
# =============================================================================


class TestProvenanceThreadSafety:
    """Verify thread safety of ProvenanceTracker operations."""

    def test_concurrent_add_to_chain(self):
        tracker = ProvenanceTracker()
        errors = []
        barrier = threading.Barrier(10)

        def worker(idx):
            try:
                barrier.wait()
                for i in range(50):
                    tracker.add_to_chain(f"op_{idx}_{i}", f"ih_{idx}_{i}", f"oh_{idx}_{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.get_chain_length() == 500  # 10 threads * 50 ops

    def test_concurrent_record(self):
        tracker = ProvenanceTracker()
        errors = []

        def worker(idx):
            try:
                for i in range(20):
                    tracker.record("job", f"j_{idx}", f"op_{i}", f"h_{idx}_{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.entry_count == 100  # 5 threads * 20 ops

    def test_concurrent_read_and_write(self):
        tracker = ProvenanceTracker()
        errors = []

        def writer():
            try:
                for i in range(50):
                    tracker.add_to_chain(f"write_{i}", f"ih_{i}", f"oh_{i}")
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(50):
                    tracker.get_chain()
                    tracker.get_chain_length()
                    tracker.verify_chain()
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


# =============================================================================
# Test deterministic hashing (same input = same hash)
# =============================================================================


class TestDeterministicHashing:
    """Verify deterministic behavior across tracker instances."""

    def test_hash_record_same_across_instances(self):
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        data = {"name": "Alice", "amount": 100.5}
        assert t1.hash_record(data) == t2.hash_record(data)

    def test_build_hash_same_across_instances(self):
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        data = [1, "two", 3.0]
        assert t1.build_hash(data) == t2.build_hash(data)

    def test_hash_record_with_datetime_uses_default_str(self):
        """datetime values should be serialized via default=str."""
        from datetime import datetime, timezone
        tracker = ProvenanceTracker()
        data = {"ts": datetime(2025, 1, 1, tzinfo=timezone.utc)}
        result = tracker.hash_record(data)
        assert len(result) == 64

    def test_hash_record_datetime_deterministic(self):
        from datetime import datetime, timezone
        tracker = ProvenanceTracker()
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        data = {"ts": ts}
        h1 = tracker.hash_record(data)
        h2 = tracker.hash_record(data)
        assert h1 == h2
