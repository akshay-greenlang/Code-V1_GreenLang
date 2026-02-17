# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-DATA-012

Tests SHA-256 chain hashing, record/verify/get operations, reset,
export, threading safety, and genesis hash determinism.
Target: 20+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

from __future__ import annotations

import hashlib
import json
import threading

import pytest

from greenlang.missing_value_imputer.provenance import ProvenanceTracker


class TestGenesisHash:
    """Test the GENESIS_HASH class constant."""

    def test_genesis_hash_is_sha256(self):
        expected = hashlib.sha256(
            b"greenlang-missing-value-imputer-genesis"
        ).hexdigest()
        assert ProvenanceTracker.GENESIS_HASH == expected

    def test_genesis_hash_length(self):
        assert len(ProvenanceTracker.GENESIS_HASH) == 64


class TestHashRecord:
    """Test the hash_record method."""

    def test_deterministic_hashing(self):
        tracker = ProvenanceTracker()
        data = {"a": 1, "b": "hello"}
        h1 = tracker.hash_record(data)
        h2 = tracker.hash_record(data)
        assert h1 == h2

    def test_key_order_independent(self):
        tracker = ProvenanceTracker()
        h1 = tracker.hash_record({"b": 2, "a": 1})
        h2 = tracker.hash_record({"a": 1, "b": 2})
        assert h1 == h2

    def test_different_data_different_hash(self):
        tracker = ProvenanceTracker()
        h1 = tracker.hash_record({"x": 1})
        h2 = tracker.hash_record({"x": 2})
        assert h1 != h2

    def test_hash_is_64_hex_chars(self):
        tracker = ProvenanceTracker()
        h = tracker.hash_record({"key": "val"})
        assert len(h) == 64
        int(h, 16)  # should not raise


class TestBuildHash:
    """Test the build_hash method."""

    def test_build_hash_list(self):
        tracker = ProvenanceTracker()
        h = tracker.build_hash([1, 2, 3])
        assert len(h) == 64

    def test_build_hash_deterministic(self):
        tracker = ProvenanceTracker()
        h1 = tracker.build_hash({"x": 42})
        h2 = tracker.build_hash({"x": 42})
        assert h1 == h2


class TestAddToChain:
    """Test the add_to_chain method."""

    def test_returns_chain_hash(self):
        tracker = ProvenanceTracker()
        chain_hash = tracker.add_to_chain("analyze", "abc", "def")
        assert len(chain_hash) == 64

    def test_chain_increments_length(self):
        tracker = ProvenanceTracker()
        assert tracker.get_chain_length() == 0
        tracker.add_to_chain("op1", "in1", "out1")
        assert tracker.get_chain_length() == 1
        tracker.add_to_chain("op2", "in2", "out2")
        assert tracker.get_chain_length() == 2

    def test_chain_with_metadata(self):
        tracker = ProvenanceTracker()
        chain_hash = tracker.add_to_chain(
            "impute", "aaa", "bbb", metadata={"method": "mean"}
        )
        assert len(chain_hash) == 64
        chain = tracker.get_chain()
        assert chain[0]["metadata"]["method"] == "mean"


class TestRecord:
    """Test the record method."""

    def test_record_creates_entity_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("job", "j001", "analyze", "hash123")
        chain = tracker.get_chain("job", "j001")
        assert len(chain) == 1
        assert chain[0]["entity_type"] == "job"
        assert chain[0]["entity_id"] == "j001"
        assert chain[0]["action"] == "analyze"

    def test_record_default_user_id(self):
        tracker = ProvenanceTracker()
        tracker.record("job", "j002", "create", "h1")
        chain = tracker.get_chain("job", "j002")
        assert chain[0]["user_id"] == "system"

    def test_record_custom_user_id(self):
        tracker = ProvenanceTracker()
        tracker.record("job", "j003", "create", "h2", user_id="admin")
        chain = tracker.get_chain("job", "j003")
        assert chain[0]["user_id"] == "admin"

    def test_record_adds_to_global_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("rule", "r1", "create", "h1")
        assert tracker.get_chain_length() == 1

    def test_multiple_entities(self):
        tracker = ProvenanceTracker()
        tracker.record("job", "j1", "create", "h1")
        tracker.record("job", "j2", "create", "h2")
        tracker.record("rule", "r1", "create", "h3")
        assert tracker.entity_count == 3
        assert tracker.entry_count == 3


class TestVerifyChain:
    """Test the verify_chain method."""

    def test_empty_chain_is_valid(self):
        tracker = ProvenanceTracker()
        is_valid, chain = tracker.verify_chain("job", "nonexist")
        assert is_valid is True
        assert chain == []

    def test_single_entry_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("job", "j1", "create", "h1")
        is_valid, chain = tracker.verify_chain("job", "j1")
        assert is_valid is True
        assert len(chain) == 1

    def test_multi_entry_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("job", "j1", "analyze", "h1")
        tracker.record("job", "j1", "impute", "h2")
        tracker.record("job", "j1", "validate", "h3")
        is_valid, chain = tracker.verify_chain("job", "j1")
        assert is_valid is True
        assert len(chain) == 3

    def test_global_chain_verify(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("op1", "in1", "out1")
        tracker.record("job", "j1", "create", "h1")
        is_valid, chain = tracker.verify_chain()
        assert is_valid is True
        assert len(chain) == 2


class TestGetChainAndGlobalChain:
    """Test get_chain, get_global_chain, and get_chain_length."""

    def test_get_global_chain_newest_first(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("first", "a", "b")
        tracker.add_to_chain("second", "c", "d")
        global_chain = tracker.get_global_chain(limit=10)
        assert global_chain[0]["operation"] == "second"
        assert global_chain[1]["operation"] == "first"

    def test_get_global_chain_with_limit(self):
        tracker = ProvenanceTracker()
        for i in range(20):
            tracker.add_to_chain(f"op{i}", "a", "b")
        global_chain = tracker.get_global_chain(limit=5)
        assert len(global_chain) == 5


class TestProperties:
    """Test entry_count and entity_count properties."""

    def test_entry_count_zero(self):
        tracker = ProvenanceTracker()
        assert tracker.entry_count == 0

    def test_entity_count_zero(self):
        tracker = ProvenanceTracker()
        assert tracker.entity_count == 0

    def test_entry_count_after_operations(self):
        tracker = ProvenanceTracker()
        tracker.record("job", "j1", "create", "h1")
        tracker.add_to_chain("op", "a", "b")
        assert tracker.entry_count == 2


class TestReset:
    """Test the reset method."""

    def test_reset_clears_chains(self):
        tracker = ProvenanceTracker()
        tracker.record("job", "j1", "create", "h1")
        tracker.add_to_chain("op", "a", "b")
        assert tracker.entry_count == 2
        tracker.reset()
        assert tracker.entry_count == 0
        assert tracker.entity_count == 0

    def test_reset_restores_genesis(self):
        tracker = ProvenanceTracker()
        tracker.add_to_chain("op", "a", "b")
        tracker.reset()
        assert tracker._last_chain_hash == ProvenanceTracker.GENESIS_HASH


class TestExportJson:
    """Test the export_json method."""

    def test_export_json_empty(self):
        tracker = ProvenanceTracker()
        exported = tracker.export_json()
        data = json.loads(exported)
        assert data == []

    def test_export_json_with_entries(self):
        tracker = ProvenanceTracker()
        tracker.record("job", "j1", "create", "h1")
        exported = tracker.export_json()
        data = json.loads(exported)
        assert len(data) == 1
        assert data[0]["entity_type"] == "job"


class TestThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_records(self):
        tracker = ProvenanceTracker()
        errors = []

        def worker(entity_id):
            try:
                for i in range(10):
                    tracker.record("job", entity_id, f"op{i}", f"hash{i}")
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

    def test_concurrent_add_to_chain(self):
        tracker = ProvenanceTracker()
        errors = []

        def worker():
            try:
                for i in range(10):
                    tracker.add_to_chain(f"op{i}", "a", "b")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.get_chain_length() == 50
