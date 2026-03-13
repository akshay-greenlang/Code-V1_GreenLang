# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-EUDR-032

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.grievance_mechanism_manager.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


class TestProvenanceTracker:
    def test_genesis_hash_length(self):
        assert len(GENESIS_HASH) == 64

    def test_genesis_hash_all_zeros(self):
        assert GENESIS_HASH == "0" * 64

    def test_init_default(self, provenance_tracker):
        assert provenance_tracker._algorithm == "sha256"

    def test_init_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported"):
            ProvenanceTracker(algorithm="md5")

    def test_compute_hash_deterministic(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash({"key": "value"})
        h2 = provenance_tracker.compute_hash({"key": "value"})
        assert h1 == h2

    def test_compute_hash_length(self, provenance_tracker):
        h = provenance_tracker.compute_hash({"test": 123})
        assert len(h) == 64

    def test_compute_hash_different_inputs(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash({"a": 1})
        h2 = provenance_tracker.compute_hash({"a": 2})
        assert h1 != h2

    def test_compute_hash_bytes(self, provenance_tracker):
        h = provenance_tracker.compute_hash_bytes(b"test data")
        assert len(h) == 64

    def test_create_entry(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="test", source="src", input_hash=GENESIS_HASH,
            output_hash="abc123",
        )
        assert entry["step"] == "test"
        assert entry["input_hash"] == GENESIS_HASH
        assert entry["output_hash"] == "abc123"

    def test_record(self, provenance_tracker):
        entry = provenance_tracker.record(
            entity_type="analytics",
            action="create",
            entity_id="ana-001",
            actor="AGENT-EUDR-032",
        )
        assert entry["step"] == "analytics:create"
        assert entry["source"] == "ana-001"

    def test_chain_accumulation(self, provenance_tracker):
        provenance_tracker.record("a", "create", "1", "system")
        provenance_tracker.record("b", "update", "2", "system")
        chain = provenance_tracker.get_chain()
        assert len(chain) == 2

    def test_verify_chain_empty(self, provenance_tracker):
        assert provenance_tracker.verify_chain([]) is True

    def test_verify_chain_valid(self, provenance_tracker):
        provenance_tracker.record("a", "create", "1", "system")
        provenance_tracker.record("b", "update", "2", "system")
        chain = provenance_tracker.get_chain()
        assert provenance_tracker.verify_chain(chain) is True

    def test_verify_chain_broken(self, provenance_tracker):
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "aaa"},
            {"input_hash": "bbb", "output_hash": "ccc"},
        ]
        assert provenance_tracker.verify_chain(entries) is False

    def test_build_chain(self, provenance_tracker):
        steps = [
            {"step": "s1", "source": "src", "data": {"k": "v1"}},
            {"step": "s2", "source": "src", "data": {"k": "v2"}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert len(chain) == 2
        assert chain[0]["input_hash"] == GENESIS_HASH
        assert chain[1]["input_hash"] == chain[0]["output_hash"]

    def test_reset(self, provenance_tracker):
        provenance_tracker.record("a", "create", "1", "system")
        provenance_tracker.reset()
        assert len(provenance_tracker.get_chain()) == 0

    def test_record_with_metadata(self, provenance_tracker):
        entry = provenance_tracker.record(
            "test", "create", "id-1", "actor",
            metadata={"extra": "data"},
        )
        assert entry["metadata"]["extra"] == "data"
