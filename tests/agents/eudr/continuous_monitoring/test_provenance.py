# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-EUDR-033

Tests SHA-256 hash chain creation, verification, deterministic hashing,
chain building, and metadata recording for the Continuous Monitoring
Agent's audit trail.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.continuous_monitoring.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


class TestProvenanceTracker:
    """Test suite for ProvenanceTracker."""

    def test_genesis_hash_length(self):
        assert len(GENESIS_HASH) == 64

    def test_genesis_hash_all_zeros(self):
        assert GENESIS_HASH == "0" * 64

    def test_init_default(self, provenance_tracker):
        assert provenance_tracker._algorithm == "sha256"

    def test_init_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported"):
            ProvenanceTracker(algorithm="md5")

    def test_init_sha256_explicit(self):
        tracker = ProvenanceTracker(algorithm="sha256")
        assert tracker._algorithm == "sha256"

    def test_compute_hash_deterministic(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash({"key": "value"})
        h2 = provenance_tracker.compute_hash({"key": "value"})
        assert h1 == h2

    def test_compute_hash_length(self, provenance_tracker):
        h = provenance_tracker.compute_hash({"test": 123})
        assert len(h) == 64

    def test_compute_hash_hex_characters(self, provenance_tracker):
        h = provenance_tracker.compute_hash({"data": "test"})
        assert all(c in "0123456789abcdef" for c in h)

    def test_compute_hash_different_inputs(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash({"a": 1})
        h2 = provenance_tracker.compute_hash({"a": 2})
        assert h1 != h2

    def test_compute_hash_key_order_independent(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash({"a": 1, "b": 2})
        h2 = provenance_tracker.compute_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_compute_hash_nested_dict(self, provenance_tracker):
        data = {"outer": {"inner": "value", "num": 42}}
        h = provenance_tracker.compute_hash(data)
        assert len(h) == 64

    def test_compute_hash_with_decimal(self, provenance_tracker):
        from decimal import Decimal
        data = {"score": Decimal("92.5")}
        h = provenance_tracker.compute_hash(data)
        assert len(h) == 64

    def test_compute_hash_with_datetime(self, provenance_tracker):
        from datetime import datetime, timezone
        data = {"timestamp": datetime.now(timezone.utc)}
        h = provenance_tracker.compute_hash(data)
        assert len(h) == 64

    def test_compute_hash_bytes(self, provenance_tracker):
        h = provenance_tracker.compute_hash_bytes(b"test data")
        assert len(h) == 64

    def test_compute_hash_bytes_deterministic(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash_bytes(b"identical")
        h2 = provenance_tracker.compute_hash_bytes(b"identical")
        assert h1 == h2

    def test_compute_hash_bytes_different_input(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash_bytes(b"data_a")
        h2 = provenance_tracker.compute_hash_bytes(b"data_b")
        assert h1 != h2

    def test_create_entry(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="monitor", source="supply_chain",
            input_hash=GENESIS_HASH, output_hash="abc123",
        )
        assert entry["step"] == "monitor"
        assert entry["input_hash"] == GENESIS_HASH
        assert entry["output_hash"] == "abc123"

    def test_create_entry_default_actor(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="detect", source="deforestation",
            input_hash=GENESIS_HASH, output_hash="xyz789",
        )
        assert entry["actor"] == "AGENT-EUDR-033"

    def test_create_entry_custom_actor(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            step="audit", source="compliance",
            input_hash=GENESIS_HASH, output_hash="def456",
            actor="user:admin",
        )
        assert entry["actor"] == "user:admin"

    def test_record(self, provenance_tracker):
        entry = provenance_tracker.record(
            entity_type="supply_chain_event",
            action="create",
            entity_id="SCE-001",
            actor="AGENT-EUDR-033",
        )
        assert entry["step"] == "supply_chain_event:create"
        assert entry["source"] == "SCE-001"

    def test_record_with_metadata(self, provenance_tracker):
        entry = provenance_tracker.record(
            "deforestation_alert", "escalate", "DEF-001", "AGENT-EUDR-033",
            metadata={"priority": "critical", "area_hectares": "15.3"},
        )
        assert entry["metadata"]["priority"] == "critical"

    def test_chain_accumulation(self, provenance_tracker):
        provenance_tracker.record("a", "create", "1", "system")
        provenance_tracker.record("b", "update", "2", "system")
        chain = provenance_tracker.get_chain()
        assert len(chain) == 2

    def test_chain_accumulation_three_entries(self, provenance_tracker):
        provenance_tracker.record("a", "create", "1", "system")
        provenance_tracker.record("b", "update", "2", "system")
        provenance_tracker.record("c", "escalate", "3", "system")
        chain = provenance_tracker.get_chain()
        assert len(chain) == 3

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

    def test_verify_chain_single_entry(self, provenance_tracker):
        entries = [{"input_hash": GENESIS_HASH, "output_hash": "aaa"}]
        assert provenance_tracker.verify_chain(entries) is True

    def test_build_chain(self, provenance_tracker):
        steps = [
            {"step": "s1", "source": "supply_chain", "data": {"k": "v1"}},
            {"step": "s2", "source": "deforestation", "data": {"k": "v2"}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert len(chain) == 2
        assert chain[0]["input_hash"] == GENESIS_HASH
        assert chain[1]["input_hash"] == chain[0]["output_hash"]

    def test_build_chain_three_steps(self, provenance_tracker):
        steps = [
            {"step": "s1", "source": "src", "data": {"a": 1}},
            {"step": "s2", "source": "src", "data": {"b": 2}},
            {"step": "s3", "source": "src", "data": {"c": 3}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert len(chain) == 3
        assert chain[2]["input_hash"] == chain[1]["output_hash"]

    def test_build_chain_verifiable(self, provenance_tracker):
        steps = [
            {"step": "s1", "source": "src", "data": {"x": 1}},
            {"step": "s2", "source": "src", "data": {"y": 2}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert provenance_tracker.verify_chain(chain) is True

    def test_reset(self, provenance_tracker):
        provenance_tracker.record("a", "create", "1", "system")
        provenance_tracker.reset()
        assert len(provenance_tracker.get_chain()) == 0

    def test_get_chain_returns_copy(self, provenance_tracker):
        provenance_tracker.record("a", "create", "1", "system")
        chain1 = provenance_tracker.get_chain()
        chain2 = provenance_tracker.get_chain()
        assert chain1 == chain2
        assert chain1 is not chain2
