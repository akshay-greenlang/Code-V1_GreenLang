# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceTracker - AGENT-EUDR-039

Tests SHA-256 hash computation, chain creation, verification,
and genesis hash constant for customs declaration provenance.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from greenlang.agents.eudr.customs_declaration_support.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


class TestGenesisHash:
    def test_genesis_hash_is_64_zeros(self):
        assert GENESIS_HASH == "0" * 64

    def test_genesis_hash_length(self):
        assert len(GENESIS_HASH) == 64


class TestProvenanceTrackerInit:
    def test_default_init(self, provenance_tracker):
        assert provenance_tracker._algorithm == "sha256"

    def test_unsupported_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            ProvenanceTracker(algorithm="md5")

    def test_chain_starts_empty(self, provenance_tracker):
        assert provenance_tracker.get_chain() == []


class TestComputeHash:
    def test_returns_64_char_hex(self, provenance_tracker):
        h = provenance_tracker.compute_hash({"key": "value"})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self, provenance_tracker):
        d = {"declaration_id": "DECL-001", "mrn": "26NL0003960000001A"}
        h1 = provenance_tracker.compute_hash(d)
        h2 = provenance_tracker.compute_hash(d)
        assert h1 == h2

    def test_key_order_independence(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash({"a": 1, "b": 2})
        h2 = provenance_tracker.compute_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_data_different_hash(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash({"mrn": "26NL0003960000001A"})
        h2 = provenance_tracker.compute_hash({"mrn": "26DE0003960000002B"})
        assert h1 != h2

    def test_manual_verification(self, provenance_tracker):
        data = {"cn_code": "18010000", "commodity": "cocoa"}
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        actual = provenance_tracker.compute_hash(data)
        assert actual == expected

    def test_empty_dict(self, provenance_tracker):
        h = provenance_tracker.compute_hash({})
        assert len(h) == 64


class TestComputeHashBytes:
    def test_returns_64_char_hex(self, provenance_tracker):
        h = provenance_tracker.compute_hash_bytes(b"customs_data")
        assert len(h) == 64

    def test_deterministic(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash_bytes(b"sad_form_content")
        h2 = provenance_tracker.compute_hash_bytes(b"sad_form_content")
        assert h1 == h2

    def test_different_bytes_different_hash(self, provenance_tracker):
        h1 = provenance_tracker.compute_hash_bytes(b"declaration_a")
        h2 = provenance_tracker.compute_hash_bytes(b"declaration_b")
        assert h1 != h2


class TestCreateEntry:
    def test_basic_entry(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            "generate_declaration", "eudr-039", GENESIS_HASH, "abc123" * 10 + "abcd")
        assert entry["step"] == "generate_declaration"
        assert entry["source"] == "eudr-039"
        assert entry["input_hash"] == GENESIS_HASH

    def test_entry_appends_to_chain(self, provenance_tracker):
        provenance_tracker.create_entry("s1", "src", "ih", "oh")
        assert len(provenance_tracker.get_chain()) == 1

    def test_custom_actor(self, provenance_tracker):
        entry = provenance_tracker.create_entry(
            "step", "src", "ih", "oh", actor="customs-agent")
        assert entry["actor"] == "customs-agent"

    def test_default_actor(self, provenance_tracker):
        entry = provenance_tracker.create_entry("step", "src", "ih", "oh")
        assert entry["actor"] == "AGENT-EUDR-039"


class TestRecord:
    def test_basic_record(self, provenance_tracker):
        entry = provenance_tracker.record(
            entity_type="declaration", action="create",
            entity_id="DECL-001", actor="OP-001")
        assert entry["step"] == "declaration:create"
        assert entry["source"] == "DECL-001"

    def test_record_with_metadata(self, provenance_tracker):
        entry = provenance_tracker.record(
            entity_type="declaration", action="submit",
            entity_id="DECL-001", actor="OP-001",
            metadata={"customs_system": "ncts", "port": "NLRTM"})
        assert entry["metadata"]["customs_system"] == "ncts"
        assert entry["metadata"]["port"] == "NLRTM"

    def test_record_chains_hashes(self, provenance_tracker):
        e1 = provenance_tracker.record("declaration", "create", "D1", "A1")
        e2 = provenance_tracker.record("declaration", "submit", "D1", "A1")
        assert e2["input_hash"] == e1["output_hash"]

    def test_first_record_uses_genesis(self, provenance_tracker):
        entry = provenance_tracker.record("declaration", "create", "D1", "A1")
        assert entry["input_hash"] == GENESIS_HASH


class TestVerifyChain:
    def test_empty_chain_valid(self, provenance_tracker):
        assert provenance_tracker.verify_chain([]) is True

    def test_single_entry_valid(self, provenance_tracker):
        entries = [{"input_hash": GENESIS_HASH, "output_hash": "abc"}]
        assert provenance_tracker.verify_chain(entries) is True

    def test_valid_chain(self, provenance_tracker):
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "hash1"},
            {"input_hash": "hash1", "output_hash": "hash2"},
            {"input_hash": "hash2", "output_hash": "hash3"},
        ]
        assert provenance_tracker.verify_chain(entries) is True

    def test_broken_chain(self, provenance_tracker):
        entries = [
            {"input_hash": GENESIS_HASH, "output_hash": "hash1"},
            {"input_hash": "WRONG", "output_hash": "hash2"},
        ]
        assert provenance_tracker.verify_chain(entries) is False


class TestBuildChain:
    def test_build_from_steps(self, provenance_tracker):
        steps = [
            {"step": "map_cn_code", "source": "s1", "data": {"cn": "18010000"}},
            {"step": "validate_hs", "source": "s2", "data": {"hs": "180100"}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert len(chain) == 2
        assert chain[0]["input_hash"] == GENESIS_HASH
        assert chain[1]["input_hash"] == chain[0]["output_hash"]

    def test_build_chain_verifiable(self, provenance_tracker):
        steps = [
            {"step": "s1", "source": "a", "data": {"x": 1}},
            {"step": "s2", "source": "b", "data": {"x": 2}},
            {"step": "s3", "source": "c", "data": {"x": 3}},
        ]
        chain = provenance_tracker.build_chain(steps)
        assert provenance_tracker.verify_chain(chain) is True

    def test_build_chain_custom_genesis(self, provenance_tracker):
        custom_genesis = "f" * 64
        steps = [{"step": "s1", "source": "a", "data": {"x": 1}}]
        chain = provenance_tracker.build_chain(steps, genesis_hash=custom_genesis)
        assert chain[0]["input_hash"] == custom_genesis


class TestGetChainAndReset:
    def test_get_chain_returns_copy(self, provenance_tracker):
        provenance_tracker.record("declaration", "create", "D1", "A1")
        chain = provenance_tracker.get_chain()
        assert len(chain) == 1
        chain.pop()
        assert len(provenance_tracker.get_chain()) == 1

    def test_reset_clears_chain(self, provenance_tracker):
        provenance_tracker.record("declaration", "create", "D1", "A1")
        assert len(provenance_tracker.get_chain()) == 1
        provenance_tracker.reset()
        assert len(provenance_tracker.get_chain()) == 0
