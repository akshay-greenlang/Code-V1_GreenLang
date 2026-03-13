# -*- coding: utf-8 -*-
"""
Unit tests for Provenance Tracking -- AGENT-EUDR-024

Tests provenance chain integrity, SHA-256 hash computation, entity type
and action validation, chain verification, genesis hash, parent hash
linking, entry ordering, deterministic hash generation, and tracker
singleton management.

Target: ~50 tests
Author: GreenLang Platform Team
Date: March 2026
"""

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.third_party_audit_manager.provenance import (
    ProvenanceRecord,
    ProvenanceTracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
    get_tracker,
    reset_tracker,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    SHA256_HEX_LENGTH,
    compute_test_hash,
)


class TestProvenanceTrackerInit:
    """Test tracker initialization."""

    def test_tracker_creation(self, provenance_tracker):
        assert provenance_tracker is not None

    def test_tracker_has_genesis_hash(self, provenance_tracker):
        assert provenance_tracker.genesis_hash is not None
        assert len(provenance_tracker.genesis_hash) > 0

    def test_tracker_starts_empty(self, provenance_tracker):
        assert provenance_tracker.entry_count == 0

    def test_singleton_pattern(self):
        reset_tracker()
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    def test_reset_clears_tracker(self):
        tracker = get_tracker()
        tracker.record("audit", "create", "AUD-001")
        reset_tracker()
        new_tracker = get_tracker()
        assert new_tracker.entry_count == 0


class TestProvenanceRecording:
    """Test provenance entry recording."""

    def test_record_single_entry(self, provenance_tracker):
        entry = provenance_tracker.record("audit", "create", "AUD-001")
        assert entry is not None
        assert provenance_tracker.entry_count == 1

    def test_record_returns_hash(self, provenance_tracker):
        entry = provenance_tracker.record("audit", "create", "AUD-001")
        assert entry.hash_value is not None
        assert len(entry.hash_value) == SHA256_HEX_LENGTH

    def test_record_links_to_genesis(self, provenance_tracker):
        entry = provenance_tracker.record("audit", "create", "AUD-001")
        assert entry.parent_hash == provenance_tracker.genesis_hash

    def test_second_entry_links_to_first(self, provenance_tracker):
        e1 = provenance_tracker.record("audit", "create", "AUD-001")
        e2 = provenance_tracker.record("audit", "update", "AUD-001")
        assert e2.parent_hash == e1.hash_value

    def test_chain_integrity(self, provenance_tracker):
        provenance_tracker.record("audit", "create", "AUD-001")
        provenance_tracker.record("nc", "classify", "NC-001")
        provenance_tracker.record("car", "issue", "CAR-001")
        assert provenance_tracker.verify_chain() is True

    def test_entry_has_timestamp(self, provenance_tracker):
        entry = provenance_tracker.record("audit", "create", "AUD-001")
        assert entry.timestamp is not None

    def test_entry_has_entity_type(self, provenance_tracker):
        entry = provenance_tracker.record("audit", "create", "AUD-001")
        assert entry.entity_type == "audit"

    def test_entry_has_action(self, provenance_tracker):
        entry = provenance_tracker.record("audit", "create", "AUD-001")
        assert entry.action == "create"

    def test_entry_has_entity_id(self, provenance_tracker):
        entry = provenance_tracker.record("audit", "create", "AUD-001")
        assert entry.entity_id == "AUD-001"


class TestValidEntityTypes:
    """Test valid entity type validation."""

    @pytest.mark.parametrize("entity_type", [
        "audit", "auditor", "checklist", "evidence", "nc",
        "car", "certificate", "report", "authority_interaction", "analytics",
    ])
    def test_valid_entity_types(self, provenance_tracker, entity_type):
        entry = provenance_tracker.record(entity_type, "create", "TEST-001")
        assert entry.entity_type == entity_type

    def test_valid_entity_types_constant(self):
        assert "audit" in VALID_ENTITY_TYPES
        assert "nc" in VALID_ENTITY_TYPES
        assert "car" in VALID_ENTITY_TYPES
        assert "certificate" in VALID_ENTITY_TYPES


class TestValidActions:
    """Test valid action validation."""

    @pytest.mark.parametrize("action", [
        "create", "update", "classify", "issue", "close",
        "verify", "generate", "sync", "log",
    ])
    def test_valid_actions(self, provenance_tracker, action):
        entry = provenance_tracker.record("audit", action, "TEST-001")
        assert entry.action == action

    def test_valid_actions_constant(self):
        assert "create" in VALID_ACTIONS
        assert "classify" in VALID_ACTIONS
        assert "issue" in VALID_ACTIONS


class TestHashDeterminism:
    """Test SHA-256 hash computation determinism."""

    def test_same_inputs_same_hash(self, provenance_tracker):
        reset_tracker()
        t1 = get_tracker()
        e1 = t1.record("audit", "create", "AUD-DET-001")

        reset_tracker()
        t2 = get_tracker()
        e2 = t2.record("audit", "create", "AUD-DET-001")

        assert e1.hash_value == e2.hash_value

    def test_different_entity_id_different_hash(self, provenance_tracker):
        e1 = provenance_tracker.record("audit", "create", "AUD-001")
        reset_tracker()
        t2 = get_tracker()
        e2 = t2.record("audit", "create", "AUD-002")
        assert e1.hash_value != e2.hash_value

    def test_different_action_different_hash(self, provenance_tracker):
        e1 = provenance_tracker.record("audit", "create", "AUD-001")
        reset_tracker()
        t2 = get_tracker()
        e2 = t2.record("audit", "update", "AUD-001")
        assert e1.hash_value != e2.hash_value

    def test_hash_is_hex_string(self, provenance_tracker):
        entry = provenance_tracker.record("audit", "create", "AUD-001")
        assert all(c in "0123456789abcdef" for c in entry.hash_value)


class TestChainVerification:
    """Test provenance chain verification."""

    def test_empty_chain_valid(self, provenance_tracker):
        assert provenance_tracker.verify_chain() is True

    def test_single_entry_chain_valid(self, provenance_tracker):
        provenance_tracker.record("audit", "create", "AUD-001")
        assert provenance_tracker.verify_chain() is True

    def test_multi_entry_chain_valid(self, provenance_tracker):
        for i in range(10):
            provenance_tracker.record("audit", "create", f"AUD-{i:03d}")
        assert provenance_tracker.verify_chain() is True

    def test_entry_count_tracked(self, provenance_tracker):
        for i in range(5):
            provenance_tracker.record("audit", "create", f"AUD-{i:03d}")
        assert provenance_tracker.entry_count == 5


class TestProvenanceRecord:
    """Test ProvenanceRecord model."""

    def test_record_fields(self, provenance_tracker):
        entry = provenance_tracker.record("audit", "create", "AUD-001")
        assert hasattr(entry, "hash_value")
        assert hasattr(entry, "parent_hash")
        assert hasattr(entry, "entity_type")
        assert hasattr(entry, "action")
        assert hasattr(entry, "entity_id")
        assert hasattr(entry, "timestamp")
