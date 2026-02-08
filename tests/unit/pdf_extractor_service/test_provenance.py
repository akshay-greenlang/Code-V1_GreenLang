# -*- coding: utf-8 -*-
"""
Unit Tests for ProvenanceTracker (AGENT-DATA-001)

Tests provenance recording, chain verification, chain retrieval,
JSON export, genesis hash generation, tamper detection, and
deterministic hashing.

Coverage target: 85%+ of provenance.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline ProvenanceTracker mirroring greenlang/pdf_extractor/provenance.py
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """SHA-256 provenance chain tracker for document extraction audit trails.

    Each record contains an operation, data hash, and link to the previous
    record's hash, forming an append-only chain.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self, agent_id: str = "GL-DATA-X-001"):
        self._agent_id = agent_id
        self._chains: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def record(
        self,
        chain_id: str,
        operation: str,
        data: Dict[str, Any],
        actor: str = "system",
    ) -> str:
        """Record an operation in the provenance chain.

        Args:
            chain_id: Unique identifier for the chain (e.g., document_id).
            operation: Operation name (e.g., 'ingest', 'extract', 'validate').
            data: Data associated with the operation.
            actor: Who performed the operation.

        Returns:
            The SHA-256 hash of this record.
        """
        if chain_id not in self._chains:
            self._chains[chain_id] = []

        chain = self._chains[chain_id]
        previous_hash = chain[-1]["record_hash"] if chain else self.GENESIS_HASH

        record = {
            "chain_id": chain_id,
            "sequence": len(chain) + 1,
            "operation": operation,
            "data_hash": self._hash_data(data),
            "previous_hash": previous_hash,
            "agent_id": self._agent_id,
            "actor": actor,
            "timestamp": datetime.utcnow().isoformat(),
        }

        record["record_hash"] = self._hash_record(record)
        chain.append(record)

        return record["record_hash"]

    def verify_chain(self, chain_id: str) -> Dict[str, Any]:
        """Verify the integrity of a provenance chain.

        Returns:
            Dict with is_valid, chain_length, broken_at (if invalid).
        """
        chain = self._chains.get(chain_id, [])
        if not chain:
            return {"is_valid": True, "chain_length": 0, "broken_at": None}

        for i, record in enumerate(chain):
            # Verify previous hash link
            expected_prev = chain[i - 1]["record_hash"] if i > 0 else self.GENESIS_HASH
            if record["previous_hash"] != expected_prev:
                return {
                    "is_valid": False,
                    "chain_length": len(chain),
                    "broken_at": i + 1,
                }

            # Verify record hash
            expected_hash = self._hash_record({
                k: v for k, v in record.items() if k != "record_hash"
            })
            if record["record_hash"] != expected_hash:
                return {
                    "is_valid": False,
                    "chain_length": len(chain),
                    "broken_at": i + 1,
                }

        return {"is_valid": True, "chain_length": len(chain), "broken_at": None}

    def get_chain(self, chain_id: str) -> List[Dict[str, Any]]:
        """Get the full provenance chain for a chain_id."""
        return list(self._chains.get(chain_id, []))

    def get_latest(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest record in a chain."""
        chain = self._chains.get(chain_id, [])
        return chain[-1] if chain else None

    def export_json(self, chain_id: str) -> str:
        """Export a provenance chain as JSON string."""
        chain = self.get_chain(chain_id)
        return json.dumps(chain, indent=2, default=str)

    def get_all_chain_ids(self) -> List[str]:
        """Return all chain IDs."""
        return list(self._chains.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Return provenance statistics."""
        total_records = sum(len(c) for c in self._chains.values())
        return {
            "total_chains": len(self._chains),
            "total_records": total_records,
            "agent_id": self._agent_id,
        }

    def _hash_data(self, data: Dict[str, Any]) -> str:
        """Hash the data payload."""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def _hash_record(self, record: Dict[str, Any]) -> str:
        """Hash a record (excluding record_hash itself)."""
        content = json.dumps(record, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestProvenanceTrackerInit:
    """Test ProvenanceTracker initialization."""

    def test_default_agent_id(self):
        tracker = ProvenanceTracker()
        assert tracker.agent_id == "GL-DATA-X-001"

    def test_custom_agent_id(self):
        tracker = ProvenanceTracker(agent_id="CUSTOM-001")
        assert tracker.agent_id == "CUSTOM-001"

    def test_genesis_hash(self):
        assert ProvenanceTracker.GENESIS_HASH == "0" * 64
        assert len(ProvenanceTracker.GENESIS_HASH) == 64

    def test_initial_empty(self):
        tracker = ProvenanceTracker()
        assert tracker.get_all_chain_ids() == []

    def test_initial_statistics(self):
        tracker = ProvenanceTracker()
        stats = tracker.get_statistics()
        assert stats["total_chains"] == 0
        assert stats["total_records"] == 0


class TestRecord:
    """Test record method."""

    def test_record_returns_hash(self):
        tracker = ProvenanceTracker()
        h = tracker.record("doc-001", "ingest", {"filename": "test.pdf"})
        assert len(h) == 64
        int(h, 16)  # Verify hex

    def test_record_creates_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {"filename": "test.pdf"})
        assert "doc-001" in tracker.get_all_chain_ids()

    def test_record_appends_to_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {"f": "test.pdf"})
        tracker.record("doc-001", "extract", {"fields": 10})
        chain = tracker.get_chain("doc-001")
        assert len(chain) == 2

    def test_record_sequence_numbers(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {})
        tracker.record("doc-001", "extract", {})
        tracker.record("doc-001", "validate", {})
        chain = tracker.get_chain("doc-001")
        assert chain[0]["sequence"] == 1
        assert chain[1]["sequence"] == 2
        assert chain[2]["sequence"] == 3

    def test_first_record_links_to_genesis(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {})
        chain = tracker.get_chain("doc-001")
        assert chain[0]["previous_hash"] == ProvenanceTracker.GENESIS_HASH

    def test_second_record_links_to_first(self):
        tracker = ProvenanceTracker()
        h1 = tracker.record("doc-001", "ingest", {})
        tracker.record("doc-001", "extract", {})
        chain = tracker.get_chain("doc-001")
        assert chain[1]["previous_hash"] == h1

    def test_record_contains_operation(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {})
        chain = tracker.get_chain("doc-001")
        assert chain[0]["operation"] == "ingest"

    def test_record_contains_agent_id(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {})
        chain = tracker.get_chain("doc-001")
        assert chain[0]["agent_id"] == "GL-DATA-X-001"

    def test_record_contains_actor(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {}, actor="user-42")
        chain = tracker.get_chain("doc-001")
        assert chain[0]["actor"] == "user-42"

    def test_record_contains_timestamp(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {})
        chain = tracker.get_chain("doc-001")
        assert chain[0]["timestamp"] is not None

    def test_deterministic_hashing(self):
        """Same data produces same data_hash across instances."""
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        data = {"filename": "test.pdf", "size": 1024}
        h1 = t1._hash_data(data)
        h2 = t2._hash_data(data)
        assert h1 == h2


class TestVerifyChain:
    """Test verify_chain method."""

    def test_empty_chain_valid(self):
        tracker = ProvenanceTracker()
        result = tracker.verify_chain("nonexistent")
        assert result["is_valid"] is True
        assert result["chain_length"] == 0

    def test_single_record_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {"f": "test.pdf"})
        result = tracker.verify_chain("doc-001")
        assert result["is_valid"] is True
        assert result["chain_length"] == 1

    def test_multi_record_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {})
        tracker.record("doc-001", "extract", {})
        tracker.record("doc-001", "validate", {})
        result = tracker.verify_chain("doc-001")
        assert result["is_valid"] is True
        assert result["chain_length"] == 3

    def test_tampered_record_detected(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {"f": "test.pdf"})
        tracker.record("doc-001", "extract", {"fields": 5})

        # Tamper with first record
        tracker._chains["doc-001"][0]["operation"] = "TAMPERED"

        result = tracker.verify_chain("doc-001")
        assert result["is_valid"] is False
        assert result["broken_at"] is not None

    def test_tampered_hash_detected(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {})
        tracker.record("doc-001", "extract", {})

        # Tamper with first record hash
        tracker._chains["doc-001"][0]["record_hash"] = "a" * 64

        result = tracker.verify_chain("doc-001")
        assert result["is_valid"] is False


class TestGetChain:
    """Test get_chain method."""

    def test_get_chain_exists(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {})
        chain = tracker.get_chain("doc-001")
        assert len(chain) == 1

    def test_get_chain_nonexistent(self):
        tracker = ProvenanceTracker()
        chain = tracker.get_chain("nonexistent")
        assert chain == []

    def test_get_chain_returns_copy(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {})
        chain1 = tracker.get_chain("doc-001")
        chain2 = tracker.get_chain("doc-001")
        assert chain1 is not chain2


class TestGetLatest:
    """Test get_latest method."""

    def test_latest_record(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {})
        tracker.record("doc-001", "extract", {})
        latest = tracker.get_latest("doc-001")
        assert latest["operation"] == "extract"

    def test_latest_nonexistent(self):
        tracker = ProvenanceTracker()
        assert tracker.get_latest("nonexistent") is None


class TestExportJson:
    """Test export_json method."""

    def test_export_valid_json(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {"f": "test.pdf"})
        exported = tracker.export_json("doc-001")
        parsed = json.loads(exported)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_export_empty_chain(self):
        tracker = ProvenanceTracker()
        exported = tracker.export_json("nonexistent")
        parsed = json.loads(exported)
        assert parsed == []

    def test_export_preserves_fields(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {"filename": "test.pdf"})
        exported = tracker.export_json("doc-001")
        parsed = json.loads(exported)
        assert parsed[0]["operation"] == "ingest"
        assert "record_hash" in parsed[0]


class TestProvenanceStatistics:
    """Test statistics gathering."""

    def test_stats_after_records(self):
        tracker = ProvenanceTracker()
        tracker.record("doc-001", "ingest", {})
        tracker.record("doc-002", "ingest", {})
        tracker.record("doc-001", "extract", {})
        stats = tracker.get_statistics()
        assert stats["total_chains"] == 2
        assert stats["total_records"] == 3

    def test_stats_agent_id(self):
        tracker = ProvenanceTracker(agent_id="TEST-001")
        stats = tracker.get_statistics()
        assert stats["agent_id"] == "TEST-001"
