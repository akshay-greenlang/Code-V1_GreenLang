# -*- coding: utf-8 -*-
"""
Test suite for audit_trail_lineage.audit_event_engine - AGENT-MRV-030.

Tests Engine 1: AuditEventEngine -- immutable SHA-256 hash chain event
recording for the Audit Trail & Lineage Agent (GL-MRV-X-042).

Coverage:
- Singleton pattern (__new__ returns same instance)
- record_event with valid data (all event types)
- Hash chain creation (genesis -> first event -> subsequent events)
- Hash chain verification (valid chain passes)
- Hash chain verification detects tampering
- Batch event recording (record_batch)
- Event querying (by type, scope, agent, time range, category)
- get_event by ID
- get_events_by_calculation
- get_events_by_scope
- Chain export (export_chain)
- Event statistics (get_event_statistics)
- Chain head and length queries
- Concurrent event recording (thread safety)
- Invalid event type rejection
- Canonical JSON determinism
- Chain isolation between organizations
- Reset functionality

Target: ~100 tests, 85%+ coverage.

Author: GL-TestEngineer
Date: March 2026
"""

import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.audit_trail_lineage.audit_event_engine import (
        AuditEventEngine,
        AuditEventRecord,
        AuditEventType,
        get_audit_event_engine,
        _decimal_serializer,
        GENESIS_HASH,
        ENCODING,
        _VALID_EVENT_TYPES,
        _MAX_BATCH_SIZE,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not ENGINE_AVAILABLE,
    reason="AuditEventEngine not available",
)

# ==============================================================================
# HELPER
# ==============================================================================

ORG_ID = "org-test-audit"
YEAR = 2025
AGENT = "GL-MRV-S1-001"


def _record_simple_event(
    engine: "AuditEventEngine",
    event_type: str = "DATA_INGESTED",
    org_id: str = ORG_ID,
    year: int = YEAR,
    agent_id: str = AGENT,
    scope: str = "scope_1",
    category: int = None,
    calculation_id: str = None,
    payload: dict = None,
    dq_score: Decimal = None,
) -> Dict[str, Any]:
    """Helper to record a simple event with minimal boilerplate."""
    return engine.record_event(
        event_type=event_type,
        agent_id=agent_id,
        scope=scope,
        category=category,
        organization_id=org_id,
        reporting_year=year,
        calculation_id=calculation_id,
        payload=payload or {"test": True},
        data_quality_score=dq_score if dq_score is not None else Decimal("0.80"),
    )


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================


@_SKIP
class TestAuditEventEngineSingleton:
    """Test singleton pattern of AuditEventEngine."""

    def test_singleton_same_instance(self, audit_event_engine):
        """Test AuditEventEngine returns the same singleton instance."""
        engine2 = AuditEventEngine()
        assert engine2 is audit_event_engine

    def test_get_audit_event_engine_accessor(self, audit_event_engine):
        """Test get_audit_event_engine() returns the singleton."""
        engine = get_audit_event_engine()
        assert engine is audit_event_engine

    def test_singleton_thread_safe(self, audit_event_engine):
        """Test singleton is thread-safe under concurrent instantiation."""
        instances = []

        def _create():
            instances.append(AuditEventEngine())

        threads = [threading.Thread(target=_create) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)


# ==============================================================================
# RECORD EVENT TESTS
# ==============================================================================


@_SKIP
class TestRecordEvent:
    """Test record_event functionality."""

    def test_record_event_success(self, audit_event_engine):
        """Test recording a single event returns success."""
        result = _record_simple_event(audit_event_engine)
        assert result["success"] is True

    def test_record_event_returns_event_id(self, audit_event_engine):
        """Test event_id starts with 'atl-' prefix."""
        result = _record_simple_event(audit_event_engine)
        assert result["event_id"].startswith("atl-")

    def test_record_event_returns_hash(self, audit_event_engine):
        """Test event_hash is a 64-character hex string."""
        result = _record_simple_event(audit_event_engine)
        assert len(result["event_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in result["event_hash"])

    def test_record_event_chain_position_zero(self, audit_event_engine):
        """Test first event in chain has position 0."""
        result = _record_simple_event(audit_event_engine)
        assert result["chain_position"] == 0

    def test_record_event_chain_key(self, audit_event_engine):
        """Test chain_key is org_id:year format."""
        result = _record_simple_event(audit_event_engine)
        assert result["chain_key"] == f"{ORG_ID}:{YEAR}"

    def test_record_event_prev_hash_genesis(self, audit_event_engine):
        """Test first event's prev_hash is genesis hash."""
        result = _record_simple_event(audit_event_engine)
        assert result["prev_event_hash"] == GENESIS_HASH

    def test_record_event_timestamp_iso(self, audit_event_engine):
        """Test timestamp is ISO 8601 format."""
        result = _record_simple_event(audit_event_engine)
        assert "T" in result["timestamp"]
        assert "+" in result["timestamp"] or "Z" in result["timestamp"]

    def test_record_event_processing_time(self, audit_event_engine):
        """Test processing_time_ms is a positive number."""
        result = _record_simple_event(audit_event_engine)
        assert result["processing_time_ms"] >= 0

    def test_record_event_event_type_in_result(self, audit_event_engine):
        """Test event_type is included in result."""
        result = _record_simple_event(audit_event_engine, event_type="CALCULATION_COMPLETED")
        assert result["event_type"] == "CALCULATION_COMPLETED"

    @pytest.mark.parametrize("event_type", [
        "DATA_INGESTED",
        "DATA_VALIDATED",
        "DATA_TRANSFORMED",
        "EMISSION_FACTOR_RESOLVED",
        "CALCULATION_STARTED",
        "CALCULATION_COMPLETED",
        "CALCULATION_FAILED",
        "COMPLIANCE_CHECKED",
        "REPORT_GENERATED",
        "PROVENANCE_SEALED",
        "MANUAL_OVERRIDE",
        "CHAIN_VERIFIED",
    ])
    def test_record_all_12_event_types(self, audit_event_engine, event_type):
        """Test recording events of all 12 types."""
        result = _record_simple_event(audit_event_engine, event_type=event_type)
        assert result["success"] is True
        assert result["event_type"] == event_type

    def test_record_event_scope_none(self, audit_event_engine):
        """Test recording event with scope=None."""
        result = _record_simple_event(audit_event_engine, scope=None)
        assert result["success"] is True

    def test_record_event_scope_3_with_category(self, audit_event_engine):
        """Test recording Scope 3 event with category."""
        result = _record_simple_event(
            audit_event_engine,
            scope="scope_3",
            category=6,
        )
        assert result["success"] is True

    def test_record_event_with_calculation_id(self, audit_event_engine):
        """Test recording event with calculation_id."""
        result = _record_simple_event(
            audit_event_engine,
            calculation_id="calc-test-123",
        )
        assert result["success"] is True

    def test_record_event_with_empty_payload(self, audit_event_engine):
        """Test recording event with empty payload."""
        result = _record_simple_event(audit_event_engine, payload={})
        assert result["success"] is True

    def test_record_event_none_dq_score_defaults_zero(self, audit_event_engine):
        """Test None data_quality_score defaults to zero."""
        # Call engine.record_event directly (not via helper) to pass None
        result = audit_event_engine.record_event(
            event_type="DATA_INGESTED",
            agent_id=AGENT,
            scope="scope_1",
            category=None,
            organization_id=ORG_ID,
            reporting_year=YEAR,
            calculation_id=None,
            payload={"test": True},
            data_quality_score=None,
        )
        assert result["success"] is True
        evt = audit_event_engine.get_event(result["event_id"])
        # Engine _normalize_dq_score returns _ZERO=Decimal("0") for None
        # (early return bypasses quantize to 2dp), so str representation is "0"
        assert Decimal(evt["data_quality_score"]) == Decimal("0")


# ==============================================================================
# VALIDATION ERROR TESTS
# ==============================================================================


@_SKIP
class TestRecordEventValidation:
    """Test input validation for record_event."""

    def test_invalid_event_type(self, audit_event_engine):
        """Test invalid event type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid event_type"):
            _record_simple_event(audit_event_engine, event_type="INVALID_TYPE")

    def test_empty_event_type(self, audit_event_engine):
        """Test empty event type raises ValueError."""
        with pytest.raises(ValueError, match="event_type"):
            _record_simple_event(audit_event_engine, event_type="")

    def test_empty_organization_id(self, audit_event_engine):
        """Test empty organization_id raises ValueError."""
        with pytest.raises(ValueError, match="organization_id"):
            _record_simple_event(audit_event_engine, org_id="")

    def test_invalid_reporting_year_too_low(self, audit_event_engine):
        """Test reporting_year < 1990 raises ValueError."""
        with pytest.raises(ValueError, match="reporting_year"):
            _record_simple_event(audit_event_engine, year=1989)

    def test_invalid_reporting_year_too_high(self, audit_event_engine):
        """Test reporting_year > 2100 raises ValueError."""
        with pytest.raises(ValueError, match="reporting_year"):
            _record_simple_event(audit_event_engine, year=2101)

    def test_invalid_scope(self, audit_event_engine):
        """Test invalid scope string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scope"):
            _record_simple_event(audit_event_engine, scope="scope_4")

    def test_invalid_category(self, audit_event_engine):
        """Test invalid category number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid category"):
            _record_simple_event(audit_event_engine, scope="scope_3", category=16)

    def test_category_zero(self, audit_event_engine):
        """Test category=0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid category"):
            _record_simple_event(audit_event_engine, scope="scope_3", category=0)

    def test_empty_agent_id(self, audit_event_engine):
        """Test empty agent_id raises ValueError."""
        with pytest.raises(ValueError, match="agent_id"):
            _record_simple_event(audit_event_engine, agent_id="")


# ==============================================================================
# HASH CHAIN TESTS
# ==============================================================================


@_SKIP
class TestHashChain:
    """Test SHA-256 hash chain creation and integrity."""

    def test_genesis_anchored(self, audit_event_engine):
        """Test first event is anchored to genesis hash."""
        result = _record_simple_event(audit_event_engine)
        assert result["prev_event_hash"] == GENESIS_HASH

    def test_chain_links_consecutively(self, audit_event_engine):
        """Test second event's prev_hash equals first event's hash."""
        r1 = _record_simple_event(audit_event_engine)
        r2 = _record_simple_event(audit_event_engine)
        assert r2["prev_event_hash"] == r1["event_hash"]

    def test_chain_positions_increment(self, audit_event_engine):
        """Test chain positions increment from 0."""
        r1 = _record_simple_event(audit_event_engine)
        r2 = _record_simple_event(audit_event_engine)
        r3 = _record_simple_event(audit_event_engine)
        assert r1["chain_position"] == 0
        assert r2["chain_position"] == 1
        assert r3["chain_position"] == 2

    def test_chain_three_events_linked(self, audit_event_engine):
        """Test three events are properly linked."""
        r1 = _record_simple_event(audit_event_engine)
        r2 = _record_simple_event(audit_event_engine)
        r3 = _record_simple_event(audit_event_engine)
        assert r2["prev_event_hash"] == r1["event_hash"]
        assert r3["prev_event_hash"] == r2["event_hash"]

    def test_event_hashes_are_unique(self, audit_event_engine):
        """Test each event produces a unique hash."""
        results = [_record_simple_event(audit_event_engine) for _ in range(5)]
        hashes = [r["event_hash"] for r in results]
        assert len(set(hashes)) == 5

    def test_event_hash_is_sha256(self, audit_event_engine):
        """Test event hash is 64 hex chars (SHA-256)."""
        result = _record_simple_event(audit_event_engine)
        assert len(result["event_hash"]) == 64

    def test_chain_head_after_events(self, audit_event_engine):
        """Test get_chain_head returns latest hash."""
        r1 = _record_simple_event(audit_event_engine)
        assert audit_event_engine.get_chain_head(ORG_ID, YEAR) == r1["event_hash"]
        r2 = _record_simple_event(audit_event_engine)
        assert audit_event_engine.get_chain_head(ORG_ID, YEAR) == r2["event_hash"]

    def test_chain_length_increments(self, audit_event_engine):
        """Test get_chain_length increments correctly."""
        assert audit_event_engine.get_chain_length(ORG_ID, YEAR) == 0
        _record_simple_event(audit_event_engine)
        assert audit_event_engine.get_chain_length(ORG_ID, YEAR) == 1
        _record_simple_event(audit_event_engine)
        assert audit_event_engine.get_chain_length(ORG_ID, YEAR) == 2

    def test_chain_head_nonexistent(self, audit_event_engine):
        """Test get_chain_head returns None for missing chain."""
        assert audit_event_engine.get_chain_head("no-org", 2025) is None

    def test_chain_length_nonexistent(self, audit_event_engine):
        """Test get_chain_length returns 0 for missing chain."""
        assert audit_event_engine.get_chain_length("no-org", 2025) == 0


# ==============================================================================
# CHAIN VERIFICATION TESTS
# ==============================================================================


@_SKIP
class TestChainVerification:
    """Test hash chain verification."""

    def test_verify_empty_chain(self, audit_event_engine):
        """Test verify_chain on empty chain returns valid."""
        result = audit_event_engine.verify_chain(ORG_ID, YEAR)
        assert result["valid"] is True
        assert result["verified_count"] == 0

    def test_verify_single_event(self, audit_event_engine):
        """Test verify_chain with single event passes."""
        _record_simple_event(audit_event_engine)
        result = audit_event_engine.verify_chain(ORG_ID, YEAR)
        assert result["valid"] is True
        assert result["verified_count"] == 1

    def test_verify_multiple_events(self, audit_event_engine):
        """Test verify_chain with multiple events passes."""
        for _ in range(10):
            _record_simple_event(audit_event_engine)
        result = audit_event_engine.verify_chain(ORG_ID, YEAR)
        assert result["valid"] is True
        assert result["verified_count"] == 10

    def test_verify_chain_returns_timing(self, audit_event_engine):
        """Test verify_chain includes verification_time_ms."""
        _record_simple_event(audit_event_engine)
        result = audit_event_engine.verify_chain(ORG_ID, YEAR)
        assert "verification_time_ms" in result
        assert result["verification_time_ms"] >= 0

    def test_verify_chain_key_in_result(self, audit_event_engine):
        """Test verify_chain includes chain_key."""
        result = audit_event_engine.verify_chain(ORG_ID, YEAR)
        assert result["chain_key"] == f"{ORG_ID}:{YEAR}"

    def test_verify_chain_range(self, audit_event_engine):
        """Test verify_chain with start/end position range."""
        for _ in range(5):
            _record_simple_event(audit_event_engine)
        result = audit_event_engine.verify_chain(
            ORG_ID, YEAR, start_position=1, end_position=3,
        )
        assert result["valid"] is True
        assert result["verified_count"] == 3

    def test_verify_detects_tampered_hash(self, audit_event_engine):
        """Test verification detects a tampered event hash."""
        _record_simple_event(audit_event_engine)
        _record_simple_event(audit_event_engine)

        # Tamper with first event's hash in the chain
        chain_key = f"{ORG_ID}:{YEAR}"
        with audit_event_engine._lock:
            original = audit_event_engine._chains[chain_key][0]
            tampered = AuditEventRecord(
                event_id=original.event_id,
                event_type=original.event_type,
                agent_id=original.agent_id,
                scope=original.scope,
                category=original.category,
                organization_id=original.organization_id,
                reporting_year=original.reporting_year,
                calculation_id=original.calculation_id,
                data_quality_score=original.data_quality_score,
                payload=original.payload,
                prev_event_hash=original.prev_event_hash,
                event_hash="TAMPERED_HASH_" + "0" * 50,
                chain_position=original.chain_position,
                timestamp=original.timestamp,
                metadata=original.metadata,
            )
            audit_event_engine._chains[chain_key][0] = tampered

        result = audit_event_engine.verify_chain(ORG_ID, YEAR)
        assert result["valid"] is False
        assert len(result["errors"]) >= 1
        assert result["first_invalid_position"] is not None


# ==============================================================================
# BATCH RECORDING TESTS
# ==============================================================================


@_SKIP
class TestRecordBatch:
    """Test batch event recording."""

    def _make_batch_event(self, idx: int) -> Dict[str, Any]:
        """Helper to create a batch event dict."""
        return {
            "event_type": "DATA_INGESTED",
            "agent_id": AGENT,
            "scope": "scope_1",
            "organization_id": ORG_ID,
            "reporting_year": YEAR,
            "payload": {"index": idx},
        }

    def test_batch_success(self, audit_event_engine):
        """Test batch recording with valid events."""
        events = [self._make_batch_event(i) for i in range(5)]
        result = audit_event_engine.record_batch(events)
        assert result["success"] is True
        assert result["total_recorded"] == 5

    def test_batch_returns_event_ids(self, audit_event_engine):
        """Test batch returns list of event_ids."""
        events = [self._make_batch_event(i) for i in range(3)]
        result = audit_event_engine.record_batch(events)
        assert len(result["event_ids"]) == 3
        for eid in result["event_ids"]:
            assert eid.startswith("atl-")

    def test_batch_chain_integrity(self, audit_event_engine):
        """Test batch events maintain chain integrity."""
        events = [self._make_batch_event(i) for i in range(10)]
        audit_event_engine.record_batch(events)
        verify = audit_event_engine.verify_chain(ORG_ID, YEAR)
        assert verify["valid"] is True
        assert verify["verified_count"] == 10

    def test_batch_empty_raises(self, audit_event_engine):
        """Test empty batch raises ValueError."""
        with pytest.raises(ValueError, match="at least one event"):
            audit_event_engine.record_batch([])

    def test_batch_exceeds_max_size(self, audit_event_engine):
        """Test batch exceeding max size raises ValueError."""
        events = [self._make_batch_event(i) for i in range(_MAX_BATCH_SIZE + 1)]
        with pytest.raises(ValueError, match="exceeds maximum"):
            audit_event_engine.record_batch(events)

    def test_batch_invalid_event_rejects_all(self, audit_event_engine):
        """Test batch with invalid event rejects entire batch."""
        events = [
            self._make_batch_event(0),
            {"event_type": "INVALID_TYPE", "agent_id": AGENT, "organization_id": ORG_ID, "reporting_year": YEAR},
        ]
        with pytest.raises(ValueError, match="failed validation"):
            audit_event_engine.record_batch(events)
        # No events should have been recorded
        assert audit_event_engine.get_chain_length(ORG_ID, YEAR) == 0

    def test_batch_missing_required_key(self, audit_event_engine):
        """Test batch event missing required key raises ValueError."""
        events = [{"event_type": "DATA_INGESTED"}]  # missing agent_id, org, year
        with pytest.raises(ValueError, match="missing required keys"):
            audit_event_engine.record_batch(events)

    def test_batch_non_dict_event(self, audit_event_engine):
        """Test batch with non-dict event raises ValueError."""
        events = ["not a dict"]
        with pytest.raises(ValueError):
            audit_event_engine.record_batch(events)

    def test_batch_processing_time(self, audit_event_engine):
        """Test batch returns processing_time_ms."""
        events = [self._make_batch_event(i) for i in range(5)]
        result = audit_event_engine.record_batch(events)
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0


# ==============================================================================
# EVENT RETRIEVAL TESTS
# ==============================================================================


@_SKIP
class TestEventRetrieval:
    """Test event retrieval and querying."""

    def test_get_event_by_id(self, audit_event_engine):
        """Test get_event returns event dictionary."""
        result = _record_simple_event(audit_event_engine)
        evt = audit_event_engine.get_event(result["event_id"])
        assert evt is not None
        assert evt["event_id"] == result["event_id"]
        assert evt["event_hash"] == result["event_hash"]

    def test_get_event_nonexistent(self, audit_event_engine):
        """Test get_event returns None for missing event."""
        evt = audit_event_engine.get_event("atl-nonexistent")
        assert evt is None

    def test_get_event_empty_id(self, audit_event_engine):
        """Test get_event returns None for empty string."""
        assert audit_event_engine.get_event("") is None

    def test_get_event_none_id(self, audit_event_engine):
        """Test get_event returns None for None."""
        assert audit_event_engine.get_event(None) is None

    def test_get_events_returns_all(self, audit_event_engine):
        """Test get_events returns all events for org/year."""
        for _ in range(5):
            _record_simple_event(audit_event_engine)
        result = audit_event_engine.get_events(ORG_ID, YEAR)
        assert result["success"] is True
        assert result["total_matching"] == 5
        assert result["returned_count"] == 5

    def test_get_events_filter_by_type(self, audit_event_engine):
        """Test get_events filters by event_type."""
        _record_simple_event(audit_event_engine, event_type="DATA_INGESTED")
        _record_simple_event(audit_event_engine, event_type="CALCULATION_COMPLETED")
        _record_simple_event(audit_event_engine, event_type="DATA_INGESTED")
        result = audit_event_engine.get_events(
            ORG_ID, YEAR, event_type="DATA_INGESTED",
        )
        assert result["total_matching"] == 2

    def test_get_events_filter_by_agent(self, audit_event_engine):
        """Test get_events filters by agent_id."""
        _record_simple_event(audit_event_engine, agent_id="GL-MRV-S1-001")
        _record_simple_event(audit_event_engine, agent_id="GL-MRV-S3-006")
        result = audit_event_engine.get_events(
            ORG_ID, YEAR, agent_id="GL-MRV-S3-006",
        )
        assert result["total_matching"] == 1

    def test_get_events_filter_by_scope(self, audit_event_engine):
        """Test get_events filters by scope."""
        _record_simple_event(audit_event_engine, scope="scope_1")
        _record_simple_event(audit_event_engine, scope="scope_2")
        _record_simple_event(audit_event_engine, scope="scope_1")
        result = audit_event_engine.get_events(
            ORG_ID, YEAR, scope="scope_2",
        )
        assert result["total_matching"] == 1

    def test_get_events_filter_by_category(self, audit_event_engine):
        """Test get_events filters by category."""
        _record_simple_event(audit_event_engine, scope="scope_3", category=6)
        _record_simple_event(audit_event_engine, scope="scope_3", category=14)
        result = audit_event_engine.get_events(
            ORG_ID, YEAR, category=6,
        )
        assert result["total_matching"] == 1

    def test_get_events_pagination_limit(self, audit_event_engine):
        """Test get_events respects limit parameter."""
        for _ in range(10):
            _record_simple_event(audit_event_engine)
        result = audit_event_engine.get_events(ORG_ID, YEAR, limit=3)
        assert result["returned_count"] == 3
        assert result["has_more"] is True

    def test_get_events_pagination_offset(self, audit_event_engine):
        """Test get_events respects offset parameter."""
        for _ in range(10):
            _record_simple_event(audit_event_engine)
        result = audit_event_engine.get_events(ORG_ID, YEAR, offset=8)
        assert result["returned_count"] == 2

    def test_get_events_filters_applied_in_result(self, audit_event_engine):
        """Test filters_applied dict in result."""
        _record_simple_event(audit_event_engine)
        result = audit_event_engine.get_events(
            ORG_ID, YEAR, event_type="DATA_INGESTED", scope="scope_1",
        )
        assert result["filters_applied"]["event_type"] == "DATA_INGESTED"
        assert result["filters_applied"]["scope"] == "scope_1"

    def test_get_events_empty_chain(self, audit_event_engine):
        """Test get_events on empty chain returns 0 events."""
        result = audit_event_engine.get_events(ORG_ID, YEAR)
        assert result["total_matching"] == 0
        assert result["returned_count"] == 0

    def test_get_events_by_calculation(self, audit_event_engine):
        """Test get_events_by_calculation returns linked events."""
        calc_id = "calc-test-xyz"
        _record_simple_event(audit_event_engine, calculation_id=calc_id, event_type="CALCULATION_STARTED")
        _record_simple_event(audit_event_engine, calculation_id=calc_id, event_type="CALCULATION_COMPLETED")
        _record_simple_event(audit_event_engine, calculation_id="other-calc")
        events = audit_event_engine.get_events_by_calculation(calc_id)
        assert len(events) == 2

    def test_get_events_by_calculation_empty(self, audit_event_engine):
        """Test get_events_by_calculation returns empty for missing calc."""
        events = audit_event_engine.get_events_by_calculation("nonexistent")
        assert events == []

    def test_get_events_by_calculation_none(self, audit_event_engine):
        """Test get_events_by_calculation returns empty for None."""
        events = audit_event_engine.get_events_by_calculation(None)
        assert events == []

    def test_get_events_by_scope(self, audit_event_engine):
        """Test get_events_by_scope returns filtered events."""
        _record_simple_event(audit_event_engine, scope="scope_1")
        _record_simple_event(audit_event_engine, scope="scope_2")
        events = audit_event_engine.get_events_by_scope(ORG_ID, YEAR, "scope_1")
        assert len(events) == 1


# ==============================================================================
# CHAIN OPERATIONS TESTS
# ==============================================================================


@_SKIP
class TestChainOperations:
    """Test chain retrieval and export operations."""

    def test_get_chain_success(self, audit_event_engine):
        """Test get_chain returns full chain data."""
        _record_simple_event(audit_event_engine)
        chain = audit_event_engine.get_chain(ORG_ID, YEAR)
        assert chain["success"] is True
        assert chain["genesis_hash"] == GENESIS_HASH
        assert chain["length"] == 1
        assert len(chain["events"]) == 1

    def test_get_chain_empty(self, audit_event_engine):
        """Test get_chain on empty chain."""
        chain = audit_event_engine.get_chain(ORG_ID, YEAR)
        assert chain["length"] == 0
        assert chain["events"] == []
        assert chain["head_hash"] is None

    def test_export_chain_success(self, audit_event_engine):
        """Test export_chain returns comprehensive export."""
        for _ in range(3):
            _record_simple_event(audit_event_engine)
        export = audit_event_engine.export_chain(ORG_ID, YEAR)
        assert export["success"] is True
        assert export["chain_length"] == 3
        assert export["verification"]["valid"] is True
        assert export["agent_id"] == "GL-MRV-X-042"
        assert "export_id" in export
        assert "exported_at" in export

    def test_export_chain_empty(self, audit_event_engine):
        """Test export_chain on empty chain."""
        export = audit_event_engine.export_chain(ORG_ID, YEAR)
        assert export["chain_length"] == 0
        assert export["verification"]["valid"] is True

    def test_export_chain_includes_engine_info(self, audit_event_engine):
        """Test export includes engine_id and engine_version."""
        _record_simple_event(audit_event_engine)
        export = audit_event_engine.export_chain(ORG_ID, YEAR)
        assert export["engine_id"] == "gl_atl_audit_event_engine"
        assert export["engine_version"] == "1.0.0"


# ==============================================================================
# STATISTICS TESTS
# ==============================================================================


@_SKIP
class TestEventStatistics:
    """Test event statistics computation."""

    def test_statistics_empty_chain(self, audit_event_engine):
        """Test statistics on empty chain."""
        stats = audit_event_engine.get_event_statistics(ORG_ID, YEAR)
        assert stats["total_events"] == 0
        assert stats["by_event_type"] == {}
        assert stats["avg_data_quality_score"] is None

    def test_statistics_by_event_type(self, audit_event_engine):
        """Test statistics groups by event_type."""
        _record_simple_event(audit_event_engine, event_type="DATA_INGESTED")
        _record_simple_event(audit_event_engine, event_type="DATA_INGESTED")
        _record_simple_event(audit_event_engine, event_type="CALCULATION_COMPLETED")
        stats = audit_event_engine.get_event_statistics(ORG_ID, YEAR)
        assert stats["by_event_type"]["DATA_INGESTED"] == 2
        assert stats["by_event_type"]["CALCULATION_COMPLETED"] == 1

    def test_statistics_by_scope(self, audit_event_engine):
        """Test statistics groups by scope."""
        _record_simple_event(audit_event_engine, scope="scope_1")
        _record_simple_event(audit_event_engine, scope="scope_2")
        stats = audit_event_engine.get_event_statistics(ORG_ID, YEAR)
        assert stats["by_scope"]["scope_1"] == 1
        assert stats["by_scope"]["scope_2"] == 1

    def test_statistics_by_agent(self, audit_event_engine):
        """Test statistics groups by agent."""
        _record_simple_event(audit_event_engine, agent_id="GL-MRV-S1-001")
        _record_simple_event(audit_event_engine, agent_id="GL-MRV-S3-006")
        stats = audit_event_engine.get_event_statistics(ORG_ID, YEAR)
        assert len(stats["by_agent"]) == 2

    def test_statistics_avg_dq_score(self, audit_event_engine):
        """Test average data quality score computation."""
        _record_simple_event(audit_event_engine, dq_score=Decimal("0.80"))
        _record_simple_event(audit_event_engine, dq_score=Decimal("0.60"))
        stats = audit_event_engine.get_event_statistics(ORG_ID, YEAR)
        assert stats["avg_data_quality_score"] is not None
        avg = Decimal(stats["avg_data_quality_score"])
        assert avg == Decimal("0.7000")

    def test_statistics_earliest_latest(self, audit_event_engine):
        """Test earliest and latest event timestamps."""
        _record_simple_event(audit_event_engine)
        time.sleep(0.01)
        _record_simple_event(audit_event_engine)
        stats = audit_event_engine.get_event_statistics(ORG_ID, YEAR)
        assert stats["earliest_event"] is not None
        assert stats["latest_event"] is not None
        assert stats["earliest_event"] <= stats["latest_event"]


# ==============================================================================
# CHAIN ISOLATION TESTS
# ==============================================================================


@_SKIP
class TestChainIsolation:
    """Test that chains are isolated between organizations and years."""

    def test_different_orgs_separate_chains(self, audit_event_engine):
        """Test different organizations have separate chains."""
        _record_simple_event(audit_event_engine, org_id="org-A")
        _record_simple_event(audit_event_engine, org_id="org-B")
        assert audit_event_engine.get_chain_length("org-A", YEAR) == 1
        assert audit_event_engine.get_chain_length("org-B", YEAR) == 1

    def test_different_years_separate_chains(self, audit_event_engine):
        """Test different years have separate chains."""
        _record_simple_event(audit_event_engine, year=2024)
        _record_simple_event(audit_event_engine, year=2025)
        assert audit_event_engine.get_chain_length(ORG_ID, 2024) == 1
        assert audit_event_engine.get_chain_length(ORG_ID, 2025) == 1

    def test_cross_org_chain_heads_independent(self, audit_event_engine):
        """Test chain heads are independent across organizations."""
        r1 = _record_simple_event(audit_event_engine, org_id="org-A")
        r2 = _record_simple_event(audit_event_engine, org_id="org-B")
        assert r1["event_hash"] != r2["event_hash"]
        assert audit_event_engine.get_chain_head("org-A", YEAR) == r1["event_hash"]
        assert audit_event_engine.get_chain_head("org-B", YEAR) == r2["event_hash"]


# ==============================================================================
# CANONICAL JSON DETERMINISM TESTS
# ==============================================================================


@_SKIP
class TestCanonicalJsonDeterminism:
    """Test that canonical JSON serialization is deterministic."""

    def test_same_payload_same_hash(self, audit_event_engine):
        """Test identical payloads produce identical canonical JSON."""
        payload1 = {"b": 2, "a": 1}
        payload2 = {"a": 1, "b": 2}
        json1 = audit_event_engine._canonical_json(payload1)
        json2 = audit_event_engine._canonical_json(payload2)
        assert json1 == json2

    def test_canonical_json_sorted_keys(self, audit_event_engine):
        """Test canonical JSON sorts keys."""
        payload = {"z": 26, "a": 1, "m": 13}
        result = audit_event_engine._canonical_json(payload)
        assert result == '{"a":1,"m":13,"z":26}'

    def test_canonical_json_no_whitespace(self, audit_event_engine):
        """Test canonical JSON has no extra whitespace."""
        payload = {"key": "value"}
        result = audit_event_engine._canonical_json(payload)
        assert " " not in result

    def test_canonical_json_nested_sorted(self, audit_event_engine):
        """Test nested dict keys are also sorted."""
        payload = {"b": {"d": 4, "c": 3}, "a": 1}
        result = audit_event_engine._canonical_json(payload)
        parsed = json.loads(result)
        assert list(parsed.keys()) == ["a", "b"]

    def test_canonical_json_with_decimal(self, audit_event_engine):
        """Test canonical JSON handles Decimal values."""
        payload = {"value": Decimal("3.14")}
        result = audit_event_engine._canonical_json(payload)
        assert '"3.14"' in result


# ==============================================================================
# DQ SCORE NORMALIZATION TESTS
# ==============================================================================


@_SKIP
class TestDQScoreNormalization:
    """Test data quality score normalization."""

    def test_none_defaults_to_zero(self, audit_event_engine):
        """Test None score defaults to 0.00."""
        result = audit_event_engine._normalize_dq_score(None)
        assert result == Decimal("0.00")

    def test_decimal_passthrough(self, audit_event_engine):
        """Test Decimal value passes through with quantization."""
        result = audit_event_engine._normalize_dq_score(Decimal("0.85"))
        assert result == Decimal("0.85")

    def test_float_conversion(self, audit_event_engine):
        """Test float is converted to Decimal."""
        result = audit_event_engine._normalize_dq_score(0.75)
        assert result == Decimal("0.75")

    def test_int_conversion(self, audit_event_engine):
        """Test integer is converted to Decimal."""
        result = audit_event_engine._normalize_dq_score(1)
        assert result == Decimal("1.00")

    def test_string_conversion(self, audit_event_engine):
        """Test string is converted to Decimal."""
        result = audit_event_engine._normalize_dq_score("0.50")
        assert result == Decimal("0.50")

    def test_clamp_above_one(self, audit_event_engine):
        """Test values above 1 are clamped to 1.00."""
        result = audit_event_engine._normalize_dq_score(Decimal("1.5"))
        assert result == Decimal("1.00")

    def test_clamp_below_zero(self, audit_event_engine):
        """Test values below 0 are clamped to 0.00."""
        result = audit_event_engine._normalize_dq_score(Decimal("-0.5"))
        assert result == Decimal("0.00")

    def test_invalid_string_defaults_zero(self, audit_event_engine):
        """Test invalid string defaults to 0.00."""
        result = audit_event_engine._normalize_dq_score("not_a_number")
        assert result == Decimal("0.00")


# ==============================================================================
# THREAD SAFETY TESTS
# ==============================================================================


@_SKIP
class TestThreadSafety:
    """Test thread safety of concurrent event recording."""

    def test_concurrent_recording(self, audit_event_engine):
        """Test concurrent event recording produces correct chain."""
        num_threads = 10
        events_per_thread = 5
        errors = []

        def _record(thread_idx: int):
            try:
                for i in range(events_per_thread):
                    _record_simple_event(
                        audit_event_engine,
                        payload={"thread": thread_idx, "index": i},
                    )
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=_record, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        expected_total = num_threads * events_per_thread
        assert audit_event_engine.get_chain_length(ORG_ID, YEAR) == expected_total

    def test_concurrent_chain_integrity(self, audit_event_engine):
        """Test chain integrity after concurrent recording."""
        num_threads = 5
        errors = []

        def _record(t: int):
            try:
                for i in range(10):
                    _record_simple_event(audit_event_engine)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=_record, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        verify = audit_event_engine.verify_chain(ORG_ID, YEAR)
        assert verify["valid"] is True


# ==============================================================================
# RESET TESTS
# ==============================================================================


@_SKIP
class TestReset:
    """Test engine reset functionality."""

    def test_reset_clears_chains(self, audit_event_engine):
        """Test reset clears all chain data."""
        _record_simple_event(audit_event_engine)
        assert audit_event_engine.get_chain_length(ORG_ID, YEAR) == 1
        audit_event_engine.reset()
        assert audit_event_engine.get_chain_length(ORG_ID, YEAR) == 0

    def test_reset_clears_event_index(self, audit_event_engine):
        """Test reset clears event index."""
        result = _record_simple_event(audit_event_engine)
        assert audit_event_engine.get_event(result["event_id"]) is not None
        audit_event_engine.reset()
        assert audit_event_engine.get_event(result["event_id"]) is None

    def test_reset_allows_new_chain(self, audit_event_engine):
        """Test recording after reset starts fresh chain."""
        _record_simple_event(audit_event_engine)
        audit_event_engine.reset()
        result = _record_simple_event(audit_event_engine)
        assert result["chain_position"] == 0
        assert result["prev_event_hash"] == GENESIS_HASH


# ==============================================================================
# ADDITIONAL EDGE CASE TESTS
# ==============================================================================


@_SKIP
class TestEdgeCases:
    """Additional edge case tests for comprehensive coverage."""

    def test_record_event_with_large_payload(self, audit_event_engine):
        """Test recording event with large payload."""
        payload = {"data": "x" * 10000, "nested": {"key": list(range(100))}}
        result = _record_simple_event(audit_event_engine, payload=payload)
        assert result["success"] is True

    def test_record_event_with_unicode_payload(self, audit_event_engine):
        """Test recording event with unicode characters in payload."""
        payload = {"description": "Emissions from cafe -- 2025", "notes": "CO2e"}
        result = _record_simple_event(audit_event_engine, payload=payload)
        assert result["success"] is True

    def test_record_event_with_special_chars_org_id(self, audit_event_engine):
        """Test recording event with special characters in org_id."""
        result = _record_simple_event(audit_event_engine, org_id="org-test_123.abc")
        assert result["success"] is True

    def test_record_event_boundary_year_1990(self, audit_event_engine):
        """Test recording event with minimum valid year 1990."""
        result = _record_simple_event(audit_event_engine, year=1990)
        assert result["success"] is True

    def test_record_event_boundary_year_2100(self, audit_event_engine):
        """Test recording event with maximum valid year 2100."""
        result = _record_simple_event(audit_event_engine, year=2100)
        assert result["success"] is True

    @pytest.mark.parametrize("category", [1, 5, 10, 15])
    def test_record_event_valid_categories(self, audit_event_engine, category):
        """Test recording events with valid Scope 3 categories."""
        result = _record_simple_event(
            audit_event_engine, scope="scope_3", category=category,
        )
        assert result["success"] is True

    @pytest.mark.parametrize("scope", ["scope_1", "scope_2", "scope_3"])
    def test_record_event_all_scopes(self, audit_event_engine, scope):
        """Test recording events with all valid scopes."""
        result = _record_simple_event(audit_event_engine, scope=scope)
        assert result["success"] is True

    def test_dq_score_boundary_zero(self, audit_event_engine):
        """Test DQ score exactly 0."""
        result = audit_event_engine._normalize_dq_score(Decimal("0"))
        assert result == Decimal("0.00")

    def test_dq_score_boundary_one(self, audit_event_engine):
        """Test DQ score exactly 1."""
        result = audit_event_engine._normalize_dq_score(Decimal("1"))
        assert result == Decimal("1.00")

    def test_dq_score_high_precision(self, audit_event_engine):
        """Test DQ score with high precision is quantized to 2dp."""
        result = audit_event_engine._normalize_dq_score(Decimal("0.8567"))
        assert result == Decimal("0.86")

    def test_chain_key_format(self, audit_event_engine):
        """Test chain key format is correct."""
        key = audit_event_engine._get_chain_key("org-test", 2025)
        assert key == "org-test:2025"

    def test_event_id_format(self, audit_event_engine):
        """Test event ID format has atl- prefix."""
        eid = audit_event_engine._generate_event_id()
        assert eid.startswith("atl-")
        assert len(eid) > 4

    def test_event_id_uniqueness(self, audit_event_engine):
        """Test event IDs are unique across multiple generations."""
        ids = [audit_event_engine._generate_event_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_get_events_max_query_limit_clamped(self, audit_event_engine):
        """Test get_events clamps limit to max query limit."""
        for _ in range(5):
            _record_simple_event(audit_event_engine)
        result = audit_event_engine.get_events(ORG_ID, YEAR, limit=50000)
        assert result["limit"] <= 10000

    def test_get_events_negative_offset_clamped(self, audit_event_engine):
        """Test get_events clamps negative offset to 0."""
        _record_simple_event(audit_event_engine)
        result = audit_event_engine.get_events(ORG_ID, YEAR, offset=-5)
        assert result["offset"] == 0

    def test_verify_chain_range_start_only(self, audit_event_engine):
        """Test verify_chain with only start_position."""
        for _ in range(5):
            _record_simple_event(audit_event_engine)
        result = audit_event_engine.verify_chain(ORG_ID, YEAR, start_position=2)
        assert result["valid"] is True
        assert result["verified_count"] == 3

    def test_verify_chain_range_end_only(self, audit_event_engine):
        """Test verify_chain with only end_position."""
        for _ in range(5):
            _record_simple_event(audit_event_engine)
        result = audit_event_engine.verify_chain(ORG_ID, YEAR, end_position=2)
        assert result["valid"] is True
        assert result["verified_count"] == 3

    def test_batch_with_different_orgs(self, audit_event_engine):
        """Test batch recording events for different organizations."""
        events = [
            {
                "event_type": "DATA_INGESTED",
                "agent_id": AGENT,
                "organization_id": "org-A",
                "reporting_year": YEAR,
            },
            {
                "event_type": "DATA_INGESTED",
                "agent_id": AGENT,
                "organization_id": "org-B",
                "reporting_year": YEAR,
            },
        ]
        result = audit_event_engine.record_batch(events)
        assert result["success"] is True
        assert audit_event_engine.get_chain_length("org-A", YEAR) == 1
        assert audit_event_engine.get_chain_length("org-B", YEAR) == 1

    def test_batch_with_scope_3_categories(self, audit_event_engine):
        """Test batch recording Scope 3 events with categories."""
        events = [
            {
                "event_type": "CALCULATION_COMPLETED",
                "agent_id": "GL-MRV-S3-006",
                "scope": "scope_3",
                "category": cat,
                "organization_id": ORG_ID,
                "reporting_year": YEAR,
            }
            for cat in [1, 6, 14, 15]
        ]
        result = audit_event_engine.record_batch(events)
        assert result["total_recorded"] == 4

    def test_get_events_time_range_filter(self, audit_event_engine):
        """Test get_events with start_time and end_time filters."""
        _record_simple_event(audit_event_engine)
        result = audit_event_engine.get_events(
            ORG_ID, YEAR,
            start_time="2020-01-01T00:00:00+00:00",
            end_time="2030-12-31T23:59:59+00:00",
        )
        assert result["total_matching"] >= 1

    def test_get_events_future_time_range(self, audit_event_engine):
        """Test get_events with future time range returns 0 events."""
        _record_simple_event(audit_event_engine)
        result = audit_event_engine.get_events(
            ORG_ID, YEAR,
            start_time="2099-01-01T00:00:00+00:00",
        )
        assert result["total_matching"] == 0

    def test_chain_50_events_verification(self, audit_event_engine):
        """Test 50-event chain integrity."""
        for i in range(50):
            _record_simple_event(audit_event_engine, payload={"index": i})
        assert audit_event_engine.get_chain_length(ORG_ID, YEAR) == 50
        verify = audit_event_engine.verify_chain(ORG_ID, YEAR)
        assert verify["valid"] is True
        assert verify["verified_count"] == 50

    def test_multiple_calculations_indexing(self, audit_event_engine):
        """Test multiple calculations are correctly indexed."""
        for i in range(3):
            _record_simple_event(
                audit_event_engine,
                calculation_id=f"calc-{i}",
                event_type="CALCULATION_STARTED",
            )
            _record_simple_event(
                audit_event_engine,
                calculation_id=f"calc-{i}",
                event_type="CALCULATION_COMPLETED",
            )
        for i in range(3):
            events = audit_event_engine.get_events_by_calculation(f"calc-{i}")
            assert len(events) == 2

    def test_statistics_by_category(self, audit_event_engine):
        """Test statistics include category breakdown."""
        _record_simple_event(audit_event_engine, scope="scope_3", category=6)
        _record_simple_event(audit_event_engine, scope="scope_3", category=14)
        _record_simple_event(audit_event_engine, scope="scope_3", category=6)
        stats = audit_event_engine.get_event_statistics(ORG_ID, YEAR)
        assert stats["by_category"]["6"] == 2
        assert stats["by_category"]["14"] == 1

    def test_export_chain_verification_included(self, audit_event_engine):
        """Test chain export includes verification result."""
        for _ in range(5):
            _record_simple_event(audit_event_engine)
        export = audit_event_engine.export_chain(ORG_ID, YEAR)
        assert "verification" in export
        assert export["verification"]["valid"] is True
        assert export["verification"]["verified_count"] == 5

    def test_record_event_metadata_preserved(self, audit_event_engine):
        """Test event metadata is preserved in retrieval."""
        metadata = {"tag": "test", "priority": "high", "reviewer": "qa-bot"}
        result = audit_event_engine.record_event(
            event_type="DATA_INGESTED",
            agent_id=AGENT,
            scope="scope_1",
            category=None,
            organization_id=ORG_ID,
            reporting_year=YEAR,
            metadata=metadata,
        )
        evt = audit_event_engine.get_event(result["event_id"])
        assert evt["metadata"] == metadata

    def test_get_events_combined_filters(self, audit_event_engine):
        """Test get_events with multiple simultaneous filters."""
        _record_simple_event(audit_event_engine, event_type="DATA_INGESTED", scope="scope_1", agent_id="GL-MRV-S1-001")
        _record_simple_event(audit_event_engine, event_type="DATA_INGESTED", scope="scope_2", agent_id="GL-MRV-S1-001")
        _record_simple_event(audit_event_engine, event_type="CALCULATION_COMPLETED", scope="scope_1", agent_id="GL-MRV-S3-006")
        result = audit_event_engine.get_events(
            ORG_ID, YEAR,
            event_type="DATA_INGESTED",
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
        )
        assert result["total_matching"] == 1
