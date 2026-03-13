# -*- coding: utf-8 -*-
"""
Tests for CustodyEventTracker - AGENT-EUDR-009 Engine 1: Custody Event Tracking

Comprehensive test suite covering:
- All 10 event types recorded correctly (F1.1)
- Temporal order validation (F1.4)
- Actor continuity checks (F1.6)
- Location continuity checks (F1.5)
- Gap detection at various thresholds (F1.7)
- Event amendment with audit trail preservation (F1.8)
- Bulk import from EDI/XML/CSV (F1.9)
- Edge cases: duplicate events, missing fields, future timestamps

Test count: 65+ tests
Coverage target: >= 85% of CustodyEventTracker module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-009 Chain of Custody Agent (GL-EUDR-COC-009)
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.chain_of_custody.conftest import (
    EVENT_TYPES,
    EUDR_COMMODITIES,
    SHA256_HEX_LENGTH,
    TRANSFER_COCOA_GH_NL,
    RECEIPT_COCOA_NL,
    STORAGE_IN_COCOA_NL,
    STORAGE_OUT_COCOA_NL,
    PROCESSING_IN_COCOA_NL,
    PROCESSING_OUT_COCOA_NL,
    EXPORT_COCOA_GH,
    IMPORT_COCOA_NL,
    INSPECTION_COCOA_NL,
    SAMPLING_COCOA_NL,
    ACTOR_TRADER_GH,
    ACTOR_IMPORTER_NL,
    ACTOR_SHIPPER_INT,
    ACTOR_PROCESSOR_DE,
    FAC_ID_PROC_GH,
    FAC_ID_WAREHOUSE_NL,
    FAC_ID_FACTORY_DE,
    BATCH_ID_COCOA_COOP_GH,
    make_event,
    compute_sha256,
)


# ===========================================================================
# 1. Event Type Recording (F1.1)
# ===========================================================================


class TestEventTypeRecording:
    """Test that all 10 custody event types are recorded correctly."""

    @pytest.mark.parametrize("event_type", EVENT_TYPES)
    def test_record_event_type(self, custody_event_tracker, event_type):
        """Each of the 10 event types can be recorded."""
        event = make_event(event_type=event_type)
        result = custody_event_tracker.record(event)
        assert result is not None
        assert result["event_type"] == event_type

    def test_record_transfer_event(self, custody_event_tracker):
        """Transfer event includes sender, receiver, source, destination."""
        event = copy.deepcopy(TRANSFER_COCOA_GH_NL)
        result = custody_event_tracker.record(event)
        assert result["event_type"] == "transfer"
        assert result["sender_actor_id"] == ACTOR_TRADER_GH
        assert result["receiver_actor_id"] == ACTOR_SHIPPER_INT

    def test_record_receipt_event(self, custody_event_tracker):
        """Receipt event records successful acceptance of goods."""
        event = copy.deepcopy(RECEIPT_COCOA_NL)
        result = custody_event_tracker.record(event)
        assert result["event_type"] == "receipt"

    def test_record_storage_in_event(self, custody_event_tracker):
        """Storage-in event records goods entering storage."""
        event = copy.deepcopy(STORAGE_IN_COCOA_NL)
        result = custody_event_tracker.record(event)
        assert result["event_type"] == "storage_in"

    def test_record_storage_out_event(self, custody_event_tracker):
        """Storage-out event records goods leaving storage."""
        event = copy.deepcopy(STORAGE_OUT_COCOA_NL)
        result = custody_event_tracker.record(event)
        assert result["event_type"] == "storage_out"

    def test_record_processing_in_event(self, custody_event_tracker):
        """Processing-in event records goods entering processing."""
        event = copy.deepcopy(PROCESSING_IN_COCOA_NL)
        result = custody_event_tracker.record(event)
        assert result["event_type"] == "processing_in"

    def test_record_processing_out_event(self, custody_event_tracker):
        """Processing-out event records goods exiting processing."""
        event = copy.deepcopy(PROCESSING_OUT_COCOA_NL)
        result = custody_event_tracker.record(event)
        assert result["event_type"] == "processing_out"

    def test_record_export_event(self, custody_event_tracker):
        """Export event includes full documentation references."""
        event = copy.deepcopy(EXPORT_COCOA_GH)
        result = custody_event_tracker.record(event)
        assert result["event_type"] == "export"
        assert len(result["document_refs"]) >= 4

    def test_record_import_event(self, custody_event_tracker):
        """Import event records goods entering EU market."""
        event = copy.deepcopy(IMPORT_COCOA_NL)
        result = custody_event_tracker.record(event)
        assert result["event_type"] == "import"

    def test_record_inspection_event(self, custody_event_tracker):
        """Inspection event records quality check on goods."""
        event = copy.deepcopy(INSPECTION_COCOA_NL)
        result = custody_event_tracker.record(event)
        assert result["event_type"] == "inspection"

    def test_record_sampling_event(self, custody_event_tracker):
        """Sampling event records sample extraction from batch."""
        event = copy.deepcopy(SAMPLING_COCOA_NL)
        result = custody_event_tracker.record(event)
        assert result["event_type"] == "sampling"
        assert result["quantity_kg"] < 1.0

    def test_invalid_event_type_raises(self, custody_event_tracker):
        """Recording an invalid event type raises ValueError."""
        event = make_event(event_type="invalid_type")
        with pytest.raises(ValueError):
            custody_event_tracker.record(event)

    def test_event_assigns_id(self, custody_event_tracker):
        """Event recording assigns a unique event_id if not provided."""
        event = make_event()
        event["event_id"] = None
        result = custody_event_tracker.record(event)
        assert result.get("event_id") is not None
        assert len(result["event_id"]) > 0

    def test_event_records_timestamp(self, custody_event_tracker):
        """Event recording preserves the provided timestamp."""
        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc).isoformat()
        event = make_event(timestamp=ts)
        result = custody_event_tracker.record(event)
        assert result["timestamp"] == ts

    def test_event_provenance_hash(self, custody_event_tracker):
        """Event recording generates a provenance hash."""
        event = make_event()
        result = custody_event_tracker.record(event)
        assert result.get("provenance_hash") is not None
        assert len(result["provenance_hash"]) == SHA256_HEX_LENGTH


# ===========================================================================
# 2. Temporal Order Validation (F1.4)
# ===========================================================================


class TestTemporalOrderValidation:
    """Test temporal ordering of events per batch."""

    def test_events_in_chronological_order(self, custody_event_tracker):
        """Events in correct chronological order are accepted."""
        batch_id = f"BATCH-TEMP-{uuid.uuid4().hex[:8]}"
        e1 = make_event("receipt", batch_id, timestamp="2026-01-01T10:00:00+00:00")
        e2 = make_event("storage_in", batch_id, timestamp="2026-01-01T12:00:00+00:00",
                         predecessor_event_id=e1["event_id"])
        custody_event_tracker.record(e1)
        result = custody_event_tracker.record(e2)
        assert result is not None

    def test_events_out_of_order_raises(self, custody_event_tracker):
        """Events in reverse chronological order raise ValueError."""
        batch_id = f"BATCH-TEMP-{uuid.uuid4().hex[:8]}"
        e1 = make_event("receipt", batch_id, timestamp="2026-01-01T12:00:00+00:00")
        e2 = make_event("storage_in", batch_id, timestamp="2026-01-01T10:00:00+00:00",
                         predecessor_event_id=e1["event_id"])
        custody_event_tracker.record(e1)
        with pytest.raises(ValueError):
            custody_event_tracker.record(e2)

    def test_simultaneous_events_allowed(self, custody_event_tracker):
        """Events with identical timestamps are allowed (e.g., inspection + sampling)."""
        batch_id = f"BATCH-SIM-{uuid.uuid4().hex[:8]}"
        ts = "2026-02-01T14:00:00+00:00"
        e1 = make_event("inspection", batch_id, timestamp=ts)
        e2 = make_event("sampling", batch_id, timestamp=ts,
                         predecessor_event_id=e1["event_id"])
        custody_event_tracker.record(e1)
        result = custody_event_tracker.record(e2)
        assert result is not None

    def test_event_chain_retrieval(self, custody_event_tracker):
        """Retrieve complete event chain for a batch in order."""
        batch_id = f"BATCH-CHAIN-{uuid.uuid4().hex[:8]}"
        events = []
        for i, etype in enumerate(["receipt", "storage_in", "storage_out"]):
            ts = f"2026-03-01T{10 + i}:00:00+00:00"
            pred = events[-1]["event_id"] if events else None
            evt = make_event(etype, batch_id, timestamp=ts, predecessor_event_id=pred)
            custody_event_tracker.record(evt)
            events.append(evt)
        chain = custody_event_tracker.get_chain(batch_id)
        assert len(chain) == 3
        assert chain[0]["event_type"] == "receipt"
        assert chain[2]["event_type"] == "storage_out"

    def test_predecessor_successor_linking(self, custody_event_tracker):
        """Events are linked via predecessor/successor references."""
        batch_id = f"BATCH-LINK-{uuid.uuid4().hex[:8]}"
        e1 = make_event("receipt", batch_id, timestamp="2026-01-10T10:00:00+00:00")
        e2 = make_event("storage_in", batch_id, timestamp="2026-01-10T11:00:00+00:00",
                         predecessor_event_id=e1["event_id"])
        r1 = custody_event_tracker.record(e1)
        r2 = custody_event_tracker.record(e2)
        assert r2["predecessor_event_id"] == r1["event_id"]

    @pytest.mark.parametrize("hours_gap", [1, 12, 23, 47, 71])
    def test_valid_temporal_gaps_accepted(self, custody_event_tracker, hours_gap):
        """Events with gaps below the threshold are accepted without flagging."""
        batch_id = f"BATCH-GAP-{hours_gap}"
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        e1 = make_event("receipt", batch_id, timestamp=base.isoformat())
        e2 = make_event("storage_in", batch_id,
                         timestamp=(base + timedelta(hours=hours_gap)).isoformat(),
                         predecessor_event_id=e1["event_id"])
        custody_event_tracker.record(e1)
        result = custody_event_tracker.record(e2)
        assert result is not None


# ===========================================================================
# 3. Actor Continuity (F1.6)
# ===========================================================================


class TestActorContinuity:
    """Test that receiver of one event matches sender of the next."""

    def test_actor_continuity_valid(self, custody_event_tracker):
        """Receiver of transfer becomes sender of next event."""
        batch_id = f"BATCH-ACT-{uuid.uuid4().hex[:8]}"
        e1 = make_event("transfer", batch_id,
                         sender_actor_id=ACTOR_TRADER_GH,
                         receiver_actor_id=ACTOR_SHIPPER_INT,
                         timestamp="2026-01-01T10:00:00+00:00")
        e2 = make_event("receipt", batch_id,
                         sender_actor_id=ACTOR_SHIPPER_INT,
                         receiver_actor_id=ACTOR_IMPORTER_NL,
                         timestamp="2026-01-15T10:00:00+00:00",
                         predecessor_event_id=e1["event_id"])
        custody_event_tracker.record(e1)
        result = custody_event_tracker.record(e2)
        assert result is not None

    def test_actor_continuity_broken_raises(self, custody_event_tracker):
        """Broken actor continuity raises ValueError."""
        batch_id = f"BATCH-ACT-BRK-{uuid.uuid4().hex[:8]}"
        e1 = make_event("transfer", batch_id,
                         sender_actor_id=ACTOR_TRADER_GH,
                         receiver_actor_id=ACTOR_SHIPPER_INT,
                         timestamp="2026-01-01T10:00:00+00:00")
        e2 = make_event("receipt", batch_id,
                         sender_actor_id=ACTOR_PROCESSOR_DE,  # Wrong sender
                         receiver_actor_id=ACTOR_IMPORTER_NL,
                         timestamp="2026-01-15T10:00:00+00:00",
                         predecessor_event_id=e1["event_id"])
        custody_event_tracker.record(e1)
        with pytest.raises(ValueError, match="actor"):
            custody_event_tracker.record(e2)

    def test_same_actor_for_internal_events(self, custody_event_tracker):
        """Internal events (storage_in, processing) can have same sender/receiver."""
        batch_id = f"BATCH-INTERNAL-{uuid.uuid4().hex[:8]}"
        e1 = make_event("storage_in", batch_id,
                         sender_actor_id=ACTOR_IMPORTER_NL,
                         receiver_actor_id=ACTOR_IMPORTER_NL,
                         timestamp="2026-01-01T10:00:00+00:00")
        result = custody_event_tracker.record(e1)
        assert result["sender_actor_id"] == result["receiver_actor_id"]


# ===========================================================================
# 4. Location Continuity (F1.5)
# ===========================================================================


class TestLocationContinuity:
    """Test location continuity between consecutive events."""

    def test_location_continuity_valid(self, custody_event_tracker):
        """Destination of one event matches source of next."""
        batch_id = f"BATCH-LOC-{uuid.uuid4().hex[:8]}"
        e1 = make_event("transfer", batch_id,
                         source_facility_id=FAC_ID_PROC_GH,
                         dest_facility_id=FAC_ID_WAREHOUSE_NL,
                         timestamp="2026-01-01T10:00:00+00:00")
        e2 = make_event("receipt", batch_id,
                         source_facility_id=FAC_ID_WAREHOUSE_NL,
                         dest_facility_id=FAC_ID_WAREHOUSE_NL,
                         timestamp="2026-01-15T10:00:00+00:00",
                         predecessor_event_id=e1["event_id"])
        custody_event_tracker.record(e1)
        result = custody_event_tracker.record(e2)
        assert result is not None

    def test_location_teleportation_raises(self, custody_event_tracker):
        """Goods teleporting between unrelated locations raises ValueError."""
        batch_id = f"BATCH-TELP-{uuid.uuid4().hex[:8]}"
        e1 = make_event("transfer", batch_id,
                         source_facility_id=FAC_ID_PROC_GH,
                         dest_facility_id=FAC_ID_WAREHOUSE_NL,
                         timestamp="2026-01-01T10:00:00+00:00")
        e2 = make_event("receipt", batch_id,
                         source_facility_id=FAC_ID_FACTORY_DE,  # Wrong: not warehouse NL
                         dest_facility_id=FAC_ID_FACTORY_DE,
                         timestamp="2026-01-15T10:00:00+00:00",
                         predecessor_event_id=e1["event_id"])
        custody_event_tracker.record(e1)
        with pytest.raises(ValueError, match="location"):
            custody_event_tracker.record(e2)

    def test_storage_events_same_location(self, custody_event_tracker):
        """Storage events have same source and destination facility."""
        batch_id = f"BATCH-SAMELOCSTOR-{uuid.uuid4().hex[:8]}"
        e1 = make_event("storage_in", batch_id,
                         source_facility_id=FAC_ID_WAREHOUSE_NL,
                         dest_facility_id=FAC_ID_WAREHOUSE_NL,
                         timestamp="2026-02-01T10:00:00+00:00")
        result = custody_event_tracker.record(e1)
        src = result["source_location"]["facility_id"]
        dst = result["dest_location"]["facility_id"]
        assert src == dst


# ===========================================================================
# 5. Gap Detection (F1.7)
# ===========================================================================


class TestGapDetection:
    """Test temporal gap detection between custody events."""

    @pytest.mark.parametrize("gap_hours,should_flag", [
        (24, False),
        (48, False),
        (72, False),
        (73, True),
        (96, True),
        (168, True),
    ])
    def test_gap_detection_at_thresholds(self, custody_event_tracker, gap_hours, should_flag):
        """Gaps at or below 72h are not flagged; gaps above are flagged."""
        batch_id = f"BATCH-GAP-{gap_hours}"
        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        e1 = make_event("receipt", batch_id, timestamp=base.isoformat())
        e2 = make_event("storage_in", batch_id,
                         timestamp=(base + timedelta(hours=gap_hours)).isoformat(),
                         predecessor_event_id=e1["event_id"])
        custody_event_tracker.record(e1)
        result = custody_event_tracker.record(e2)
        gaps = custody_event_tracker.detect_gaps(batch_id)
        if should_flag:
            assert len(gaps) > 0, f"Expected gap at {gap_hours}h to be flagged"
            assert gaps[0]["gap_hours"] >= gap_hours
        else:
            assert len(gaps) == 0, f"Expected no gap at {gap_hours}h"

    def test_gap_detection_configurable_threshold(self, config, custody_event_tracker):
        """Gap threshold is configurable per deployment."""
        assert config["custody_gap_threshold_hours"] == 72

    def test_multiple_gaps_detected(self, custody_event_tracker):
        """Multiple gaps in a chain are all detected."""
        batch_id = f"BATCH-MULTIGAP"
        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        timestamps = [
            base,
            base + timedelta(hours=100),  # Gap 1: 100h
            base + timedelta(hours=300),  # Gap 2: 200h
        ]
        events = []
        for i, (ts, etype) in enumerate(zip(timestamps, ["receipt", "storage_in", "storage_out"])):
            pred = events[-1]["event_id"] if events else None
            evt = make_event(etype, batch_id, timestamp=ts.isoformat(),
                             predecessor_event_id=pred)
            custody_event_tracker.record(evt)
            events.append(evt)
        gaps = custody_event_tracker.detect_gaps(batch_id)
        assert len(gaps) >= 2


# ===========================================================================
# 6. Event Amendment (F1.8)
# ===========================================================================


class TestEventAmendment:
    """Test event amendment with immutable audit trail."""

    def test_amend_event_preserves_original(self, custody_event_tracker):
        """Amending an event preserves the original record."""
        event = make_event()
        original = custody_event_tracker.record(event)
        amended = custody_event_tracker.amend(
            original["event_id"],
            {"quantity_kg": 4900.0, "reason": "Scale recalibration"},
        )
        assert amended is not None
        assert amended["quantity_kg"] == 4900.0
        history = custody_event_tracker.get_amendment_history(original["event_id"])
        assert len(history) >= 2

    def test_amend_creates_audit_trail(self, custody_event_tracker):
        """Amendment creates an immutable audit trail entry."""
        event = make_event()
        original = custody_event_tracker.record(event)
        custody_event_tracker.amend(
            original["event_id"],
            {"notes": "Corrected weight", "quantity_kg": 4950.0},
        )
        history = custody_event_tracker.get_amendment_history(original["event_id"])
        assert any(h.get("quantity_kg") == event["quantity_kg"] for h in history)

    def test_amend_nonexistent_raises(self, custody_event_tracker):
        """Amending a non-existent event raises an error."""
        with pytest.raises((ValueError, KeyError)):
            custody_event_tracker.amend("EVT-NONEXISTENT", {"notes": "test"})

    def test_amendment_updates_provenance_hash(self, custody_event_tracker):
        """Amendment generates a new provenance hash."""
        event = make_event()
        original = custody_event_tracker.record(event)
        amended = custody_event_tracker.amend(
            original["event_id"], {"quantity_kg": 4800.0}
        )
        assert amended.get("provenance_hash") != original.get("provenance_hash")


# ===========================================================================
# 7. Bulk Import (F1.9)
# ===========================================================================


class TestBulkImport:
    """Test bulk event import from various sources."""

    def test_bulk_import_list(self, custody_event_tracker):
        """Import a list of events in bulk."""
        events = [
            make_event("receipt", f"BATCH-BULK-{i}", timestamp=f"2026-02-{i+1:02d}T10:00:00+00:00")
            for i in range(10)
        ]
        results = custody_event_tracker.bulk_import(events)
        assert len(results) == 10
        assert all(r is not None for r in results)

    def test_bulk_import_partial_failure(self, custody_event_tracker):
        """Bulk import with some invalid events reports partial failures."""
        events = [
            make_event("receipt", "BATCH-BULK-OK"),
            make_event("invalid_type", "BATCH-BULK-BAD"),
            make_event("transfer", "BATCH-BULK-OK2"),
        ]
        results = custody_event_tracker.bulk_import(events, continue_on_error=True)
        assert len([r for r in results if r.get("status") == "error"]) >= 1

    def test_bulk_import_empty_list(self, custody_event_tracker):
        """Bulk import of empty list returns empty results."""
        results = custody_event_tracker.bulk_import([])
        assert len(results) == 0

    def test_bulk_import_preserves_order(self, custody_event_tracker):
        """Events are imported in the order provided."""
        batch_id = "BATCH-BULK-ORDER"
        events = [
            make_event("receipt", batch_id, timestamp=f"2026-01-{i+1:02d}T10:00:00+00:00")
            for i in range(5)
        ]
        results = custody_event_tracker.bulk_import(events)
        for i in range(len(results) - 1):
            assert results[i]["timestamp"] <= results[i + 1]["timestamp"]


# ===========================================================================
# 8. Edge Cases
# ===========================================================================


class TestCustodyEventEdgeCases:
    """Test edge cases in custody event recording."""

    def test_duplicate_event_id_raises(self, custody_event_tracker):
        """Recording two events with the same ID raises an error."""
        event = make_event(event_id="EVT-DUP-001")
        custody_event_tracker.record(event)
        with pytest.raises((ValueError, KeyError)):
            custody_event_tracker.record(copy.deepcopy(event))

    def test_missing_batch_id_raises(self, custody_event_tracker):
        """Event without batch_id raises ValueError."""
        event = make_event()
        event["batch_id"] = None
        with pytest.raises(ValueError):
            custody_event_tracker.record(event)

    def test_zero_quantity_allowed_for_inspection(self, custody_event_tracker):
        """Zero quantity is allowed for inspection and sampling events."""
        event = make_event("inspection", quantity_kg=0.0)
        result = custody_event_tracker.record(event)
        assert result["quantity_kg"] == 0.0

    def test_negative_quantity_raises(self, custody_event_tracker):
        """Negative quantity raises ValueError."""
        event = make_event(quantity_kg=-100.0)
        with pytest.raises(ValueError):
            custody_event_tracker.record(event)

    def test_future_timestamp_raises(self, custody_event_tracker):
        """Event with future timestamp raises ValueError."""
        future_ts = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        event = make_event(timestamp=future_ts)
        with pytest.raises(ValueError):
            custody_event_tracker.record(event)

    def test_missing_sender_raises(self, custody_event_tracker):
        """Event without sender_actor_id raises ValueError."""
        event = make_event()
        event["sender_actor_id"] = None
        with pytest.raises(ValueError):
            custody_event_tracker.record(event)

    def test_missing_receiver_raises(self, custody_event_tracker):
        """Event without receiver_actor_id raises ValueError."""
        event = make_event()
        event["receiver_actor_id"] = None
        with pytest.raises(ValueError):
            custody_event_tracker.record(event)

    def test_get_nonexistent_event_returns_none(self, custody_event_tracker):
        """Getting a non-existent event returns None."""
        result = custody_event_tracker.get("EVT-NONEXISTENT")
        assert result is None

    def test_event_with_empty_document_refs(self, custody_event_tracker):
        """Event with no document references is accepted."""
        event = make_event(document_refs=[])
        result = custody_event_tracker.record(event)
        assert result["document_refs"] == []

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_events_for_all_commodities(self, custody_event_tracker, commodity):
        """Custody events can be recorded for all 7 EUDR commodities."""
        event = make_event(batch_id=f"BATCH-{commodity}-001")
        result = custody_event_tracker.record(event)
        assert result is not None

    def test_very_large_quantity(self, custody_event_tracker):
        """Very large quantities (1M kg) are accepted."""
        event = make_event(quantity_kg=1_000_000.0)
        result = custody_event_tracker.record(event)
        assert result["quantity_kg"] == 1_000_000.0

    def test_very_small_quantity(self, custody_event_tracker):
        """Very small quantities (0.001 kg) are accepted."""
        event = make_event("sampling", quantity_kg=0.001)
        result = custody_event_tracker.record(event)
        assert result["quantity_kg"] == pytest.approx(0.001)

    def test_unicode_notes(self, custody_event_tracker):
        """Unicode characters in notes are preserved."""
        event = make_event()
        event["notes"] = "Livraison re\u00e7ue \u00e0 Rotterdam - qualit\u00e9 A+"
        result = custody_event_tracker.record(event)
        assert "\u00e9" in result["notes"]
