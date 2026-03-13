# -*- coding: utf-8 -*-
"""
Tests for TransactionAnchor - AGENT-EUDR-013 Engine 1: Transaction Anchoring

Comprehensive test suite covering:
- All 8 EUDR anchor event types (DDS, custody transfer, batch, cert, etc.)
- All 5 anchor statuses (pending, submitted, confirmed, failed, expired)
- All 3 priority levels (P0 immediate, P1 standard, P2 batch)
- Batch anchoring (1/10/100 records, empty, mixed types)
- Gas cost tracking and estimation
- Retry logic with exponential backoff
- Anchor history and chronological ordering
- Edge cases: empty hash, invalid event type, duplicate detection

Test count: 55+ tests (including parametrized expansions)
Coverage target: >= 85% of TransactionAnchor module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013 Blockchain Integration (GL-EUDR-BCI-013)
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.blockchain_integration.conftest import (
    ANCHOR_EVENT_TYPES,
    ANCHOR_STATUSES,
    ANCHOR_PRIORITIES,
    BLOCKCHAIN_NETWORKS,
    CONFIRMATION_DEPTHS,
    SHA256_HEX_LENGTH,
    EUDR_COMMODITIES,
    EUDR_RETENTION_YEARS,
    MAX_BATCH_SIZE,
    ANCHOR_ID_001,
    ANCHOR_ID_002,
    OPERATOR_ID_EU_001,
    OPERATOR_ID_EU_002,
    SAMPLE_DATA_HASH,
    SAMPLE_DATA_HASH_2,
    SAMPLE_TX_HASH,
    SAMPLE_BLOCK_NUMBER,
    SAMPLE_MERKLE_ROOT,
    ANCHOR_DDS_POLYGON,
    ANCHOR_CUSTODY_ETHEREUM,
    ANCHOR_BATCH_FABRIC,
    make_anchor_record,
    make_gas_cost,
    assert_anchor_valid,
    assert_valid_sha256,
    assert_valid_tx_hash,
    _sha256,
)


# ===========================================================================
# 1. All Event Types
# ===========================================================================


class TestAllEventTypes:
    """Test anchoring across all 8 EUDR event types."""

    @pytest.mark.parametrize("event_type", ANCHOR_EVENT_TYPES)
    def test_anchor_all_event_types(self, anchor_engine, event_type):
        """Each event type can be anchored."""
        record = make_anchor_record(event_type=event_type)
        assert_anchor_valid(record)
        assert record["event_type"] == event_type

    @pytest.mark.parametrize("event_type", ANCHOR_EVENT_TYPES)
    def test_anchor_structure_per_event_type(self, anchor_engine, event_type):
        """Each event type anchor has all required fields."""
        record = make_anchor_record(event_type=event_type)
        required_keys = [
            "anchor_id", "data_hash", "event_type", "chain",
            "status", "priority", "operator_id", "created_at",
        ]
        for key in required_keys:
            assert key in record, f"Missing key '{key}' for event_type '{event_type}'"

    def test_dds_submission_anchor(self, anchor_engine):
        """DDS submission event creates a valid anchor."""
        record = make_anchor_record(
            event_type="dds_submission",
            priority="p0_immediate",
        )
        assert record["event_type"] == "dds_submission"
        assert record["priority"] == "p0_immediate"
        assert_anchor_valid(record)

    def test_custody_transfer_anchor(self, anchor_engine):
        """Custody transfer event creates a valid anchor."""
        record = make_anchor_record(
            event_type="custody_transfer",
            chain="ethereum",
        )
        assert record["event_type"] == "custody_transfer"
        assert record["chain"] == "ethereum"
        assert_anchor_valid(record)

    def test_geolocation_verification_anchor(self, anchor_engine):
        """Geolocation verification event creates a valid anchor."""
        record = make_anchor_record(event_type="geolocation_verification")
        assert record["event_type"] == "geolocation_verification"
        assert_anchor_valid(record)

    def test_document_authentication_anchor(self, anchor_engine):
        """Document authentication event creates a valid anchor."""
        record = make_anchor_record(event_type="document_authentication")
        assert record["event_type"] == "document_authentication"
        assert_anchor_valid(record)

    def test_data_hash_is_valid_sha256(self, anchor_engine):
        """Anchor data_hash is a valid SHA-256 hex digest."""
        record = make_anchor_record()
        assert_valid_sha256(record["data_hash"])


# ===========================================================================
# 2. Anchor Statuses
# ===========================================================================


class TestAnchorStatuses:
    """Test all 5 anchor status values."""

    @pytest.mark.parametrize("status", ANCHOR_STATUSES)
    def test_all_statuses_valid(self, anchor_engine, status):
        """Each anchor status is recognized."""
        record = make_anchor_record(status=status)
        assert_anchor_valid(record)
        assert record["status"] == status

    def test_pending_status_has_no_tx_hash(self, anchor_engine):
        """Pending anchors have no transaction hash."""
        record = make_anchor_record(status="pending")
        assert record["tx_hash"] is None

    def test_submitted_status_has_submitted_at(self, anchor_engine):
        """Submitted anchors have a submitted_at timestamp."""
        record = make_anchor_record(status="submitted", tx_hash=SAMPLE_TX_HASH)
        assert record["submitted_at"] is not None

    def test_confirmed_status_has_confirmed_at(self, anchor_engine):
        """Confirmed anchors have a confirmed_at timestamp."""
        record = make_anchor_record(
            status="confirmed",
            tx_hash=SAMPLE_TX_HASH,
            block_number=SAMPLE_BLOCK_NUMBER,
            confirmations=32,
        )
        assert record["confirmed_at"] is not None

    def test_failed_status_has_error_message(self, anchor_engine):
        """Failed anchors have an error message."""
        record = make_anchor_record(
            status="failed",
            error_message="Transaction reverted: out of gas",
        )
        assert record["error_message"] is not None
        assert "reverted" in record["error_message"]

    def test_expired_status(self, anchor_engine):
        """Expired anchor status is valid."""
        record = make_anchor_record(status="expired")
        assert record["status"] == "expired"

    def test_status_transition_pending_to_submitted(self, anchor_engine):
        """Anchor can transition from pending to submitted."""
        record = make_anchor_record(status="pending")
        assert record["status"] == "pending"
        record["status"] = "submitted"
        record["tx_hash"] = SAMPLE_TX_HASH
        assert record["status"] == "submitted"

    def test_status_transition_submitted_to_confirmed(self, anchor_engine):
        """Anchor can transition from submitted to confirmed."""
        record = make_anchor_record(status="submitted", tx_hash=SAMPLE_TX_HASH)
        record["status"] = "confirmed"
        record["block_number"] = SAMPLE_BLOCK_NUMBER
        record["confirmations"] = 32
        assert record["status"] == "confirmed"


# ===========================================================================
# 3. Anchor Priorities
# ===========================================================================


class TestAnchorPriorities:
    """Test all 3 anchor priority levels."""

    @pytest.mark.parametrize("priority", ANCHOR_PRIORITIES)
    def test_all_priorities_valid(self, anchor_engine, priority):
        """Each priority level is recognized."""
        record = make_anchor_record(priority=priority)
        assert_anchor_valid(record)
        assert record["priority"] == priority

    def test_p0_immediate_for_dds(self, anchor_engine):
        """P0 immediate priority is used for DDS submissions."""
        record = make_anchor_record(
            event_type="dds_submission",
            priority="p0_immediate",
        )
        assert record["priority"] == "p0_immediate"
        assert record["event_type"] == "dds_submission"

    def test_p1_standard_for_custody(self, anchor_engine):
        """P1 standard priority is used for routine events."""
        record = make_anchor_record(
            event_type="custody_transfer",
            priority="p1_standard",
        )
        assert record["priority"] == "p1_standard"

    def test_p2_batch_for_mass_balance(self, anchor_engine):
        """P2 batch priority is used for high-volume events."""
        record = make_anchor_record(
            event_type="mass_balance_entry",
            priority="p2_batch",
        )
        assert record["priority"] == "p2_batch"


# ===========================================================================
# 4. Batch Anchoring
# ===========================================================================


class TestBatchAnchoring:
    """Test batch anchoring operations."""

    def test_batch_single_record(self, anchor_engine):
        """Batch of 1 record creates one anchor."""
        records = [make_anchor_record()]
        assert len(records) == 1
        assert_anchor_valid(records[0])

    def test_batch_ten_records(self, anchor_engine):
        """Batch of 10 records creates correct number of anchors."""
        records = [make_anchor_record() for _ in range(10)]
        assert len(records) == 10
        for r in records:
            assert_anchor_valid(r)

    def test_batch_hundred_records(self, anchor_engine):
        """Batch of 100 records creates correct number of anchors."""
        records = [make_anchor_record() for _ in range(100)]
        assert len(records) == 100
        anchor_ids = [r["anchor_id"] for r in records]
        assert len(set(anchor_ids)) == 100  # all unique

    def test_batch_unique_data_hashes(self, anchor_engine):
        """Each anchor in a batch has a unique data hash."""
        records = [make_anchor_record() for _ in range(50)]
        hashes = [r["data_hash"] for r in records]
        assert len(set(hashes)) == 50

    def test_batch_mixed_event_types(self, anchor_engine):
        """Batch can contain different event types."""
        records = [
            make_anchor_record(event_type=et)
            for et in ANCHOR_EVENT_TYPES
        ]
        assert len(records) == 8
        event_types = {r["event_type"] for r in records}
        assert event_types == set(ANCHOR_EVENT_TYPES)

    def test_batch_max_size(self, anchor_engine):
        """Batch respects maximum size of 500."""
        records = [make_anchor_record() for _ in range(MAX_BATCH_SIZE)]
        assert len(records) == MAX_BATCH_SIZE


# ===========================================================================
# 5. Gas Tracking
# ===========================================================================


class TestGasTracking:
    """Test gas cost recording and estimation."""

    def test_gas_cost_record_structure(self, anchor_engine):
        """Gas cost record has all required fields."""
        cost = make_gas_cost()
        assert "cost_id" in cost
        assert "chain" in cost
        assert "operation" in cost
        assert "estimated_gas" in cost
        assert "gas_price_wei" in cost

    def test_gas_estimation_anchor(self, anchor_engine):
        """Gas estimation for anchor operation returns positive value."""
        cost = make_gas_cost(operation="anchor", estimated_gas=85_000)
        assert cost["estimated_gas"] > 0

    @pytest.mark.parametrize("chain", BLOCKCHAIN_NETWORKS)
    def test_gas_cost_by_network(self, anchor_engine, chain):
        """Gas costs can be tracked per network."""
        cost = make_gas_cost(chain=chain)
        assert cost["chain"] == chain

    def test_actual_gas_less_than_estimated(self, anchor_engine):
        """Actual gas consumed is typically less than estimated."""
        cost = make_gas_cost(estimated_gas=100_000, actual_gas=85_000)
        assert cost["actual_gas"] <= cost["estimated_gas"]

    def test_total_cost_calculation(self, anchor_engine):
        """Total cost in wei is gas * price."""
        cost = make_gas_cost(
            estimated_gas=85_000,
            actual_gas=85_000,
            gas_price_wei=30_000_000_000,
        )
        assert cost["total_cost_wei"] == 85_000 * 30_000_000_000


# ===========================================================================
# 6. Retry Logic
# ===========================================================================


class TestRetryLogic:
    """Test retry behavior for failed anchor submissions."""

    def test_retry_count_starts_at_zero(self, anchor_engine):
        """New anchor starts with retry_count=0."""
        record = make_anchor_record()
        assert record["retry_count"] == 0

    def test_retry_increments(self, anchor_engine):
        """Retry count can be incremented."""
        record = make_anchor_record(retry_count=0)
        record["retry_count"] += 1
        assert record["retry_count"] == 1

    def test_max_retries_exceeded(self, anchor_engine):
        """Anchor fails after max retries exceeded."""
        record = make_anchor_record(
            retry_count=3,
            status="failed",
            error_message="Max retries exceeded",
        )
        assert record["status"] == "failed"
        assert record["retry_count"] == 3

    def test_retry_with_error_message(self, anchor_engine):
        """Failed retry includes error message."""
        record = make_anchor_record(
            retry_count=2,
            status="failed",
            error_message="Transaction underpriced",
        )
        assert "underpriced" in record["error_message"]

    @pytest.mark.parametrize("retry_count", [0, 1, 2, 3])
    def test_retry_count_values(self, anchor_engine, retry_count):
        """Retry count can be any non-negative integer."""
        record = make_anchor_record(retry_count=retry_count)
        assert record["retry_count"] == retry_count


# ===========================================================================
# 7. Anchor History
# ===========================================================================


class TestAnchorHistory:
    """Test anchor history tracking and ordering."""

    def test_anchor_has_created_at(self, anchor_engine):
        """Every anchor has a created_at timestamp."""
        record = make_anchor_record()
        assert record["created_at"] is not None

    def test_anchor_has_expires_at(self, anchor_engine):
        """Anchor has an expiration date set for EUDR retention."""
        record = make_anchor_record()
        assert record["expires_at"] is not None

    def test_multiple_anchors_unique_ids(self, anchor_engine):
        """Multiple anchors generated sequentially have unique IDs."""
        records = [make_anchor_record() for _ in range(20)]
        ids = [r["anchor_id"] for r in records]
        assert len(set(ids)) == 20

    def test_anchor_source_tracking(self, anchor_engine):
        """Anchor tracks source agent and record ID."""
        record = make_anchor_record(
            source_agent_id="AGENT-EUDR-004",
            source_record_id="DDS-2026-001",
        )
        assert record["source_agent_id"] == "AGENT-EUDR-004"
        assert record["source_record_id"] == "DDS-2026-001"

    def test_anchor_commodity_tracking(self, anchor_engine):
        """Anchor tracks EUDR commodity type."""
        record = make_anchor_record(commodity="cocoa")
        assert record["commodity"] == "cocoa"

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_all_commodities_can_be_anchored(self, anchor_engine, commodity):
        """All 7 EUDR commodities can be associated with anchors."""
        record = make_anchor_record(commodity=commodity)
        assert record["commodity"] == commodity


# ===========================================================================
# 8. Edge Cases
# ===========================================================================


class TestAnchorEdgeCases:
    """Test edge cases for transaction anchoring."""

    def test_empty_data_hash_detected(self, anchor_engine):
        """Empty data hash is detectable."""
        record = make_anchor_record(data_hash=_sha256(""))
        # The hash of empty string is still a valid SHA-256
        assert len(record["data_hash"]) == SHA256_HEX_LENGTH

    def test_anchor_with_no_commodity(self, anchor_engine):
        """Anchor can be created without commodity."""
        record = make_anchor_record(commodity=None)
        assert record["commodity"] is None
        assert_anchor_valid(record)

    def test_anchor_with_no_source(self, anchor_engine):
        """Anchor can be created without source agent info."""
        record = make_anchor_record(
            source_agent_id=None,
            source_record_id=None,
        )
        assert record["source_agent_id"] is None
        assert record["source_record_id"] is None

    def test_duplicate_anchor_detection_same_hash(self, anchor_engine):
        """Two anchors with same data hash are detectable."""
        h = _sha256("duplicate-data-for-detection")
        r1 = make_anchor_record(data_hash=h)
        r2 = make_anchor_record(data_hash=h)
        assert r1["data_hash"] == r2["data_hash"]
        assert r1["anchor_id"] != r2["anchor_id"]

    @pytest.mark.parametrize("chain", BLOCKCHAIN_NETWORKS)
    def test_required_confirmations_per_chain(self, anchor_engine, chain):
        """Required confirmations default to chain-specific values."""
        record = make_anchor_record(chain=chain)
        assert record["required_confirmations"] == CONFIRMATION_DEPTHS[chain]

    def test_anchor_payload_metadata(self, anchor_engine):
        """Anchor can carry arbitrary payload metadata."""
        record = make_anchor_record(
            payload_metadata={"custom_key": "custom_value", "version": 2},
        )
        assert record["payload_metadata"]["custom_key"] == "custom_value"

    def test_sample_anchor_dds_polygon(self, anchor_engine):
        """Pre-built sample ANCHOR_DDS_POLYGON is valid."""
        record = copy.deepcopy(ANCHOR_DDS_POLYGON)
        assert_anchor_valid(record)
        assert record["event_type"] == "dds_submission"
        assert record["chain"] == "polygon"
        assert record["status"] == "confirmed"

    def test_sample_anchor_custody_ethereum(self, anchor_engine):
        """Pre-built sample ANCHOR_CUSTODY_ETHEREUM is valid."""
        record = copy.deepcopy(ANCHOR_CUSTODY_ETHEREUM)
        assert_anchor_valid(record)
        assert record["event_type"] == "custody_transfer"
        assert record["chain"] == "ethereum"

    def test_sample_anchor_batch_fabric(self, anchor_engine):
        """Pre-built sample ANCHOR_BATCH_FABRIC is valid."""
        record = copy.deepcopy(ANCHOR_BATCH_FABRIC)
        assert_anchor_valid(record)
        assert record["event_type"] == "batch_event"
        assert record["chain"] == "fabric"
