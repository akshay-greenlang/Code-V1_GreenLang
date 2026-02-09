# -*- coding: utf-8 -*-
"""
Unit tests for DistributionEngine
==================================

AGENT-DATA-008: Supplier Questionnaire Processor
Tests all methods of DistributionEngine with comprehensive coverage.
Validates campaign creation, batch distribution, status tracking,
access-token generation, cancellation, redistribution, campaign
summaries, and SHA-256 provenance hashing.

Total: ~70 tests
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from greenlang.supplier_questionnaire.distribution import DistributionEngine
from greenlang.supplier_questionnaire.models import (
    Distribution,
    DistributionChannel,
    DistributionStatus,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def engine() -> DistributionEngine:
    """Fresh DistributionEngine per test."""
    return DistributionEngine()


@pytest.fixture
def small_engine() -> DistributionEngine:
    """Engine with a small distribution limit."""
    return DistributionEngine({"max_distributions": 5, "max_batch_size": 3})


@pytest.fixture
def sample_suppliers() -> List[Dict[str, str]]:
    """Three sample supplier dicts."""
    return [
        {"id": "SUP001", "name": "Acme Corp", "email": "acme@example.com"},
        {"id": "SUP002", "name": "Globex Inc", "email": "globex@example.com"},
        {"id": "SUP003", "name": "Initech LLC", "email": "initech@example.com"},
    ]


@pytest.fixture
def campaign_id(engine: DistributionEngine) -> str:
    """Create a basic campaign and return its ID."""
    return engine.create_campaign(
        name="Q1 2025 CDP",
        template_id="tmpl-001",
        supplier_ids=["SUP001", "SUP002"],
        channel="email",
        deadline_days=30,
    )


# ===================================================================
# TEST CLASS: Create Campaign
# ===================================================================

class TestCreateCampaign:
    """Tests for DistributionEngine.create_campaign()."""

    def test_create_campaign_returns_id(self, engine):
        cid = engine.create_campaign(
            name="Camp", template_id="t1", supplier_ids=["s1"],
        )
        assert cid.startswith("camp-")

    def test_create_campaign_stores_campaign(self, engine):
        cid = engine.create_campaign(
            name="Camp", template_id="t1", supplier_ids=["s1"],
        )
        camp = engine.get_campaign(cid)
        assert camp["campaign_id"] == cid

    def test_create_campaign_creates_distributions(self, engine):
        cid = engine.create_campaign(
            name="Camp", template_id="t1", supplier_ids=["s1", "s2", "s3"],
        )
        dists = engine.list_distributions(campaign_id=cid)
        assert len(dists) == 3

    def test_create_campaign_name_stored(self, engine):
        cid = engine.create_campaign(
            name="My Camp", template_id="t1", supplier_ids=["s1"],
        )
        camp = engine.get_campaign(cid)
        assert camp["name"] == "My Camp"

    def test_create_campaign_template_id_stored(self, engine):
        cid = engine.create_campaign(
            name="C", template_id="tmpl-xyz", supplier_ids=["s1"],
        )
        camp = engine.get_campaign(cid)
        assert camp["template_id"] == "tmpl-xyz"

    def test_create_campaign_empty_name_raises(self, engine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.create_campaign(
                name="", template_id="t1", supplier_ids=["s1"],
            )

    def test_create_campaign_empty_template_raises(self, engine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.create_campaign(
                name="C", template_id="", supplier_ids=["s1"],
            )

    def test_create_campaign_empty_suppliers_raises(self, engine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.create_campaign(
                name="C", template_id="t1", supplier_ids=[],
            )

    def test_create_campaign_stats_increment(self, engine):
        engine.create_campaign(
            name="C", template_id="t1", supplier_ids=["s1"],
        )
        stats = engine.get_statistics()
        assert stats["campaigns_created"] == 1

    def test_create_campaign_provenance_hash_set(self, engine):
        cid = engine.create_campaign(
            name="C", template_id="t1", supplier_ids=["s1"],
        )
        camp = engine.get_campaign(cid)
        assert len(camp["provenance_hash"]) == 64

    def test_create_campaign_deadline_computed(self, engine):
        cid = engine.create_campaign(
            name="C", template_id="t1", supplier_ids=["s1"],
            deadline_days=60,
        )
        camp = engine.get_campaign(cid)
        assert "deadline" in camp

    def test_create_campaign_channel_applied(self, engine):
        cid = engine.create_campaign(
            name="C", template_id="t1", supplier_ids=["s1"],
            channel="portal",
        )
        dists = engine.list_distributions(campaign_id=cid)
        assert dists[0].channel == DistributionChannel.PORTAL


# ===================================================================
# TEST CLASS: Distribute
# ===================================================================

class TestDistribute:
    """Tests for DistributionEngine.distribute()."""

    def test_distribute_single_supplier(self, engine):
        dists = engine.distribute(
            template_id="t1",
            supplier_list=[{"id": "S1", "name": "A", "email": "a@b.com"}],
        )
        assert len(dists) == 1
        assert dists[0].supplier_id == "S1"

    def test_distribute_multiple_suppliers(self, engine, sample_suppliers):
        dists = engine.distribute(
            template_id="t1", supplier_list=sample_suppliers,
        )
        assert len(dists) == 3

    def test_distribute_via_email(self, engine):
        dists = engine.distribute(
            template_id="t1",
            supplier_list=[{"id": "S1", "name": "A", "email": "a@b.com"}],
            channel="email",
        )
        assert dists[0].channel == DistributionChannel.EMAIL

    def test_distribute_via_portal(self, engine):
        dists = engine.distribute(
            template_id="t1",
            supplier_list=[{"id": "S1", "name": "A", "email": "a@b.com"}],
            channel="portal",
        )
        assert dists[0].channel == DistributionChannel.PORTAL

    def test_distribute_via_api(self, engine):
        dists = engine.distribute(
            template_id="t1",
            supplier_list=[{"id": "S1", "name": "A", "email": "a@b.com"}],
            channel="api",
        )
        assert dists[0].channel == DistributionChannel.API

    def test_distribute_via_bulk_upload(self, engine):
        dists = engine.distribute(
            template_id="t1",
            supplier_list=[{"id": "S1", "name": "A", "email": "a@b.com"}],
            channel="bulk_upload",
        )
        assert dists[0].channel == DistributionChannel.BULK_UPLOAD

    def test_distribute_empty_template_raises(self, engine, sample_suppliers):
        with pytest.raises(ValueError, match="non-empty"):
            engine.distribute(template_id="", supplier_list=sample_suppliers)

    def test_distribute_empty_list_raises(self, engine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.distribute(template_id="t1", supplier_list=[])

    def test_distribute_invalid_channel_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown channel"):
            engine.distribute(
                template_id="t1",
                supplier_list=[{"id": "S1", "name": "A", "email": "a@b.com"}],
                channel="carrier_pigeon",
            )

    def test_distribute_batch_exceeds_max_raises(self, small_engine):
        suppliers = [{"id": f"S{i}", "name": f"N{i}", "email": f"e{i}@x.com"}
                     for i in range(10)]
        with pytest.raises(ValueError, match="exceeds"):
            small_engine.distribute(template_id="t1", supplier_list=suppliers)

    def test_distribute_skips_empty_supplier_id(self, engine):
        suppliers = [
            {"id": "", "name": "Bad", "email": "bad@x.com"},
            {"id": "S1", "name": "Good", "email": "good@x.com"},
        ]
        dists = engine.distribute(template_id="t1", supplier_list=suppliers)
        assert len(dists) == 1
        assert dists[0].supplier_id == "S1"

    def test_distribute_initial_status_is_sent(self, engine):
        dists = engine.distribute(
            template_id="t1",
            supplier_list=[{"id": "S1", "name": "A", "email": "a@b.com"}],
        )
        assert dists[0].status == DistributionStatus.SENT

    def test_distribute_sets_sent_at_timestamp(self, engine):
        dists = engine.distribute(
            template_id="t1",
            supplier_list=[{"id": "S1", "name": "A", "email": "a@b.com"}],
        )
        assert dists[0].sent_at is not None

    def test_distribute_generates_access_token(self, engine):
        dists = engine.distribute(
            template_id="t1",
            supplier_list=[{"id": "S1", "name": "A", "email": "a@b.com"}],
        )
        assert len(dists[0].access_token) == 64

    def test_distribute_sets_deadline(self, engine):
        dists = engine.distribute(
            template_id="t1",
            supplier_list=[{"id": "S1", "name": "A", "email": "a@b.com"}],
            deadline_days=45,
        )
        assert dists[0].deadline is not None

    def test_distribute_provenance_hash_set(self, engine):
        dists = engine.distribute(
            template_id="t1",
            supplier_list=[{"id": "S1", "name": "A", "email": "a@b.com"}],
        )
        assert len(dists[0].provenance_hash) == 64

    def test_distribute_with_campaign_id(self, engine):
        dists = engine.distribute(
            template_id="t1",
            supplier_list=[{"id": "S1", "name": "A", "email": "a@b.com"}],
            campaign_id="camp-manual-001",
        )
        assert dists[0].campaign_id == "camp-manual-001"

    def test_distribute_stats_increment(self, engine, sample_suppliers):
        engine.distribute(template_id="t1", supplier_list=sample_suppliers)
        stats = engine.get_statistics()
        assert stats["distributions_created"] == 3


# ===================================================================
# TEST CLASS: Get Distribution
# ===================================================================

class TestGetDistribution:
    """Tests for DistributionEngine.get_distribution()."""

    def test_get_distribution_existing(self, engine, campaign_id):
        dists = engine.list_distributions(campaign_id=campaign_id)
        fetched = engine.get_distribution(dists[0].distribution_id)
        assert fetched.distribution_id == dists[0].distribution_id

    def test_get_distribution_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown distribution"):
            engine.get_distribution("non-existent-id")

    def test_get_distribution_preserves_fields(self, engine, campaign_id):
        dists = engine.list_distributions(campaign_id=campaign_id)
        fetched = engine.get_distribution(dists[0].distribution_id)
        assert fetched.template_id == "tmpl-001"
        assert fetched.campaign_id == campaign_id


# ===================================================================
# TEST CLASS: List Distributions
# ===================================================================

class TestListDistributions:
    """Tests for DistributionEngine.list_distributions()."""

    def test_list_distributions_empty(self, engine):
        assert engine.list_distributions() == []

    def test_list_distributions_returns_all(self, engine, campaign_id):
        dists = engine.list_distributions()
        assert len(dists) == 2

    def test_list_distributions_filter_campaign(self, engine, campaign_id):
        engine.create_campaign(
            name="Other", template_id="t2", supplier_ids=["s99"],
        )
        filtered = engine.list_distributions(campaign_id=campaign_id)
        assert len(filtered) == 2
        for d in filtered:
            assert d.campaign_id == campaign_id

    def test_list_distributions_filter_supplier(self, engine, campaign_id):
        filtered = engine.list_distributions(supplier_id="SUP001")
        assert len(filtered) == 1
        assert filtered[0].supplier_id == "SUP001"

    def test_list_distributions_filter_status(self, engine, campaign_id):
        filtered = engine.list_distributions(status="sent")
        assert len(filtered) == 2

    def test_list_distributions_no_match(self, engine, campaign_id):
        filtered = engine.list_distributions(supplier_id="NONEXISTENT")
        assert filtered == []

    def test_list_distributions_combined_filters(self, engine, campaign_id):
        filtered = engine.list_distributions(
            campaign_id=campaign_id, supplier_id="SUP001",
        )
        assert len(filtered) == 1


# ===================================================================
# TEST CLASS: Update Status
# ===================================================================

class TestUpdateStatus:
    """Tests for DistributionEngine.update_status()."""

    def test_update_status_to_delivered(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        updated = engine.update_status(dist.distribution_id, "delivered")
        assert updated.status == DistributionStatus.DELIVERED

    def test_update_status_sets_delivered_at(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        updated = engine.update_status(dist.distribution_id, "delivered")
        assert updated.delivered_at is not None

    def test_update_status_to_opened(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        updated = engine.update_status(dist.distribution_id, "opened")
        assert updated.status == DistributionStatus.OPENED
        assert updated.opened_at is not None

    def test_update_status_to_submitted(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        updated = engine.update_status(dist.distribution_id, "submitted")
        assert updated.status == DistributionStatus.SUBMITTED
        assert updated.submitted_at is not None

    def test_update_status_sent_sets_sent_at(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        assert dist.sent_at is not None

    def test_update_status_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown distribution"):
            engine.update_status("bad-id", "delivered")

    def test_update_status_invalid_status_raises(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        with pytest.raises(ValueError, match="[Uu]nknown status"):
            engine.update_status(dist.distribution_id, "flying")

    def test_update_status_provenance_changes(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        old_hash = dist.provenance_hash
        updated = engine.update_status(dist.distribution_id, "delivered")
        assert updated.provenance_hash != old_hash

    def test_update_status_idempotent_timestamps(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        updated1 = engine.update_status(dist.distribution_id, "delivered")
        ts1 = updated1.delivered_at
        updated2 = engine.update_status(dist.distribution_id, "delivered")
        assert updated2.delivered_at == ts1


# ===================================================================
# TEST CLASS: Generate Access Token
# ===================================================================

class TestGenerateAccessToken:
    """Tests for DistributionEngine.generate_access_token()."""

    def test_generate_access_token_returns_hex(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        token = engine.generate_access_token(dist.distribution_id)
        assert re.fullmatch(r"[0-9a-f]{64}", token)

    def test_generate_access_token_stored_on_distribution(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        token = engine.generate_access_token(dist.distribution_id)
        refreshed = engine.get_distribution(dist.distribution_id)
        assert refreshed.access_token == token

    def test_generate_access_token_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown distribution"):
            engine.generate_access_token("bad-id")

    def test_generate_access_token_stats_increment(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        engine.generate_access_token(dist.distribution_id)
        stats = engine.get_statistics()
        assert stats["tokens_generated"] >= 1

    def test_generate_access_token_unique_per_call(self, engine, campaign_id):
        dists = engine.list_distributions(campaign_id=campaign_id)
        t1 = engine.generate_access_token(dists[0].distribution_id)
        t2 = engine.generate_access_token(dists[1].distribution_id)
        assert t1 != t2


# ===================================================================
# TEST CLASS: Get Campaign
# ===================================================================

class TestGetCampaign:
    """Tests for DistributionEngine.get_campaign()."""

    def test_get_campaign_existing(self, engine, campaign_id):
        camp = engine.get_campaign(campaign_id)
        assert camp["campaign_id"] == campaign_id

    def test_get_campaign_has_distributions_list(self, engine, campaign_id):
        camp = engine.get_campaign(campaign_id)
        assert "distributions" in camp
        assert isinstance(camp["distributions"], list)
        assert len(camp["distributions"]) == 2

    def test_get_campaign_has_distribution_count(self, engine, campaign_id):
        camp = engine.get_campaign(campaign_id)
        assert camp["distribution_count"] == 2

    def test_get_campaign_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown campaign"):
            engine.get_campaign("camp-fake")


# ===================================================================
# TEST CLASS: Cancel Distribution
# ===================================================================

class TestCancelDistribution:
    """Tests for DistributionEngine.cancel_distribution()."""

    def test_cancel_sent_distribution(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        cancelled = engine.cancel_distribution(dist.distribution_id)
        assert cancelled.status == DistributionStatus.CANCELLED

    def test_cancel_already_cancelled_raises(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        engine.cancel_distribution(dist.distribution_id)
        with pytest.raises(ValueError, match="cannot be cancelled"):
            engine.cancel_distribution(dist.distribution_id)

    def test_cancel_submitted_raises(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        engine.update_status(dist.distribution_id, "submitted")
        with pytest.raises(ValueError, match="cannot be cancelled"):
            engine.cancel_distribution(dist.distribution_id)

    def test_cancel_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown distribution"):
            engine.cancel_distribution("bad-id")

    def test_cancel_stats_increment(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        engine.cancel_distribution(dist.distribution_id)
        stats = engine.get_statistics()
        assert stats["distributions_cancelled"] == 1

    def test_cancel_provenance_set(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        cancelled = engine.cancel_distribution(dist.distribution_id)
        assert len(cancelled.provenance_hash) == 64

    def test_cancel_delivered_distribution(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        engine.update_status(dist.distribution_id, "delivered")
        cancelled = engine.cancel_distribution(dist.distribution_id)
        assert cancelled.status == DistributionStatus.CANCELLED

    def test_cancel_opened_distribution(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        engine.update_status(dist.distribution_id, "opened")
        cancelled = engine.cancel_distribution(dist.distribution_id)
        assert cancelled.status == DistributionStatus.CANCELLED


# ===================================================================
# TEST CLASS: Redistribute
# ===================================================================

class TestRedistribute:
    """Tests for DistributionEngine.redistribute()."""

    def test_redistribute_bounced(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        engine.update_status(dist.distribution_id, "bounced")
        new_dist = engine.redistribute(dist.distribution_id)
        assert new_dist.distribution_id != dist.distribution_id
        assert new_dist.status == DistributionStatus.SENT

    def test_redistribute_expired(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        engine.update_status(dist.distribution_id, "expired")
        new_dist = engine.redistribute(dist.distribution_id)
        assert new_dist.status == DistributionStatus.SENT

    def test_redistribute_cancelled(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        engine.cancel_distribution(dist.distribution_id)
        new_dist = engine.redistribute(dist.distribution_id)
        assert new_dist.status == DistributionStatus.SENT

    def test_redistribute_sent_raises(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        with pytest.raises(ValueError, match="cannot be redistributed"):
            engine.redistribute(dist.distribution_id)

    def test_redistribute_preserves_supplier(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        engine.update_status(dist.distribution_id, "bounced")
        new_dist = engine.redistribute(dist.distribution_id)
        assert new_dist.supplier_id == dist.supplier_id

    def test_redistribute_preserves_template(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        engine.update_status(dist.distribution_id, "bounced")
        new_dist = engine.redistribute(dist.distribution_id)
        assert new_dist.template_id == dist.template_id

    def test_redistribute_preserves_campaign(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        engine.update_status(dist.distribution_id, "bounced")
        new_dist = engine.redistribute(dist.distribution_id)
        assert new_dist.campaign_id == dist.campaign_id

    def test_redistribute_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown distribution"):
            engine.redistribute("bad-id")

    def test_redistribute_stats_increment(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        engine.update_status(dist.distribution_id, "bounced")
        engine.redistribute(dist.distribution_id)
        stats = engine.get_statistics()
        assert stats["distributions_redistributed"] == 1


# ===================================================================
# TEST CLASS: Campaign Summary
# ===================================================================

class TestGetCampaignSummary:
    """Tests for DistributionEngine.get_campaign_summary()."""

    def test_campaign_summary_basic(self, engine, campaign_id):
        summary = engine.get_campaign_summary(campaign_id)
        assert summary["campaign_id"] == campaign_id
        assert summary["total_distributions"] == 2

    def test_campaign_summary_response_rate_zero(self, engine, campaign_id):
        summary = engine.get_campaign_summary(campaign_id)
        assert summary["response_rate"] == 0.0

    def test_campaign_summary_after_submission(self, engine, campaign_id):
        dists = engine.list_distributions(campaign_id=campaign_id)
        engine.update_status(dists[0].distribution_id, "submitted")
        summary = engine.get_campaign_summary(campaign_id)
        assert summary["submitted_count"] == 1
        assert summary["response_rate"] == 50.0

    def test_campaign_summary_open_rate(self, engine, campaign_id):
        dists = engine.list_distributions(campaign_id=campaign_id)
        engine.update_status(dists[0].distribution_id, "opened")
        summary = engine.get_campaign_summary(campaign_id)
        assert summary["open_rate"] == 50.0

    def test_campaign_summary_has_provenance(self, engine, campaign_id):
        summary = engine.get_campaign_summary(campaign_id)
        assert len(summary["provenance_hash"]) == 64

    def test_campaign_summary_has_timestamp(self, engine, campaign_id):
        summary = engine.get_campaign_summary(campaign_id)
        assert "timestamp" in summary

    def test_campaign_summary_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown campaign"):
            engine.get_campaign_summary("camp-fake")

    def test_campaign_summary_status_breakdown(self, engine, campaign_id):
        summary = engine.get_campaign_summary(campaign_id)
        assert isinstance(summary["status_breakdown"], dict)
        assert "sent" in summary["status_breakdown"]

    def test_campaign_summary_overdue_count_zero(self, engine, campaign_id):
        summary = engine.get_campaign_summary(campaign_id)
        assert summary["overdue_count"] == 0


# ===================================================================
# TEST CLASS: Delivery Tracking Timestamps
# ===================================================================

class TestDeliveryTracking:
    """Tests for timestamp tracking through the delivery lifecycle."""

    def test_sent_at_set_on_creation(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        assert dist.sent_at is not None

    def test_delivered_at_set_on_update(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        updated = engine.update_status(dist.distribution_id, "delivered")
        assert updated.delivered_at is not None

    def test_opened_at_set_on_update(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        updated = engine.update_status(dist.distribution_id, "opened")
        assert updated.opened_at is not None

    def test_submitted_at_set_on_update(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        updated = engine.update_status(dist.distribution_id, "submitted")
        assert updated.submitted_at is not None


# ===================================================================
# TEST CLASS: Statistics
# ===================================================================

class TestStatistics:
    """Tests for get_statistics()."""

    def test_statistics_initial_zeros(self, engine):
        stats = engine.get_statistics()
        assert stats["distributions_created"] == 0
        assert stats["campaigns_created"] == 0

    def test_statistics_active_distributions(self, engine, campaign_id):
        stats = engine.get_statistics()
        assert stats["active_distributions"] == 2

    def test_statistics_active_campaigns(self, engine, campaign_id):
        stats = engine.get_statistics()
        assert stats["active_campaigns"] == 1

    def test_statistics_has_timestamp(self, engine):
        stats = engine.get_statistics()
        assert "timestamp" in stats


# ===================================================================
# TEST CLASS: Provenance on All Operations
# ===================================================================

class TestProvenance:
    """Tests for SHA-256 provenance hashing on all operations."""

    def test_distribute_provenance(self, engine):
        dists = engine.distribute(
            template_id="t1",
            supplier_list=[{"id": "S1", "name": "A", "email": "a@b.com"}],
        )
        assert len(dists[0].provenance_hash) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", dists[0].provenance_hash)

    def test_update_status_provenance(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        updated = engine.update_status(dist.distribution_id, "delivered")
        assert len(updated.provenance_hash) == 64

    def test_cancel_provenance(self, engine, campaign_id):
        dist = engine.list_distributions(campaign_id=campaign_id)[0]
        cancelled = engine.cancel_distribution(dist.distribution_id)
        assert len(cancelled.provenance_hash) == 64

    def test_campaign_provenance(self, engine, campaign_id):
        camp = engine.get_campaign(campaign_id)
        assert len(camp["provenance_hash"]) == 64


# ===================================================================
# TEST CLASS: Thread Safety
# ===================================================================

class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_distributes(self, engine):
        errors = []

        def distribute_one(idx):
            try:
                engine.distribute(
                    template_id="t1",
                    supplier_list=[{"id": f"S{idx}", "name": f"N{idx}", "email": f"e{idx}@x.com"}],
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=distribute_one, args=(i,)) for i in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(engine.list_distributions()) == 15


# ===================================================================
# TEST CLASS: Batch Processing
# ===================================================================

class TestBatchProcessing:
    """Tests for batch distribution limits."""

    def test_batch_within_limit(self, small_engine):
        suppliers = [{"id": f"S{i}", "name": f"N{i}", "email": f"e{i}@x.com"}
                     for i in range(3)]
        dists = small_engine.distribute(template_id="t1", supplier_list=suppliers)
        assert len(dists) == 3

    def test_batch_exceeds_limit_raises(self, small_engine):
        suppliers = [{"id": f"S{i}", "name": f"N{i}", "email": f"e{i}@x.com"}
                     for i in range(4)]
        with pytest.raises(ValueError, match="exceeds"):
            small_engine.distribute(template_id="t1", supplier_list=suppliers)

    def test_default_batch_limit_is_5000(self, engine):
        assert engine._max_batch_size == 5000
