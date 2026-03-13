# -*- coding: utf-8 -*-
"""
Unit tests for StatusTracker engine - AGENT-EUDR-036

Tests status checking, status change recording, history tracking,
timeline generation, and submission timeout detection.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.eu_information_system_interface.config import (
    EUInformationSystemInterfaceConfig,
)
from greenlang.agents.eudr.eu_information_system_interface.status_tracker import (
    StatusTracker,
)
from greenlang.agents.eudr.eu_information_system_interface.models import (
    DDSStatus,
)
from greenlang.agents.eudr.eu_information_system_interface.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def tracker() -> StatusTracker:
    """Create a StatusTracker instance."""
    config = EUInformationSystemInterfaceConfig()
    return StatusTracker(config=config, provenance=ProvenanceTracker())


class TestCheckStatus:
    """Test StatusTracker.check_status()."""

    @pytest.mark.asyncio
    async def test_check_status_basic(self, tracker):
        result = await tracker.check_status(
            dds_id="dds-001",
            eu_reference="EUDR-2026-DE-00012345",
        )
        assert result.check_id.startswith("chk-")
        assert result.dds_id == "dds-001"
        assert result.eu_reference == "EUDR-2026-DE-00012345"

    @pytest.mark.asyncio
    async def test_check_status_default_submitted(self, tracker):
        result = await tracker.check_status(
            dds_id="dds-001",
            eu_reference="EUDR-REF-001",
        )
        # Default previous status is SUBMITTED
        assert result.previous_status == DDSStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_check_status_with_known_status(self, tracker):
        result = await tracker.check_status(
            dds_id="dds-001",
            eu_reference="EUDR-REF-001",
            current_known_status="received",
        )
        assert result.previous_status == DDSStatus.RECEIVED

    @pytest.mark.asyncio
    async def test_check_status_has_provenance(self, tracker):
        result = await tracker.check_status(
            dds_id="dds-001",
            eu_reference="EUDR-REF-001",
        )
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_check_status_tracks_dds(self, tracker):
        await tracker.check_status("dds-001", "EUDR-REF-001")
        await tracker.check_status("dds-002", "EUDR-REF-002")
        summary = await tracker.get_pending_submissions_summary()
        assert summary["total_tracked"] == 2


class TestRecordStatusChange:
    """Test StatusTracker.record_status_change()."""

    @pytest.mark.asyncio
    async def test_valid_transition(self, tracker):
        result = await tracker.record_status_change(
            dds_id="dds-001",
            old_status="submitted",
            new_status="received",
        )
        assert result.status_changed is True
        assert result.previous_status == DDSStatus.SUBMITTED
        assert result.current_status == DDSStatus.RECEIVED

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self, tracker):
        with pytest.raises(ValueError, match="Invalid transition"):
            await tracker.record_status_change(
                dds_id="dds-001",
                old_status="draft",
                new_status="accepted",
            )

    @pytest.mark.asyncio
    async def test_invalid_status_value_raises(self, tracker):
        with pytest.raises(ValueError, match="Invalid status"):
            await tracker.record_status_change(
                dds_id="dds-001",
                old_status="submitted",
                new_status="nonexistent",
            )

    @pytest.mark.asyncio
    async def test_valid_transitions_accepted(self, tracker):
        # SUBMITTED -> RECEIVED
        await tracker.record_status_change("dds-001", "submitted", "received")
        # RECEIVED -> UNDER_REVIEW
        await tracker.record_status_change("dds-001", "received", "under_review")
        # UNDER_REVIEW -> ACCEPTED
        result = await tracker.record_status_change("dds-001", "under_review", "accepted")
        assert result.current_status == DDSStatus.ACCEPTED

    @pytest.mark.asyncio
    async def test_withdrawn_no_transitions(self, tracker):
        with pytest.raises(ValueError, match="Invalid transition"):
            await tracker.record_status_change("dds-001", "withdrawn", "submitted")


class TestGetStatusHistory:
    """Test StatusTracker.get_status_history()."""

    @pytest.mark.asyncio
    async def test_empty_history(self, tracker):
        history = await tracker.get_status_history("dds-unknown")
        assert history == []

    @pytest.mark.asyncio
    async def test_history_after_checks(self, tracker):
        await tracker.check_status("dds-001", "EUDR-REF-001")
        history = await tracker.get_status_history("dds-001")
        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_history_after_changes(self, tracker):
        await tracker.record_status_change("dds-001", "submitted", "received")
        await tracker.record_status_change("dds-001", "received", "under_review")
        history = await tracker.get_status_history("dds-001")
        assert len(history) >= 2


class TestGetStatusTimeline:
    """Test StatusTracker.get_status_timeline()."""

    @pytest.mark.asyncio
    async def test_timeline_empty(self, tracker):
        timeline = await tracker.get_status_timeline("dds-unknown")
        assert timeline["dds_id"] == "dds-unknown"
        assert timeline["current_status"] == "unknown"
        assert timeline["total_transitions"] == 0

    @pytest.mark.asyncio
    async def test_timeline_with_data(self, tracker):
        await tracker.record_status_change("dds-001", "submitted", "received")
        await tracker.record_status_change("dds-001", "received", "under_review")
        timeline = await tracker.get_status_timeline("dds-001")
        assert timeline["current_status"] == "under_review"
        assert timeline["total_transitions"] >= 2


class TestCheckSubmissionTimeout:
    """Test StatusTracker.check_submission_timeout()."""

    @pytest.mark.asyncio
    async def test_not_timed_out(self, tracker, sample_submission):
        result = await tracker.check_submission_timeout(sample_submission)
        assert result["is_timed_out"] is False

    @pytest.mark.asyncio
    async def test_timed_out(self, tracker, old_submission):
        result = await tracker.check_submission_timeout(old_submission)
        assert result["is_timed_out"] is True
        assert result["elapsed_hours"] > 72


class TestHealthCheck:
    """Test StatusTracker.health_check()."""

    @pytest.mark.asyncio
    async def test_health_check(self, tracker):
        health = await tracker.health_check()
        assert health["engine"] == "StatusTracker"
        assert health["status"] == "available"
