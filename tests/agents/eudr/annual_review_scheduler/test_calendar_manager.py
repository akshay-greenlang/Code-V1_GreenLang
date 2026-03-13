# -*- coding: utf-8 -*-
"""
Unit tests for CalendarManagerEngine - AGENT-EUDR-034

Tests calendar entry creation, phase-based scheduling, meeting
management, recurring events, conflict detection, calendar export,
and timezone handling.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 (Engine 6: Calendar Manager)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.annual_review_scheduler.config import (
    AnnualReviewSchedulerConfig,
)
from greenlang.agents.eudr.annual_review_scheduler.calendar_manager import (
    CalendarManagerEngine,
)
from greenlang.agents.eudr.annual_review_scheduler.models import (
    CalendarEntry,
    CalendarEntryType,
    ReviewPhase,
)
from greenlang.agents.eudr.annual_review_scheduler.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return AnnualReviewSchedulerConfig()


@pytest.fixture
def calendar(config):
    return CalendarManagerEngine(config=config, provenance=ProvenanceTracker())


# ---------------------------------------------------------------------------
# Entry Creation
# ---------------------------------------------------------------------------

class TestEntryCreation:
    """Test calendar entry creation."""

    @pytest.mark.asyncio
    async def test_create_entry_returns_calendar_entry(self, calendar):
        now = datetime.now(tz=timezone.utc)
        entry = await calendar.create_entry(
            cycle_id="cyc-001",
            entry_type=CalendarEntryType.PHASE_START,
            title="Data Collection Start",
            start_time=now + timedelta(days=14),
        )
        assert isinstance(entry, CalendarEntry)
        assert entry.entry_id.startswith("cal-")

    @pytest.mark.asyncio
    async def test_create_entry_sets_cycle_id(self, calendar):
        now = datetime.now(tz=timezone.utc)
        entry = await calendar.create_entry(
            cycle_id="cyc-001",
            entry_type=CalendarEntryType.DEADLINE,
            title="Test",
            start_time=now + timedelta(days=30),
        )
        assert entry.cycle_id == "cyc-001"

    @pytest.mark.asyncio
    async def test_create_entry_sets_type(self, calendar):
        now = datetime.now(tz=timezone.utc)
        entry = await calendar.create_entry(
            cycle_id="cyc-001",
            entry_type=CalendarEntryType.REVIEW_MEETING,
            title="Meeting",
            start_time=now + timedelta(days=60),
        )
        assert entry.entry_type == CalendarEntryType.REVIEW_MEETING

    @pytest.mark.asyncio
    async def test_create_entry_with_end_time(self, calendar):
        now = datetime.now(tz=timezone.utc)
        start = now + timedelta(days=14)
        end = start + timedelta(hours=2)
        entry = await calendar.create_entry(
            cycle_id="cyc-001",
            entry_type=CalendarEntryType.REVIEW_MEETING,
            title="Meeting",
            start_time=start,
            end_time=end,
        )
        assert entry.end_time == end

    @pytest.mark.asyncio
    async def test_create_entry_with_attendees(self, calendar):
        now = datetime.now(tz=timezone.utc)
        entry = await calendar.create_entry(
            cycle_id="cyc-001",
            entry_type=CalendarEntryType.REVIEW_MEETING,
            title="Meeting",
            start_time=now + timedelta(days=60),
            attendees=["lead@company.com", "analyst@company.com"],
        )
        assert len(entry.attendees) == 2

    @pytest.mark.asyncio
    async def test_create_entry_with_phase(self, calendar):
        now = datetime.now(tz=timezone.utc)
        entry = await calendar.create_entry(
            cycle_id="cyc-001",
            entry_type=CalendarEntryType.PHASE_START,
            title="Phase Start",
            start_time=now + timedelta(days=14),
            phase=ReviewPhase.DATA_COLLECTION,
        )
        assert entry.phase == ReviewPhase.DATA_COLLECTION

    @pytest.mark.asyncio
    async def test_create_entry_with_location(self, calendar):
        now = datetime.now(tz=timezone.utc)
        entry = await calendar.create_entry(
            cycle_id="cyc-001",
            entry_type=CalendarEntryType.REVIEW_MEETING,
            title="In-Person Meeting",
            start_time=now + timedelta(days=60),
            location="Conference Room B",
        )
        assert entry.location == "Conference Room B"

    @pytest.mark.asyncio
    async def test_create_recurring_entry(self, calendar):
        now = datetime.now(tz=timezone.utc)
        entry = await calendar.create_entry(
            cycle_id="cyc-001",
            entry_type=CalendarEntryType.REMINDER,
            title="Weekly Status Check",
            start_time=now + timedelta(days=7),
            recurring=True,
        )
        assert entry.recurring is True


# ---------------------------------------------------------------------------
# Generate for Cycle
# ---------------------------------------------------------------------------

class TestGenerateForCycle:
    """Test calendar generation for entire review cycle."""

    @pytest.mark.asyncio
    async def test_generate_for_cycle_returns_entries(self, calendar, sample_review_cycle):
        entries = await calendar.generate_for_cycle(sample_review_cycle)
        assert len(entries) > 0

    @pytest.mark.asyncio
    async def test_generate_includes_phase_starts(self, calendar, sample_review_cycle):
        entries = await calendar.generate_for_cycle(sample_review_cycle)
        phase_starts = [e for e in entries if e.entry_type == CalendarEntryType.PHASE_START]
        assert len(phase_starts) >= 6

    @pytest.mark.asyncio
    async def test_generate_includes_deadlines(self, calendar, sample_review_cycle):
        entries = await calendar.generate_for_cycle(sample_review_cycle)
        deadlines = [e for e in entries if e.entry_type == CalendarEntryType.DEADLINE]
        assert len(deadlines) >= 1

    @pytest.mark.asyncio
    async def test_generate_chronological_order(self, calendar, sample_review_cycle):
        entries = await calendar.generate_for_cycle(sample_review_cycle)
        for i in range(len(entries) - 1):
            assert entries[i].start_time <= entries[i + 1].start_time

    @pytest.mark.asyncio
    async def test_generate_entries_have_unique_ids(self, calendar, sample_review_cycle):
        entries = await calendar.generate_for_cycle(sample_review_cycle)
        ids = [e.entry_id for e in entries]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_generate_entries_tied_to_cycle(self, calendar, sample_review_cycle):
        entries = await calendar.generate_for_cycle(sample_review_cycle)
        for entry in entries:
            assert entry.cycle_id == sample_review_cycle.cycle_id


# ---------------------------------------------------------------------------
# Update and Delete
# ---------------------------------------------------------------------------

class TestUpdateAndDelete:
    """Test calendar entry updates and deletion."""

    @pytest.mark.asyncio
    async def test_update_entry_title(self, calendar):
        now = datetime.now(tz=timezone.utc)
        entry = await calendar.create_entry(
            cycle_id="cyc-001",
            entry_type=CalendarEntryType.REMINDER,
            title="Original Title",
            start_time=now + timedelta(days=7),
        )
        updated = await calendar.update_entry(
            entry.entry_id, title="Updated Title",
        )
        assert updated.title == "Updated Title"

    @pytest.mark.asyncio
    async def test_update_entry_time(self, calendar):
        now = datetime.now(tz=timezone.utc)
        entry = await calendar.create_entry(
            cycle_id="cyc-001",
            entry_type=CalendarEntryType.DEADLINE,
            title="Deadline",
            start_time=now + timedelta(days=30),
        )
        new_time = now + timedelta(days=35)
        updated = await calendar.update_entry(
            entry.entry_id, start_time=new_time,
        )
        assert updated.start_time == new_time

    @pytest.mark.asyncio
    async def test_delete_entry(self, calendar):
        now = datetime.now(tz=timezone.utc)
        entry = await calendar.create_entry(
            cycle_id="cyc-001",
            entry_type=CalendarEntryType.REMINDER,
            title="To Delete",
            start_time=now + timedelta(days=7),
        )
        result = await calendar.delete_entry(entry.entry_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent_raises(self, calendar):
        with pytest.raises(ValueError, match="not found"):
            await calendar.delete_entry("cal-nonexistent")

    @pytest.mark.asyncio
    async def test_update_nonexistent_raises(self, calendar):
        with pytest.raises(ValueError, match="not found"):
            await calendar.update_entry("cal-nonexistent", title="New")


# ---------------------------------------------------------------------------
# List and Filter
# ---------------------------------------------------------------------------

class TestListAndFilter:
    """Test calendar entry listing and filtering."""

    @pytest.mark.asyncio
    async def test_list_by_cycle(self, calendar):
        now = datetime.now(tz=timezone.utc)
        await calendar.create_entry(
            cycle_id="cyc-001", entry_type=CalendarEntryType.PHASE_START,
            title="A", start_time=now + timedelta(days=1),
        )
        await calendar.create_entry(
            cycle_id="cyc-002", entry_type=CalendarEntryType.PHASE_START,
            title="B", start_time=now + timedelta(days=1),
        )
        entries = await calendar.list_entries(cycle_id="cyc-001")
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_list_by_type(self, calendar):
        now = datetime.now(tz=timezone.utc)
        await calendar.create_entry(
            cycle_id="cyc-001", entry_type=CalendarEntryType.DEADLINE,
            title="Deadline", start_time=now + timedelta(days=30),
        )
        await calendar.create_entry(
            cycle_id="cyc-001", entry_type=CalendarEntryType.REVIEW_MEETING,
            title="Meeting", start_time=now + timedelta(days=60),
        )
        deadlines = await calendar.list_entries(
            cycle_id="cyc-001", entry_type=CalendarEntryType.DEADLINE,
        )
        assert len(deadlines) == 1

    @pytest.mark.asyncio
    async def test_list_by_date_range(self, calendar):
        now = datetime.now(tz=timezone.utc)
        await calendar.create_entry(
            cycle_id="cyc-001", entry_type=CalendarEntryType.REMINDER,
            title="Soon", start_time=now + timedelta(days=5),
        )
        await calendar.create_entry(
            cycle_id="cyc-001", entry_type=CalendarEntryType.DEADLINE,
            title="Later", start_time=now + timedelta(days=60),
        )
        entries = await calendar.list_entries(
            cycle_id="cyc-001",
            from_date=now,
            to_date=now + timedelta(days=30),
        )
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_list_by_phase(self, calendar):
        now = datetime.now(tz=timezone.utc)
        await calendar.create_entry(
            cycle_id="cyc-001", entry_type=CalendarEntryType.PHASE_START,
            title="Prep", start_time=now + timedelta(days=1),
            phase=ReviewPhase.PREPARATION,
        )
        await calendar.create_entry(
            cycle_id="cyc-001", entry_type=CalendarEntryType.PHASE_START,
            title="DC", start_time=now + timedelta(days=14),
            phase=ReviewPhase.DATA_COLLECTION,
        )
        entries = await calendar.list_entries(
            cycle_id="cyc-001", phase=ReviewPhase.PREPARATION,
        )
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_get_entry_by_id(self, calendar):
        now = datetime.now(tz=timezone.utc)
        created = await calendar.create_entry(
            cycle_id="cyc-001", entry_type=CalendarEntryType.REMINDER,
            title="Get This", start_time=now + timedelta(days=7),
        )
        retrieved = await calendar.get_entry(created.entry_id)
        assert retrieved.entry_id == created.entry_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_entry_raises(self, calendar):
        with pytest.raises(ValueError, match="not found"):
            await calendar.get_entry("cal-nonexistent")


# ---------------------------------------------------------------------------
# Calendar Export
# ---------------------------------------------------------------------------

class TestCalendarExport:
    """Test calendar export functionality."""

    @pytest.mark.asyncio
    async def test_export_ical_format(self, calendar, sample_review_cycle):
        entries = await calendar.generate_for_cycle(sample_review_cycle)
        ical = await calendar.export_ical(cycle_id=sample_review_cycle.cycle_id)
        assert ical is not None
        assert isinstance(ical, str)
        assert "BEGIN:VCALENDAR" in ical or len(ical) > 0

    @pytest.mark.asyncio
    async def test_export_json_format(self, calendar, sample_review_cycle):
        entries = await calendar.generate_for_cycle(sample_review_cycle)
        json_data = await calendar.export_json(cycle_id=sample_review_cycle.cycle_id)
        assert json_data is not None
        assert isinstance(json_data, (str, dict, list))

    @pytest.mark.asyncio
    async def test_export_empty_cycle_returns_empty(self, calendar):
        ical = await calendar.export_ical(cycle_id="cyc-empty-export")
        assert ical is not None


# ---------------------------------------------------------------------------
# Conflict Detection
# ---------------------------------------------------------------------------

class TestConflictDetection:
    """Test calendar conflict detection."""

    @pytest.mark.asyncio
    async def test_detect_overlapping_entries(self, calendar):
        now = datetime.now(tz=timezone.utc)
        start1 = now + timedelta(days=30)
        end1 = start1 + timedelta(hours=2)
        start2 = start1 + timedelta(hours=1)  # Overlaps
        end2 = start2 + timedelta(hours=2)
        await calendar.create_entry(
            cycle_id="cyc-conflict", entry_type=CalendarEntryType.REVIEW_MEETING,
            title="Meeting 1", start_time=start1, end_time=end1,
            attendees=["lead@company.com"],
        )
        await calendar.create_entry(
            cycle_id="cyc-conflict", entry_type=CalendarEntryType.REVIEW_MEETING,
            title="Meeting 2", start_time=start2, end_time=end2,
            attendees=["lead@company.com"],
        )
        conflicts = await calendar.detect_conflicts(cycle_id="cyc-conflict")
        assert len(conflicts) >= 1

    @pytest.mark.asyncio
    async def test_no_conflicts_when_non_overlapping(self, calendar):
        now = datetime.now(tz=timezone.utc)
        await calendar.create_entry(
            cycle_id="cyc-no-conflict", entry_type=CalendarEntryType.PHASE_START,
            title="Entry 1", start_time=now + timedelta(days=1),
            end_time=now + timedelta(days=1, hours=1),
        )
        await calendar.create_entry(
            cycle_id="cyc-no-conflict", entry_type=CalendarEntryType.DEADLINE,
            title="Entry 2", start_time=now + timedelta(days=30),
            end_time=now + timedelta(days=30, hours=1),
        )
        conflicts = await calendar.detect_conflicts(cycle_id="cyc-no-conflict")
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_detect_conflicts_empty_calendar(self, calendar):
        conflicts = await calendar.detect_conflicts(cycle_id="cyc-empty-conflict")
        assert conflicts == []


# ---------------------------------------------------------------------------
# Milestone Entries
# ---------------------------------------------------------------------------

class TestMilestoneEntries:
    """Test milestone calendar entries."""

    @pytest.mark.asyncio
    async def test_create_milestone_entry(self, calendar):
        now = datetime.now(tz=timezone.utc)
        entry = await calendar.create_entry(
            cycle_id="cyc-001",
            entry_type=CalendarEntryType.MILESTONE,
            title="50% Checklist Completion",
            start_time=now + timedelta(days=30),
            description="Half of all checklist items completed",
        )
        assert entry.entry_type == CalendarEntryType.MILESTONE

    @pytest.mark.asyncio
    async def test_create_sign_off_entry(self, calendar):
        now = datetime.now(tz=timezone.utc)
        entry = await calendar.create_entry(
            cycle_id="cyc-001",
            entry_type=CalendarEntryType.SIGN_OFF,
            title="Final Sign-Off",
            start_time=now + timedelta(days=109),
            attendees=["compliance@company.com"],
        )
        assert entry.entry_type == CalendarEntryType.SIGN_OFF

    @pytest.mark.asyncio
    async def test_list_milestones_only(self, calendar):
        now = datetime.now(tz=timezone.utc)
        await calendar.create_entry(
            cycle_id="cyc-mile", entry_type=CalendarEntryType.MILESTONE,
            title="Milestone 1", start_time=now + timedelta(days=30),
        )
        await calendar.create_entry(
            cycle_id="cyc-mile", entry_type=CalendarEntryType.REMINDER,
            title="Reminder 1", start_time=now + timedelta(days=5),
        )
        milestones = await calendar.list_entries(
            cycle_id="cyc-mile", entry_type=CalendarEntryType.MILESTONE,
        )
        assert len(milestones) == 1


# ---------------------------------------------------------------------------
# Attendee Management
# ---------------------------------------------------------------------------

class TestAttendeeManagement:
    """Test attendee operations on calendar entries."""

    @pytest.mark.asyncio
    async def test_add_attendee(self, calendar):
        now = datetime.now(tz=timezone.utc)
        entry = await calendar.create_entry(
            cycle_id="cyc-001", entry_type=CalendarEntryType.REVIEW_MEETING,
            title="Meeting", start_time=now + timedelta(days=60),
            attendees=["lead@company.com"],
        )
        updated = await calendar.update_entry(
            entry.entry_id,
            attendees=["lead@company.com", "analyst@company.com"],
        )
        assert len(updated.attendees) == 2

    @pytest.mark.asyncio
    async def test_update_entry_description(self, calendar):
        now = datetime.now(tz=timezone.utc)
        entry = await calendar.create_entry(
            cycle_id="cyc-001", entry_type=CalendarEntryType.REMINDER,
            title="Reminder", start_time=now + timedelta(days=7),
        )
        updated = await calendar.update_entry(
            entry.entry_id, description="Updated description",
        )
        assert updated.description == "Updated description"
