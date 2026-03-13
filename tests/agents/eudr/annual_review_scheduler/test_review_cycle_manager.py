# -*- coding: utf-8 -*-
"""
Unit tests for ReviewCycleManager - AGENT-EUDR-034

Tests review cycle creation, scheduling, phase advancement, status
transitions, cycle completion, cancellation, pause/resume, commodity
scope management, and auto-scheduling.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 (Engine 1: Review Cycle Manager)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.annual_review_scheduler.config import (
    AnnualReviewSchedulerConfig,
)
from greenlang.agents.eudr.annual_review_scheduler.review_cycle_manager import (
    ReviewCycleManager,
)
from greenlang.agents.eudr.annual_review_scheduler.models import (
    CommodityScope,
    EUDRCommodity,
    ReviewCycle,
    ReviewCycleStatus,
    ReviewPhase,
    ReviewPhaseConfig,
    ReviewType,
)
from greenlang.agents.eudr.annual_review_scheduler.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return AnnualReviewSchedulerConfig()


@pytest.fixture
def manager(config):
    return ReviewCycleManager(config=config, provenance=ProvenanceTracker())


@pytest.fixture
def coffee_scope():
    return CommodityScope(
        commodity=EUDRCommodity.COFFEE,
        supplier_count=15,
        shipment_count=120,
    )


@pytest.fixture
def multi_commodity_scope():
    return [
        CommodityScope(commodity=EUDRCommodity.COFFEE, supplier_count=15, shipment_count=120),
        CommodityScope(commodity=EUDRCommodity.COCOA, supplier_count=8, shipment_count=45),
        CommodityScope(commodity=EUDRCommodity.WOOD, supplier_count=5, shipment_count=20),
    ]


# ---------------------------------------------------------------------------
# Cycle Creation
# ---------------------------------------------------------------------------

class TestCreateCycle:
    """Test review cycle creation."""

    @pytest.mark.asyncio
    async def test_create_cycle_returns_review_cycle(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        assert isinstance(cycle, ReviewCycle)
        assert cycle.cycle_id.startswith("cyc-")

    @pytest.mark.asyncio
    async def test_create_cycle_sets_operator(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        assert cycle.operator_id == "operator-001"

    @pytest.mark.asyncio
    async def test_create_cycle_sets_year(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        assert cycle.review_year == 2026

    @pytest.mark.asyncio
    async def test_create_cycle_sets_review_type(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.SEMI_ANNUAL,
            commodity_scope=[coffee_scope],
        )
        assert cycle.review_type == ReviewType.SEMI_ANNUAL

    @pytest.mark.asyncio
    async def test_create_cycle_initial_status_is_draft(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        assert cycle.status == ReviewCycleStatus.DRAFT

    @pytest.mark.asyncio
    async def test_create_cycle_initial_phase_is_preparation(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        assert cycle.current_phase == ReviewPhase.PREPARATION

    @pytest.mark.asyncio
    async def test_create_cycle_has_phase_configs(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        assert len(cycle.phase_configs) == 6

    @pytest.mark.asyncio
    async def test_create_cycle_provenance_hash(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        assert len(cycle.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_create_cycle_with_multi_commodity(self, manager, multi_commodity_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=multi_commodity_scope,
        )
        assert len(cycle.commodity_scope) == 3

    @pytest.mark.asyncio
    async def test_create_cycle_scheduled_dates_set(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        assert cycle.scheduled_start is not None
        assert cycle.scheduled_end is not None
        assert cycle.scheduled_end > cycle.scheduled_start

    @pytest.mark.asyncio
    async def test_create_cycle_created_by_agent(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        assert cycle.created_by == "AGENT-EUDR-034"


# ---------------------------------------------------------------------------
# Schedule Cycle
# ---------------------------------------------------------------------------

class TestScheduleCycle:
    """Test cycle scheduling."""

    @pytest.mark.asyncio
    async def test_schedule_cycle_transitions_to_scheduled(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        scheduled = await manager.schedule_cycle(cycle.cycle_id)
        assert scheduled.status == ReviewCycleStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_schedule_cycle_sets_start_date(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        scheduled = await manager.schedule_cycle(
            cycle.cycle_id,
            start_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
        assert scheduled.scheduled_start.year == 2026
        assert scheduled.scheduled_start.month == 4

    @pytest.mark.asyncio
    async def test_schedule_nonexistent_cycle_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            await manager.schedule_cycle("cyc-nonexistent")

    @pytest.mark.asyncio
    async def test_schedule_already_active_cycle_raises(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.schedule_cycle(cycle.cycle_id)
        await manager.start_cycle(cycle.cycle_id)
        with pytest.raises(ValueError, match="Cannot schedule"):
            await manager.schedule_cycle(cycle.cycle_id)


# ---------------------------------------------------------------------------
# Start Cycle
# ---------------------------------------------------------------------------

class TestStartCycle:
    """Test cycle starting."""

    @pytest.mark.asyncio
    async def test_start_cycle_transitions_to_in_progress(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.schedule_cycle(cycle.cycle_id)
        started = await manager.start_cycle(cycle.cycle_id)
        assert started.status == ReviewCycleStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_start_cycle_sets_actual_start(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.schedule_cycle(cycle.cycle_id)
        started = await manager.start_cycle(cycle.cycle_id)
        assert started.actual_start is not None

    @pytest.mark.asyncio
    async def test_start_draft_cycle_raises(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        with pytest.raises(ValueError, match="Cannot start"):
            await manager.start_cycle(cycle.cycle_id)


# ---------------------------------------------------------------------------
# Phase Advancement
# ---------------------------------------------------------------------------

class TestAdvancePhase:
    """Test phase advancement logic."""

    @pytest.mark.asyncio
    async def test_advance_from_preparation_to_data_collection(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.schedule_cycle(cycle.cycle_id)
        await manager.start_cycle(cycle.cycle_id)
        advanced = await manager.advance_phase(cycle.cycle_id)
        assert advanced.current_phase == ReviewPhase.DATA_COLLECTION

    @pytest.mark.asyncio
    async def test_advance_through_all_phases(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.schedule_cycle(cycle.cycle_id)
        await manager.start_cycle(cycle.cycle_id)

        expected_phases = [
            ReviewPhase.DATA_COLLECTION,
            ReviewPhase.ANALYSIS,
            ReviewPhase.REVIEW_MEETING,
            ReviewPhase.REMEDIATION,
            ReviewPhase.SIGN_OFF,
        ]
        for expected_phase in expected_phases:
            result = await manager.advance_phase(cycle.cycle_id)
            assert result.current_phase == expected_phase

    @pytest.mark.asyncio
    async def test_advance_past_sign_off_completes_cycle(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.schedule_cycle(cycle.cycle_id)
        await manager.start_cycle(cycle.cycle_id)
        for _ in range(5):
            await manager.advance_phase(cycle.cycle_id)
        result = await manager.advance_phase(cycle.cycle_id)
        assert result.status == ReviewCycleStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_advance_draft_cycle_raises(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        with pytest.raises(ValueError, match="Cannot advance"):
            await manager.advance_phase(cycle.cycle_id)

    @pytest.mark.asyncio
    async def test_advance_updates_provenance_hash(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.schedule_cycle(cycle.cycle_id)
        await manager.start_cycle(cycle.cycle_id)
        original_hash = cycle.provenance_hash
        advanced = await manager.advance_phase(cycle.cycle_id)
        assert advanced.provenance_hash != original_hash
        assert len(advanced.provenance_hash) == 64


# ---------------------------------------------------------------------------
# Pause / Resume
# ---------------------------------------------------------------------------

class TestPauseResumeCycle:
    """Test cycle pause and resume."""

    @pytest.mark.asyncio
    async def test_pause_active_cycle(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.schedule_cycle(cycle.cycle_id)
        await manager.start_cycle(cycle.cycle_id)
        paused = await manager.pause_cycle(cycle.cycle_id, reason="Budget review")
        assert paused.status == ReviewCycleStatus.PAUSED

    @pytest.mark.asyncio
    async def test_resume_paused_cycle(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.schedule_cycle(cycle.cycle_id)
        await manager.start_cycle(cycle.cycle_id)
        await manager.pause_cycle(cycle.cycle_id, reason="Budget review")
        resumed = await manager.resume_cycle(cycle.cycle_id)
        assert resumed.status == ReviewCycleStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_pause_draft_cycle_raises(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        with pytest.raises(ValueError, match="Cannot pause"):
            await manager.pause_cycle(cycle.cycle_id, reason="test")

    @pytest.mark.asyncio
    async def test_resume_non_paused_cycle_raises(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.schedule_cycle(cycle.cycle_id)
        await manager.start_cycle(cycle.cycle_id)
        with pytest.raises(ValueError, match="Cannot resume"):
            await manager.resume_cycle(cycle.cycle_id)


# ---------------------------------------------------------------------------
# Cancel Cycle
# ---------------------------------------------------------------------------

class TestCancelCycle:
    """Test cycle cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_draft_cycle(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        cancelled = await manager.cancel_cycle(cycle.cycle_id, reason="Not needed")
        assert cancelled.status == ReviewCycleStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_active_cycle(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.schedule_cycle(cycle.cycle_id)
        await manager.start_cycle(cycle.cycle_id)
        cancelled = await manager.cancel_cycle(cycle.cycle_id, reason="Regulatory change")
        assert cancelled.status == ReviewCycleStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_completed_cycle_raises(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.schedule_cycle(cycle.cycle_id)
        await manager.start_cycle(cycle.cycle_id)
        for _ in range(6):
            await manager.advance_phase(cycle.cycle_id)
        with pytest.raises(ValueError, match="Cannot cancel"):
            await manager.cancel_cycle(cycle.cycle_id, reason="Too late")


# ---------------------------------------------------------------------------
# List / Get
# ---------------------------------------------------------------------------

class TestListAndGetCycles:
    """Test cycle listing and retrieval."""

    @pytest.mark.asyncio
    async def test_get_cycle_by_id(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        retrieved = await manager.get_cycle(cycle.cycle_id)
        assert retrieved.cycle_id == cycle.cycle_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_cycle_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            await manager.get_cycle("cyc-nonexistent")

    @pytest.mark.asyncio
    async def test_list_cycles_by_operator(self, manager, coffee_scope):
        for i in range(3):
            await manager.create_cycle(
                operator_id="operator-001",
                review_year=2024 + i,
                review_type=ReviewType.ANNUAL,
                commodity_scope=[coffee_scope],
            )
        cycles = await manager.list_cycles(operator_id="operator-001")
        assert len(cycles) == 3

    @pytest.mark.asyncio
    async def test_list_cycles_by_status(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.schedule_cycle(cycle.cycle_id)
        drafts = await manager.list_cycles(status=ReviewCycleStatus.DRAFT)
        scheduled = await manager.list_cycles(status=ReviewCycleStatus.SCHEDULED)
        assert len(drafts) == 0
        assert len(scheduled) == 1

    @pytest.mark.asyncio
    async def test_list_cycles_by_year(self, manager, coffee_scope):
        await manager.create_cycle(
            operator_id="operator-001",
            review_year=2026,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.create_cycle(
            operator_id="operator-001",
            review_year=2025,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        cycles_2026 = await manager.list_cycles(review_year=2026)
        assert len(cycles_2026) == 1
        assert cycles_2026[0].review_year == 2026


# ---------------------------------------------------------------------------
# Auto-Scheduling
# ---------------------------------------------------------------------------

class TestAutoScheduling:
    """Test auto-scheduling capabilities."""

    @pytest.mark.asyncio
    async def test_auto_schedule_creates_next_cycle(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2025,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        await manager.schedule_cycle(cycle.cycle_id)
        await manager.start_cycle(cycle.cycle_id)
        for _ in range(6):
            await manager.advance_phase(cycle.cycle_id)
        next_cycle = await manager.auto_schedule_next(cycle.cycle_id)
        assert next_cycle.review_year == 2026
        assert next_cycle.status in (ReviewCycleStatus.DRAFT, ReviewCycleStatus.SCHEDULED)

    @pytest.mark.asyncio
    async def test_auto_schedule_preserves_commodity_scope(self, manager, multi_commodity_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2025,
            review_type=ReviewType.ANNUAL,
            commodity_scope=multi_commodity_scope,
        )
        await manager.schedule_cycle(cycle.cycle_id)
        await manager.start_cycle(cycle.cycle_id)
        for _ in range(6):
            await manager.advance_phase(cycle.cycle_id)
        next_cycle = await manager.auto_schedule_next(cycle.cycle_id)
        assert len(next_cycle.commodity_scope) == 3

    @pytest.mark.asyncio
    async def test_auto_schedule_incomplete_cycle_raises(self, manager, coffee_scope):
        cycle = await manager.create_cycle(
            operator_id="operator-001",
            review_year=2025,
            review_type=ReviewType.ANNUAL,
            commodity_scope=[coffee_scope],
        )
        with pytest.raises(ValueError, match="not completed"):
            await manager.auto_schedule_next(cycle.cycle_id)
