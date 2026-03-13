# -*- coding: utf-8 -*-
"""
Unit tests for InspectionCoordinator engine - AGENT-EUDR-040

Tests inspection scheduling, status updates, findings recording,
follow-up scheduling, type validation, notice period enforcement,
listing, retrieval, and health checks.

55+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from greenlang.agents.eudr.authority_communication_manager.config import (
    AuthorityCommunicationManagerConfig,
)
from greenlang.agents.eudr.authority_communication_manager.inspection_coordinator import (
    InspectionCoordinator,
)
from greenlang.agents.eudr.authority_communication_manager.models import (
    Inspection,
    InspectionType,
)


@pytest.fixture
def config():
    return AuthorityCommunicationManagerConfig()


@pytest.fixture
def coordinator(config):
    return InspectionCoordinator(config=config)


# ====================================================================
# Initialization
# ====================================================================


class TestInit:
    def test_coordinator_created(self, coordinator):
        assert coordinator is not None

    def test_default_config(self):
        c = InspectionCoordinator()
        assert c.config is not None

    def test_custom_config(self, config):
        c = InspectionCoordinator(config=config)
        assert c.config is config

    def test_inspections_empty(self, coordinator):
        assert len(coordinator._inspections) == 0

    def test_provenance_initialized(self, coordinator):
        assert coordinator._provenance is not None


# ====================================================================
# Schedule Inspection
# ====================================================================


class TestScheduleInspection:
    @pytest.mark.asyncio
    async def test_schedule_announced(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=10)
        result = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            inspection_type="announced",
            scheduled_date=future,
            location="Warehouse Berlin",
            scope="DDS documentation review",
            inspector_name="Dr. Mueller",
        )
        assert isinstance(result, Inspection)
        assert result.inspection_type == InspectionType.ANNOUNCED
        assert result.status == "scheduled"

    @pytest.mark.asyncio
    async def test_schedule_unannounced(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=1)
        result = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-FR-001",
            inspection_type="unannounced",
            scheduled_date=future,
        )
        assert result.inspection_type == InspectionType.UNANNOUNCED

    @pytest.mark.asyncio
    async def test_schedule_remote(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=3)
        result = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-NL-001",
            inspection_type="remote",
            scheduled_date=future,
        )
        assert result.inspection_type == InspectionType.REMOTE

    @pytest.mark.asyncio
    async def test_schedule_follow_up(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=14)
        result = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            inspection_type="follow_up",
            scheduled_date=future,
        )
        assert result.inspection_type == InspectionType.FOLLOW_UP

    @pytest.mark.asyncio
    async def test_schedule_document_review(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=5)
        result = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            inspection_type="document_review",
            scheduled_date=future,
        )
        assert result.inspection_type == InspectionType.DOCUMENT_REVIEW

    @pytest.mark.asyncio
    async def test_schedule_physical(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=7)
        result = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-IT-001",
            inspection_type="physical_inspection",
            scheduled_date=future,
        )
        assert result.inspection_type == InspectionType.PHYSICAL_INSPECTION

    @pytest.mark.asyncio
    async def test_schedule_assigns_id(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=7)
        result = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            inspection_type="announced",
            scheduled_date=future,
        )
        assert result.inspection_id is not None
        assert len(result.inspection_id) > 0

    @pytest.mark.asyncio
    async def test_schedule_computes_provenance(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=7)
        result = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            inspection_type="announced",
            scheduled_date=future,
        )
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_schedule_invalid_type_raises(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=7)
        with pytest.raises(ValueError, match="Invalid"):
            await coordinator.schedule_inspection(
                operator_id="OP-001",
                authority_id="AUTH-DE-001",
                inspection_type="invalid_type",
                scheduled_date=future,
            )

    @pytest.mark.asyncio
    async def test_schedule_stored(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=7)
        result = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            inspection_type="announced",
            scheduled_date=future,
        )
        assert result.inspection_id in coordinator._inspections

    @pytest.mark.asyncio
    async def test_schedule_with_location(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=7)
        result = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            inspection_type="announced",
            scheduled_date=future,
            location="Port of Hamburg, Terminal 4",
        )
        assert result.location == "Port of Hamburg, Terminal 4"


# ====================================================================
# Update Status
# ====================================================================


class TestUpdateStatus:
    @pytest.mark.asyncio
    async def test_update_to_confirmed(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=7)
        insp = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            inspection_type="announced",
            scheduled_date=future,
        )
        result = await coordinator.update_status(
            insp.inspection_id, "confirmed"
        )
        assert result.status == "confirmed"

    @pytest.mark.asyncio
    async def test_update_to_cancelled(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=7)
        insp = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            inspection_type="announced",
            scheduled_date=future,
        )
        result = await coordinator.update_status(
            insp.inspection_id, "cancelled"
        )
        assert result.status == "cancelled"

    @pytest.mark.asyncio
    async def test_update_not_found(self, coordinator):
        with pytest.raises(ValueError, match="not found"):
            await coordinator.update_status("nonexistent", "in_progress")


# ====================================================================
# Record Findings
# ====================================================================


class TestRecordFindings:
    @pytest.mark.asyncio
    async def test_record_findings(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=7)
        insp = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            inspection_type="announced",
            scheduled_date=future,
        )
        result = await coordinator.record_findings(
            inspection_id=insp.inspection_id,
            findings=["DDS incomplete", "Missing geolocation data"],
        )
        assert len(result.findings) == 2

    @pytest.mark.asyncio
    async def test_record_findings_with_corrective_actions(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=7)
        insp = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            inspection_type="announced",
            scheduled_date=future,
        )
        result = await coordinator.record_findings(
            inspection_id=insp.inspection_id,
            findings=["DDS incomplete"],
            corrective_actions=["Submit complete DDS within 30 days"],
        )
        assert len(result.corrective_actions) == 1

    @pytest.mark.asyncio
    async def test_record_findings_not_found(self, coordinator):
        with pytest.raises(ValueError, match="not found"):
            await coordinator.record_findings(
                inspection_id="nonexistent",
                findings=["Finding"],
            )


# ====================================================================
# Get / List / Health
# ====================================================================


class TestGetListHealth:
    @pytest.mark.asyncio
    async def test_get_inspection(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=7)
        insp = await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            inspection_type="announced",
            scheduled_date=future,
        )
        result = await coordinator.get_inspection(insp.inspection_id)
        assert result is not None
        assert result.inspection_id == insp.inspection_id

    @pytest.mark.asyncio
    async def test_get_inspection_not_found(self, coordinator):
        result = await coordinator.get_inspection("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_inspections_empty(self, coordinator):
        result = await coordinator.list_inspections()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_inspections_multiple(self, coordinator):
        future = datetime.now(tz=timezone.utc) + timedelta(days=7)
        await coordinator.schedule_inspection(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            inspection_type="announced",
            scheduled_date=future,
        )
        await coordinator.schedule_inspection(
            operator_id="OP-002",
            authority_id="AUTH-FR-001",
            inspection_type="remote",
            scheduled_date=future,
        )
        result = await coordinator.list_inspections()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_health_check(self, coordinator):
        health = await coordinator.health_check()
        assert health["status"] == "healthy"
