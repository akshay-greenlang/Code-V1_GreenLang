# -*- coding: utf-8 -*-
"""
Unit tests for AuditRecorder engine - AGENT-EUDR-036

Tests audit event recording, Article 31 retention, entity querying,
date range filtering, audit report generation, and record purging.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from greenlang.agents.eudr.eu_information_system_interface.config import (
    EUInformationSystemInterfaceConfig,
)
from greenlang.agents.eudr.eu_information_system_interface.audit_recorder import (
    AuditRecorder,
)
from greenlang.agents.eudr.eu_information_system_interface.models import (
    AuditEventType,
)
from greenlang.agents.eudr.eu_information_system_interface.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def recorder() -> AuditRecorder:
    """Create an AuditRecorder instance."""
    config = EUInformationSystemInterfaceConfig()
    return AuditRecorder(config=config, provenance=ProvenanceTracker())


@pytest.fixture
def summary_recorder() -> AuditRecorder:
    """Create an AuditRecorder with summary detail level."""
    config = EUInformationSystemInterfaceConfig(audit_detail_level="summary")
    return AuditRecorder(config=config, provenance=ProvenanceTracker())


@pytest.fixture
def minimal_recorder() -> AuditRecorder:
    """Create an AuditRecorder with minimal detail level."""
    config = EUInformationSystemInterfaceConfig(audit_detail_level="minimal")
    return AuditRecorder(config=config, provenance=ProvenanceTracker())


class TestRecordEvent:
    """Test AuditRecorder.record_event()."""

    @pytest.mark.asyncio
    async def test_record_basic_event(self, recorder):
        record = await recorder.record_event(
            event_type="dds_created",
            entity_type="dds",
            entity_id="dds-001",
            actor="system",
            action="create",
        )
        assert record.audit_id.startswith("aud-")
        assert record.event_type == AuditEventType.DDS_CREATED
        assert record.entity_type == "dds"
        assert record.entity_id == "dds-001"
        assert record.actor == "system"

    @pytest.mark.asyncio
    async def test_record_with_details(self, recorder):
        record = await recorder.record_event(
            event_type="dds_submitted",
            entity_type="dds",
            entity_id="dds-001",
            actor="user@example.com",
            action="submit",
            details={"commodity": "cocoa", "quantity_kg": 50000},
        )
        assert "commodity" in record.details

    @pytest.mark.asyncio
    async def test_retention_until_set(self, recorder):
        record = await recorder.record_event(
            event_type="dds_created",
            entity_type="dds",
            entity_id="dds-001",
            actor="system",
            action="create",
        )
        assert record.retention_until is not None
        # At least 5 years retention per Article 31
        delta = record.retention_until - record.timestamp
        assert delta.days >= 5 * 365 - 1  # Allow 1 day tolerance

    @pytest.mark.asyncio
    async def test_provenance_hash_computed(self, recorder):
        record = await recorder.record_event(
            event_type="dds_created",
            entity_type="dds",
            entity_id="dds-001",
            actor="system",
            action="create",
        )
        assert len(record.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_record_count_increments(self, recorder):
        assert recorder.record_count == 0
        await recorder.record_event(
            event_type="dds_created",
            entity_type="dds",
            entity_id="dds-001",
            actor="system",
            action="create",
        )
        assert recorder.record_count == 1

    @pytest.mark.asyncio
    async def test_unknown_event_type_defaults(self, recorder):
        record = await recorder.record_event(
            event_type="unknown_event_xyz",
            entity_type="dds",
            entity_id="dds-001",
            actor="system",
            action="unknown",
        )
        assert record.event_type == AuditEventType.API_CALL_MADE

    @pytest.mark.asyncio
    async def test_request_response_inclusion(self, recorder):
        record = await recorder.record_event(
            event_type="api_call_made",
            entity_type="api_call",
            entity_id="POST:/dds/submit",
            actor="AGENT-EUDR-036",
            action="POST /dds/submit",
            request_summary="DDS submission payload",
            response_summary="200 OK",
        )
        assert record.request_summary == "DDS submission payload"
        assert record.response_summary == "200 OK"


class TestRecordDDSEvent:
    """Test AuditRecorder.record_dds_event()."""

    @pytest.mark.asyncio
    async def test_dds_event_convenience(self, recorder):
        record = await recorder.record_dds_event(
            dds_id="dds-001",
            event_type="dds_validated",
            actor="system",
        )
        assert record.entity_type == "dds"
        assert record.entity_id == "dds-001"


class TestRecordAPICallEvent:
    """Test AuditRecorder.record_api_call_event()."""

    @pytest.mark.asyncio
    async def test_successful_api_call(self, recorder):
        record = await recorder.record_api_call_event(
            method="POST",
            endpoint="/dds/submit",
            status_code=200,
            duration_ms=125.5,
            success=True,
        )
        assert record.entity_type == "api_call"
        assert record.details["success"] is True

    @pytest.mark.asyncio
    async def test_failed_api_call(self, recorder):
        record = await recorder.record_api_call_event(
            method="POST",
            endpoint="/dds/submit",
            status_code=500,
            duration_ms=3000.0,
            success=False,
            error_message="Internal Server Error",
        )
        assert record.event_type == AuditEventType.API_CALL_FAILED
        assert record.details["error"] == "Internal Server Error"


class TestGetRecordsForEntity:
    """Test AuditRecorder.get_records_for_entity()."""

    @pytest.mark.asyncio
    async def test_get_empty(self, recorder):
        records = await recorder.get_records_for_entity("dds", "dds-unknown")
        assert records == []

    @pytest.mark.asyncio
    async def test_get_entity_records(self, recorder):
        await recorder.record_event(
            event_type="dds_created", entity_type="dds",
            entity_id="dds-001", actor="system", action="create",
        )
        await recorder.record_event(
            event_type="dds_validated", entity_type="dds",
            entity_id="dds-001", actor="system", action="validate",
        )
        await recorder.record_event(
            event_type="dds_created", entity_type="dds",
            entity_id="dds-002", actor="system", action="create",
        )
        records = await recorder.get_records_for_entity("dds", "dds-001")
        assert len(records) == 2


class TestGetRecordsByEventType:
    """Test AuditRecorder.get_records_by_event_type()."""

    @pytest.mark.asyncio
    async def test_filter_by_event_type(self, recorder):
        await recorder.record_event(
            event_type="dds_created", entity_type="dds",
            entity_id="dds-001", actor="system", action="create",
        )
        await recorder.record_event(
            event_type="dds_submitted", entity_type="dds",
            entity_id="dds-001", actor="system", action="submit",
        )
        records = await recorder.get_records_by_event_type("dds_created")
        assert len(records) == 1
        assert records[0].event_type == AuditEventType.DDS_CREATED


class TestGetRecordsByDateRange:
    """Test AuditRecorder.get_records_by_date_range()."""

    @pytest.mark.asyncio
    async def test_date_range_filter(self, recorder):
        await recorder.record_event(
            event_type="dds_created", entity_type="dds",
            entity_id="dds-001", actor="system", action="create",
        )
        now = datetime.now(timezone.utc)
        records = await recorder.get_records_by_date_range(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
        )
        assert len(records) >= 1

    @pytest.mark.asyncio
    async def test_date_range_entity_filter(self, recorder):
        await recorder.record_event(
            event_type="dds_created", entity_type="dds",
            entity_id="dds-001", actor="system", action="create",
        )
        await recorder.record_event(
            event_type="operator_registered", entity_type="operator",
            entity_id="op-001", actor="system", action="register",
        )
        now = datetime.now(timezone.utc)
        records = await recorder.get_records_by_date_range(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
            entity_type="dds",
        )
        assert len(records) == 1


class TestGenerateAuditReport:
    """Test AuditRecorder.generate_audit_report()."""

    @pytest.mark.asyncio
    async def test_empty_report(self, recorder):
        report = await recorder.generate_audit_report("dds", "dds-unknown")
        assert report["total_events"] == 0
        assert report["report_id"].startswith("rpt-")

    @pytest.mark.asyncio
    async def test_report_with_events(self, recorder):
        await recorder.record_event(
            event_type="dds_created", entity_type="dds",
            entity_id="dds-001", actor="system", action="create",
        )
        await recorder.record_event(
            event_type="dds_submitted", entity_type="dds",
            entity_id="dds-001", actor="system", action="submit",
        )
        report = await recorder.generate_audit_report("dds", "dds-001")
        assert report["total_events"] == 2
        assert len(report["records"]) == 2
        assert report["regulation_reference"] == "EU 2023/1115 Article 31"
        assert len(report["provenance_hash"]) == 64


class TestPurgeExpiredRecords:
    """Test AuditRecorder.purge_expired_records()."""

    @pytest.mark.asyncio
    async def test_purge_no_expired(self, recorder):
        await recorder.record_event(
            event_type="dds_created", entity_type="dds",
            entity_id="dds-001", actor="system", action="create",
        )
        result = await recorder.purge_expired_records()
        assert result["purged_count"] == 0
        assert result["remaining_count"] == 1

    @pytest.mark.asyncio
    async def test_purge_with_expired(self, recorder):
        record = await recorder.record_event(
            event_type="dds_created", entity_type="dds",
            entity_id="dds-001", actor="system", action="create",
        )
        # Manually set retention to past
        record.retention_until = datetime.now(timezone.utc) - timedelta(days=1)
        result = await recorder.purge_expired_records()
        assert result["purged_count"] == 1
        assert result["remaining_count"] == 0


class TestDetailFiltering:
    """Test audit detail level filtering."""

    @pytest.mark.asyncio
    async def test_full_details(self, recorder):
        record = await recorder.record_event(
            event_type="dds_created", entity_type="dds",
            entity_id="dds-001", actor="system", action="create",
            details={"commodity": "cocoa", "nested": {"key": "val"}, "count": 5},
        )
        assert "nested" in record.details

    @pytest.mark.asyncio
    async def test_summary_details(self, summary_recorder):
        record = await summary_recorder.record_event(
            event_type="dds_created", entity_type="dds",
            entity_id="dds-001", actor="system", action="create",
            details={"commodity": "cocoa", "nested": {"key": "val"}, "count": 5},
        )
        # Summary keeps only string/number/bool at top level
        assert "commodity" in record.details
        assert "count" in record.details
        assert "nested" not in record.details

    @pytest.mark.asyncio
    async def test_minimal_details(self, minimal_recorder):
        record = await minimal_recorder.record_event(
            event_type="dds_created", entity_type="dds",
            entity_id="dds-001", actor="system", action="create",
            details={"dds_id": "dds-001", "commodity": "cocoa", "status": "draft"},
        )
        # Minimal keeps only *_id and status
        assert "dds_id" in record.details
        assert "status" in record.details
        assert "commodity" not in record.details


class TestHealthCheck:
    """Test AuditRecorder.health_check()."""

    @pytest.mark.asyncio
    async def test_health_check(self, recorder):
        health = await recorder.health_check()
        assert health["engine"] == "AuditRecorder"
        assert health["status"] == "available"
        assert health["record_count"] == 0
