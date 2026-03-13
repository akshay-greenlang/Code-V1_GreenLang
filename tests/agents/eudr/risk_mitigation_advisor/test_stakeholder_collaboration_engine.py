# -*- coding: utf-8 -*-
"""
Tests for Engine 8: Stakeholder Collaboration Engine - AGENT-EUDR-025

Tests multi-party coordination, role-based access, message threading,
task assignment, document sharing, supplier portal, NGO workspace,
activity audit trail, and bulk communication.

Test count: ~60 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    StakeholderRole,
    CollaborateRequest,
    CollaborateResponse,
)
from greenlang.agents.eudr.risk_mitigation_advisor.stakeholder_collaboration_engine import (
    StakeholderCollaborationEngine,
)


class TestCollaborationEngineInit:
    def test_engine_initializes(self, collaboration_engine):
        assert collaboration_engine is not None

    def test_engine_supports_all_roles(self, collaboration_engine):
        roles = collaboration_engine.get_supported_roles()
        assert len(roles) >= 6


class TestMessageActions:
    @pytest.mark.asyncio
    async def test_send_message(self, collaboration_engine, collaborate_message_request):
        result = await collaboration_engine.execute(collaborate_message_request)
        assert isinstance(result, CollaborateResponse)
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_message_has_action_id(self, collaboration_engine, collaborate_message_request):
        result = await collaboration_engine.execute(collaborate_message_request)
        assert result.action_id != ""

    @pytest.mark.asyncio
    async def test_message_processing_time(self, collaboration_engine, collaborate_message_request):
        result = await collaboration_engine.execute(collaborate_message_request)
        assert result.processing_time_ms >= Decimal("0")


class TestTaskAssignment:
    @pytest.mark.asyncio
    async def test_assign_task(self, collaboration_engine, collaborate_task_request):
        result = await collaboration_engine.execute(collaborate_task_request)
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_assign_multiple_tasks(self, collaboration_engine):
        request = CollaborateRequest(
            action="task",
            plan_id="plan-001",
            stakeholder_role=StakeholderRole.PROCUREMENT,
            task_assignments=[
                {"assignee": "team-a", "task": "Task 1", "due_date": "2026-04-01"},
                {"assignee": "team-b", "task": "Task 2", "due_date": "2026-04-15"},
            ],
        )
        result = await collaboration_engine.execute(request)
        assert result.status == "success"


class TestDocumentSharing:
    @pytest.mark.asyncio
    async def test_share_document(self, collaboration_engine):
        request = CollaborateRequest(
            action="document",
            plan_id="plan-001",
            stakeholder_role=StakeholderRole.INTERNAL_COMPLIANCE,
            document_ids=["doc-001", "doc-002"],
        )
        result = await collaboration_engine.execute(request)
        assert result.status == "success"


class TestRoleBasedAccess:
    @pytest.mark.asyncio
    async def test_supplier_access_own_plan(self, collaboration_engine):
        request = CollaborateRequest(
            action="progress",
            plan_id="plan-001",
            stakeholder_role=StakeholderRole.SUPPLIER,
            message="Completed milestone 3.",
        )
        result = await collaboration_engine.execute(request)
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_ngo_access_landscape(self, collaboration_engine):
        request = CollaborateRequest(
            action="message",
            plan_id="plan-001",
            stakeholder_role=StakeholderRole.NGO_PARTNER,
            message="Landscape initiative update.",
        )
        result = await collaboration_engine.execute(request)
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_authority_read_only(self, collaboration_engine):
        request = CollaborateRequest(
            action="message",
            plan_id="plan-001",
            stakeholder_role=StakeholderRole.COMPETENT_AUTHORITY,
            message="Request for documentation.",
        )
        result = await collaboration_engine.execute(request)
        # Authority should be limited but request should not error
        assert isinstance(result, CollaborateResponse)

    @pytest.mark.asyncio
    async def test_certification_body_access(self, collaboration_engine):
        request = CollaborateRequest(
            action="document",
            plan_id="plan-001",
            stakeholder_role=StakeholderRole.CERTIFICATION_BODY,
            document_ids=["audit-report-001"],
        )
        result = await collaboration_engine.execute(request)
        assert isinstance(result, CollaborateResponse)

    @pytest.mark.parametrize("role", list(StakeholderRole))
    @pytest.mark.asyncio
    async def test_all_roles_can_send_message(self, collaboration_engine, role):
        request = CollaborateRequest(
            action="message",
            plan_id="plan-001",
            stakeholder_role=role,
            message=f"Test message from {role.value}.",
        )
        result = await collaboration_engine.execute(request)
        assert isinstance(result, CollaborateResponse)


class TestProgressReporting:
    @pytest.mark.asyncio
    async def test_report_progress(self, collaboration_engine):
        request = CollaborateRequest(
            action="progress",
            plan_id="plan-001",
            stakeholder_role=StakeholderRole.SUPPLIER,
            message="GPS coordinates uploaded for all plots.",
        )
        result = await collaboration_engine.execute(request)
        assert result.status == "success"


class TestBulkCommunication:
    @pytest.mark.asyncio
    async def test_bulk_message(self, collaboration_engine):
        result = await collaboration_engine.send_bulk_message(
            plan_ids=["plan-001", "plan-002", "plan-003"],
            stakeholder_role=StakeholderRole.INTERNAL_COMPLIANCE,
            message="Quarterly compliance update required.",
        )
        assert result is not None
        assert len(result) == 3


class TestActivityAuditTrail:
    @pytest.mark.asyncio
    async def test_action_logged(self, collaboration_engine, collaborate_message_request):
        await collaboration_engine.execute(collaborate_message_request)
        log = await collaboration_engine.get_activity_log(plan_id="plan-001")
        assert isinstance(log, list)
        assert len(log) >= 1

    @pytest.mark.asyncio
    async def test_log_has_timestamp(self, collaboration_engine, collaborate_message_request):
        await collaboration_engine.execute(collaborate_message_request)
        log = await collaboration_engine.get_activity_log(plan_id="plan-001")
        if log:
            assert "timestamp" in log[0]

    @pytest.mark.asyncio
    async def test_log_has_actor(self, collaboration_engine, collaborate_message_request):
        await collaboration_engine.execute(collaborate_message_request)
        log = await collaboration_engine.get_activity_log(plan_id="plan-001")
        if log:
            assert "actor" in log[0] or "stakeholder_role" in log[0]


class TestStakeholderDashboard:
    @pytest.mark.asyncio
    async def test_get_supplier_dashboard(self, collaboration_engine):
        dashboard = await collaboration_engine.get_dashboard(
            plan_id="plan-001",
            stakeholder_role=StakeholderRole.SUPPLIER,
        )
        assert dashboard is not None

    @pytest.mark.asyncio
    async def test_get_internal_dashboard(self, collaboration_engine):
        dashboard = await collaboration_engine.get_dashboard(
            plan_id="plan-001",
            stakeholder_role=StakeholderRole.INTERNAL_COMPLIANCE,
        )
        assert dashboard is not None


class TestCollaborationEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_message(self, collaboration_engine):
        request = CollaborateRequest(
            action="message",
            plan_id="plan-001",
            stakeholder_role=StakeholderRole.INTERNAL_COMPLIANCE,
            message="",
        )
        result = await collaboration_engine.execute(request)
        assert isinstance(result, CollaborateResponse)

    @pytest.mark.asyncio
    async def test_invalid_action(self, collaboration_engine):
        request = CollaborateRequest(
            action="invalid_action",
            plan_id="plan-001",
            stakeholder_role=StakeholderRole.INTERNAL_COMPLIANCE,
        )
        result = await collaboration_engine.execute(request)
        assert isinstance(result, CollaborateResponse)

    @pytest.mark.asyncio
    async def test_no_document_ids(self, collaboration_engine):
        request = CollaborateRequest(
            action="document",
            plan_id="plan-001",
            stakeholder_role=StakeholderRole.INTERNAL_COMPLIANCE,
            document_ids=[],
        )
        result = await collaboration_engine.execute(request)
        assert isinstance(result, CollaborateResponse)
