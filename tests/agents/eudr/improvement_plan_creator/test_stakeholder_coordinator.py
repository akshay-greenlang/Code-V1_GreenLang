# -*- coding: utf-8 -*-
"""
Unit tests for StakeholderCoordinator - AGENT-EUDR-035

Tests stakeholder assignment, RACI validation, notification dispatch,
acknowledgment tracking, and bulk operations. Validates RACI governance
rules (exactly 1 A, >= 1 R per action).

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (Engine 7: Stakeholder Coordinator)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.improvement_plan_creator.config import (
    ImprovementPlanCreatorConfig,
)
from greenlang.agents.eudr.improvement_plan_creator.stakeholder_coordinator import (
    StakeholderCoordinator,
)
from greenlang.agents.eudr.improvement_plan_creator.models import (
    ActionStatus,
    ActionType,
    EisenhowerQuadrant,
    ImprovementAction,
    NotificationChannel,
    NotificationRecord,
    RACIRole,
    StakeholderAssignment,
)


@pytest.fixture
def config():
    return ImprovementPlanCreatorConfig()


@pytest.fixture
def coordinator(config):
    return StakeholderCoordinator(config=config)


@pytest.fixture
def sample_action():
    """Create a sample action for assignment tests."""
    now = datetime.now(tz=timezone.utc)
    return ImprovementAction(
        action_id="act-test-001",
        plan_id="plan-001",
        gap_id="gap-001",
        action_type=ActionType.CORRECTIVE,
        title="Test Action",
        status=ActionStatus.PROPOSED,
        time_bound_deadline=now + timedelta(days=60),
        priority_score=Decimal("80.00"),
        eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
        urgency_score=Decimal("85.00"),
        importance_score=Decimal("75.00"),
        provenance_hash="a" * 64,
    )


@pytest.fixture
def multiple_actions():
    """Create multiple actions for batch tests."""
    now = datetime.now(tz=timezone.utc)
    return [
        ImprovementAction(
            action_id=f"act-{i}",
            plan_id="plan-001",
            gap_id="gap-001",
            action_type=ActionType.CORRECTIVE,
            title=f"Action {i}",
            status=ActionStatus.PROPOSED,
            time_bound_deadline=now + timedelta(days=30 * i),
            priority_score=Decimal("80.00"),
            eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
            urgency_score=Decimal("85.00"),
            importance_score=Decimal("75.00"),
            provenance_hash="a" * 64,
        )
        for i in range(1, 4)
    ]


# ---------------------------------------------------------------------------
# Assign Stakeholders
# ---------------------------------------------------------------------------

class TestAssignStakeholders:
    """Test stakeholder assignment to actions."""

    @pytest.mark.asyncio
    async def test_assign_returns_list(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "responsible",
            }
        ]
        assignments = await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )
        assert isinstance(assignments, list)
        assert len(assignments) == 1

    @pytest.mark.asyncio
    async def test_assignment_has_id(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "responsible",
            }
        ]
        assignments = await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )
        assert assignments[0].assignment_id.startswith("ASSIGN-")

    @pytest.mark.asyncio
    async def test_assignment_links_stakeholder(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "responsible",
            }
        ]
        assignments = await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )
        assert assignments[0].stakeholder_id == "sh-001"

    @pytest.mark.asyncio
    async def test_assignment_links_action(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "responsible",
            }
        ]
        assignments = await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )
        assert assignments[0].action_id == sample_action.action_id

    @pytest.mark.asyncio
    async def test_assignment_has_timestamp(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "responsible",
            }
        ]
        assignments = await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )
        assert assignments[0].assigned_at is not None

    @pytest.mark.asyncio
    async def test_assignment_preserves_role(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "accountable",
            }
        ]
        assignments = await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )
        assert assignments[0].role == RACIRole.ACCOUNTABLE

    @pytest.mark.asyncio
    async def test_assign_multiple_stakeholders(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "accountable",
            },
            {
                "id": "sh-002",
                "name": "Bob Worker",
                "email": "bob@company.com",
                "role": "responsible",
            },
        ]
        assignments = await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )
        assert len(assignments) == 2


# ---------------------------------------------------------------------------
# RACI Validation
# ---------------------------------------------------------------------------

class TestRACIValidation:
    """Test RACI completeness validation."""

    @pytest.mark.asyncio
    async def test_valid_raci_assignment(self, coordinator, sample_action, config):
        # Enable RACI validation
        config.raci_validation_enabled = True
        coordinator.config = config

        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "accountable",
            },
            {
                "id": "sh-002",
                "name": "Bob Worker",
                "email": "bob@company.com",
                "role": "responsible",
            },
        ]
        # This should not raise an error
        assignments = await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )
        assert len(assignments) == 2

    @pytest.mark.asyncio
    async def test_assign_defaults_to_informed(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                # No role specified
            }
        ]
        assignments = await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )
        assert assignments[0].role == RACIRole.INFORMED


# ---------------------------------------------------------------------------
# Notification Dispatch
# ---------------------------------------------------------------------------

class TestNotificationDispatch:
    """Test stakeholder notification dispatch."""

    @pytest.mark.asyncio
    async def test_send_notification_returns_record(self, coordinator, sample_action):
        # First assign a stakeholder
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "responsible",
            }
        ]
        await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )

        # Then send notification
        record = await coordinator.send_notification(
            action_id=sample_action.action_id,
            stakeholder_id="sh-001",
            subject="Action Assigned",
            body="You have been assigned a new action.",
        )
        assert isinstance(record, NotificationRecord)

    @pytest.mark.asyncio
    async def test_notification_has_id(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "responsible",
            }
        ]
        await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )

        record = await coordinator.send_notification(
            action_id=sample_action.action_id,
            stakeholder_id="sh-001",
            subject="Test",
            body="Test message",
        )
        assert record.notification_id.startswith("NOTIF-")

    @pytest.mark.asyncio
    async def test_notification_marked_delivered(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "responsible",
            }
        ]
        await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )

        record = await coordinator.send_notification(
            action_id=sample_action.action_id,
            stakeholder_id="sh-001",
            subject="Test",
            body="Test message",
        )
        assert record.delivered is True

    @pytest.mark.asyncio
    async def test_send_bulk_notifications(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "accountable",
            },
            {
                "id": "sh-002",
                "name": "Bob Worker",
                "email": "bob@company.com",
                "role": "responsible",
            },
        ]
        await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )

        records = await coordinator.send_bulk_notifications(
            action=sample_action,
            subject="Plan Update",
            body="The improvement plan has been updated.",
        )
        assert len(records) == 2


# ---------------------------------------------------------------------------
# Acknowledgment Tracking
# ---------------------------------------------------------------------------

class TestAcknowledgmentTracking:
    """Test stakeholder acknowledgment tracking."""

    @pytest.mark.asyncio
    async def test_acknowledge_returns_assignment(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "responsible",
            }
        ]
        await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )

        assignment = await coordinator.acknowledge(
            action_id=sample_action.action_id,
            stakeholder_id="sh-001",
        )
        assert isinstance(assignment, StakeholderAssignment)

    @pytest.mark.asyncio
    async def test_acknowledge_sets_timestamp(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "responsible",
            }
        ]
        await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )

        assignment = await coordinator.acknowledge(
            action_id=sample_action.action_id,
            stakeholder_id="sh-001",
        )
        assert assignment.acknowledged_at is not None

    @pytest.mark.asyncio
    async def test_acknowledge_unknown_returns_none(self, coordinator):
        assignment = await coordinator.acknowledge(
            action_id="unknown-action",
            stakeholder_id="unknown-stakeholder",
        )
        assert assignment is None

    @pytest.mark.asyncio
    async def test_get_pending_acknowledgments(self, coordinator, multiple_actions):
        # Assign stakeholders and notify
        for action in multiple_actions:
            stakeholders = [
                {
                    "id": f"sh-{action.action_id}",
                    "name": "Test Stakeholder",
                    "email": "test@company.com",
                    "role": "responsible",
                }
            ]
            await coordinator.assign_stakeholders(
                action=action,
                stakeholders=stakeholders,
            )
            await coordinator.send_notification(
                action_id=action.action_id,
                stakeholder_id=f"sh-{action.action_id}",
                subject="Test",
                body="Test",
            )

        pending = await coordinator.get_pending_acknowledgments(multiple_actions)
        assert len(pending) == len(multiple_actions)


# ---------------------------------------------------------------------------
# Assignment Retrieval
# ---------------------------------------------------------------------------

class TestAssignmentRetrieval:
    """Test assignment retrieval methods."""

    @pytest.mark.asyncio
    async def test_get_assignments_returns_list(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "responsible",
            }
        ]
        await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )

        assignments = await coordinator.get_assignments(sample_action.action_id)
        assert isinstance(assignments, list)
        assert len(assignments) == 1

    @pytest.mark.asyncio
    async def test_get_assignments_empty_for_unknown(self, coordinator):
        assignments = await coordinator.get_assignments("unknown-action")
        assert assignments == []

    @pytest.mark.asyncio
    async def test_get_notifications_returns_list(self, coordinator, sample_action):
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "responsible",
            }
        ]
        await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )
        await coordinator.send_notification(
            action_id=sample_action.action_id,
            stakeholder_id="sh-001",
            subject="Test",
            body="Test",
        )

        notifications = await coordinator.get_notifications(sample_action.action_id)
        assert isinstance(notifications, list)
        assert len(notifications) == 1

    @pytest.mark.asyncio
    async def test_get_notifications_empty_for_unknown(self, coordinator):
        notifications = await coordinator.get_notifications("unknown-action")
        assert notifications == []


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Test engine health check."""

    @pytest.mark.asyncio
    async def test_health_check_returns_dict(self, coordinator):
        result = await coordinator.health_check()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_health_check_has_status(self, coordinator):
        result = await coordinator.health_check()
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_has_engine_name(self, coordinator):
        result = await coordinator.health_check()
        assert result["engine"] == "StakeholderCoordinator"

    @pytest.mark.asyncio
    async def test_health_check_tracks_metrics(self, coordinator, sample_action):
        # Add some assignments
        stakeholders = [
            {
                "id": "sh-001",
                "name": "Alice Manager",
                "email": "alice@company.com",
                "role": "responsible",
            }
        ]
        await coordinator.assign_stakeholders(
            action=sample_action,
            stakeholders=stakeholders,
        )

        result = await coordinator.health_check()
        assert "total_assignments" in result
        assert result["total_assignments"] > 0
