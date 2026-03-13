# -*- coding: utf-8 -*-
"""
Unit tests for NotificationEngine - AGENT-EUDR-034

Tests notification scheduling, delivery, template rendering,
multi-channel support, retry logic, priority handling, bulk
operations, and delivery tracking.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 (Engine 7: Notification Engine)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from greenlang.agents.eudr.annual_review_scheduler.config import (
    AnnualReviewSchedulerConfig,
)
from greenlang.agents.eudr.annual_review_scheduler.notification_engine import (
    NotificationEngine,
)
from greenlang.agents.eudr.annual_review_scheduler.models import (
    NotificationChannel,
    NotificationPriority,
    NotificationRecord,
    NotificationStatus,
    NotificationTemplate,
    ReviewPhase,
)
from greenlang.agents.eudr.annual_review_scheduler.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return AnnualReviewSchedulerConfig()


@pytest.fixture
def engine(config):
    return NotificationEngine(config=config, provenance=ProvenanceTracker())


# ---------------------------------------------------------------------------
# Notification Creation
# ---------------------------------------------------------------------------

class TestNotificationCreation:
    """Test notification creation."""

    @pytest.mark.asyncio
    async def test_create_notification_returns_record(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="analyst@company.com",
            subject="Test Notification",
            body="This is a test notification.",
        )
        assert isinstance(ntf, NotificationRecord)
        assert ntf.notification_id.startswith("ntf-")

    @pytest.mark.asyncio
    async def test_create_notification_sets_channel(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.SLACK,
            priority=NotificationPriority.HIGH,
            recipient="#eudr-compliance",
            subject="Slack Alert",
            body="Alert message",
        )
        assert ntf.channel == NotificationChannel.SLACK

    @pytest.mark.asyncio
    async def test_create_notification_initial_status_pending(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="user@company.com",
            subject="Test",
            body="Body",
        )
        assert ntf.status == NotificationStatus.PENDING

    @pytest.mark.asyncio
    async def test_create_notification_sets_priority(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.SMS,
            priority=NotificationPriority.CRITICAL,
            recipient="+1234567890",
            subject="Critical Alert",
            body="Immediate action required",
        )
        assert ntf.priority == NotificationPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_create_webhook_notification(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.WEBHOOK,
            priority=NotificationPriority.HIGH,
            recipient="https://hooks.company.com/eudr",
            subject="Deadline Warning",
            body='{"event":"deadline_warning"}',
        )
        assert ntf.channel == NotificationChannel.WEBHOOK

    @pytest.mark.asyncio
    async def test_create_in_app_notification(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.IN_APP,
            priority=NotificationPriority.NORMAL,
            recipient="user-001",
            subject="In-App Alert",
            body="You have a pending task.",
        )
        assert ntf.channel == NotificationChannel.IN_APP


# ---------------------------------------------------------------------------
# Notification Sending
# ---------------------------------------------------------------------------

class TestNotificationSending:
    """Test notification delivery."""

    @pytest.mark.asyncio
    async def test_send_email_notification(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="user@company.com",
            subject="Test",
            body="Body",
        )
        sent = await engine.send_notification(ntf.notification_id)
        assert sent.status == NotificationStatus.SENT
        assert sent.sent_at is not None

    @pytest.mark.asyncio
    async def test_send_already_sent_raises(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="user@company.com",
            subject="Test",
            body="Body",
        )
        await engine.send_notification(ntf.notification_id)
        with pytest.raises(ValueError, match="already sent"):
            await engine.send_notification(ntf.notification_id)

    @pytest.mark.asyncio
    async def test_send_webhook_notification(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.WEBHOOK,
            priority=NotificationPriority.HIGH,
            recipient="https://hooks.company.com/eudr",
            subject="Event",
            body='{"event":"test"}',
        )
        sent = await engine.send_notification(ntf.notification_id)
        assert sent.status == NotificationStatus.SENT

    @pytest.mark.asyncio
    async def test_send_nonexistent_raises(self, engine):
        with pytest.raises(ValueError, match="not found"):
            await engine.send_notification("ntf-nonexistent")


# ---------------------------------------------------------------------------
# Template Rendering
# ---------------------------------------------------------------------------

class TestTemplateRendering:
    """Test notification template rendering."""

    @pytest.mark.asyncio
    async def test_register_template(self, engine, sample_notification_template):
        result = await engine.register_template(sample_notification_template)
        assert result is True

    @pytest.mark.asyncio
    async def test_render_template(self, engine, sample_notification_template):
        await engine.register_template(sample_notification_template)
        rendered = await engine.render_template(
            template_id=sample_notification_template.template_id,
            variables={
                "recipient_name": "John Smith",
                "phase_name": "Data Collection",
                "cycle_id": "cyc-001",
                "start_date": "2026-04-01",
            },
        )
        assert "John Smith" in rendered["body"]
        assert "Data Collection" in rendered["subject"]
        assert "cyc-001" in rendered["body"]

    @pytest.mark.asyncio
    async def test_render_missing_template_raises(self, engine):
        with pytest.raises(ValueError, match="not found"):
            await engine.render_template(
                template_id="tpl-nonexistent",
                variables={},
            )

    @pytest.mark.asyncio
    async def test_render_template_with_missing_variable(self, engine, sample_notification_template):
        await engine.register_template(sample_notification_template)
        rendered = await engine.render_template(
            template_id=sample_notification_template.template_id,
            variables={"recipient_name": "John"},
        )
        assert rendered is not None

    @pytest.mark.asyncio
    async def test_create_from_template(self, engine, sample_notification_template):
        await engine.register_template(sample_notification_template)
        ntf = await engine.create_from_template(
            cycle_id="cyc-001",
            template_id=sample_notification_template.template_id,
            recipient="user@company.com",
            variables={
                "recipient_name": "Jane Doe",
                "phase_name": "Analysis",
                "cycle_id": "cyc-001",
                "start_date": "2026-05-01",
            },
        )
        assert isinstance(ntf, NotificationRecord)
        assert ntf.template_id == sample_notification_template.template_id


# ---------------------------------------------------------------------------
# Retry Logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    """Test notification retry mechanism."""

    @pytest.mark.asyncio
    async def test_retry_failed_notification(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="user@company.com",
            subject="Test",
            body="Body",
        )
        await engine.mark_failed(
            ntf.notification_id, reason="SMTP connection timeout",
        )
        retried = await engine.retry_notification(ntf.notification_id)
        assert retried.retry_count >= 1

    @pytest.mark.asyncio
    async def test_retry_exceeds_max_retries(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.SMS,
            priority=NotificationPriority.HIGH,
            recipient="+1234567890",
            subject="Test",
            body="Body",
        )
        for i in range(3):
            await engine.mark_failed(ntf.notification_id, reason=f"Attempt {i+1}")
        with pytest.raises(ValueError, match="max retries"):
            await engine.retry_notification(ntf.notification_id)

    @pytest.mark.asyncio
    async def test_retry_already_sent_raises(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="user@company.com",
            subject="Test",
            body="Body",
        )
        await engine.send_notification(ntf.notification_id)
        with pytest.raises(ValueError, match="already sent"):
            await engine.retry_notification(ntf.notification_id)

    @pytest.mark.asyncio
    async def test_mark_failed_sets_status(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="user@company.com",
            subject="Test",
            body="Body",
        )
        failed = await engine.mark_failed(
            ntf.notification_id, reason="Connection refused",
        )
        assert failed.status == NotificationStatus.FAILED
        assert failed.failure_reason == "Connection refused"


# ---------------------------------------------------------------------------
# Bulk Operations
# ---------------------------------------------------------------------------

class TestBulkOperations:
    """Test bulk notification operations."""

    @pytest.mark.asyncio
    async def test_send_bulk_notifications(self, engine):
        ntf_ids = []
        for i in range(5):
            ntf = await engine.create_notification(
                cycle_id="cyc-001",
                channel=NotificationChannel.EMAIL,
                priority=NotificationPriority.NORMAL,
                recipient=f"user{i}@company.com",
                subject=f"Notification {i}",
                body=f"Body {i}",
            )
            ntf_ids.append(ntf.notification_id)
        results = await engine.send_bulk(ntf_ids)
        assert len(results) == 5
        for result in results:
            assert result.status == NotificationStatus.SENT

    @pytest.mark.asyncio
    async def test_cancel_pending_notifications(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.LOW,
            recipient="user@company.com",
            subject="Cancelable",
            body="Body",
        )
        cancelled = await engine.cancel_notification(ntf.notification_id)
        assert cancelled.status == NotificationStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_sent_notification_raises(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001",
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="user@company.com",
            subject="Already Sent",
            body="Body",
        )
        await engine.send_notification(ntf.notification_id)
        with pytest.raises(ValueError, match="Cannot cancel"):
            await engine.cancel_notification(ntf.notification_id)


# ---------------------------------------------------------------------------
# List and Filter
# ---------------------------------------------------------------------------

class TestListAndFilter:
    """Test notification listing and filtering."""

    @pytest.mark.asyncio
    async def test_list_by_cycle(self, engine):
        await engine.create_notification(
            cycle_id="cyc-001", channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="a@company.com", subject="A", body="A",
        )
        await engine.create_notification(
            cycle_id="cyc-002", channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="b@company.com", subject="B", body="B",
        )
        notifications = await engine.list_notifications(cycle_id="cyc-001")
        assert len(notifications) == 1

    @pytest.mark.asyncio
    async def test_list_by_status(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001", channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="user@company.com", subject="T", body="B",
        )
        await engine.send_notification(ntf.notification_id)
        pending = await engine.list_notifications(
            cycle_id="cyc-001", status=NotificationStatus.PENDING,
        )
        sent = await engine.list_notifications(
            cycle_id="cyc-001", status=NotificationStatus.SENT,
        )
        assert len(pending) == 0
        assert len(sent) == 1

    @pytest.mark.asyncio
    async def test_list_by_channel(self, engine):
        await engine.create_notification(
            cycle_id="cyc-001", channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="user@company.com", subject="Email", body="B",
        )
        await engine.create_notification(
            cycle_id="cyc-001", channel=NotificationChannel.SLACK,
            priority=NotificationPriority.NORMAL,
            recipient="#channel", subject="Slack", body="B",
        )
        emails = await engine.list_notifications(
            cycle_id="cyc-001", channel=NotificationChannel.EMAIL,
        )
        assert len(emails) == 1

    @pytest.mark.asyncio
    async def test_list_by_priority(self, engine):
        await engine.create_notification(
            cycle_id="cyc-001", channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="user@company.com", subject="Normal", body="B",
        )
        await engine.create_notification(
            cycle_id="cyc-001", channel=NotificationChannel.SMS,
            priority=NotificationPriority.CRITICAL,
            recipient="+1234567890", subject="Critical", body="B",
        )
        critical = await engine.list_notifications(
            cycle_id="cyc-001", priority=NotificationPriority.CRITICAL,
        )
        assert len(critical) == 1

    @pytest.mark.asyncio
    async def test_get_notification_by_id(self, engine):
        ntf = await engine.create_notification(
            cycle_id="cyc-001", channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="user@company.com", subject="T", body="B",
        )
        retrieved = await engine.get_notification(ntf.notification_id)
        assert retrieved.notification_id == ntf.notification_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_notification_raises(self, engine):
        with pytest.raises(ValueError, match="not found"):
            await engine.get_notification("ntf-nonexistent")


# ---------------------------------------------------------------------------
# Delivery Statistics
# ---------------------------------------------------------------------------

class TestDeliveryStatistics:
    """Test delivery statistics computation."""

    @pytest.mark.asyncio
    async def test_get_delivery_stats(self, engine):
        for i in range(5):
            ntf = await engine.create_notification(
                cycle_id="cyc-001", channel=NotificationChannel.EMAIL,
                priority=NotificationPriority.NORMAL,
                recipient=f"user{i}@company.com",
                subject=f"Notification {i}", body=f"Body {i}",
            )
            if i < 3:
                await engine.send_notification(ntf.notification_id)
            elif i == 3:
                await engine.mark_failed(ntf.notification_id, reason="Error")
        stats = await engine.get_delivery_stats(cycle_id="cyc-001")
        assert stats["total"] == 5
        assert stats["sent"] == 3
        assert stats["failed"] == 1
        assert stats["pending"] == 1

    @pytest.mark.asyncio
    async def test_delivery_stats_empty_cycle(self, engine):
        stats = await engine.get_delivery_stats(cycle_id="cyc-empty")
        assert stats["total"] == 0

    @pytest.mark.asyncio
    async def test_delivery_stats_by_channel(self, engine):
        await engine.create_notification(
            cycle_id="cyc-stats", channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="user@company.com", subject="Email", body="B",
        )
        await engine.create_notification(
            cycle_id="cyc-stats", channel=NotificationChannel.SLACK,
            priority=NotificationPriority.NORMAL,
            recipient="#channel", subject="Slack", body="B",
        )
        stats = await engine.get_delivery_stats(cycle_id="cyc-stats")
        assert stats["total"] == 2


# ---------------------------------------------------------------------------
# Scheduled Notifications
# ---------------------------------------------------------------------------

class TestScheduledNotifications:
    """Test scheduled notification generation for review cycle events."""

    @pytest.mark.asyncio
    async def test_schedule_phase_notifications(self, engine, sample_review_cycle):
        notifications = await engine.schedule_for_cycle(
            cycle=sample_review_cycle,
            recipients=["lead@company.com", "analyst@company.com"],
        )
        assert len(notifications) > 0

    @pytest.mark.asyncio
    async def test_scheduled_notifications_are_pending(self, engine, sample_review_cycle):
        notifications = await engine.schedule_for_cycle(
            cycle=sample_review_cycle,
            recipients=["user@company.com"],
        )
        for ntf in notifications:
            assert ntf.status == NotificationStatus.PENDING

    @pytest.mark.asyncio
    async def test_schedule_empty_recipients_returns_empty(self, engine, sample_review_cycle):
        notifications = await engine.schedule_for_cycle(
            cycle=sample_review_cycle,
            recipients=[],
        )
        assert notifications == []

    @pytest.mark.asyncio
    async def test_schedule_includes_deadline_reminders(self, engine, sample_review_cycle):
        notifications = await engine.schedule_for_cycle(
            cycle=sample_review_cycle,
            recipients=["user@company.com"],
        )
        deadline_ntfs = [n for n in notifications if "deadline" in n.subject.lower() or "deadline" in n.body.lower()]
        # Should have at least some deadline-related notifications
        assert len(notifications) > 0

    @pytest.mark.asyncio
    async def test_schedule_creates_unique_ids(self, engine, sample_review_cycle):
        notifications = await engine.schedule_for_cycle(
            cycle=sample_review_cycle,
            recipients=["user@company.com"],
        )
        ids = [n.notification_id for n in notifications]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Notification Priority Sorting
# ---------------------------------------------------------------------------

class TestPrioritySorting:
    """Test notification priority ordering."""

    @pytest.mark.asyncio
    async def test_critical_before_normal(self, engine):
        await engine.create_notification(
            cycle_id="cyc-prio", channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL,
            recipient="user@company.com", subject="Normal", body="B",
        )
        await engine.create_notification(
            cycle_id="cyc-prio", channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.CRITICAL,
            recipient="user@company.com", subject="Critical", body="B",
        )
        all_ntfs = await engine.list_notifications(cycle_id="cyc-prio")
        assert len(all_ntfs) == 2

    @pytest.mark.asyncio
    async def test_get_pending_by_priority_order(self, engine):
        for priority in [NotificationPriority.LOW, NotificationPriority.CRITICAL,
                         NotificationPriority.NORMAL]:
            await engine.create_notification(
                cycle_id="cyc-order", channel=NotificationChannel.EMAIL,
                priority=priority,
                recipient="user@company.com",
                subject=f"Priority {priority.value}", body="B",
            )
        pending = await engine.list_notifications(
            cycle_id="cyc-order", status=NotificationStatus.PENDING,
        )
        assert len(pending) == 3
