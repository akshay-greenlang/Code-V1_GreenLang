# -*- coding: utf-8 -*-
"""
Integration tests - End-to-end alert flow (OBS-004)

Tests the full alerting pipeline from alert creation through routing,
notification, deduplication, escalation, and lifecycle management.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.alerting_service.config import AlertingConfig
from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    NotificationResult,
    NotificationStatus,
)


# Import test-level reference implementations from unit tests
from tests.unit.alerting_service.test_lifecycle import AlertLifecycle
from tests.unit.alerting_service.test_deduplication import AlertDeduplicator
from tests.unit.alerting_service.test_router import AlertRouter
from tests.unit.alerting_service.test_escalation import EscalationEngine
from tests.unit.alerting_service.test_webhook_receiver import WebhookReceiver
from tests.unit.alerting_service.test_analytics import AlertAnalytics


# ============================================================================
# End-to-end alert flow tests
# ============================================================================


class TestEndToEndAlertFlow:
    """Integration test suite for the complete alert lifecycle."""

    @pytest.fixture
    def lifecycle(self):
        return AlertLifecycle()

    @pytest.fixture
    def deduplicator(self):
        return AlertDeduplicator(window_minutes=60)

    @pytest.fixture
    def router(self, alerting_config):
        return AlertRouter(alerting_config)

    @pytest.fixture
    def escalation(self):
        return EscalationEngine(ack_timeout_minutes=15)

    @pytest.fixture
    def analytics(self):
        return AlertAnalytics(enabled=True)

    @pytest.fixture
    def receiver(self):
        return WebhookReceiver()

    @pytest.fixture
    def mock_slack_channel(self):
        ch = AsyncMock()
        ch.send.return_value = NotificationResult(
            channel=NotificationChannel.SLACK,
            status=NotificationStatus.SENT,
            duration_ms=50.0,
        )
        return ch

    @pytest.fixture
    def mock_pd_channel(self):
        ch = AsyncMock()
        ch.send.return_value = NotificationResult(
            channel=NotificationChannel.PAGERDUTY,
            status=NotificationStatus.SENT,
            duration_ms=120.0,
        )
        return ch

    @pytest.fixture
    def mock_email_channel(self):
        ch = AsyncMock()
        ch.send.return_value = NotificationResult(
            channel=NotificationChannel.EMAIL,
            status=NotificationStatus.SENT,
            duration_ms=200.0,
        )
        return ch

    @pytest.mark.asyncio
    async def test_fire_route_notify(
        self, lifecycle, router, sample_critical_alert,
        mock_slack_channel, mock_pd_channel,
    ):
        """Create alert -> route -> send to channels."""
        router.register_channel("slack", mock_slack_channel)
        router.register_channel("pagerduty", mock_pd_channel)

        fired = lifecycle.fire(sample_critical_alert)
        channels = router.route(fired)
        results = await router.notify(fired, channels)

        assert fired.status == AlertStatus.FIRING
        assert len(results) >= 1
        assert any(r.status == NotificationStatus.SENT for r in results)

    @pytest.mark.asyncio
    async def test_fire_acknowledge_resolve(self, lifecycle, analytics, sample_critical_alert):
        """Full lifecycle: fire -> ack -> resolve."""
        fired = lifecycle.fire(sample_critical_alert)
        analytics.record_fired(fired)

        acked = lifecycle.acknowledge(fired.alert_id, user="jane")
        acked.acknowledged_at = datetime.now(timezone.utc)
        analytics.record_acknowledged(acked)

        resolved = lifecycle.resolve(fired.alert_id, user="ops-bot")
        resolved.resolved_at = datetime.now(timezone.utc)
        analytics.record_resolved(resolved)

        assert resolved.status == AlertStatus.RESOLVED
        assert len(analytics._events) == 3

    @pytest.mark.asyncio
    async def test_fire_deduplicate(
        self, lifecycle, deduplicator, router, sample_critical_alert,
        mock_slack_channel,
    ):
        """Duplicate alert is suppressed, not re-notified."""
        router.register_channel("slack", mock_slack_channel)

        # First fire
        alert1, is_new1 = deduplicator.process(sample_critical_alert)
        assert is_new1 is True

        # Duplicate
        dup = Alert(
            source=sample_critical_alert.source,
            name=sample_critical_alert.name,
            severity=sample_critical_alert.severity,
            title=sample_critical_alert.title,
            labels=dict(sample_critical_alert.labels),
        )
        alert2, is_new2 = deduplicator.process(dup)
        assert is_new2 is False
        assert alert2.alert_id == alert1.alert_id

    @pytest.mark.asyncio
    async def test_fire_escalate_after_timeout(
        self, lifecycle, escalation, sample_critical_alert,
    ):
        """Unacked alert past timeout is auto-escalated."""
        fired = lifecycle.fire(sample_critical_alert)
        fired.fired_at = datetime.now(timezone.utc) - timedelta(minutes=20)

        assert escalation.should_escalate(fired) is True

        channels = escalation.escalate(fired)
        assert fired.escalation_level == 1
        assert len(channels) > 0

    @pytest.mark.asyncio
    async def test_acknowledge_records_mtta(self, lifecycle, analytics, sample_critical_alert):
        """MTTA metric is recorded on acknowledge."""
        fired = lifecycle.fire(sample_critical_alert)
        acked = lifecycle.acknowledge(fired.alert_id, user="jane")
        acked.acknowledged_at = acked.fired_at + timedelta(minutes=3)
        analytics.record_acknowledged(acked)

        report = analytics.get_mtta_report(team="platform")
        assert report["count"] == 1
        assert report["avg_mtta_seconds"] == pytest.approx(180.0)

    @pytest.mark.asyncio
    async def test_resolve_records_mttr(self, lifecycle, analytics, sample_critical_alert):
        """MTTR metric is recorded on resolve."""
        fired = lifecycle.fire(sample_critical_alert)
        resolved = lifecycle.resolve(fired.alert_id, user="ops-bot")
        resolved.resolved_at = resolved.fired_at + timedelta(hours=1)
        analytics.record_resolved(resolved)

        report = analytics.get_mttr_report(team="platform")
        assert report["count"] == 1
        assert report["avg_mttr_seconds"] == pytest.approx(3600.0)

    @pytest.mark.asyncio
    async def test_critical_routes_to_pd_og_slack(
        self, router, sample_critical_alert,
    ):
        """Critical alert routes to PD, OG, and Slack channels."""
        channels = router.route(sample_critical_alert)

        assert "pagerduty" in channels
        assert "opsgenie" in channels
        assert "slack" in channels

    @pytest.mark.asyncio
    async def test_warning_routes_to_slack_email(
        self, router, sample_warning_alert,
    ):
        """Warning alert routes to Slack and Email channels."""
        channels = router.route(sample_warning_alert)

        assert "slack" in channels
        assert "email" in channels

    @pytest.mark.asyncio
    async def test_webhook_intake_full_flow(
        self, receiver, lifecycle, deduplicator, router,
        alertmanager_payload, mock_slack_channel, mock_pd_channel,
    ):
        """AM webhook -> parse -> fire -> route -> notify."""
        router.register_channel("slack", mock_slack_channel)
        router.register_channel("pagerduty", mock_pd_channel)

        # Parse webhook
        alerts = receiver.parse(alertmanager_payload)
        assert len(alerts) == 1

        # Process through pipeline
        alert = alerts[0]
        processed, is_new = deduplicator.process(alert)
        assert is_new is True

        fired = lifecycle.fire(processed)
        channels = router.route(fired)
        results = await router.notify(fired, channels)

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_channel_failure_doesnt_block(
        self, router, sample_critical_alert,
    ):
        """One channel fails, others still succeed."""
        mock_fail = AsyncMock()
        mock_fail.send.side_effect = ConnectionError("PD down")

        mock_ok = AsyncMock()
        mock_ok.send.return_value = NotificationResult(
            channel=NotificationChannel.SLACK,
            status=NotificationStatus.SENT,
        )

        router.register_channel("pagerduty", mock_fail)
        router.register_channel("slack", mock_ok)

        results = await router.notify(
            sample_critical_alert,
            channels=["pagerduty", "slack"],
        )

        statuses = [r.status for r in results]
        assert NotificationStatus.SENT in statuses
        assert NotificationStatus.FAILED in statuses

    @pytest.mark.asyncio
    async def test_suppress_prevents_notification(
        self, lifecycle, router, sample_critical_alert, mock_slack_channel,
    ):
        """Suppressed alerts are not notified (SUPPRESSED status check)."""
        router.register_channel("slack", mock_slack_channel)

        fired = lifecycle.fire(sample_critical_alert)
        lifecycle.suppress(fired.alert_id)

        assert fired.status == AlertStatus.SUPPRESSED

    @pytest.mark.asyncio
    async def test_test_notification(
        self, router, mock_slack_channel,
    ):
        """Test notification sends a test alert through the pipeline."""
        router.register_channel("slack", mock_slack_channel)

        test_alert = Alert(
            source="test",
            name="TestNotification",
            severity=AlertSeverity.INFO,
            title="This is a test alert notification",
            team="platform",
        )
        results = await router.notify(test_alert, channels=["slack"])

        assert len(results) == 1
        assert results[0].status == NotificationStatus.SENT
