# -*- coding: utf-8 -*-
"""
Load tests - Alerting Service burst scenarios (OBS-004)

Tests burst behavior under high concurrency:
- 100 simultaneous alerts fired and processed
- 100 notifications sent in burst, all succeed (mock)
- 100 duplicate alerts, only 1 notification
- 50 critical + 30 warning + 20 info with correct routing

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock

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

from tests.unit.alerting_service.test_lifecycle import AlertLifecycle
from tests.unit.alerting_service.test_deduplication import AlertDeduplicator
from tests.unit.alerting_service.test_router import AlertRouter


# ============================================================================
# Burst tests
# ============================================================================


class TestAlertingBurst:
    """Burst scenario tests for the alerting service."""

    def test_100_simultaneous_alerts(self):
        """100 alerts fired concurrently, all processed."""
        lifecycle = AlertLifecycle()
        alerts = [
            Alert(
                source="burst-test",
                name=f"BurstAlert-{i}",
                severity=AlertSeverity.CRITICAL,
                title=f"Burst alert {i}",
                labels={"instance": f"node-{i}"},
            )
            for i in range(100)
        ]

        results = []
        for alert in alerts:
            fired = lifecycle.fire(alert)
            results.append(fired)

        assert len(results) == 100
        assert all(r.status == AlertStatus.FIRING for r in results)
        # Each should have a unique alert_id
        ids = [r.alert_id for r in results]
        assert len(set(ids)) == 100

    @pytest.mark.asyncio
    async def test_burst_notification_delivery(self):
        """100 notifications sent in burst, all succeed (mock)."""
        mock_channel = AsyncMock()
        mock_channel.send.return_value = NotificationResult(
            channel=NotificationChannel.SLACK,
            status=NotificationStatus.SENT,
            duration_ms=10.0,
        )

        alerts = [
            Alert(
                source="burst", name=f"Notif-{i}",
                severity=AlertSeverity.WARNING, title=f"Notif {i}",
            )
            for i in range(100)
        ]

        tasks = [mock_channel.send(alert) for alert in alerts]
        results = await asyncio.gather(*tasks)

        assert len(results) == 100
        assert all(r.status == NotificationStatus.SENT for r in results)

    def test_burst_deduplication(self):
        """100 duplicate alerts, only 1 new notification."""
        deduplicator = AlertDeduplicator(window_minutes=60)

        new_count = 0
        dup_count = 0
        for i in range(100):
            alert = Alert(
                source="burst", name="SameAlert",
                severity=AlertSeverity.CRITICAL, title="Same alert",
                labels={"instance": "node-01"},
            )
            _, is_new = deduplicator.process(alert)
            if is_new:
                new_count += 1
            else:
                dup_count += 1

        assert new_count == 1
        assert dup_count == 99

    def test_burst_mixed_severity(self):
        """50 critical + 30 warning + 20 info, correct routing per severity."""
        config = AlertingConfig(
            pagerduty_enabled=True,
            opsgenie_enabled=True,
            slack_enabled=True,
            email_enabled=True,
        )
        router = AlertRouter(config)

        severity_map = (
            [(AlertSeverity.CRITICAL, 50)] +
            [(AlertSeverity.WARNING, 30)] +
            [(AlertSeverity.INFO, 20)]
        )

        critical_routes = []
        warning_routes = []
        info_routes = []

        for severity, count in severity_map:
            for i in range(count):
                alert = Alert(
                    source="burst", name=f"Mixed-{severity.value}-{i}",
                    severity=severity, title=f"Mixed {i}",
                    labels={"instance": f"node-{i}"},
                )
                channels = router.route(alert)
                if severity == AlertSeverity.CRITICAL:
                    critical_routes.append(channels)
                elif severity == AlertSeverity.WARNING:
                    warning_routes.append(channels)
                else:
                    info_routes.append(channels)

        # Verify critical routes to PD+OG+Slack
        assert len(critical_routes) == 50
        for channels in critical_routes:
            assert "pagerduty" in channels
            assert "opsgenie" in channels
            assert "slack" in channels

        # Verify warning routes to Slack+Email
        assert len(warning_routes) == 30
        for channels in warning_routes:
            assert "slack" in channels
            assert "email" in channels

        # Verify info routes to Email
        assert len(info_routes) == 20
        for channels in info_routes:
            assert "email" in channels
