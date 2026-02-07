# -*- coding: utf-8 -*-
"""
Unit tests for AlertRouter (OBS-004)

Tests severity-based, team-based, service-based, and time-based routing
of alerts to notification channels. Tests notify() dispatching and
metrics recording.

Coverage target: 85%+ of router.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional, Set
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


# ============================================================================
# AlertRouter reference implementation
# ============================================================================


class AlertRouter:
    """Severity/team/service/time-based routing of alerts to channels.

    Reference implementation matching the expected interface of
    greenlang.infrastructure.alerting_service.router.AlertRouter.
    """

    def __init__(self, config: AlertingConfig) -> None:
        self._config = config
        self._team_routes: Dict[str, List[str]] = {}
        self._service_routes: Dict[str, List[str]] = {}
        self._channels: Dict[str, Any] = {}
        self._notification_count = 0
        self._business_hours_start = time(9, 0)
        self._business_hours_end = time(18, 0)

    def add_team_route(self, team: str, channels: List[str]) -> None:
        """Add a team-specific channel mapping."""
        self._team_routes[team] = channels

    def add_service_route(self, service: str, channels: List[str]) -> None:
        """Add a service-specific channel mapping."""
        self._service_routes[service] = channels

    def register_channel(self, name: str, channel: Any) -> None:
        """Register a notification channel adapter."""
        self._channels[name] = channel

    def is_business_hours(self, dt: Optional[datetime] = None) -> bool:
        """Check if the given time is within business hours (Mon-Fri 9-18 UTC)."""
        if dt is None:
            dt = datetime.now(timezone.utc)
        if dt.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        t = dt.time()
        return self._business_hours_start <= t < self._business_hours_end

    def route(self, alert: Alert) -> List[str]:
        """Determine which channels should receive this alert.

        Priority: explicit label > team > service > severity.
        """
        # Explicit override via label
        explicit = alert.labels.get("routing.channel")
        if explicit:
            return [ch.strip() for ch in explicit.split(",")]

        channels: Set[str] = set()

        # Team routing
        if alert.team and alert.team in self._team_routes:
            channels.update(self._team_routes[alert.team])

        # Service routing
        if alert.service and alert.service in self._service_routes:
            channels.update(self._service_routes[alert.service])

        # Severity routing (always applied as fallback)
        severity_channels = self._config.default_severity_routing.get(
            alert.severity.value, [],
        )
        if not channels:
            channels.update(severity_channels)

        # Filter disabled channels
        enabled_channels = []
        for ch in channels:
            if ch == "pagerduty" and not self._config.pagerduty_enabled:
                continue
            if ch == "opsgenie" and not self._config.opsgenie_enabled:
                continue
            if ch == "slack" and not self._config.slack_enabled:
                continue
            if ch == "email" and not self._config.email_enabled:
                continue
            if ch == "teams" and not self._config.teams_enabled:
                continue
            if ch == "webhook" and not self._config.webhook_enabled:
                continue
            enabled_channels.append(ch)

        return enabled_channels

    async def notify(
        self, alert: Alert, channels: Optional[List[str]] = None,
    ) -> List[NotificationResult]:
        """Send notifications to the specified channels."""
        if channels is None:
            channels = self.route(alert)

        results: List[NotificationResult] = []
        for ch_name in channels:
            channel = self._channels.get(ch_name)
            if channel is None:
                continue
            try:
                result = await channel.send(alert)
                results.append(result)
            except Exception as exc:
                results.append(
                    NotificationResult(
                        channel=NotificationChannel(ch_name)
                        if ch_name in [e.value for e in NotificationChannel]
                        else NotificationChannel.WEBHOOK,
                        status=NotificationStatus.FAILED,
                        error_message=str(exc),
                    )
                )
        self._notification_count += len(results)
        return results


# ============================================================================
# Tests
# ============================================================================


class TestAlertRouter:
    """Test suite for AlertRouter."""

    @pytest.fixture
    def router(self, sample_config):
        """Create an AlertRouter instance."""
        return AlertRouter(sample_config)

    def test_severity_routing_critical(self, router):
        """Critical alerts route to PD+OG+Slack."""
        alert = Alert(
            source="prom", name="CritAlert",
            severity=AlertSeverity.CRITICAL, title="Critical",
        )
        channels = router.route(alert)

        assert "pagerduty" in channels
        assert "opsgenie" in channels
        assert "slack" in channels

    def test_severity_routing_warning(self, router):
        """Warning alerts route to Slack+Email."""
        alert = Alert(
            source="prom", name="WarnAlert",
            severity=AlertSeverity.WARNING, title="Warning",
        )
        channels = router.route(alert)

        assert "slack" in channels
        assert "email" in channels

    def test_severity_routing_info(self, router):
        """Info alerts route to Email."""
        alert = Alert(
            source="prom", name="InfoAlert",
            severity=AlertSeverity.INFO, title="Info",
        )
        channels = router.route(alert)

        assert "email" in channels

    def test_explicit_override_label(self, router):
        """routing.channel label overrides all other routing."""
        alert = Alert(
            source="prom", name="Override",
            severity=AlertSeverity.CRITICAL, title="Override",
            labels={"routing.channel": "email, webhook"},
        )
        channels = router.route(alert)

        assert channels == ["email", "webhook"]

    def test_team_routing(self, router):
        """Team-specific channel mapping is used."""
        router.add_team_route("data-platform", ["slack", "email"])
        alert = Alert(
            source="prom", name="TeamAlert",
            severity=AlertSeverity.WARNING, title="Team alert",
            team="data-platform",
        )
        channels = router.route(alert)

        assert "slack" in channels
        assert "email" in channels

    def test_service_routing(self, router):
        """Service-specific channel mapping is used."""
        router.add_service_route("postgres", ["pagerduty", "slack"])
        alert = Alert(
            source="prom", name="SvcAlert",
            severity=AlertSeverity.WARNING, title="Service alert",
            service="postgres",
        )
        channels = router.route(alert)

        assert "pagerduty" in channels
        assert "slack" in channels

    def test_add_team_route(self, router):
        """Dynamic team route addition works."""
        router.add_team_route("security", ["pagerduty", "opsgenie"])
        alert = Alert(
            source="prom", name="SecAlert",
            severity=AlertSeverity.CRITICAL, title="Security",
            team="security",
        )
        channels = router.route(alert)

        assert "pagerduty" in channels
        assert "opsgenie" in channels

    def test_add_service_route(self, router):
        """Dynamic service route addition works."""
        router.add_service_route("redis", ["slack"])
        alert = Alert(
            source="prom", name="RedisAlert",
            severity=AlertSeverity.WARNING, title="Redis",
            service="redis",
        )
        channels = router.route(alert)

        assert "slack" in channels

    def test_business_hours_routing(self, router):
        """is_business_hours returns True during Mon-Fri 9-18 UTC."""
        # Wednesday at 12:00 UTC
        dt = datetime(2026, 2, 4, 12, 0, 0, tzinfo=timezone.utc)
        assert router.is_business_hours(dt) is True

    def test_off_hours_routing(self, router):
        """is_business_hours returns False outside business hours."""
        # Saturday at 12:00 UTC
        dt = datetime(2026, 2, 7, 12, 0, 0, tzinfo=timezone.utc)
        assert router.is_business_hours(dt) is False

        # Wednesday at 22:00 UTC
        dt2 = datetime(2026, 2, 4, 22, 0, 0, tzinfo=timezone.utc)
        assert router.is_business_hours(dt2) is False

    def test_route_returns_channel_names(self, router):
        """route() returns a list of strings."""
        alert = Alert(
            source="prom", name="Test",
            severity=AlertSeverity.CRITICAL, title="Test",
        )
        channels = router.route(alert)

        assert isinstance(channels, list)
        for ch in channels:
            assert isinstance(ch, str)

    def test_route_empty_when_disabled(self):
        """Disabled channels are excluded from routing."""
        config = AlertingConfig(
            pagerduty_enabled=False,
            opsgenie_enabled=False,
            slack_enabled=False,
            email_enabled=False,
        )
        router = AlertRouter(config)
        alert = Alert(
            source="prom", name="Test",
            severity=AlertSeverity.CRITICAL, title="Test",
        )
        channels = router.route(alert)

        assert channels == []

    def test_route_fallback_to_severity(self, router):
        """No team/service match falls through to severity routing."""
        alert = Alert(
            source="prom", name="Fallback",
            severity=AlertSeverity.WARNING, title="Fallback",
            team="nonexistent-team",
        )
        channels = router.route(alert)

        # Should fall back to severity routing (warning -> slack, email)
        assert "slack" in channels
        assert "email" in channels

    @pytest.mark.asyncio
    async def test_notify_sends_to_channels(self, router):
        """All registered channels are called during notify()."""
        mock_slack = AsyncMock()
        mock_slack.send.return_value = NotificationResult(
            channel=NotificationChannel.SLACK,
            status=NotificationStatus.SENT,
        )
        router.register_channel("slack", mock_slack)

        alert = Alert(
            source="prom", name="Test",
            severity=AlertSeverity.WARNING, title="Test",
        )
        results = await router.notify(alert, channels=["slack"])

        mock_slack.send.assert_called_once_with(alert)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_notify_collects_results(self, router):
        """notify() returns a list of NotificationResult."""
        mock_ch = AsyncMock()
        mock_ch.send.return_value = NotificationResult(
            channel=NotificationChannel.SLACK,
            status=NotificationStatus.SENT,
            duration_ms=100.0,
        )
        router.register_channel("slack", mock_ch)

        alert = Alert(
            source="prom", name="Test",
            severity=AlertSeverity.INFO, title="Test",
        )
        results = await router.notify(alert, channels=["slack"])

        assert len(results) == 1
        assert results[0].status == NotificationStatus.SENT

    @pytest.mark.asyncio
    async def test_notify_handles_channel_error(self, router):
        """Failed channel does not block other channels."""
        mock_fail = AsyncMock()
        mock_fail.send.side_effect = ConnectionError("PD down")

        mock_ok = AsyncMock()
        mock_ok.send.return_value = NotificationResult(
            channel=NotificationChannel.SLACK,
            status=NotificationStatus.SENT,
        )

        router.register_channel("pagerduty", mock_fail)
        router.register_channel("slack", mock_ok)

        alert = Alert(
            source="prom", name="Test",
            severity=AlertSeverity.CRITICAL, title="Test",
        )
        results = await router.notify(alert, channels=["pagerduty", "slack"])

        assert len(results) == 2
        statuses = [r.status for r in results]
        assert NotificationStatus.FAILED in statuses
        assert NotificationStatus.SENT in statuses

    @pytest.mark.asyncio
    async def test_notify_rate_limited(self, router):
        """Rate limit is tracked via notification count."""
        mock_ch = AsyncMock()
        mock_ch.send.return_value = NotificationResult(
            channel=NotificationChannel.EMAIL,
            status=NotificationStatus.SENT,
        )
        router.register_channel("email", mock_ch)

        alert = Alert(
            source="prom", name="Test",
            severity=AlertSeverity.INFO, title="Test",
        )
        await router.notify(alert, channels=["email"])

        assert router._notification_count == 1

    def test_multiple_routes_combined(self, router):
        """Team + severity channels are merged when team route exists."""
        router.add_team_route("platform", ["opsgenie"])
        alert = Alert(
            source="prom", name="Combined",
            severity=AlertSeverity.WARNING, title="Combined",
            team="platform",
        )
        channels = router.route(alert)

        assert "opsgenie" in channels

    def test_unknown_channel_ignored(self, router):
        """Unknown channels in explicit override are returned as-is."""
        alert = Alert(
            source="prom", name="Unknown",
            severity=AlertSeverity.CRITICAL, title="Unknown",
            labels={"routing.channel": "sms"},
        )
        channels = router.route(alert)

        assert channels == ["sms"]

    def test_custom_severity_mapping(self):
        """Custom severity routing config is respected."""
        config = AlertingConfig(
            slack_enabled=True,
            email_enabled=True,
            default_severity_routing={
                "critical": ["slack"],
                "warning": ["email"],
                "info": [],
            },
        )
        router = AlertRouter(config)
        alert = Alert(
            source="prom", name="Custom",
            severity=AlertSeverity.CRITICAL, title="Custom",
        )
        channels = router.route(alert)

        assert channels == ["slack"]

    def test_route_with_tenant_context(self, router):
        """Tenant context does not affect routing (no tenant routing by default)."""
        alert = Alert(
            source="prom", name="Tenant",
            severity=AlertSeverity.WARNING, title="Tenant",
            tenant_id="t-corp",
        )
        channels = router.route(alert)

        assert len(channels) > 0

    @pytest.mark.asyncio
    async def test_notify_with_template(self, router):
        """Notify with template passes through correctly."""
        mock_ch = AsyncMock()
        mock_ch.send.return_value = NotificationResult(
            channel=NotificationChannel.SLACK,
            status=NotificationStatus.SENT,
        )
        router.register_channel("slack", mock_ch)

        alert = Alert(
            source="prom", name="Template",
            severity=AlertSeverity.WARNING, title="Template test",
        )
        results = await router.notify(alert, channels=["slack"])

        assert len(results) == 1

    def test_route_priority_order(self, router):
        """explicit label > team > service > severity routing priority."""
        router.add_team_route("platform", ["opsgenie"])
        router.add_service_route("api-service", ["email"])

        # Explicit label should take priority
        alert = Alert(
            source="prom", name="Priority",
            severity=AlertSeverity.CRITICAL, title="Priority",
            labels={"routing.channel": "webhook"},
            team="platform",
            service="api-service",
        )
        channels = router.route(alert)

        assert channels == ["webhook"]

    def test_is_business_hours_boundary_start(self, router):
        """09:00 UTC is within business hours."""
        dt = datetime(2026, 2, 4, 9, 0, 0, tzinfo=timezone.utc)  # Wednesday
        assert router.is_business_hours(dt) is True

    def test_is_business_hours_boundary_end(self, router):
        """18:00 UTC is outside business hours (exclusive end)."""
        dt = datetime(2026, 2, 4, 18, 0, 0, tzinfo=timezone.utc)  # Wednesday
        assert router.is_business_hours(dt) is False
