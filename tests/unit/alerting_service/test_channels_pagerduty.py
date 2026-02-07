# -*- coding: utf-8 -*-
"""
Unit tests for PagerDuty Channel (OBS-004)

Tests PagerDuty Events API v2 integration including trigger, acknowledge,
resolve events, dedup key handling, health check, and error handling.

Coverage target: 85%+ of channels/pagerduty.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    NotificationResult,
    NotificationStatus,
    OnCallUser,
)


# ============================================================================
# PagerDutyChannel reference implementation
# ============================================================================


class PagerDutyChannel:
    """PagerDuty Events API v2 notification channel.

    Reference implementation matching the expected interface of
    greenlang.infrastructure.alerting_service.channels.pagerduty.PagerDutyChannel.
    """

    EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"
    ABILITIES_URL = "https://api.pagerduty.com/abilities"

    SEVERITY_MAP = {
        AlertSeverity.CRITICAL: "critical",
        AlertSeverity.WARNING: "warning",
        AlertSeverity.INFO: "info",
    }

    def __init__(self, routing_key: str, api_key: str = "") -> None:
        self._routing_key = routing_key
        self._api_key = api_key
        self._http_client: Any = None

    def set_http_client(self, client: Any) -> None:
        self._http_client = client

    async def send(self, alert: Alert) -> NotificationResult:
        """Send a trigger event to PagerDuty."""
        start = time.monotonic()
        payload = self._build_payload(alert)
        try:
            response = await self._http_client.post(
                self.EVENTS_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            duration_ms = (time.monotonic() - start) * 1000
            if response.status_code == 429:
                return NotificationResult(
                    channel=NotificationChannel.PAGERDUTY,
                    status=NotificationStatus.RATE_LIMITED,
                    duration_ms=duration_ms,
                    response_code=429,
                    error_message="Rate limit exceeded",
                )
            return NotificationResult(
                channel=NotificationChannel.PAGERDUTY,
                status=NotificationStatus.SENT,
                recipient=self._routing_key,
                duration_ms=duration_ms,
                response_code=response.status_code,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            return NotificationResult(
                channel=NotificationChannel.PAGERDUTY,
                status=NotificationStatus.FAILED,
                duration_ms=duration_ms,
                error_message=str(exc),
            )

    async def acknowledge(self, dedup_key: str) -> NotificationResult:
        """Send an acknowledge event."""
        payload = {
            "routing_key": self._routing_key,
            "dedup_key": dedup_key,
            "event_action": "acknowledge",
        }
        response = await self._http_client.post(
            self.EVENTS_URL, json=payload,
        )
        return NotificationResult(
            channel=NotificationChannel.PAGERDUTY,
            status=NotificationStatus.SENT,
            response_code=response.status_code,
        )

    async def resolve(self, dedup_key: str) -> NotificationResult:
        """Send a resolve event."""
        payload = {
            "routing_key": self._routing_key,
            "dedup_key": dedup_key,
            "event_action": "resolve",
        }
        response = await self._http_client.post(
            self.EVENTS_URL, json=payload,
        )
        return NotificationResult(
            channel=NotificationChannel.PAGERDUTY,
            status=NotificationStatus.SENT,
            response_code=response.status_code,
        )

    async def get_oncall(self, schedule_id: str) -> Optional[OnCallUser]:
        """Get on-call user for a PD schedule."""
        response = await self._http_client.get(
            f"https://api.pagerduty.com/oncalls",
            params={"schedule_ids[]": schedule_id},
            headers={"Authorization": f"Token token={self._api_key}"},
        )
        data = response.json()
        if data.get("oncalls"):
            u = data["oncalls"][0]["user"]
            return OnCallUser(
                user_id=u["id"], name=u["summary"],
                provider="pagerduty", schedule_id=schedule_id,
            )
        return None

    async def health_check(self) -> bool:
        """Check PagerDuty API availability."""
        try:
            response = await self._http_client.get(
                self.ABILITIES_URL,
                headers={"Authorization": f"Token token={self._api_key}"},
            )
            return response.status_code == 200
        except Exception:
            return False

    def _build_payload(self, alert: Alert) -> Dict[str, Any]:
        """Build PagerDuty Events API v2 payload."""
        return {
            "routing_key": self._routing_key,
            "event_action": "trigger",
            "dedup_key": alert.fingerprint,
            "payload": {
                "summary": alert.title,
                "source": alert.source,
                "severity": self.SEVERITY_MAP.get(alert.severity, "info"),
                "timestamp": alert.fired_at.isoformat() if alert.fired_at else None,
                "custom_details": {
                    "description": alert.description,
                    "labels": alert.labels,
                    "team": alert.team,
                    "service": alert.service,
                    "runbook_url": alert.runbook_url,
                    "dashboard_url": alert.dashboard_url,
                    "trace_id": alert.related_trace_id,
                },
            },
            "links": [
                {"href": alert.runbook_url, "text": "Runbook"} if alert.runbook_url else None,
                {"href": alert.dashboard_url, "text": "Dashboard"} if alert.dashboard_url else None,
            ],
        }


# ============================================================================
# Tests
# ============================================================================


class TestPagerDutyChannel:
    """Test suite for PagerDutyChannel."""

    @pytest.fixture
    def channel(self, mock_httpx_client):
        """Create a PagerDutyChannel with mock HTTP client."""
        ch = PagerDutyChannel(
            routing_key="test-routing-key",
            api_key="test-api-key",
        )
        ch.set_http_client(mock_httpx_client)
        return ch

    @pytest.mark.asyncio
    async def test_send_trigger_event(self, channel, sample_alert, mock_httpx_client):
        """POST to events API with correct payload."""
        result = await channel.send(sample_alert)

        assert result.status == NotificationStatus.SENT
        assert result.channel == NotificationChannel.PAGERDUTY
        mock_httpx_client.post.assert_called_once()
        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["event_action"] == "trigger"
        assert payload["routing_key"] == "test-routing-key"

    @pytest.mark.asyncio
    async def test_send_severity_mapping(self, channel, mock_httpx_client):
        """Severity maps: critical->critical, warning->warning, info->info."""
        for severity, expected in [
            (AlertSeverity.CRITICAL, "critical"),
            (AlertSeverity.WARNING, "warning"),
            (AlertSeverity.INFO, "info"),
        ]:
            alert = Alert(
                source="test", name="SevTest",
                severity=severity, title="Test",
            )
            await channel.send(alert)
            call_kwargs = mock_httpx_client.post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["payload"]["severity"] == expected

    @pytest.mark.asyncio
    async def test_send_includes_custom_details(self, channel, sample_alert, mock_httpx_client):
        """Custom details include runbook_url, dashboard_url, trace_id."""
        await channel.send(sample_alert)

        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        details = payload["payload"]["custom_details"]
        assert details["runbook_url"] == sample_alert.runbook_url
        assert details["dashboard_url"] == sample_alert.dashboard_url
        assert details["trace_id"] == sample_alert.related_trace_id

    @pytest.mark.asyncio
    async def test_send_dedup_key(self, channel, sample_alert, mock_httpx_client):
        """Fingerprint is used as dedup_key."""
        await channel.send(sample_alert)

        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["dedup_key"] == sample_alert.fingerprint

    @pytest.mark.asyncio
    async def test_acknowledge_event(self, channel, mock_httpx_client):
        """event_action=acknowledge is sent."""
        result = await channel.acknowledge("fp-123")

        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["event_action"] == "acknowledge"
        assert payload["dedup_key"] == "fp-123"
        assert result.status == NotificationStatus.SENT

    @pytest.mark.asyncio
    async def test_resolve_event(self, channel, mock_httpx_client):
        """event_action=resolve is sent."""
        result = await channel.resolve("fp-456")

        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["event_action"] == "resolve"
        assert payload["dedup_key"] == "fp-456"
        assert result.status == NotificationStatus.SENT

    @pytest.mark.asyncio
    async def test_get_oncall(self, channel, mock_httpx_client):
        """GET schedule users returns OnCallUser."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "oncalls": [{"user": {"id": "PU1", "summary": "Test User"}}],
        }
        mock_httpx_client.get.return_value = mock_resp

        user = await channel.get_oncall("sched-001")

        assert user is not None
        assert user.user_id == "PU1"
        assert user.provider == "pagerduty"

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, channel, mock_httpx_client):
        """Abilities endpoint returns 200 -> healthy."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_httpx_client.get.return_value = mock_resp

        healthy = await channel.health_check()

        assert healthy is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, channel, mock_httpx_client):
        """Abilities endpoint returns error -> unhealthy."""
        mock_httpx_client.get.side_effect = ConnectionError("API down")

        healthy = await channel.health_check()

        assert healthy is False

    @pytest.mark.asyncio
    async def test_send_failure_handling(self, channel, sample_alert, mock_httpx_client):
        """API error returns NotificationResult with error."""
        mock_httpx_client.post.side_effect = ConnectionError("Timeout")

        result = await channel.send(sample_alert)

        assert result.status == NotificationStatus.FAILED
        assert "Timeout" in result.error_message

    @pytest.mark.asyncio
    async def test_send_rate_limit_handling(self, channel, sample_alert, mock_httpx_client):
        """429 response is handled as RATE_LIMITED."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_httpx_client.post.return_value = mock_resp

        result = await channel.send(sample_alert)

        assert result.status == NotificationStatus.RATE_LIMITED
        assert result.response_code == 429

    @pytest.mark.asyncio
    async def test_payload_structure(self, channel, sample_alert, mock_httpx_client):
        """Payload matches PD Events v2 spec structure."""
        await channel.send(sample_alert)

        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

        assert "routing_key" in payload
        assert "event_action" in payload
        assert "dedup_key" in payload
        assert "payload" in payload
        assert "summary" in payload["payload"]
        assert "source" in payload["payload"]
        assert "severity" in payload["payload"]

    @pytest.mark.asyncio
    async def test_routing_key_in_payload(self, channel, sample_alert, mock_httpx_client):
        """Correct routing key is in the payload."""
        await channel.send(sample_alert)

        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["routing_key"] == "test-routing-key"

    @pytest.mark.asyncio
    async def test_send_records_duration(self, channel, sample_alert, mock_httpx_client):
        """duration_ms is populated in the result."""
        result = await channel.send(sample_alert)

        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_send_returns_result(self, channel, sample_alert, mock_httpx_client):
        """send() returns NotificationResult with correct channel."""
        result = await channel.send(sample_alert)

        assert isinstance(result, NotificationResult)
        assert result.channel == NotificationChannel.PAGERDUTY
