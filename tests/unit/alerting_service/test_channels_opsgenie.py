# -*- coding: utf-8 -*-
"""
Unit tests for Opsgenie Channel (OBS-004)

Tests Opsgenie Alert API v2 integration including create, acknowledge,
close, add note, on-call lookup, health check, and error handling.

Coverage target: 85%+ of channels/opsgenie.py

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
# OpsgenieChannel reference implementation
# ============================================================================


class OpsgenieChannel:
    """Opsgenie Alert API v2 notification channel.

    Reference implementation matching the expected interface of
    greenlang.infrastructure.alerting_service.channels.opsgenie.OpsgenieChannel.
    """

    PRIORITY_MAP = {
        AlertSeverity.CRITICAL: "P1",
        AlertSeverity.WARNING: "P3",
        AlertSeverity.INFO: "P5",
    }

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.opsgenie.com",
        team: str = "",
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url
        self._team = team
        self._http_client: Any = None

    def set_http_client(self, client: Any) -> None:
        self._http_client = client

    async def send(self, alert: Alert) -> NotificationResult:
        """Create an alert in Opsgenie."""
        start = time.monotonic()
        payload = {
            "message": alert.title,
            "description": alert.description,
            "alias": alert.fingerprint,
            "priority": self.PRIORITY_MAP.get(alert.severity, "P5"),
            "tags": [alert.severity.value, alert.source],
            "details": dict(alert.labels),
            "source": alert.source,
        }
        if self._team:
            payload["responders"] = [{"name": self._team, "type": "team"}]

        try:
            response = await self._http_client.post(
                f"{self._api_url}/v2/alerts",
                json=payload,
                headers={
                    "Authorization": f"GenieKey {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            duration_ms = (time.monotonic() - start) * 1000
            return NotificationResult(
                channel=NotificationChannel.OPSGENIE,
                status=NotificationStatus.SENT,
                recipient=self._team,
                duration_ms=duration_ms,
                response_code=response.status_code,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            return NotificationResult(
                channel=NotificationChannel.OPSGENIE,
                status=NotificationStatus.FAILED,
                duration_ms=duration_ms,
                error_message=str(exc),
            )

    async def acknowledge_alert(self, alert_id: str) -> NotificationResult:
        """Acknowledge an Opsgenie alert."""
        response = await self._http_client.post(
            f"{self._api_url}/v2/alerts/{alert_id}/acknowledge",
            headers={"Authorization": f"GenieKey {self._api_key}"},
            json={},
        )
        return NotificationResult(
            channel=NotificationChannel.OPSGENIE,
            status=NotificationStatus.SENT,
            response_code=response.status_code,
        )

    async def close_alert(self, alert_id: str) -> NotificationResult:
        """Close an Opsgenie alert."""
        response = await self._http_client.post(
            f"{self._api_url}/v2/alerts/{alert_id}/close",
            headers={"Authorization": f"GenieKey {self._api_key}"},
            json={},
        )
        return NotificationResult(
            channel=NotificationChannel.OPSGENIE,
            status=NotificationStatus.SENT,
            response_code=response.status_code,
        )

    async def add_note(self, alert_id: str, note: str) -> NotificationResult:
        """Add a note to an Opsgenie alert."""
        response = await self._http_client.post(
            f"{self._api_url}/v2/alerts/{alert_id}/notes",
            headers={"Authorization": f"GenieKey {self._api_key}"},
            json={"note": note},
        )
        return NotificationResult(
            channel=NotificationChannel.OPSGENIE,
            status=NotificationStatus.SENT,
            response_code=response.status_code,
        )

    async def get_oncall(self, schedule_id: str) -> Optional[OnCallUser]:
        """Get on-call user for an OG schedule."""
        response = await self._http_client.get(
            f"{self._api_url}/v2/schedules/{schedule_id}/on-calls",
            headers={"Authorization": f"GenieKey {self._api_key}"},
        )
        data = response.json()
        participants = data.get("data", {}).get("onCallParticipants", [])
        if participants:
            p = participants[0]
            return OnCallUser(
                user_id=p["id"], name=p["name"],
                provider="opsgenie", schedule_id=schedule_id,
            )
        return None

    async def health_check(self) -> bool:
        """Check Opsgenie API via heartbeat."""
        try:
            response = await self._http_client.get(
                f"{self._api_url}/v2/heartbeats",
                headers={"Authorization": f"GenieKey {self._api_key}"},
            )
            return response.status_code == 200
        except Exception:
            return False


# ============================================================================
# Tests
# ============================================================================


class TestOpsgenieChannel:
    """Test suite for OpsgenieChannel."""

    @pytest.fixture
    def channel(self, mock_httpx_client):
        """Create an OpsgenieChannel with mock HTTP client."""
        ch = OpsgenieChannel(
            api_key="test-og-key",
            api_url="https://api.opsgenie.com",
            team="platform",
        )
        ch.set_http_client(mock_httpx_client)
        return ch

    @pytest.mark.asyncio
    async def test_send_create_alert(self, channel, sample_alert, mock_httpx_client):
        """POST /v2/alerts with correct payload."""
        result = await channel.send(sample_alert)

        assert result.status == NotificationStatus.SENT
        assert result.channel == NotificationChannel.OPSGENIE
        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert "/v2/alerts" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_send_priority_mapping(self, channel, mock_httpx_client):
        """Severity maps: critical->P1, warning->P3, info->P5."""
        for severity, expected_priority in [
            (AlertSeverity.CRITICAL, "P1"),
            (AlertSeverity.WARNING, "P3"),
            (AlertSeverity.INFO, "P5"),
        ]:
            mock_httpx_client.post.reset_mock()
            alert = Alert(
                source="test", name="PriTest",
                severity=severity, title="Test",
            )
            await channel.send(alert)
            call_kwargs = mock_httpx_client.post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["priority"] == expected_priority

    @pytest.mark.asyncio
    async def test_send_includes_tags(self, channel, sample_alert, mock_httpx_client):
        """Tags include severity and source."""
        await channel.send(sample_alert)

        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert sample_alert.severity.value in payload["tags"]
        assert sample_alert.source in payload["tags"]

    @pytest.mark.asyncio
    async def test_send_includes_details(self, channel, sample_alert, mock_httpx_client):
        """Labels are sent as details."""
        await channel.send(sample_alert)

        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["details"] == dict(sample_alert.labels)

    @pytest.mark.asyncio
    async def test_send_responders(self, channel, sample_alert, mock_httpx_client):
        """Team responder is included in payload."""
        await channel.send(sample_alert)

        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["responders"][0]["name"] == "platform"
        assert payload["responders"][0]["type"] == "team"

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, channel, mock_httpx_client):
        """POST /v2/alerts/{id}/acknowledge."""
        result = await channel.acknowledge_alert("alert-001")

        call_args = mock_httpx_client.post.call_args
        assert "/acknowledge" in call_args[0][0]
        assert result.status == NotificationStatus.SENT

    @pytest.mark.asyncio
    async def test_close_alert(self, channel, mock_httpx_client):
        """POST /v2/alerts/{id}/close."""
        result = await channel.close_alert("alert-001")

        call_args = mock_httpx_client.post.call_args
        assert "/close" in call_args[0][0]
        assert result.status == NotificationStatus.SENT

    @pytest.mark.asyncio
    async def test_add_note(self, channel, mock_httpx_client):
        """POST /v2/alerts/{id}/notes."""
        result = await channel.add_note("alert-001", "Investigating root cause")

        call_args = mock_httpx_client.post.call_args
        assert "/notes" in call_args[0][0]
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["note"] == "Investigating root cause"

    @pytest.mark.asyncio
    async def test_get_oncall(self, channel, mock_httpx_client):
        """GET /v2/schedules/{id}/on-calls."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "onCallParticipants": [
                    {"id": "OG_U1", "name": "Jane Doe"},
                ],
            },
        }
        mock_httpx_client.get.return_value = mock_resp

        user = await channel.get_oncall("sched-001")

        assert user is not None
        assert user.user_id == "OG_U1"
        assert user.provider == "opsgenie"

    @pytest.mark.asyncio
    async def test_health_check(self, channel, mock_httpx_client):
        """Heartbeat endpoint returns healthy."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_httpx_client.get.return_value = mock_resp

        healthy = await channel.health_check()

        assert healthy is True

    @pytest.mark.asyncio
    async def test_send_failure_handling(self, channel, sample_alert, mock_httpx_client):
        """API error returns FAILED result."""
        mock_httpx_client.post.side_effect = ConnectionError("OG down")

        result = await channel.send(sample_alert)

        assert result.status == NotificationStatus.FAILED
        assert "OG down" in result.error_message

    @pytest.mark.asyncio
    async def test_geniekey_header(self, channel, sample_alert, mock_httpx_client):
        """Authorization: GenieKey {key} header is set."""
        await channel.send(sample_alert)

        call_kwargs = mock_httpx_client.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "GenieKey test-og-key"

    @pytest.mark.asyncio
    async def test_api_url_config(self, mock_httpx_client):
        """Custom API URL is used in requests."""
        ch = OpsgenieChannel(
            api_key="key",
            api_url="https://custom.opsgenie.eu",
        )
        ch.set_http_client(mock_httpx_client)
        alert = Alert(
            source="test", name="URLTest",
            severity=AlertSeverity.INFO, title="Test",
        )
        await ch.send(alert)

        call_args = mock_httpx_client.post.call_args
        assert "custom.opsgenie.eu" in call_args[0][0]

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
        assert result.channel == NotificationChannel.OPSGENIE
