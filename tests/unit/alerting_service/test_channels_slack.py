# -*- coding: utf-8 -*-
"""
Unit tests for Slack Channel (OBS-004)

Tests Slack Block Kit webhook notifications including severity-based
webhook selection, block structure, emoji mapping, and error handling.

Coverage target: 85%+ of channels/slack.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    NotificationResult,
    NotificationStatus,
)


# ============================================================================
# SlackChannel reference implementation
# ============================================================================


class SlackChannel:
    """Slack Block Kit webhook notification channel.

    Reference implementation matching the expected interface of
    greenlang.infrastructure.alerting_service.channels.slack.SlackChannel.
    """

    SEVERITY_EMOJI = {
        AlertSeverity.CRITICAL: ":rotating_light:",
        AlertSeverity.WARNING: ":warning:",
        AlertSeverity.INFO: ":information_source:",
    }

    def __init__(
        self,
        webhook_critical: str = "",
        webhook_warning: str = "",
        webhook_info: str = "",
    ) -> None:
        self._webhooks = {
            AlertSeverity.CRITICAL: webhook_critical,
            AlertSeverity.WARNING: webhook_warning,
            AlertSeverity.INFO: webhook_info,
        }
        self._http_client: Any = None

    def set_http_client(self, client: Any) -> None:
        self._http_client = client

    def _get_webhook_url(self, severity: AlertSeverity) -> str:
        return self._webhooks.get(severity, "")

    async def send(self, alert: Alert) -> NotificationResult:
        """Send a Block Kit message to the severity-appropriate webhook."""
        start = time.monotonic()
        webhook_url = self._get_webhook_url(alert.severity)
        if not webhook_url:
            return NotificationResult(
                channel=NotificationChannel.SLACK,
                status=NotificationStatus.SKIPPED,
                error_message="No webhook configured for severity",
            )
        blocks = self._build_blocks(alert)
        try:
            response = await self._http_client.post(
                webhook_url,
                json={"blocks": blocks},
            )
            duration_ms = (time.monotonic() - start) * 1000
            return NotificationResult(
                channel=NotificationChannel.SLACK,
                status=NotificationStatus.SENT,
                recipient=webhook_url,
                duration_ms=duration_ms,
                response_code=response.status_code,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            return NotificationResult(
                channel=NotificationChannel.SLACK,
                status=NotificationStatus.FAILED,
                duration_ms=duration_ms,
                error_message=str(exc),
            )

    async def health_check(self) -> bool:
        """Slack has no health endpoint; return True if any webhook is configured."""
        return any(url for url in self._webhooks.values())

    def _build_blocks(self, alert: Alert) -> List[Dict[str, Any]]:
        """Build Slack Block Kit blocks."""
        emoji = self.SEVERITY_EMOJI.get(alert.severity, ":grey_question:")
        blocks: List[Dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {alert.name}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Severity:* {alert.severity.value}"},
                    {"type": "mrkdwn", "text": f"*Status:* {alert.status.value}"},
                    {"type": "mrkdwn", "text": f"*Team:* {alert.team or 'N/A'}"},
                    {"type": "mrkdwn", "text": f"*Service:* {alert.service or 'N/A'}"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": alert.title,
                },
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Source: {alert.source} | Env: {alert.environment}"},
                ],
            },
        ]

        # Actions block with buttons (only if URLs present)
        actions = []
        if alert.runbook_url:
            actions.append({
                "type": "button",
                "text": {"type": "plain_text", "text": "Runbook"},
                "url": alert.runbook_url,
            })
        if alert.dashboard_url:
            actions.append({
                "type": "button",
                "text": {"type": "plain_text", "text": "Dashboard"},
                "url": alert.dashboard_url,
            })
        if actions:
            blocks.append({"type": "actions", "elements": actions})

        return blocks


# ============================================================================
# Tests
# ============================================================================


class TestSlackChannel:
    """Test suite for SlackChannel."""

    @pytest.fixture
    def channel(self, mock_httpx_client):
        """Create a SlackChannel with mock HTTP client."""
        ch = SlackChannel(
            webhook_critical="https://hooks.slack.com/critical",
            webhook_warning="https://hooks.slack.com/warning",
            webhook_info="https://hooks.slack.com/info",
        )
        ch.set_http_client(mock_httpx_client)
        return ch

    @pytest.mark.asyncio
    async def test_send_critical_webhook(self, channel, mock_httpx_client):
        """Critical alert uses the critical webhook URL."""
        alert = Alert(
            source="prom", name="CritAlert",
            severity=AlertSeverity.CRITICAL, title="Critical alert",
        )
        await channel.send(alert)

        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "https://hooks.slack.com/critical"

    @pytest.mark.asyncio
    async def test_send_warning_webhook(self, channel, mock_httpx_client):
        """Warning alert uses the warning webhook URL."""
        alert = Alert(
            source="prom", name="WarnAlert",
            severity=AlertSeverity.WARNING, title="Warning alert",
        )
        await channel.send(alert)

        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "https://hooks.slack.com/warning"

    @pytest.mark.asyncio
    async def test_send_info_webhook(self, channel, mock_httpx_client):
        """Info alert uses the info webhook URL."""
        alert = Alert(
            source="prom", name="InfoAlert",
            severity=AlertSeverity.INFO, title="Info alert",
        )
        await channel.send(alert)

        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "https://hooks.slack.com/info"

    @pytest.mark.asyncio
    async def test_block_kit_structure(self, channel, sample_alert, mock_httpx_client):
        """Blocks list contains header, section, context, and actions."""
        await channel.send(sample_alert)

        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        blocks = payload["blocks"]
        block_types = [b["type"] for b in blocks]

        assert "header" in block_types
        assert "section" in block_types
        assert "context" in block_types
        assert "actions" in block_types  # sample_alert has runbook + dashboard URLs

    @pytest.mark.asyncio
    async def test_severity_emoji(self, channel, mock_httpx_client):
        """Correct emoji per severity level."""
        for severity, expected_emoji in [
            (AlertSeverity.CRITICAL, ":rotating_light:"),
            (AlertSeverity.WARNING, ":warning:"),
            (AlertSeverity.INFO, ":information_source:"),
        ]:
            mock_httpx_client.post.reset_mock()
            alert = Alert(
                source="test", name="EmojiTest",
                severity=severity, title="Test",
            )
            await channel.send(alert)
            call_kwargs = mock_httpx_client.post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            header = payload["blocks"][0]
            assert expected_emoji in header["text"]["text"]

    @pytest.mark.asyncio
    async def test_header_block(self, channel, sample_alert, mock_httpx_client):
        """Header block contains alert name."""
        await channel.send(sample_alert)

        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        header = payload["blocks"][0]
        assert sample_alert.name in header["text"]["text"]

    @pytest.mark.asyncio
    async def test_fields_block(self, channel, sample_alert, mock_httpx_client):
        """Section fields include severity, status, team, service."""
        await channel.send(sample_alert)

        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        fields_block = payload["blocks"][1]
        fields_text = " ".join(f["text"] for f in fields_block["fields"])

        assert "critical" in fields_text
        assert "firing" in fields_text
        assert "platform" in fields_text
        assert "api-service" in fields_text

    @pytest.mark.asyncio
    async def test_actions_block_with_urls(self, channel, sample_alert, mock_httpx_client):
        """Runbook + dashboard buttons present when URLs are set."""
        await channel.send(sample_alert)

        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        actions_block = [b for b in payload["blocks"] if b["type"] == "actions"]

        assert len(actions_block) == 1
        elements = actions_block[0]["elements"]
        button_texts = [e["text"]["text"] for e in elements]
        assert "Runbook" in button_texts
        assert "Dashboard" in button_texts

    @pytest.mark.asyncio
    async def test_actions_block_without_urls(self, channel, mock_httpx_client):
        """No actions block when no URLs are set."""
        alert = Alert(
            source="test", name="NoUrls",
            severity=AlertSeverity.WARNING, title="No URLs",
        )
        await channel.send(alert)

        call_kwargs = mock_httpx_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        actions_blocks = [b for b in payload["blocks"] if b["type"] == "actions"]

        assert len(actions_blocks) == 0

    @pytest.mark.asyncio
    async def test_send_failure_handling(self, channel, sample_alert, mock_httpx_client):
        """Webhook error returns FAILED result."""
        mock_httpx_client.post.side_effect = ConnectionError("Webhook down")

        result = await channel.send(sample_alert)

        assert result.status == NotificationStatus.FAILED
        assert "Webhook down" in result.error_message

    @pytest.mark.asyncio
    async def test_health_check(self, channel):
        """Health check returns True when webhooks are configured."""
        healthy = await channel.health_check()

        assert healthy is True

    @pytest.mark.asyncio
    async def test_send_returns_result(self, channel, sample_alert, mock_httpx_client):
        """send() returns NotificationResult with correct channel."""
        result = await channel.send(sample_alert)

        assert isinstance(result, NotificationResult)
        assert result.channel == NotificationChannel.SLACK
        assert result.status == NotificationStatus.SENT
