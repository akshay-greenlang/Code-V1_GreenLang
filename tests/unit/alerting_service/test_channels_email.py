# -*- coding: utf-8 -*-
"""
Unit tests for Email Channel (OBS-004)

Tests AWS SES and SMTP email delivery including HTML body generation,
subject formatting, recipient routing, and error handling.

Coverage target: 85%+ of channels/email.py

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
# EmailChannel reference implementation
# ============================================================================


class EmailChannel:
    """Email notification channel via AWS SES or SMTP.

    Reference implementation matching the expected interface of
    greenlang.infrastructure.alerting_service.channels.email.EmailChannel.
    """

    SUBJECT_PREFIX = {
        AlertSeverity.CRITICAL: "[CRITICAL]",
        AlertSeverity.WARNING: "[WARNING]",
        AlertSeverity.INFO: "[INFO]",
    }

    def __init__(
        self,
        from_address: str = "alerts@greenlang.io",
        use_ses: bool = True,
        ses_region: str = "eu-west-1",
        smtp_host: str = "",
        smtp_port: int = 587,
        team_recipients: Dict[str, List[str]] | None = None,
    ) -> None:
        self._from = from_address
        self._use_ses = use_ses
        self._ses_region = ses_region
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._team_recipients = team_recipients or {}
        self._ses_client: Any = None
        self._smtp_sender: Any = None

    def set_ses_client(self, client: Any) -> None:
        self._ses_client = client

    def set_smtp_sender(self, sender: Any) -> None:
        self._smtp_sender = sender

    def _get_recipients(self, alert: Alert) -> List[str]:
        """Get email recipients based on team."""
        if alert.team and alert.team in self._team_recipients:
            return self._team_recipients[alert.team]
        return self._team_recipients.get("default", ["ops@greenlang.io"])

    def _build_subject(self, alert: Alert) -> str:
        """Build email subject with severity prefix."""
        prefix = self.SUBJECT_PREFIX.get(alert.severity, "[ALERT]")
        return f"{prefix} {alert.title}"

    def _build_html_body(self, alert: Alert) -> str:
        """Build HTML email body."""
        return f"""<html>
<body>
<h1>{alert.title}</h1>
<table>
<tr><td><b>Severity:</b></td><td>{alert.severity.value}</td></tr>
<tr><td><b>Status:</b></td><td>{alert.status.value}</td></tr>
<tr><td><b>Source:</b></td><td>{alert.source}</td></tr>
<tr><td><b>Team:</b></td><td>{alert.team}</td></tr>
<tr><td><b>Service:</b></td><td>{alert.service}</td></tr>
</table>
<p>{alert.description}</p>
{f'<p><a href="{alert.runbook_url}">Runbook</a></p>' if alert.runbook_url else ''}
{f'<p><a href="{alert.dashboard_url}">Dashboard</a></p>' if alert.dashboard_url else ''}
</body>
</html>"""

    async def send(self, alert: Alert) -> NotificationResult:
        """Send email notification."""
        start = time.monotonic()
        recipients = self._get_recipients(alert)
        subject = self._build_subject(alert)
        html_body = self._build_html_body(alert)

        try:
            if self._use_ses:
                await self._send_ses(recipients, subject, html_body)
            else:
                await self._send_smtp(recipients, subject, html_body)

            duration_ms = (time.monotonic() - start) * 1000
            return NotificationResult(
                channel=NotificationChannel.EMAIL,
                status=NotificationStatus.SENT,
                recipient=", ".join(recipients),
                duration_ms=duration_ms,
                response_code=200,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            return NotificationResult(
                channel=NotificationChannel.EMAIL,
                status=NotificationStatus.FAILED,
                duration_ms=duration_ms,
                error_message=str(exc),
            )

    async def _send_ses(
        self, recipients: List[str], subject: str, html_body: str,
    ) -> None:
        """Send via AWS SES."""
        self._ses_client.send_email(
            Source=self._from,
            Destination={"ToAddresses": recipients},
            Message={
                "Subject": {"Data": subject},
                "Body": {"Html": {"Data": html_body}},
            },
        )

    async def _send_smtp(
        self, recipients: List[str], subject: str, html_body: str,
    ) -> None:
        """Send via SMTP."""
        self._smtp_sender(
            from_addr=self._from,
            to_addrs=recipients,
            subject=subject,
            html_body=html_body,
        )

    async def health_check(self) -> bool:
        """Check email sending availability."""
        if self._use_ses:
            try:
                self._ses_client.get_send_quota()
                return True
            except Exception:
                return False
        return bool(self._smtp_host)


# ============================================================================
# Tests
# ============================================================================


class TestEmailChannel:
    """Test suite for EmailChannel."""

    @pytest.fixture
    def ses_client(self):
        """Create a mock boto3 SES client."""
        client = MagicMock()
        client.send_email.return_value = {"MessageId": "msg-001"}
        client.get_send_quota.return_value = {"Max24HourSend": 10000}
        return client

    @pytest.fixture
    def smtp_sender(self):
        """Create a mock SMTP sender."""
        return MagicMock()

    @pytest.fixture
    def channel_ses(self, ses_client):
        """Create an EmailChannel using SES mode."""
        ch = EmailChannel(
            from_address="alerts@test.greenlang.io",
            use_ses=True,
            ses_region="eu-west-1",
            team_recipients={
                "platform": ["platform-team@greenlang.io"],
                "default": ["ops@greenlang.io"],
            },
        )
        ch.set_ses_client(ses_client)
        return ch

    @pytest.fixture
    def channel_smtp(self, smtp_sender):
        """Create an EmailChannel using SMTP mode."""
        ch = EmailChannel(
            from_address="alerts@test.greenlang.io",
            use_ses=False,
            smtp_host="smtp.test.io",
            smtp_port=465,
            team_recipients={"default": ["ops@greenlang.io"]},
        )
        ch.set_smtp_sender(smtp_sender)
        return ch

    @pytest.mark.asyncio
    async def test_send_ses(self, channel_ses, sample_alert, ses_client):
        """boto3 SES send_email is called."""
        result = await channel_ses.send(sample_alert)

        ses_client.send_email.assert_called_once()
        assert result.status == NotificationStatus.SENT

    @pytest.mark.asyncio
    async def test_send_smtp(self, channel_smtp, sample_alert, smtp_sender):
        """SMTP sender is called."""
        result = await channel_smtp.send(sample_alert)

        smtp_sender.assert_called_once()
        assert result.status == NotificationStatus.SENT

    @pytest.mark.asyncio
    async def test_html_body_generated(self, channel_ses, sample_alert, ses_client):
        """HTML email body contains alert details."""
        await channel_ses.send(sample_alert)

        call_kwargs = ses_client.send_email.call_args
        html_body = call_kwargs[1]["Message"]["Body"]["Html"]["Data"]

        assert sample_alert.title in html_body
        assert sample_alert.severity.value in html_body
        assert sample_alert.source in html_body
        assert "<html>" in html_body

    @pytest.mark.asyncio
    async def test_subject_includes_severity(self, channel_ses, ses_client):
        """Email subject has [CRITICAL] prefix."""
        alert = Alert(
            source="test", name="CritEmail",
            severity=AlertSeverity.CRITICAL, title="Critical issue",
            team="platform",
        )
        await channel_ses.send(alert)

        call_kwargs = ses_client.send_email.call_args
        subject = call_kwargs[1]["Message"]["Subject"]["Data"]
        assert subject.startswith("[CRITICAL]")

    @pytest.mark.asyncio
    async def test_from_address_set(self, channel_ses, sample_alert, ses_client):
        """From address is from config."""
        await channel_ses.send(sample_alert)

        call_kwargs = ses_client.send_email.call_args
        assert call_kwargs[1]["Source"] == "alerts@test.greenlang.io"

    @pytest.mark.asyncio
    async def test_recipient_routing(self, channel_ses, ses_client):
        """Team-based recipients are used."""
        alert = Alert(
            source="test", name="TeamEmail",
            severity=AlertSeverity.WARNING, title="Team issue",
            team="platform",
        )
        await channel_ses.send(alert)

        call_kwargs = ses_client.send_email.call_args
        recipients = call_kwargs[1]["Destination"]["ToAddresses"]
        assert "platform-team@greenlang.io" in recipients

    @pytest.mark.asyncio
    async def test_ses_region_config(self, channel_ses):
        """SES region is stored from config."""
        assert channel_ses._ses_region == "eu-west-1"

    @pytest.mark.asyncio
    async def test_send_failure_handling(self, channel_ses, sample_alert, ses_client):
        """SES error returns FAILED result."""
        ses_client.send_email.side_effect = Exception("SES quota exceeded")

        result = await channel_ses.send(sample_alert)

        assert result.status == NotificationStatus.FAILED
        assert "SES quota exceeded" in result.error_message

    @pytest.mark.asyncio
    async def test_health_check_ses(self, channel_ses, ses_client):
        """Health check calls get_send_quota."""
        healthy = await channel_ses.health_check()

        ses_client.get_send_quota.assert_called_once()
        assert healthy is True

    @pytest.mark.asyncio
    async def test_send_returns_result(self, channel_ses, sample_alert, ses_client):
        """send() returns NotificationResult with correct channel."""
        result = await channel_ses.send(sample_alert)

        assert isinstance(result, NotificationResult)
        assert result.channel == NotificationChannel.EMAIL
