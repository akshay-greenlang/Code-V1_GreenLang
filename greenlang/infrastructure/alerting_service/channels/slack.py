# -*- coding: utf-8 -*-
"""
Slack Notification Channel - OBS-004: Unified Alerting Service

Delivers alert notifications to Slack via incoming webhooks using the
Block Kit message format. Supports severity-based webhook routing so
critical alerts can go to a dedicated channel.

Reference:
    https://api.slack.com/reference/block-kit

Example:
    >>> channel = SlackChannel(
    ...     webhook_critical="https://hooks.slack.com/...",
    ...     webhook_warning="https://hooks.slack.com/...",
    ...     webhook_info="https://hooks.slack.com/...",
    ... )
    >>> result = await channel.send(alert, "CPU above 90%")

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.alerting_service.channels.base import (
    BaseNotificationChannel,
)
from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    NotificationResult,
    NotificationStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional httpx import
# ---------------------------------------------------------------------------

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_EMOJI: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: ":rotating_light:",
    AlertSeverity.WARNING: ":warning:",
    AlertSeverity.INFO: ":information_source:",
}

SEVERITY_COLOR: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: "#FF0000",
    AlertSeverity.WARNING: "#FFA500",
    AlertSeverity.INFO: "#36a64f",
}


# ---------------------------------------------------------------------------
# SlackChannel
# ---------------------------------------------------------------------------


class SlackChannel(BaseNotificationChannel):
    """Slack incoming-webhook notification channel using Block Kit.

    Routes alerts to severity-specific webhooks when configured.

    Attributes:
        webhook_critical: Webhook URL for critical alerts.
        webhook_warning: Webhook URL for warning alerts.
        webhook_info: Webhook URL for info alerts.
    """

    name = "slack"

    def __init__(
        self,
        webhook_critical: str = "",
        webhook_warning: str = "",
        webhook_info: str = "",
    ) -> None:
        self.webhook_critical = webhook_critical
        self.webhook_warning = webhook_warning
        self.webhook_info = webhook_info
        self.enabled = bool(
            webhook_critical or webhook_warning or webhook_info
        ) and HTTPX_AVAILABLE

    # ------------------------------------------------------------------
    # BaseNotificationChannel interface
    # ------------------------------------------------------------------

    async def send(
        self,
        alert: Alert,
        rendered_message: str,
    ) -> NotificationResult:
        """Post a Block Kit message to the appropriate Slack webhook.

        Args:
            alert: Alert to notify about.
            rendered_message: Pre-rendered description text.

        Returns:
            NotificationResult.
        """
        if not HTTPX_AVAILABLE:
            return self._make_result(
                NotificationStatus.FAILED,
                error_message="httpx not installed",
            )

        webhook_url = self._get_webhook_url(alert.severity)
        if not webhook_url:
            return self._make_result(
                NotificationStatus.SKIPPED,
                error_message=f"No webhook configured for severity {alert.severity.value}",
            )

        blocks = self._build_blocks(alert, rendered_message)
        payload: Dict[str, Any] = {"blocks": blocks}

        start = self._timed()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(webhook_url, json=payload)

            duration_ms = (self._timed() - start) * 1000

            if resp.status_code == 200:
                logger.info(
                    "Slack notification sent: alert=%s, severity=%s",
                    alert.alert_id[:8], alert.severity.value,
                )
                return self._make_result(
                    NotificationStatus.SENT,
                    recipient="slack",
                    duration_ms=duration_ms,
                    response_code=resp.status_code,
                )

            logger.warning(
                "Slack send failed: status=%d, body=%s",
                resp.status_code, resp.text[:200],
            )
            return self._make_result(
                NotificationStatus.FAILED,
                recipient="slack",
                duration_ms=duration_ms,
                response_code=resp.status_code,
                error_message=resp.text[:200],
            )

        except Exception as exc:
            duration_ms = (self._timed() - start) * 1000
            logger.error("Slack send error: %s", exc)
            return self._make_result(
                NotificationStatus.FAILED,
                duration_ms=duration_ms,
                error_message=str(exc),
            )

    async def health_check(self) -> bool:
        """Check Slack webhook reachability.

        Returns:
            True if at least one webhook responds.
        """
        if not HTTPX_AVAILABLE:
            return False

        for url in [
            self.webhook_critical,
            self.webhook_warning,
            self.webhook_info,
        ]:
            if not url:
                continue
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(
                        url,
                        json={"text": ""},
                    )
                # Slack returns 400 for empty text but proves webhook exists
                if resp.status_code in (200, 400):
                    return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------
    # Block Kit builders
    # ------------------------------------------------------------------

    def _build_blocks(
        self,
        alert: Alert,
        message: str,
    ) -> List[Dict[str, Any]]:
        """Build Slack Block Kit blocks for an alert.

        Args:
            alert: Source alert.
            message: Pre-rendered description text.

        Returns:
            List of Block Kit block dicts.
        """
        emoji = SEVERITY_EMOJI.get(alert.severity, ":bell:")
        blocks: List[Dict[str, Any]] = []

        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {alert.name}",
                "emoji": True,
            },
        })

        # Fields section: severity, status, service
        fields = [
            {
                "type": "mrkdwn",
                "text": f"*Severity:* `{alert.severity.value.upper()}`",
            },
            {
                "type": "mrkdwn",
                "text": f"*Status:* `{alert.status.value.upper()}`",
            },
        ]
        if alert.service:
            fields.append({
                "type": "mrkdwn",
                "text": f"*Service:* {alert.service}",
            })
        if alert.team:
            fields.append({
                "type": "mrkdwn",
                "text": f"*Team:* {alert.team}",
            })

        blocks.append({"type": "section", "fields": fields})

        # Title
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{alert.title}*",
            },
        })

        # Description
        if message:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message[:3000],
                },
            })

        # Context
        context_elements = []
        if alert.environment:
            context_elements.append({
                "type": "mrkdwn",
                "text": f"Env: `{alert.environment}`",
            })
        if alert.fired_at:
            context_elements.append({
                "type": "mrkdwn",
                "text": f"Fired: {alert.fired_at.strftime('%Y-%m-%d %H:%M UTC')}",
            })
        if context_elements:
            blocks.append({
                "type": "context",
                "elements": context_elements,
            })

        # Action buttons
        actions: List[Dict[str, Any]] = []
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

        # Divider
        blocks.append({"type": "divider"})

        return blocks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_webhook_url(self, severity: AlertSeverity) -> str:
        """Return the webhook URL for a given severity.

        Falls back through severity levels if a dedicated webhook is not
        configured (critical -> warning -> info).

        Args:
            severity: Alert severity.

        Returns:
            Webhook URL string, possibly empty.
        """
        if severity == AlertSeverity.CRITICAL:
            return (
                self.webhook_critical
                or self.webhook_warning
                or self.webhook_info
            )
        if severity == AlertSeverity.WARNING:
            return self.webhook_warning or self.webhook_info
        return self.webhook_info
