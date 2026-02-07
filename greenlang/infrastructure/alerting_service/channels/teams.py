# -*- coding: utf-8 -*-
"""
Microsoft Teams Notification Channel - OBS-004: Unified Alerting Service

Delivers alert notifications to Microsoft Teams via incoming webhooks
using the Adaptive Card v1.5 message format.

Reference:
    https://learn.microsoft.com/en-us/adaptive-cards/

Example:
    >>> channel = TeamsChannel(webhook_url="https://outlook.office.com/webhook/...")
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

SEVERITY_COLORS: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: "#FF0000",
    AlertSeverity.WARNING: "#FFA500",
    AlertSeverity.INFO: "#0078D4",
}

SEVERITY_LABELS: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: "CRITICAL",
    AlertSeverity.WARNING: "WARNING",
    AlertSeverity.INFO: "INFO",
}


# ---------------------------------------------------------------------------
# TeamsChannel
# ---------------------------------------------------------------------------


class TeamsChannel(BaseNotificationChannel):
    """Microsoft Teams incoming-webhook notification channel.

    Posts Adaptive Card v1.5 messages to a Teams webhook URL.

    Attributes:
        webhook_url: Teams incoming webhook URL.
    """

    name = "teams"

    def __init__(self, webhook_url: str = "") -> None:
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url) and HTTPX_AVAILABLE

    # ------------------------------------------------------------------
    # BaseNotificationChannel interface
    # ------------------------------------------------------------------

    async def send(
        self,
        alert: Alert,
        rendered_message: str,
    ) -> NotificationResult:
        """Post an Adaptive Card to the Teams webhook.

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
        if not self.webhook_url:
            return self._make_result(
                NotificationStatus.SKIPPED,
                error_message="No Teams webhook URL configured",
            )

        card = self._build_adaptive_card(alert, rendered_message)
        payload = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "contentUrl": None,
                    "content": card,
                }
            ],
        }

        start = self._timed()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

            duration_ms = (self._timed() - start) * 1000

            if resp.status_code == 200:
                logger.info(
                    "Teams notification sent: alert=%s", alert.alert_id[:8],
                )
                return self._make_result(
                    NotificationStatus.SENT,
                    recipient="teams",
                    duration_ms=duration_ms,
                    response_code=resp.status_code,
                )

            logger.warning(
                "Teams send failed: status=%d, body=%s",
                resp.status_code, resp.text[:200],
            )
            return self._make_result(
                NotificationStatus.FAILED,
                recipient="teams",
                duration_ms=duration_ms,
                response_code=resp.status_code,
                error_message=resp.text[:200],
            )

        except Exception as exc:
            duration_ms = (self._timed() - start) * 1000
            logger.error("Teams send error: %s", exc)
            return self._make_result(
                NotificationStatus.FAILED,
                duration_ms=duration_ms,
                error_message=str(exc),
            )

    async def health_check(self) -> bool:
        """Check Teams webhook reachability.

        Returns:
            True if the webhook URL responds.
        """
        if not HTTPX_AVAILABLE or not self.webhook_url:
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.head(self.webhook_url)
            return resp.status_code < 500
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Adaptive Card builder
    # ------------------------------------------------------------------

    def _build_adaptive_card(
        self,
        alert: Alert,
        message: str,
    ) -> Dict[str, Any]:
        """Build an Adaptive Card v1.5 payload.

        Args:
            alert: Source alert.
            message: Pre-rendered description text.

        Returns:
            Adaptive Card JSON dict.
        """
        color = SEVERITY_COLORS.get(alert.severity, "#0078D4")
        label = SEVERITY_LABELS.get(alert.severity, "INFO")

        body: List[Dict[str, Any]] = []

        # Header
        body.append({
            "type": "TextBlock",
            "text": f"[{label}] {alert.name}",
            "weight": "Bolder",
            "size": "Large",
            "color": "Attention" if alert.severity == AlertSeverity.CRITICAL else "Default",
            "wrap": True,
        })

        # Title
        body.append({
            "type": "TextBlock",
            "text": alert.title,
            "weight": "Bolder",
            "size": "Medium",
            "wrap": True,
        })

        # Facts
        facts: List[Dict[str, str]] = [
            {"title": "Severity", "value": label},
            {"title": "Status", "value": alert.status.value.upper()},
        ]
        if alert.team:
            facts.append({"title": "Team", "value": alert.team})
        if alert.service:
            facts.append({"title": "Service", "value": alert.service})
        if alert.environment:
            facts.append({"title": "Environment", "value": alert.environment})
        if alert.fired_at:
            facts.append({
                "title": "Fired At",
                "value": alert.fired_at.strftime("%Y-%m-%d %H:%M UTC"),
            })

        body.append({
            "type": "FactSet",
            "facts": facts,
        })

        # Description
        if message:
            body.append({
                "type": "TextBlock",
                "text": message[:2000],
                "wrap": True,
                "isSubtle": True,
            })

        # Actions
        actions: List[Dict[str, Any]] = []
        if alert.runbook_url:
            actions.append({
                "type": "Action.OpenUrl",
                "title": "View Runbook",
                "url": alert.runbook_url,
            })
        if alert.dashboard_url:
            actions.append({
                "type": "Action.OpenUrl",
                "title": "View Dashboard",
                "url": alert.dashboard_url,
            })

        card: Dict[str, Any] = {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.5",
            "body": body,
        }
        if actions:
            card["actions"] = actions

        return card
