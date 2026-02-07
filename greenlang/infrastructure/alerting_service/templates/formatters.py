# -*- coding: utf-8 -*-
"""
Channel-Specific Formatters - OBS-004: Unified Alerting Service

Produces channel-native payload structures (Slack Block Kit, Teams
Adaptive Cards, HTML email, plain text) from an Alert and a rendered
message string. These formatters are used by the template engine and
directly by channels that need structured payloads.

Example:
    >>> fmt = SlackBlockKitFormatter()
    >>> payload = fmt.format(alert, "CPU above 90%")

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slack Block Kit Formatter
# ---------------------------------------------------------------------------


class SlackBlockKitFormatter:
    """Format alerts as Slack Block Kit JSON payloads."""

    SEVERITY_EMOJI: Dict[AlertSeverity, str] = {
        AlertSeverity.CRITICAL: ":rotating_light:",
        AlertSeverity.WARNING: ":warning:",
        AlertSeverity.INFO: ":information_source:",
    }

    def format(self, alert: Alert, message: str) -> Dict[str, Any]:
        """Build a complete Slack Block Kit payload.

        Args:
            alert: Source alert.
            message: Pre-rendered description.

        Returns:
            JSON-serializable dict.
        """
        blocks = [
            self._header_block(alert),
            self._fields_block(alert),
        ]
        if message:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": message[:3000]},
            })
        blocks.append(self._context_block(alert))
        actions = self._actions_block(alert)
        if actions:
            blocks.append(actions)
        blocks.append({"type": "divider"})
        return {"blocks": blocks}

    def _header_block(self, alert: Alert) -> Dict[str, Any]:
        """Build a header block with severity emoji."""
        emoji = self.SEVERITY_EMOJI.get(alert.severity, ":bell:")
        return {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {alert.name}",
                "emoji": True,
            },
        }

    def _fields_block(self, alert: Alert) -> Dict[str, Any]:
        """Build a section block with key fields."""
        fields = [
            {"type": "mrkdwn", "text": f"*Severity:* `{alert.severity.value.upper()}`"},
            {"type": "mrkdwn", "text": f"*Status:* `{alert.status.value.upper()}`"},
        ]
        if alert.service:
            fields.append({"type": "mrkdwn", "text": f"*Service:* {alert.service}"})
        if alert.team:
            fields.append({"type": "mrkdwn", "text": f"*Team:* {alert.team}"})
        return {"type": "section", "fields": fields}

    def _context_block(self, alert: Alert) -> Dict[str, Any]:
        """Build a context block with metadata."""
        elements = []
        if alert.environment:
            elements.append({"type": "mrkdwn", "text": f"Env: `{alert.environment}`"})
        if alert.fired_at:
            elements.append({
                "type": "mrkdwn",
                "text": f"Fired: {alert.fired_at.strftime('%Y-%m-%d %H:%M UTC')}",
            })
        if not elements:
            elements.append({"type": "mrkdwn", "text": "GreenLang Alerting"})
        return {"type": "context", "elements": elements}

    def _actions_block(self, alert: Alert) -> Optional[Dict[str, Any]]:
        """Build an actions block with runbook/dashboard buttons."""
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
            return {"type": "actions", "elements": actions}
        return None


# ---------------------------------------------------------------------------
# Teams Adaptive Card Formatter
# ---------------------------------------------------------------------------


class TeamsAdaptiveCardFormatter:
    """Format alerts as Microsoft Teams Adaptive Card v1.5 payloads."""

    SEVERITY_COLORS: Dict[AlertSeverity, str] = {
        AlertSeverity.CRITICAL: "#FF0000",
        AlertSeverity.WARNING: "#FFA500",
        AlertSeverity.INFO: "#0078D4",
    }

    def format(self, alert: Alert, message: str) -> Dict[str, Any]:
        """Build a complete Adaptive Card payload.

        Args:
            alert: Source alert.
            message: Pre-rendered description.

        Returns:
            Adaptive Card JSON dict.
        """
        body: List[Dict[str, Any]] = [
            self._header(alert),
            self._facts(alert),
        ]
        if message:
            body.append(self._body(message))

        card: Dict[str, Any] = {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.5",
            "body": body,
        }
        actions = self._actions(alert)
        if actions:
            card["actions"] = actions
        return card

    def _header(self, alert: Alert) -> Dict[str, Any]:
        """Build a header TextBlock."""
        label = alert.severity.value.upper()
        return {
            "type": "TextBlock",
            "text": f"[{label}] {alert.name}",
            "weight": "Bolder",
            "size": "Large",
            "color": "Attention" if alert.severity == AlertSeverity.CRITICAL else "Default",
            "wrap": True,
        }

    def _facts(self, alert: Alert) -> Dict[str, Any]:
        """Build a FactSet with alert metadata."""
        facts = [
            {"title": "Severity", "value": alert.severity.value.upper()},
            {"title": "Status", "value": alert.status.value.upper()},
        ]
        if alert.team:
            facts.append({"title": "Team", "value": alert.team})
        if alert.service:
            facts.append({"title": "Service", "value": alert.service})
        if alert.environment:
            facts.append({"title": "Environment", "value": alert.environment})
        return {"type": "FactSet", "facts": facts}

    def _body(self, message: str) -> Dict[str, Any]:
        """Build a description TextBlock."""
        return {
            "type": "TextBlock",
            "text": message[:2000],
            "wrap": True,
            "isSubtle": True,
        }

    def _actions(self, alert: Alert) -> List[Dict[str, Any]]:
        """Build action buttons."""
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
        return actions


# ---------------------------------------------------------------------------
# Email HTML Formatter
# ---------------------------------------------------------------------------


class EmailHTMLFormatter:
    """Format alerts as HTML email bodies with inline CSS."""

    SEVERITY_COLORS: Dict[AlertSeverity, str] = {
        AlertSeverity.CRITICAL: "#DC3545",
        AlertSeverity.WARNING: "#FFC107",
        AlertSeverity.INFO: "#17A2B8",
    }

    def format(self, alert: Alert, message: str) -> str:
        """Build an HTML email body.

        Args:
            alert: Source alert.
            message: Pre-rendered description.

        Returns:
            HTML string.
        """
        color = self.SEVERITY_COLORS.get(alert.severity, "#6C757D")
        badge = self._severity_badge(alert)
        table = self._detail_table(alert)
        buttons = self._action_buttons(alert)

        return f"""\
<!DOCTYPE html>
<html><head><meta charset="utf-8"/></head>
<body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
  <div style="background:{color};color:white;padding:16px 20px;border-radius:4px 4px 0 0;">
    <h2 style="margin:0;">{badge} {alert.name}</h2>
  </div>
  <div style="border:1px solid #dee2e6;border-top:none;padding:20px;border-radius:0 0 4px 4px;">
    <h3 style="margin-top:0;">{alert.title}</h3>
    <p style="color:#495057;">{message}</p>
    {table}
    {buttons}
    <hr style="border:none;border-top:1px solid #dee2e6;margin:20px 0;"/>
    <p style="font-size:12px;color:#6c757d;">GreenLang Unified Alerting</p>
  </div>
</body></html>"""

    def _severity_badge(self, alert: Alert) -> str:
        """Return a text badge for the severity."""
        return f"[{alert.severity.value.upper()}]"

    def _detail_table(self, alert: Alert) -> str:
        """Build an HTML table of alert details."""
        rows = [
            ("Status", alert.status.value),
            ("Service", alert.service or "N/A"),
            ("Team", alert.team or "N/A"),
            ("Environment", alert.environment or "N/A"),
        ]
        if alert.fired_at:
            rows.append(("Fired At", alert.fired_at.strftime("%Y-%m-%d %H:%M UTC")))

        row_html = "".join(
            f'<tr><td style="padding:4px 8px;font-weight:bold;">{k}</td>'
            f"<td>{v}</td></tr>"
            for k, v in rows
        )
        return f'<table style="width:100%;border-collapse:collapse;margin:16px 0;">{row_html}</table>'

    def _action_buttons(self, alert: Alert) -> str:
        """Build HTML action links."""
        parts = []
        if alert.runbook_url:
            parts.append(
                f'<a href="{alert.runbook_url}" style="color:#0d6efd;margin-right:12px;">Runbook</a>'
            )
        if alert.dashboard_url:
            parts.append(
                f'<a href="{alert.dashboard_url}" style="color:#0d6efd;">Dashboard</a>'
            )
        if parts:
            return f'<div style="margin-top:16px;">{"".join(parts)}</div>'
        return ""


# ---------------------------------------------------------------------------
# Plain Text Formatter
# ---------------------------------------------------------------------------


class PlainTextFormatter:
    """Format alerts as plain text."""

    def format(self, alert: Alert, message: str) -> str:
        """Build a plain-text representation.

        Args:
            alert: Source alert.
            message: Pre-rendered description.

        Returns:
            Plain text string.
        """
        lines = [
            f"[{alert.severity.value.upper()}] {alert.name}",
            f"Title: {alert.title}",
            f"Status: {alert.status.value}",
        ]
        if alert.description:
            lines.append(f"Description: {alert.description}")
        if message and message != alert.description:
            lines.append(f"Message: {message}")
        lines.append(f"Service: {alert.service or 'N/A'}")
        lines.append(f"Team: {alert.team or 'N/A'}")
        lines.append(f"Environment: {alert.environment or 'N/A'}")
        if alert.fired_at:
            lines.append(
                f"Fired At: {alert.fired_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
        if alert.runbook_url:
            lines.append(f"Runbook: {alert.runbook_url}")
        if alert.dashboard_url:
            lines.append(f"Dashboard: {alert.dashboard_url}")
        return "\n".join(lines)
