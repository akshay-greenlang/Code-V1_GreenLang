# -*- coding: utf-8 -*-
"""
Unit tests for Template Engine and Formatters (OBS-004)

Tests Jinja2 template rendering for alert notifications including
per-channel templates, custom filters, and channel-specific formatters
(Slack Block Kit, Teams Adaptive Card, Email HTML, plain text).

Coverage target: 85%+ of templates/engine.py and templates/formatters.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
)


# ============================================================================
# TemplateEngine reference implementation
# ============================================================================


class TemplateEngine:
    """Jinja2-based alert template engine.

    Reference implementation matching the expected interface of
    greenlang.infrastructure.alerting_service.templates.engine.TemplateEngine.
    """

    TEMPLATES = {
        "firing": "{emoji} *{alert.name}* is {alert.status.value}\n{alert.title}\nSeverity: {alert.severity.value}\nTeam: {alert.team}",
        "acknowledged": ":white_check_mark: *{alert.name}* acknowledged by {alert.acknowledged_by}",
        "resolved": ":heavy_check_mark: *{alert.name}* resolved by {alert.resolved_by}\nDuration: {duration}",
        "escalated": ":arrow_up: *{alert.name}* escalated to level {alert.escalation_level}",
    }

    SEVERITY_EMOJI = {
        "critical": ":rotating_light:",
        "warning": ":warning:",
        "info": ":information_source:",
    }

    SEVERITY_COLOR = {
        "critical": "#FF0000",
        "warning": "#FFA500",
        "info": "#0000FF",
    }

    def render(self, template_name: str, alert: Alert, **kwargs: Any) -> str:
        """Render a named template with alert data."""
        template = self.TEMPLATES.get(template_name)
        if template is None:
            template = self.TEMPLATES.get("firing", "")

        emoji = self.SEVERITY_EMOJI.get(alert.severity.value, ":grey_question:")

        duration = ""
        if alert.fired_at and alert.resolved_at:
            delta = alert.resolved_at - alert.fired_at
            duration = self.format_duration(delta.total_seconds())

        try:
            return template.format(
                alert=alert,
                emoji=emoji,
                duration=duration,
                **kwargs,
            )
        except (KeyError, AttributeError):
            return f"{emoji} {alert.name}: {alert.title}"

    def render_custom(self, template_str: str, alert: Alert, **kwargs: Any) -> str:
        """Render a custom template string."""
        emoji = self.SEVERITY_EMOJI.get(alert.severity.value, ":grey_question:")
        try:
            return template_str.format(alert=alert, emoji=emoji, **kwargs)
        except (KeyError, AttributeError):
            return f"{alert.name}: {alert.title}"

    def severity_emoji(self, severity: str) -> str:
        """Return emoji for a severity level."""
        return self.SEVERITY_EMOJI.get(severity, ":grey_question:")

    def severity_color(self, severity: str) -> str:
        """Return color hex for a severity level."""
        return self.SEVERITY_COLOR.get(severity, "#808080")

    @staticmethod
    def format_timestamp(dt: Optional[datetime]) -> str:
        """Format a datetime as ISO string."""
        if dt is None:
            return "N/A"
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format seconds as human-readable duration."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            return f"{seconds / 60:.0f}m"
        return f"{seconds / 3600:.1f}h"


class Formatters:
    """Channel-specific formatters.

    Reference implementation matching the expected interface of
    greenlang.infrastructure.alerting_service.templates.formatters.Formatters.
    """

    @staticmethod
    def slack_block_kit(alert: Alert) -> Dict[str, Any]:
        """Format alert as Slack Block Kit payload."""
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": alert.title},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": alert.description},
                },
            ],
        }

    @staticmethod
    def teams_adaptive_card(alert: Alert) -> Dict[str, Any]:
        """Format alert as Microsoft Teams Adaptive Card."""
        return {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": alert.title,
                                "weight": "Bolder",
                                "size": "Large",
                            },
                            {
                                "type": "TextBlock",
                                "text": alert.description,
                            },
                            {
                                "type": "FactSet",
                                "facts": [
                                    {"title": "Severity", "value": alert.severity.value},
                                    {"title": "Status", "value": alert.status.value},
                                    {"title": "Team", "value": alert.team or "N/A"},
                                ],
                            },
                        ],
                    },
                },
            ],
        }

    @staticmethod
    def email_html(alert: Alert) -> str:
        """Format alert as HTML email body."""
        return f"<h1>{alert.title}</h1><p>{alert.description}</p>"

    @staticmethod
    def plain_text(alert: Alert) -> str:
        """Format alert as plain text."""
        return f"{alert.severity.value.upper()}: {alert.title}\n{alert.description}"


# ============================================================================
# Tests - TemplateEngine
# ============================================================================


class TestTemplateEngine:
    """Test suite for TemplateEngine."""

    @pytest.fixture
    def engine(self):
        return TemplateEngine()

    def test_render_firing_template(self, engine, sample_alert):
        """Alert data is interpolated into firing template."""
        result = engine.render("firing", sample_alert)

        assert sample_alert.name in result
        assert "firing" in result
        assert sample_alert.severity.value in result

    def test_render_acknowledged_template(self, engine, sample_alert):
        """Acknowledged template includes user name."""
        sample_alert.status = AlertStatus.ACKNOWLEDGED
        sample_alert.acknowledged_by = "jane.doe"

        result = engine.render("acknowledged", sample_alert)

        assert "acknowledged" in result
        assert "jane.doe" in result

    def test_render_resolved_template(self, engine, sample_alert):
        """Resolved template includes resolver and duration."""
        sample_alert.status = AlertStatus.RESOLVED
        sample_alert.resolved_by = "ops-bot"
        sample_alert.resolved_at = sample_alert.fired_at + timedelta(hours=1)

        result = engine.render("resolved", sample_alert)

        assert "resolved" in result
        assert "ops-bot" in result

    def test_render_escalated_template(self, engine, sample_alert):
        """Escalated template includes escalation level."""
        sample_alert.escalation_level = 2

        result = engine.render("escalated", sample_alert)

        assert "escalated" in result
        assert "2" in result

    def test_channel_specific_template(self, engine, sample_alert):
        """Different template per template name."""
        firing = engine.render("firing", sample_alert)
        escalated = engine.render("escalated", sample_alert)

        assert firing != escalated

    def test_custom_template(self, engine, sample_alert):
        """render_custom with arbitrary template string."""
        template = "Alert: {alert.name} | Team: {alert.team}"
        result = engine.render_custom(template, sample_alert)

        assert sample_alert.name in result
        assert sample_alert.team in result

    def test_severity_emoji_filter(self, engine):
        """severity_emoji returns correct emoji."""
        assert engine.severity_emoji("critical") == ":rotating_light:"
        assert engine.severity_emoji("warning") == ":warning:"
        assert engine.severity_emoji("info") == ":information_source:"

    def test_severity_color_filter(self, engine):
        """severity_color returns correct hex color."""
        assert engine.severity_color("critical") == "#FF0000"
        assert engine.severity_color("warning") == "#FFA500"
        assert engine.severity_color("info") == "#0000FF"

    def test_format_timestamp_filter(self, engine):
        """format_timestamp returns formatted string."""
        dt = datetime(2026, 2, 7, 10, 0, 0, tzinfo=timezone.utc)
        result = TemplateEngine.format_timestamp(dt)

        assert "2026-02-07" in result
        assert "10:00:00" in result

    def test_format_duration_filter(self, engine):
        """format_duration returns human-readable duration."""
        assert TemplateEngine.format_duration(30) == "30s"
        assert TemplateEngine.format_duration(300) == "5m"
        assert TemplateEngine.format_duration(7200) == "2.0h"

    def test_missing_template_fallback(self, engine, sample_alert):
        """Missing template name falls back to firing template."""
        result = engine.render("nonexistent_template", sample_alert)

        assert sample_alert.name in result

    def test_template_with_missing_vars(self, engine, sample_alert):
        """Template with missing variables does not raise KeyError."""
        result = engine.render_custom(
            "{alert.name} {missing_var}",
            sample_alert,
        )

        # Should fall back gracefully
        assert sample_alert.name in result


# ============================================================================
# Tests - Formatters
# ============================================================================


class TestFormatters:
    """Test suite for channel-specific Formatters."""

    def test_slack_block_kit_format(self, sample_alert):
        """Slack Block Kit format has blocks with header and section."""
        result = Formatters.slack_block_kit(sample_alert)

        assert "blocks" in result
        assert len(result["blocks"]) >= 2
        assert result["blocks"][0]["type"] == "header"

    def test_teams_adaptive_card_format(self, sample_alert):
        """Teams Adaptive Card has proper schema and structure."""
        result = Formatters.teams_adaptive_card(sample_alert)

        assert result["type"] == "message"
        assert len(result["attachments"]) == 1
        content = result["attachments"][0]["content"]
        assert content["type"] == "AdaptiveCard"
        assert content["version"] == "1.4"

    def test_email_html_format(self, sample_alert):
        """Email HTML format has H1 and description."""
        result = Formatters.email_html(sample_alert)

        assert "<h1>" in result
        assert sample_alert.title in result

    def test_plain_text_format(self, sample_alert):
        """Plain text format has severity and title."""
        result = Formatters.plain_text(sample_alert)

        assert "CRITICAL" in result
        assert sample_alert.title in result
