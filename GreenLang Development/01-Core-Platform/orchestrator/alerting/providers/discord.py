# -*- coding: utf-8 -*-
"""
Discord Webhook Provider
========================

Formats alert payloads for Discord webhooks using embeds.

Supports:
- Rich embed formatting
- Color-coded severity indicators
- Thumbnail and footer elements
- Field-based detail display

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: Discord Webhook Provider
"""

import logging
from typing import Any, Dict, List, Optional

from greenlang.orchestrator.alerting.webhooks import (
    AlertPayload,
    AlertSeverity,
    AlertType,
    WebhookConfig,
)

logger = logging.getLogger(__name__)


# Severity to Discord color mapping (decimal format)
SEVERITY_COLORS: Dict[AlertSeverity, int] = {
    AlertSeverity.CRITICAL: 0xDC3545,  # Red
    AlertSeverity.HIGH: 0xFD7E14,      # Orange
    AlertSeverity.MEDIUM: 0xFFC107,    # Yellow
    AlertSeverity.LOW: 0x17A2B8,       # Teal
    AlertSeverity.INFO: 0x28A745,      # Green
}

# Severity to emoji mapping
SEVERITY_EMOJIS: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: ":rotating_light:",
    AlertSeverity.HIGH: ":warning:",
    AlertSeverity.MEDIUM: ":yellow_circle:",
    AlertSeverity.LOW: ":information_source:",
    AlertSeverity.INFO: ":white_check_mark:",
}

# Alert type to emoji and title mapping
ALERT_TYPE_INFO: Dict[AlertType, Dict[str, str]] = {
    AlertType.RUN_FAILED: {
        "emoji": ":x:",
        "title": "Pipeline Run Failed",
    },
    AlertType.STEP_TIMEOUT: {
        "emoji": ":alarm_clock:",
        "title": "Step Timeout",
    },
    AlertType.POLICY_DENIAL: {
        "emoji": ":no_entry:",
        "title": "Policy Denial",
    },
    AlertType.SLO_BREACH: {
        "emoji": ":chart_with_downwards_trend:",
        "title": "SLO Breach",
    },
    AlertType.RUN_SUCCEEDED: {
        "emoji": ":white_check_mark:",
        "title": "Pipeline Run Succeeded",
    },
}


def format_discord_payload(
    alert: AlertPayload, config: WebhookConfig
) -> Dict[str, Any]:
    """
    Format alert payload for Discord embeds.

    Creates a rich embed message with:
    - Title with severity emoji
    - Description with alert message
    - Fields for details
    - Color-coded by severity
    - Footer with timestamp and alert ID

    Args:
        alert: Alert payload to format
        config: Webhook configuration

    Returns:
        Formatted Discord payload dictionary
    """
    type_info = ALERT_TYPE_INFO.get(
        alert.alert_type,
        {"emoji": ":bell:", "title": "Alert"}
    )
    severity_emoji = SEVERITY_EMOJIS.get(alert.severity, ":grey_question:")
    color = SEVERITY_COLORS.get(alert.severity, 0x808080)

    # Build embed fields
    fields: List[Dict[str, Any]] = [
        {
            "name": ":id: Run ID",
            "value": f"`{alert.run_id}`",
            "inline": True,
        },
        {
            "name": ":warning: Severity",
            "value": f"{severity_emoji} {alert.severity.value.upper()}",
            "inline": True,
        },
        {
            "name": ":globe_with_meridians: Namespace",
            "value": f"`{alert.namespace}`",
            "inline": True,
        },
    ]

    if alert.pipeline_id:
        fields.append({
            "name": ":package: Pipeline ID",
            "value": f"`{alert.pipeline_id}`",
            "inline": True,
        })

    if alert.step_id:
        fields.append({
            "name": ":footprints: Step ID",
            "value": f"`{alert.step_id}`",
            "inline": True,
        })

    # Add custom details as fields
    if alert.details:
        for key, value in list(alert.details.items())[:5]:  # Limit fields
            if isinstance(value, (dict, list)):
                continue  # Skip complex types
            fields.append({
                "name": f":small_blue_diamond: {key.replace('_', ' ').title()}",
                "value": f"`{value}`" if len(str(value)) < 100 else str(value)[:100] + "...",
                "inline": True,
            })

    # Build the embed
    embed: Dict[str, Any] = {
        "title": f"{type_info['emoji']} {type_info['title']}",
        "description": alert.message,
        "color": color,
        "fields": fields[:25],  # Discord limit
        "footer": {
            "text": f"GreenLang Orchestrator | Alert ID: {alert.alert_id}",
        },
        "timestamp": alert.timestamp.isoformat(),
    }

    # Build the final payload
    payload: Dict[str, Any] = {
        "username": "GreenLang Orchestrator",
        "avatar_url": config.metadata.get(
            "avatar_url",
            "https://greenlang.io/images/logo.png"
        ),
        "embeds": [embed],
    }

    # Add content for @mentions if critical
    if alert.severity == AlertSeverity.CRITICAL:
        role_id = config.metadata.get("critical_role_id")
        if role_id:
            payload["content"] = f"<@&{role_id}> Critical alert!"

    return payload


class DiscordWebhookProvider:
    """
    Discord webhook provider for alert notifications.

    Provides methods for formatting and sending alerts to Discord
    using webhooks with rich embeds.

    Example:
        >>> provider = DiscordWebhookProvider()
        >>> payload = provider.format_alert(alert, config)
        >>> # Send payload to Discord webhook URL
    """

    def __init__(self):
        """Initialize DiscordWebhookProvider."""
        logger.info("DiscordWebhookProvider initialized")

    def format_alert(
        self, alert: AlertPayload, config: WebhookConfig
    ) -> Dict[str, Any]:
        """
        Format an alert for Discord.

        Args:
            alert: Alert payload to format
            config: Webhook configuration

        Returns:
            Formatted Discord payload
        """
        return format_discord_payload(alert, config)

    def get_color_for_severity(self, severity: AlertSeverity) -> int:
        """Get Discord color (decimal) for severity level."""
        return SEVERITY_COLORS.get(severity, 0x808080)

    @staticmethod
    def create_button_component(
        label: str,
        url: str,
        emoji: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a button component for Discord message.

        Note: Discord webhooks have limited button support.
        This is primarily for reference when using bot-based integration.

        Args:
            label: Button label text
            url: URL to link to
            emoji: Optional emoji for the button

        Returns:
            Button component dictionary
        """
        component: Dict[str, Any] = {
            "type": 2,  # Button
            "style": 5,  # Link style
            "label": label,
            "url": url,
        }
        if emoji:
            component["emoji"] = {"name": emoji}
        return component


__all__ = [
    "DiscordWebhookProvider",
    "format_discord_payload",
    "SEVERITY_COLORS",
]
