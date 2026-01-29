# -*- coding: utf-8 -*-
"""
Slack Webhook Provider
======================

Formats alert payloads for Slack webhooks using Block Kit.

Supports:
- Rich message formatting with blocks
- Color-coded severity indicators
- Action buttons (acknowledge, view details)
- Contextual information display

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: Slack Webhook Provider
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


# Severity to Slack color mapping
SEVERITY_COLORS: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: "#dc3545",  # Red
    AlertSeverity.HIGH: "#fd7e14",      # Orange
    AlertSeverity.MEDIUM: "#ffc107",    # Yellow
    AlertSeverity.LOW: "#17a2b8",       # Teal
    AlertSeverity.INFO: "#28a745",      # Green
}

# Severity to emoji mapping
SEVERITY_EMOJIS: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: ":rotating_light:",
    AlertSeverity.HIGH: ":warning:",
    AlertSeverity.MEDIUM: ":large_yellow_circle:",
    AlertSeverity.LOW: ":information_source:",
    AlertSeverity.INFO: ":white_check_mark:",
}

# Alert type to emoji mapping
ALERT_TYPE_EMOJIS: Dict[AlertType, str] = {
    AlertType.RUN_FAILED: ":x:",
    AlertType.STEP_TIMEOUT: ":alarm_clock:",
    AlertType.POLICY_DENIAL: ":no_entry:",
    AlertType.SLO_BREACH: ":chart_with_downwards_trend:",
    AlertType.RUN_SUCCEEDED: ":white_check_mark:",
}


def format_slack_payload(
    alert: AlertPayload, config: WebhookConfig
) -> Dict[str, Any]:
    """
    Format alert payload for Slack Block Kit.

    Creates a rich message with:
    - Header with severity indicator
    - Alert details in sections
    - Contextual metadata
    - Optional action buttons

    Args:
        alert: Alert payload to format
        config: Webhook configuration

    Returns:
        Formatted Slack payload dictionary
    """
    severity_emoji = SEVERITY_EMOJIS.get(alert.severity, ":grey_question:")
    type_emoji = ALERT_TYPE_EMOJIS.get(alert.alert_type, ":bell:")
    color = SEVERITY_COLORS.get(alert.severity, "#808080")

    # Build blocks
    blocks: List[Dict[str, Any]] = []

    # Header block
    header_text = f"{severity_emoji} {type_emoji} *{alert.alert_type.value.upper().replace('_', ' ')}*"
    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"GreenLang Alert: {alert.alert_type.value.replace('_', ' ').title()}",
            "emoji": True,
        }
    })

    # Main message section
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*{alert.message}*",
        }
    })

    # Severity and namespace context
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"*Severity:* {alert.severity.value.upper()}",
            },
            {
                "type": "mrkdwn",
                "text": f"*Namespace:* `{alert.namespace}`",
            },
        ]
    })

    # Divider
    blocks.append({"type": "divider"})

    # Details section
    details_fields = [
        {
            "type": "mrkdwn",
            "text": f"*Run ID:*\n`{alert.run_id}`",
        },
    ]

    if alert.pipeline_id:
        details_fields.append({
            "type": "mrkdwn",
            "text": f"*Pipeline ID:*\n`{alert.pipeline_id}`",
        })

    if alert.step_id:
        details_fields.append({
            "type": "mrkdwn",
            "text": f"*Step ID:*\n`{alert.step_id}`",
        })

    blocks.append({
        "type": "section",
        "fields": details_fields[:10],  # Slack limit
    })

    # Additional details from payload
    if alert.details:
        detail_text_parts = []
        for key, value in list(alert.details.items())[:5]:  # Limit details
            if isinstance(value, (dict, list)):
                continue  # Skip complex types for cleaner display
            detail_text_parts.append(f"*{key}:* `{value}`")

        if detail_text_parts:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "\n".join(detail_text_parts),
                }
            })

    # Timestamp context
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"*Alert ID:* `{alert.alert_id}` | *Time:* <!date^{int(alert.timestamp.timestamp())}^{{date_short_pretty}} at {{time}}|{alert.timestamp.isoformat()}>",
            }
        ]
    })

    # Build the final payload
    payload: Dict[str, Any] = {
        "blocks": blocks,
        "attachments": [
            {
                "color": color,
                "fallback": alert.message,
            }
        ],
    }

    # Add text fallback for notifications
    payload["text"] = f"{severity_emoji} [{alert.severity.value.upper()}] {alert.message}"

    return payload


class SlackWebhookProvider:
    """
    Slack webhook provider for alert notifications.

    Provides methods for formatting and sending alerts to Slack
    using the Incoming Webhooks API.

    Example:
        >>> provider = SlackWebhookProvider()
        >>> payload = provider.format_alert(alert, config)
        >>> # Send payload to Slack webhook URL
    """

    def __init__(self):
        """Initialize SlackWebhookProvider."""
        logger.info("SlackWebhookProvider initialized")

    def format_alert(
        self, alert: AlertPayload, config: WebhookConfig
    ) -> Dict[str, Any]:
        """
        Format an alert for Slack.

        Args:
            alert: Alert payload to format
            config: Webhook configuration

        Returns:
            Formatted Slack payload
        """
        return format_slack_payload(alert, config)

    def get_color_for_severity(self, severity: AlertSeverity) -> str:
        """Get Slack color for severity level."""
        return SEVERITY_COLORS.get(severity, "#808080")

    @staticmethod
    def create_action_block(
        run_id: str,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create an action block with buttons.

        Args:
            run_id: Run ID for action context
            base_url: Base URL for the orchestrator UI

        Returns:
            Slack action block
        """
        actions = []

        if base_url:
            actions.append({
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "View Run Details",
                    "emoji": True,
                },
                "url": f"{base_url}/runs/{run_id}",
                "action_id": "view_run",
            })

        actions.append({
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": "Acknowledge",
                "emoji": True,
            },
            "style": "primary",
            "action_id": f"ack_{run_id}",
        })

        return {
            "type": "actions",
            "elements": actions,
        }


__all__ = [
    "SlackWebhookProvider",
    "format_slack_payload",
    "SEVERITY_COLORS",
    "SEVERITY_EMOJIS",
]
