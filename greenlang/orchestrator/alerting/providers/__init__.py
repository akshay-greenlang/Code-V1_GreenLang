# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator Alert Webhook Providers
==============================================

Provider-specific formatters for webhook alerts.

Supported providers:
- Slack (Slack Block Kit formatting)
- Discord (Discord embed formatting)
- PagerDuty (Events API v2)
- Custom (Generic HTTP POST)

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: Alert Webhook Providers
"""

from greenlang.orchestrator.alerting.providers.slack import (
    SlackWebhookProvider,
    format_slack_payload,
)
from greenlang.orchestrator.alerting.providers.discord import (
    DiscordWebhookProvider,
    format_discord_payload,
)
from greenlang.orchestrator.alerting.providers.pagerduty import (
    PagerDutyProvider,
    format_pagerduty_payload,
)
from greenlang.orchestrator.alerting.providers.custom import (
    CustomWebhookProvider,
    format_custom_payload,
)

__all__ = [
    # Slack
    "SlackWebhookProvider",
    "format_slack_payload",
    # Discord
    "DiscordWebhookProvider",
    "format_discord_payload",
    # PagerDuty
    "PagerDutyProvider",
    "format_pagerduty_payload",
    # Custom
    "CustomWebhookProvider",
    "format_custom_payload",
]
