# -*- coding: utf-8 -*-
"""
Notification configuration for the factors watch & release pipeline.

Environment variables:
    GL_FACTORS_SLACK_WEBHOOK_URL   - Slack incoming webhook URL
    GL_FACTORS_SLACK_CHANNEL       - Slack channel override (optional)
    GL_FACTORS_SMTP_HOST           - SMTP hostname for email notifications
    GL_FACTORS_SMTP_PORT           - SMTP port (default 587)
    GL_FACTORS_SMTP_USER           - SMTP auth username
    GL_FACTORS_SMTP_PASSWORD       - SMTP auth password
    GL_FACTORS_NOTIFY_EMAIL_FROM   - Sender email
    GL_FACTORS_NOTIFY_EMAIL_TO     - Comma-separated recipient emails
    GL_FACTORS_NOTIFICATIONS_ENABLED - Feature flag (default true)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NotificationChannel(str, Enum):
    SLACK = "slack"
    EMAIL = "email"


class NotificationEventType(str, Enum):
    WATCH_CHANGE = "watch_change"
    WATCH_ERROR = "watch_error"
    RELEASE_READY = "release_ready"
    RELEASE_PUBLISHED = "release_published"
    POLICY_CHANGE = "policy_change"


@dataclass
class NotificationRoute:
    """Maps event types to notification channels and recipients."""

    event_type: NotificationEventType
    channels: List[NotificationChannel]
    email_recipients: List[str] = field(default_factory=list)
    slack_channel: Optional[str] = None


@dataclass
class NotificationConfig:
    """Centralized notification settings loaded from environment."""

    enabled: bool = True

    # Slack settings
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None

    # Email settings
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: List[str] = field(default_factory=list)

    # Routing
    routes: List[NotificationRoute] = field(default_factory=list)

    @property
    def slack_configured(self) -> bool:
        return bool(self.slack_webhook_url)

    @property
    def email_configured(self) -> bool:
        return bool(self.smtp_host and self.email_from and self.email_to)

    @classmethod
    def from_env(cls) -> NotificationConfig:
        """Load notification config from environment variables."""
        enabled = os.environ.get("GL_FACTORS_NOTIFICATIONS_ENABLED", "true").lower() in (
            "true", "1", "yes",
        )
        email_to_raw = os.environ.get("GL_FACTORS_NOTIFY_EMAIL_TO", "")
        email_to = [e.strip() for e in email_to_raw.split(",") if e.strip()]

        config = cls(
            enabled=enabled,
            slack_webhook_url=os.environ.get("GL_FACTORS_SLACK_WEBHOOK_URL"),
            slack_channel=os.environ.get("GL_FACTORS_SLACK_CHANNEL"),
            smtp_host=os.environ.get("GL_FACTORS_SMTP_HOST"),
            smtp_port=int(os.environ.get("GL_FACTORS_SMTP_PORT", "587")),
            smtp_user=os.environ.get("GL_FACTORS_SMTP_USER"),
            smtp_password=os.environ.get("GL_FACTORS_SMTP_PASSWORD"),
            email_from=os.environ.get("GL_FACTORS_NOTIFY_EMAIL_FROM"),
            email_to=email_to,
        )

        # Default routing: all events to all configured channels
        config.routes = _default_routes(config)
        logger.debug(
            "Notification config: enabled=%s slack=%s email=%s routes=%d",
            config.enabled, config.slack_configured, config.email_configured, len(config.routes),
        )
        return config

    def routes_for_event(self, event_type: NotificationEventType) -> List[NotificationRoute]:
        return [r for r in self.routes if r.event_type == event_type]


def _default_routes(config: NotificationConfig) -> List[NotificationRoute]:
    """Build default notification routing from available channels."""
    routes: List[NotificationRoute] = []
    channels: List[NotificationChannel] = []
    if config.slack_configured:
        channels.append(NotificationChannel.SLACK)
    if config.email_configured:
        channels.append(NotificationChannel.EMAIL)
    if not channels:
        return routes

    for event_type in NotificationEventType:
        routes.append(
            NotificationRoute(
                event_type=event_type,
                channels=list(channels),
                email_recipients=list(config.email_to),
                slack_channel=config.slack_channel,
            )
        )
    return routes
