# -*- coding: utf-8 -*-
"""
Notification Channel Registry - OBS-004: Unified Alerting Service

Central registry for notification channels. The ``create_channels()``
factory reads the AlertingConfig and instantiates only the channels that
are enabled, returning a populated ChannelRegistry.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from greenlang.infrastructure.alerting_service.channels.base import (
    BaseNotificationChannel,
)
from greenlang.infrastructure.alerting_service.config import AlertingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ChannelRegistry
# ---------------------------------------------------------------------------


class ChannelRegistry:
    """Thread-safe registry mapping channel names to channel instances.

    Attributes:
        _channels: Internal mapping of name -> channel.
    """

    def __init__(self) -> None:
        self._channels: Dict[str, BaseNotificationChannel] = {}

    def register(self, name: str, channel: BaseNotificationChannel) -> None:
        """Register a channel by name.

        Args:
            name: Machine-readable channel name.
            channel: Channel implementation.
        """
        self._channels[name] = channel
        logger.info("Channel registered: %s (enabled=%s)", name, channel.enabled)

    def get(self, name: str) -> Optional[BaseNotificationChannel]:
        """Retrieve a channel by name.

        Args:
            name: Channel name.

        Returns:
            Channel instance or None.
        """
        return self._channels.get(name)

    def list_channels(self) -> List[str]:
        """List all registered channel names.

        Returns:
            Sorted list of channel names.
        """
        return sorted(self._channels.keys())

    async def get_healthy_channels(self) -> List[str]:
        """Return names of channels that pass their health check.

        Returns:
            List of healthy channel names.
        """
        healthy: List[str] = []
        for name, channel in self._channels.items():
            try:
                if await channel.health_check():
                    healthy.append(name)
            except Exception:
                logger.warning("Health check failed for channel: %s", name)
        return healthy


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_channels(config: AlertingConfig) -> ChannelRegistry:
    """Instantiate and register all enabled notification channels.

    Args:
        config: AlertingConfig with per-channel credentials.

    Returns:
        Populated ChannelRegistry.
    """
    registry = ChannelRegistry()

    if config.pagerduty_enabled:
        from greenlang.infrastructure.alerting_service.channels.pagerduty import (
            PagerDutyChannel,
        )
        registry.register(
            "pagerduty",
            PagerDutyChannel(
                routing_key=config.pagerduty_routing_key,
                api_key=config.pagerduty_api_key,
                service_id=config.pagerduty_service_id,
            ),
        )

    if config.opsgenie_enabled:
        from greenlang.infrastructure.alerting_service.channels.opsgenie import (
            OpsgenieChannel,
        )
        registry.register(
            "opsgenie",
            OpsgenieChannel(
                api_key=config.opsgenie_api_key,
                api_url=config.opsgenie_api_url,
                default_team=config.opsgenie_team,
            ),
        )

    if config.slack_enabled:
        from greenlang.infrastructure.alerting_service.channels.slack import (
            SlackChannel,
        )
        registry.register(
            "slack",
            SlackChannel(
                webhook_critical=config.slack_webhook_critical,
                webhook_warning=config.slack_webhook_warning,
                webhook_info=config.slack_webhook_info,
            ),
        )

    if config.email_enabled:
        from greenlang.infrastructure.alerting_service.channels.email import (
            EmailChannel,
        )
        registry.register(
            "email",
            EmailChannel(
                from_addr=config.email_from,
                smtp_host=config.email_smtp_host,
                smtp_port=config.email_smtp_port,
                use_ses=config.email_use_ses,
                ses_region=config.email_ses_region,
            ),
        )

    if config.teams_enabled:
        from greenlang.infrastructure.alerting_service.channels.teams import (
            TeamsChannel,
        )
        registry.register(
            "teams",
            TeamsChannel(webhook_url=config.teams_webhook_url),
        )

    if config.webhook_enabled:
        from greenlang.infrastructure.alerting_service.channels.webhook import (
            WebhookChannel,
        )
        registry.register(
            "webhook",
            WebhookChannel(
                url=config.webhook_url,
                secret=config.webhook_secret,
            ),
        )

    logger.info(
        "Channel registry created: %d channels registered (%s)",
        len(registry.list_channels()),
        ", ".join(registry.list_channels()),
    )
    return registry


__all__ = [
    "BaseNotificationChannel",
    "ChannelRegistry",
    "create_channels",
]
