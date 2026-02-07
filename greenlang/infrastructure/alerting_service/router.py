# -*- coding: utf-8 -*-
"""
Alert Router - OBS-004: Unified Alerting Service

Determines which notification channels should receive a given alert and
dispatches notifications through the channel registry. Routing rules are
evaluated in priority order:

1. Explicit override via ``alert.labels["routing.channel"]``
2. Team-based routing rules
3. Service-based routing rules
4. Severity-based default routing (from config)
5. Time-based routing (business-hours vs off-hours)

Example:
    >>> router = AlertRouter(config, channel_registry)
    >>> channels = router.route(alert)
    >>> results = await router.notify(alert, channels, template_engine)

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.alerting_service.channels import ChannelRegistry
from greenlang.infrastructure.alerting_service.config import AlertingConfig
from greenlang.infrastructure.alerting_service.metrics import (
    record_notification,
    record_notification_failure,
)
from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    NotificationResult,
    NotificationStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate Limiter (simple sliding window)
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Token-bucket style rate limiter per channel."""

    def __init__(
        self,
        global_limit: int = 120,
        per_channel_limit: int = 60,
    ) -> None:
        self._global_limit = global_limit
        self._per_channel_limit = per_channel_limit
        self._global_window: List[float] = []
        self._channel_windows: Dict[str, List[float]] = defaultdict(list)

    def allow(self, channel: str) -> bool:
        """Check if a notification is allowed under rate limits.

        Args:
            channel: Channel name.

        Returns:
            True if under limits.
        """
        now = time.time()
        cutoff = now - 60.0

        # Global window
        self._global_window = [
            t for t in self._global_window if t > cutoff
        ]
        if len(self._global_window) >= self._global_limit:
            logger.warning("Global rate limit reached: %d/min", self._global_limit)
            return False

        # Per-channel window
        self._channel_windows[channel] = [
            t for t in self._channel_windows[channel] if t > cutoff
        ]
        if len(self._channel_windows[channel]) >= self._per_channel_limit:
            logger.warning(
                "Channel rate limit reached: channel=%s, %d/min",
                channel, self._per_channel_limit,
            )
            return False

        self._global_window.append(now)
        self._channel_windows[channel].append(now)
        return True


# ---------------------------------------------------------------------------
# AlertRouter
# ---------------------------------------------------------------------------


class AlertRouter:
    """Route alerts to notification channels and dispatch deliveries.

    Attributes:
        config: AlertingConfig instance.
        channel_registry: Registry of active channels.
    """

    def __init__(
        self,
        config: AlertingConfig,
        channel_registry: ChannelRegistry,
    ) -> None:
        self.config = config
        self.channel_registry = channel_registry
        self._team_routes: Dict[str, List[str]] = {}
        self._service_routes: Dict[str, List[str]] = {}
        self._rate_limiter = _RateLimiter(
            global_limit=config.rate_limit_per_minute,
            per_channel_limit=config.rate_limit_per_channel_per_minute,
        )
        logger.info("AlertRouter initialized")

    # ------------------------------------------------------------------
    # Route configuration
    # ------------------------------------------------------------------

    def add_team_route(self, team: str, channels: List[str]) -> None:
        """Add a team-specific routing rule.

        Args:
            team: Team name.
            channels: Channel names to route to.
        """
        self._team_routes[team] = channels
        logger.info("Team route added: %s -> %s", team, channels)

    def add_service_route(self, service: str, channels: List[str]) -> None:
        """Add a service-specific routing rule.

        Args:
            service: Service name.
            channels: Channel names to route to.
        """
        self._service_routes[service] = channels
        logger.info("Service route added: %s -> %s", service, channels)

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    def route(self, alert: Alert) -> List[str]:
        """Determine which channels should receive the alert.

        Evaluation order (first match wins):
          1. Explicit ``routing.channel`` label
          2. Team routing rules
          3. Service routing rules
          4. Severity-based default routing
          5. Business-hours adjustment

        Args:
            alert: The alert to route.

        Returns:
            List of channel names.
        """
        # 1. Explicit override
        explicit = alert.labels.get("routing.channel", "")
        if explicit:
            channels = [c.strip() for c in explicit.split(",") if c.strip()]
            logger.debug("Explicit routing: %s -> %s", alert.alert_id[:8], channels)
            return channels

        # 2. Team routing
        if alert.team and alert.team in self._team_routes:
            channels = self._team_routes[alert.team]
            logger.debug("Team routing: %s -> %s", alert.team, channels)
            return channels

        # 3. Service routing
        if alert.service and alert.service in self._service_routes:
            channels = self._service_routes[alert.service]
            logger.debug("Service routing: %s -> %s", alert.service, channels)
            return channels

        # 4. Severity-based default
        severity_key = alert.severity.value
        channels = list(
            self.config.default_severity_routing.get(severity_key, ["email"])
        )

        # 5. Time-based adjustment: off-hours critical -> add pagerduty
        if not self._is_business_hours() and alert.severity == AlertSeverity.CRITICAL:
            if "pagerduty" not in channels:
                channels.append("pagerduty")

        logger.debug(
            "Default routing: severity=%s -> %s", severity_key, channels,
        )
        return channels

    # ------------------------------------------------------------------
    # Notification dispatch
    # ------------------------------------------------------------------

    async def notify(
        self,
        alert: Alert,
        channels: List[str],
        template_engine: Any = None,
    ) -> List[NotificationResult]:
        """Send notifications to the resolved channels.

        Args:
            alert: The alert to notify about.
            channels: Channel names from ``route()``.
            template_engine: Optional TemplateEngine for rendering.

        Returns:
            List of NotificationResults (one per channel).
        """
        results: List[NotificationResult] = []

        # Render message once
        rendered = alert.description
        if template_engine is not None:
            try:
                rendered = template_engine.render("firing", alert, "default")
            except Exception as exc:
                logger.warning("Template render failed: %s", exc)

        for channel_name in channels:
            channel = self.channel_registry.get(channel_name)
            if channel is None:
                logger.warning(
                    "Channel not found in registry: %s", channel_name,
                )
                continue
            if not channel.enabled:
                logger.debug("Channel disabled, skipping: %s", channel_name)
                continue

            # Rate limiting
            if not self._rate_limiter.allow(channel_name):
                result = NotificationResult(
                    channel=_to_notification_channel(channel_name),
                    status=NotificationStatus.RATE_LIMITED,
                    recipient=channel_name,
                    error_message="Rate limited",
                )
                record_notification(
                    channel_name,
                    alert.severity.value,
                    "rate_limited",
                )
                results.append(result)
                continue

            try:
                result = await channel.send(alert, rendered)
                record_notification(
                    channel_name,
                    alert.severity.value,
                    result.status.value,
                    result.duration_ms / 1000.0,
                )
                if result.status == NotificationStatus.FAILED:
                    record_notification_failure(
                        channel_name,
                        result.error_message[:50] if result.error_message else "unknown",
                    )

            except Exception as exc:
                logger.error(
                    "Channel %s send exception: %s", channel_name, exc,
                )
                from greenlang.infrastructure.alerting_service.models import (
                    NotificationChannel,
                )
                result = NotificationResult(
                    channel=_to_notification_channel(channel_name),
                    status=NotificationStatus.FAILED,
                    error_message=str(exc),
                )
                record_notification(
                    channel_name, alert.severity.value, "failed",
                )
                record_notification_failure(channel_name, "exception")

            results.append(result)

        alert.notification_count += 1
        logger.info(
            "Notifications dispatched: alert=%s, channels=%d, sent=%d, failed=%d",
            alert.alert_id[:8],
            len(channels),
            sum(1 for r in results if r.status == NotificationStatus.SENT),
            sum(1 for r in results if r.status == NotificationStatus.FAILED),
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_business_hours() -> bool:
        """Check if current UTC time is within business hours.

        Business hours: Monday-Friday 09:00-18:00 UTC.

        Returns:
            True if within business hours.
        """
        now = datetime.now(timezone.utc)
        # Monday=0, Sunday=6
        if now.weekday() >= 5:
            return False
        return 9 <= now.hour < 18


def _to_notification_channel(name: str) -> Any:
    """Convert channel name string to NotificationChannel enum."""
    from greenlang.infrastructure.alerting_service.models import NotificationChannel

    try:
        return NotificationChannel(name)
    except ValueError:
        return NotificationChannel.WEBHOOK
