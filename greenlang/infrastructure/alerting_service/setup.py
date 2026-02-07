# -*- coding: utf-8 -*-
"""
Alerting Service Setup - OBS-004: Unified Alerting Service

Provides ``configure_alerting(app)`` which wires up the full alerting
pipeline (channels, lifecycle, dedup, router, escalation, on-call,
analytics, template engine) and mounts the REST API.  Also exposes
``get_alerting_service(app)`` for programmatic access.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.alerting_service.setup import configure_alerting
    >>> app = FastAPI()
    >>> configure_alerting(app)

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.alerting_service.analytics import AlertAnalytics
from greenlang.infrastructure.alerting_service.channels import (
    ChannelRegistry,
    create_channels,
)
from greenlang.infrastructure.alerting_service.config import (
    AlertingConfig,
    get_config,
)
from greenlang.infrastructure.alerting_service.deduplication import AlertDeduplicator
from greenlang.infrastructure.alerting_service.escalation import EscalationEngine
from greenlang.infrastructure.alerting_service.lifecycle import AlertLifecycle
from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
)
from greenlang.infrastructure.alerting_service.oncall import OnCallManager
from greenlang.infrastructure.alerting_service.router import AlertRouter
from greenlang.infrastructure.alerting_service.templates.engine import TemplateEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI

    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# AlertingService facade
# ---------------------------------------------------------------------------


class AlertingService:
    """Unified facade over the complete alerting pipeline.

    Orchestrates deduplication, lifecycle management, routing,
    escalation, analytics, and on-call coordination.

    Attributes:
        config: AlertingConfig instance.
        lifecycle: Alert lifecycle state machine.
        dedup: Alert deduplicator.
        router: Alert notification router.
        escalation: Escalation engine.
        oncall: On-call manager.
        analytics: Alert analytics engine.
        template_engine: Template rendering engine.
        channel_registry: Registered notification channels.
    """

    def __init__(
        self,
        config: AlertingConfig,
        lifecycle: AlertLifecycle,
        dedup: AlertDeduplicator,
        router: AlertRouter,
        escalation: EscalationEngine,
        oncall: OnCallManager,
        analytics: AlertAnalytics,
        template_engine: TemplateEngine,
        channel_registry: ChannelRegistry,
    ) -> None:
        self.config = config
        self.lifecycle = lifecycle
        self.dedup = dedup
        self.router = router
        self.escalation = escalation
        self.oncall = oncall
        self.analytics = analytics
        self.template_engine = template_engine
        self.channel_registry = channel_registry
        logger.info("AlertingService facade created")

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def fire_alert(self, alert_data: Dict[str, Any]) -> Alert:
        """Fire a new alert through the full pipeline.

        Steps:
          1. Create / deduplicate via lifecycle + dedup
          2. Record analytics
          3. Route to channels
          4. Dispatch notifications

        Args:
            alert_data: Alert constructor kwargs.

        Returns:
            The Alert instance (new or existing).
        """
        if not self.config.enabled:
            logger.debug("Alerting disabled; ignoring fire_alert")
            return Alert(
                source=alert_data.get("source", "unknown"),
                name=alert_data.get("name", "unknown"),
                severity=AlertSeverity(alert_data.get("severity", "info")),
                title=alert_data.get("title", ""),
            )

        # 1. Lifecycle
        alert = self.lifecycle.fire(alert_data)

        # 2. Dedup
        alert, is_new = self.dedup.process(alert)

        # 3. Analytics
        self.analytics.record_alert_fired(alert)

        # 4. Route + notify (only for new alerts or first dedup hit)
        if is_new or alert.notification_count <= 1:
            channels = self.router.route(alert)
            await self.router.notify(alert, channels, self.template_engine)

        return alert

    async def acknowledge_alert(self, alert_id: str, user: str) -> Alert:
        """Acknowledge an alert.

        Args:
            alert_id: Alert identifier.
            user: User performing ack.

        Returns:
            Updated Alert.
        """
        alert = self.lifecycle.acknowledge(alert_id, user)
        self.analytics.record_alert_acknowledged(alert)

        # Notify channels that support bidirectional ack
        for ch_name in ["pagerduty", "opsgenie"]:
            channel = self.channel_registry.get(ch_name)
            if channel and channel.enabled:
                try:
                    await channel.acknowledge(alert.fingerprint)
                except Exception as exc:
                    logger.warning(
                        "Channel ack failed: %s, error=%s", ch_name, exc,
                    )

        return alert

    async def resolve_alert(self, alert_id: str, user: str) -> Alert:
        """Resolve an alert.

        Args:
            alert_id: Alert identifier.
            user: User resolving.

        Returns:
            Updated Alert.
        """
        alert = self.lifecycle.resolve(alert_id, user)
        self.analytics.record_alert_resolved(alert)

        # Notify channels that support bidirectional resolve
        for ch_name in ["pagerduty", "opsgenie"]:
            channel = self.channel_registry.get(ch_name)
            if channel and channel.enabled:
                try:
                    await channel.resolve(alert.fingerprint)
                except Exception as exc:
                    logger.warning(
                        "Channel resolve failed: %s, error=%s", ch_name, exc,
                    )

        return alert

    async def escalate_alert(
        self,
        alert_id: str,
        reason: str = "",
    ) -> Optional[Alert]:
        """Manually escalate an alert.

        Args:
            alert_id: Alert identifier.
            reason: Escalation reason.

        Returns:
            Updated Alert or None.
        """
        return await self.escalation.escalate(alert_id, reason)

    async def suppress_alert(
        self,
        alert_id: str,
        duration_minutes: int,
        reason: str = "",
    ) -> Alert:
        """Suppress (snooze) an alert.

        Args:
            alert_id: Alert identifier.
            duration_minutes: Suppression duration.
            reason: Reason for suppression.

        Returns:
            Updated Alert.
        """
        return self.lifecycle.suppress(alert_id, duration_minutes, reason)

    async def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Retrieve an alert by ID.

        Args:
            alert_id: Alert identifier.

        Returns:
            Alert or None.
        """
        return self.lifecycle.get_alert(alert_id)

    async def list_alerts(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Alert]:
        """List alerts with optional filters.

        Args:
            filters: Dict with optional keys: status, severity, team,
                     service, limit, offset.

        Returns:
            Filtered list of alerts.
        """
        filters = filters or {}

        status = filters.get("status")
        if isinstance(status, str):
            try:
                status = AlertStatus(status)
            except ValueError:
                status = None

        severity = filters.get("severity")
        if isinstance(severity, str):
            try:
                severity = AlertSeverity(severity)
            except ValueError:
                severity = None

        return self.lifecycle.list_alerts(
            status=status,
            severity=severity,
            team=filters.get("team"),
            service=filters.get("service"),
            limit=filters.get("limit", 100),
            offset=filters.get("offset", 0),
        )


# ---------------------------------------------------------------------------
# Setup function
# ---------------------------------------------------------------------------


def configure_alerting(
    app: Any = None,
    config: Optional[AlertingConfig] = None,
) -> AlertingService:
    """Wire up the full alerting service and mount the API.

    Args:
        app: FastAPI application instance (optional).
        config: AlertingConfig (loaded from env if not provided).

    Returns:
        Configured AlertingService.
    """
    if config is None:
        config = get_config()

    # Build components
    channel_registry = create_channels(config)
    lifecycle = AlertLifecycle(config)
    dedup = AlertDeduplicator(window_minutes=config.dedup_window_minutes)
    oncall = OnCallManager(config)
    template_engine = TemplateEngine()
    analytics = AlertAnalytics(config)
    router = AlertRouter(config, channel_registry)
    escalation = EscalationEngine(config, lifecycle, router, oncall)

    # Build facade
    service = AlertingService(
        config=config,
        lifecycle=lifecycle,
        dedup=dedup,
        router=router,
        escalation=escalation,
        oncall=oncall,
        analytics=analytics,
        template_engine=template_engine,
        channel_registry=channel_registry,
    )

    # Mount on FastAPI app
    if app is not None and FASTAPI_AVAILABLE:
        try:
            from greenlang.infrastructure.alerting_service.api.router import (
                alerts_router,
            )
            if alerts_router is not None:
                app.include_router(alerts_router)
                logger.info("Alerting API router mounted at /api/v1/alerts")
        except ImportError:
            logger.warning("Could not import alerts_router")

        app.state.alerting_service = service

    # Log summary
    logger.info(
        "Alerting service configured: env=%s, channels=%s, "
        "escalation=%s, dedup_window=%dm",
        config.environment,
        channel_registry.list_channels(),
        config.escalation_enabled,
        config.dedup_window_minutes,
    )

    return service


def get_alerting_service(app: Any) -> AlertingService:
    """Retrieve the AlertingService from a FastAPI application.

    Args:
        app: FastAPI application.

    Returns:
        AlertingService instance.

    Raises:
        RuntimeError: If not configured.
    """
    svc = getattr(app.state, "alerting_service", None)
    if svc is None:
        raise RuntimeError(
            "AlertingService not configured. Call configure_alerting(app) first."
        )
    return svc
