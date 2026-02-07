# -*- coding: utf-8 -*-
"""
PagerDuty Notification Channel - OBS-004: Unified Alerting Service

Integrates with PagerDuty Events API v2 for incident triggering,
acknowledgement, and resolution.  Also supports REST API v2 for
on-call schedule lookups.

Reference:
    https://developer.pagerduty.com/api-reference/send-an-event

Example:
    >>> channel = PagerDutyChannel(
    ...     routing_key="R0xxx", api_key="u+xxx", service_id="P0xxx",
    ... )
    >>> result = await channel.send(alert, "CPU above 90%")

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.alerting_service.channels.base import (
    BaseNotificationChannel,
)
from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    NotificationResult,
    NotificationStatus,
    OnCallUser,
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
    logger.debug("httpx not installed; PagerDutyChannel will not function")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PD_EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"
PD_API_BASE = "https://api.pagerduty.com"

SEVERITY_MAP: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: "critical",
    AlertSeverity.WARNING: "warning",
    AlertSeverity.INFO: "info",
}


# ---------------------------------------------------------------------------
# PagerDutyChannel
# ---------------------------------------------------------------------------


class PagerDutyChannel(BaseNotificationChannel):
    """PagerDuty Events API v2 notification channel.

    Supports trigger, acknowledge, and resolve event actions.  On-call
    lookups use the PagerDuty REST API v2.

    Attributes:
        routing_key: PD Events API v2 integration routing key.
        api_key: PD REST API v2 token (for on-call lookups).
        service_id: PD service identifier.
    """

    name = "pagerduty"

    def __init__(
        self,
        routing_key: str,
        api_key: str = "",
        service_id: str = "",
    ) -> None:
        """Initialize the PagerDuty channel.

        Args:
            routing_key: Events API v2 routing key.
            api_key: REST API v2 token for on-call lookups.
            service_id: PD service ID for context enrichment.
        """
        self.routing_key = routing_key
        self.api_key = api_key
        self.service_id = service_id
        self.enabled = bool(routing_key) and HTTPX_AVAILABLE

    # ------------------------------------------------------------------
    # BaseNotificationChannel interface
    # ------------------------------------------------------------------

    async def send(
        self,
        alert: Alert,
        rendered_message: str,
    ) -> NotificationResult:
        """Trigger a PagerDuty incident via Events API v2.

        Args:
            alert: The alert triggering the incident.
            rendered_message: Pre-rendered description text.

        Returns:
            NotificationResult with delivery outcome.
        """
        if not HTTPX_AVAILABLE:
            return self._make_result(
                NotificationStatus.FAILED,
                error_message="httpx not installed",
            )

        payload = self._build_payload(alert, rendered_message)
        start = self._timed()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(PD_EVENTS_URL, json=payload)

            duration_ms = (self._timed() - start) * 1000

            if resp.status_code in (200, 202):
                logger.info(
                    "PagerDuty trigger sent: alert_id=%s, dedup=%s",
                    alert.alert_id[:8], alert.fingerprint[:12],
                )
                return self._make_result(
                    NotificationStatus.SENT,
                    recipient=self.service_id or "pagerduty",
                    duration_ms=duration_ms,
                    response_code=resp.status_code,
                )

            logger.warning(
                "PagerDuty trigger failed: status=%d, body=%s",
                resp.status_code, resp.text[:200],
            )
            return self._make_result(
                NotificationStatus.FAILED,
                recipient=self.service_id or "pagerduty",
                duration_ms=duration_ms,
                response_code=resp.status_code,
                error_message=resp.text[:200],
            )

        except Exception as exc:
            duration_ms = (self._timed() - start) * 1000
            logger.error("PagerDuty send error: %s", exc)
            return self._make_result(
                NotificationStatus.FAILED,
                duration_ms=duration_ms,
                error_message=str(exc),
            )

    async def acknowledge(self, alert_id: str) -> NotificationResult:
        """Send an acknowledge event to PagerDuty.

        Args:
            alert_id: Dedup key (fingerprint) to acknowledge.

        Returns:
            NotificationResult.
        """
        return await self._send_event(alert_id, "acknowledge")

    async def resolve(self, alert_id: str) -> NotificationResult:
        """Send a resolve event to PagerDuty.

        Args:
            alert_id: Dedup key (fingerprint) to resolve.

        Returns:
            NotificationResult.
        """
        return await self._send_event(alert_id, "resolve")

    async def health_check(self) -> bool:
        """Verify PagerDuty REST API connectivity.

        Returns:
            True if the API is reachable.
        """
        if not HTTPX_AVAILABLE or not self.api_key:
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{PD_API_BASE}/abilities",
                    headers={
                        "Authorization": f"Token token={self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("PagerDuty health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # On-call lookup
    # ------------------------------------------------------------------

    async def get_oncall(self, schedule_id: str) -> Optional[OnCallUser]:
        """Retrieve the current on-call user for a PagerDuty schedule.

        Args:
            schedule_id: PD schedule ID.

        Returns:
            OnCallUser or None on failure.
        """
        if not HTTPX_AVAILABLE or not self.api_key:
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{PD_API_BASE}/schedules/{schedule_id}/users",
                    headers={
                        "Authorization": f"Token token={self.api_key}",
                        "Content-Type": "application/json",
                    },
                )

            if resp.status_code != 200:
                logger.warning(
                    "PD on-call lookup failed: schedule=%s, status=%d",
                    schedule_id, resp.status_code,
                )
                return None

            data = resp.json()
            users = data.get("users", [])
            if not users:
                return None

            user = users[0]
            return OnCallUser(
                user_id=user.get("id", ""),
                name=user.get("name", ""),
                email=user.get("email", ""),
                phone=user.get("contact_methods", [{}])[0].get("address", "")
                if user.get("contact_methods")
                else "",
                provider="pagerduty",
                schedule_id=schedule_id,
            )

        except Exception as exc:
            logger.error("PD on-call lookup error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        alert: Alert,
        rendered_message: str,
    ) -> Dict[str, Any]:
        """Build the Events API v2 trigger payload.

        Args:
            alert: Source alert.
            rendered_message: Pre-rendered description.

        Returns:
            JSON-serializable payload dict.
        """
        custom_details: Dict[str, Any] = {
            "alert_id": alert.alert_id,
            "team": alert.team,
            "service": alert.service,
            "environment": alert.environment,
            "labels": alert.labels,
        }
        if alert.runbook_url:
            custom_details["runbook_url"] = alert.runbook_url
        if alert.dashboard_url:
            custom_details["dashboard_url"] = alert.dashboard_url
        if alert.related_trace_id:
            custom_details["trace_id"] = alert.related_trace_id

        return {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "dedup_key": alert.fingerprint,
            "payload": {
                "summary": alert.title[:1024],
                "source": alert.source,
                "severity": SEVERITY_MAP.get(alert.severity, "info"),
                "timestamp": (
                    alert.fired_at.isoformat() if alert.fired_at else ""
                ),
                "component": alert.service,
                "group": alert.team,
                "class": alert.name,
                "custom_details": custom_details,
            },
            "links": [
                {"href": url, "text": label}
                for label, url in [
                    ("Runbook", alert.runbook_url),
                    ("Dashboard", alert.dashboard_url),
                ]
                if url
            ],
        }

    async def _send_event(
        self,
        dedup_key: str,
        action: str,
    ) -> NotificationResult:
        """Send an acknowledge or resolve event.

        Args:
            dedup_key: PD dedup key.
            action: ``acknowledge`` or ``resolve``.

        Returns:
            NotificationResult.
        """
        if not HTTPX_AVAILABLE:
            return self._make_result(
                NotificationStatus.FAILED,
                error_message="httpx not installed",
            )

        payload = {
            "routing_key": self.routing_key,
            "event_action": action,
            "dedup_key": dedup_key,
        }
        start = self._timed()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(PD_EVENTS_URL, json=payload)

            duration_ms = (self._timed() - start) * 1000

            if resp.status_code in (200, 202):
                logger.info("PagerDuty %s sent: dedup=%s", action, dedup_key[:12])
                return self._make_result(
                    NotificationStatus.SENT,
                    recipient="pagerduty",
                    duration_ms=duration_ms,
                    response_code=resp.status_code,
                )

            return self._make_result(
                NotificationStatus.FAILED,
                duration_ms=duration_ms,
                response_code=resp.status_code,
                error_message=resp.text[:200],
            )

        except Exception as exc:
            duration_ms = (self._timed() - start) * 1000
            logger.error("PagerDuty %s error: %s", action, exc)
            return self._make_result(
                NotificationStatus.FAILED,
                duration_ms=duration_ms,
                error_message=str(exc),
            )
