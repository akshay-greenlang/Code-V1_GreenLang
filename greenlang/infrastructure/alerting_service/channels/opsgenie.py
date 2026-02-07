# -*- coding: utf-8 -*-
"""
Opsgenie Notification Channel - OBS-004: Unified Alerting Service

Integrates with Opsgenie Alert API v2 for alert creation, acknowledgement,
closure, note attachment, and on-call schedule lookups.

Reference:
    https://docs.opsgenie.com/docs/alert-api

Example:
    >>> channel = OpsgenieChannel(
    ...     api_key="xxx", api_url="https://api.opsgenie.com",
    ...     default_team="platform",
    ... )
    >>> result = await channel.send(alert, "CPU above 90%")

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
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
    logger.debug("httpx not installed; OpsgenieChannel will not function")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRIORITY_MAP: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: "P1",
    AlertSeverity.WARNING: "P3",
    AlertSeverity.INFO: "P5",
}


# ---------------------------------------------------------------------------
# OpsgenieChannel
# ---------------------------------------------------------------------------


class OpsgenieChannel(BaseNotificationChannel):
    """Opsgenie Alert API v2 notification channel.

    Attributes:
        api_key: GenieKey for API authentication.
        api_url: Opsgenie API base URL (EU or US).
        default_team: Default responder team name.
    """

    name = "opsgenie"

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.opsgenie.com",
        default_team: str = "",
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.default_team = default_team
        self.enabled = bool(api_key) and HTTPX_AVAILABLE

    # ------------------------------------------------------------------
    # Headers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        """Return standard API request headers."""
        return {
            "Authorization": f"GenieKey {self.api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # BaseNotificationChannel interface
    # ------------------------------------------------------------------

    async def send(
        self,
        alert: Alert,
        rendered_message: str,
    ) -> NotificationResult:
        """Create an Opsgenie alert.

        Args:
            alert: The alert to deliver.
            rendered_message: Pre-rendered description text.

        Returns:
            NotificationResult.
        """
        if not HTTPX_AVAILABLE:
            return self._make_result(
                NotificationStatus.FAILED,
                error_message="httpx not installed",
            )

        tags = [alert.severity.value, alert.environment]
        if alert.team:
            tags.append(f"team:{alert.team}")
        if alert.service:
            tags.append(f"service:{alert.service}")

        responders: List[Dict[str, str]] = []
        team = alert.team or self.default_team
        if team:
            responders.append({"type": "team", "name": team})

        details: Dict[str, str] = {
            "alert_id": alert.alert_id,
            "source": alert.source,
            "environment": alert.environment,
            "fingerprint": alert.fingerprint,
        }
        if alert.runbook_url:
            details["runbook_url"] = alert.runbook_url
        if alert.dashboard_url:
            details["dashboard_url"] = alert.dashboard_url
        if alert.related_trace_id:
            details["trace_id"] = alert.related_trace_id
        details.update(alert.labels)

        payload: Dict[str, Any] = {
            "message": alert.title[:130],
            "alias": alert.fingerprint,
            "description": rendered_message[:15000],
            "priority": PRIORITY_MAP.get(alert.severity, "P5"),
            "tags": tags,
            "details": details,
            "source": f"greenlang-{alert.environment}",
            "entity": alert.service,
        }
        if responders:
            payload["responders"] = responders

        start = self._timed()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.api_url}/v2/alerts",
                    json=payload,
                    headers=self._headers(),
                )

            duration_ms = (self._timed() - start) * 1000

            if resp.status_code in (200, 202):
                logger.info(
                    "Opsgenie alert created: alert_id=%s, alias=%s",
                    alert.alert_id[:8], alert.fingerprint[:12],
                )
                return self._make_result(
                    NotificationStatus.SENT,
                    recipient=team or "opsgenie",
                    duration_ms=duration_ms,
                    response_code=resp.status_code,
                )

            logger.warning(
                "Opsgenie create failed: status=%d, body=%s",
                resp.status_code, resp.text[:200],
            )
            return self._make_result(
                NotificationStatus.FAILED,
                recipient=team or "opsgenie",
                duration_ms=duration_ms,
                response_code=resp.status_code,
                error_message=resp.text[:200],
            )

        except Exception as exc:
            duration_ms = (self._timed() - start) * 1000
            logger.error("Opsgenie send error: %s", exc)
            return self._make_result(
                NotificationStatus.FAILED,
                duration_ms=duration_ms,
                error_message=str(exc),
            )

    async def acknowledge(self, alert_id: str, user: str = "") -> NotificationResult:
        """Acknowledge an Opsgenie alert.

        Args:
            alert_id: Opsgenie alert alias (fingerprint).
            user: User performing ack.

        Returns:
            NotificationResult.
        """
        payload: Dict[str, Any] = {
            "source": "greenlang-alerting",
            "note": f"Acknowledged by {user}" if user else "Acknowledged via API",
        }
        if user:
            payload["user"] = user

        return await self._post_action(
            f"/v2/alerts/{alert_id}/acknowledge",
            payload,
            "acknowledge",
        )

    async def close(self, alert_id: str, user: str = "") -> NotificationResult:
        """Close an Opsgenie alert.

        Args:
            alert_id: Opsgenie alert alias.
            user: User closing.

        Returns:
            NotificationResult.
        """
        payload: Dict[str, Any] = {
            "source": "greenlang-alerting",
            "note": f"Closed by {user}" if user else "Closed via API",
        }
        if user:
            payload["user"] = user

        return await self._post_action(
            f"/v2/alerts/{alert_id}/close", payload, "close",
        )

    async def resolve(self, alert_id: str) -> NotificationResult:
        """Resolve (close) an Opsgenie alert.

        Args:
            alert_id: Alert alias to resolve.

        Returns:
            NotificationResult.
        """
        return await self.close(alert_id, user="greenlang-system")

    async def add_note(self, alert_id: str, note: str) -> NotificationResult:
        """Add a note to an Opsgenie alert.

        Args:
            alert_id: Alert alias.
            note: Note text.

        Returns:
            NotificationResult.
        """
        payload = {
            "note": note[:25000],
            "source": "greenlang-alerting",
        }
        return await self._post_action(
            f"/v2/alerts/{alert_id}/notes", payload, "add_note",
        )

    async def get_oncall(self, schedule_id: str) -> Optional[OnCallUser]:
        """Get the current on-call user from an Opsgenie schedule.

        Args:
            schedule_id: Opsgenie schedule ID or name.

        Returns:
            OnCallUser or None.
        """
        if not HTTPX_AVAILABLE:
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.api_url}/v2/schedules/{schedule_id}/on-calls",
                    headers=self._headers(),
                )

            if resp.status_code != 200:
                logger.warning(
                    "OG on-call lookup failed: schedule=%s, status=%d",
                    schedule_id, resp.status_code,
                )
                return None

            data = resp.json()
            on_call_data = data.get("data", {})
            participants = on_call_data.get("onCallParticipants", [])
            if not participants:
                return None

            user_data = participants[0]
            return OnCallUser(
                user_id=user_data.get("id", ""),
                name=user_data.get("name", ""),
                email=user_data.get("name", ""),  # OG uses name as identifier
                provider="opsgenie",
                schedule_id=schedule_id,
            )

        except Exception as exc:
            logger.error("OG on-call lookup error: %s", exc)
            return None

    async def health_check(self) -> bool:
        """Verify Opsgenie API connectivity.

        Returns:
            True if the API is reachable.
        """
        if not HTTPX_AVAILABLE or not self.api_key:
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self.api_url}/v2/heartbeats",
                    headers=self._headers(),
                )
            return resp.status_code in (200, 404)  # 404 = no heartbeats configured
        except Exception as exc:
            logger.warning("Opsgenie health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _post_action(
        self,
        path: str,
        payload: Dict[str, Any],
        action_name: str,
    ) -> NotificationResult:
        """Execute a POST action against the Opsgenie API.

        Args:
            path: API path (e.g. ``/v2/alerts/{id}/acknowledge``).
            payload: JSON body.
            action_name: Human-readable action name for logging.

        Returns:
            NotificationResult.
        """
        if not HTTPX_AVAILABLE:
            return self._make_result(
                NotificationStatus.FAILED,
                error_message="httpx not installed",
            )

        start = self._timed()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.api_url}{path}",
                    json=payload,
                    headers=self._headers(),
                )

            duration_ms = (self._timed() - start) * 1000

            if resp.status_code in (200, 202):
                logger.info("Opsgenie %s sent: path=%s", action_name, path)
                return self._make_result(
                    NotificationStatus.SENT,
                    recipient="opsgenie",
                    duration_ms=duration_ms,
                    response_code=resp.status_code,
                )

            logger.warning(
                "Opsgenie %s failed: status=%d, body=%s",
                action_name, resp.status_code, resp.text[:200],
            )
            return self._make_result(
                NotificationStatus.FAILED,
                duration_ms=duration_ms,
                response_code=resp.status_code,
                error_message=resp.text[:200],
            )

        except Exception as exc:
            duration_ms = (self._timed() - start) * 1000
            logger.error("Opsgenie %s error: %s", action_name, exc)
            return self._make_result(
                NotificationStatus.FAILED,
                duration_ms=duration_ms,
                error_message=str(exc),
            )
