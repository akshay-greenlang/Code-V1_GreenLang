# -*- coding: utf-8 -*-
"""
On-Call Manager - OBS-004: Unified Alerting Service

Provides a unified interface for querying on-call schedules across
PagerDuty and Opsgenie. Results are cached with a 5-minute TTL to
reduce upstream API calls.

Example:
    >>> oncall = OnCallManager(config)
    >>> user = await oncall.get_current_oncall("P_SCHED_1", "pagerduty")
    >>> print(user.name, user.email)

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.alerting_service.config import AlertingConfig
from greenlang.infrastructure.alerting_service.metrics import record_oncall_lookup
from greenlang.infrastructure.alerting_service.models import (
    OnCallSchedule,
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PD_API_BASE = "https://api.pagerduty.com"
CACHE_TTL_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# OnCallManager
# ---------------------------------------------------------------------------


class OnCallManager:
    """Unified on-call schedule manager with provider abstraction.

    Caches on-call lookups for 5 minutes to reduce API chatter.

    Attributes:
        config: AlertingConfig instance.
    """

    def __init__(self, config: AlertingConfig) -> None:
        self.config = config
        self._cache: Dict[str, OnCallUser] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        logger.info("OnCallManager initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_current_oncall(
        self,
        schedule_id: str,
        provider: str = "",
    ) -> Optional[OnCallUser]:
        """Get the current on-call user for a schedule.

        Args:
            schedule_id: Schedule identifier.
            provider: ``pagerduty`` or ``opsgenie``. Auto-detected if empty.

        Returns:
            OnCallUser or None on failure.
        """
        cache_key = f"{provider}:{schedule_id}"

        if self._is_cache_valid(cache_key):
            logger.debug("On-call cache hit: %s", cache_key)
            return self._cache.get(cache_key)

        resolved_provider = provider or self._detect_provider()
        user: Optional[OnCallUser] = None

        try:
            if resolved_provider == "pagerduty":
                user = await self._fetch_pagerduty_oncall(schedule_id)
            elif resolved_provider == "opsgenie":
                user = await self._fetch_opsgenie_oncall(schedule_id)
            else:
                logger.warning("Unknown on-call provider: %s", resolved_provider)
                record_oncall_lookup(resolved_provider, "error")
                return None

            if user is not None:
                self._cache[cache_key] = user
                self._cache_timestamps[cache_key] = datetime.now(timezone.utc)
                record_oncall_lookup(resolved_provider, "success")
            else:
                record_oncall_lookup(resolved_provider, "not_found")

        except Exception as exc:
            logger.error(
                "On-call lookup failed: provider=%s, schedule=%s, error=%s",
                resolved_provider, schedule_id, exc,
            )
            record_oncall_lookup(resolved_provider, "error")

        return user

    async def get_oncall_schedule(
        self,
        schedule_id: str,
        provider: str = "",
    ) -> Optional[OnCallSchedule]:
        """Get a full on-call schedule including current responder.

        Args:
            schedule_id: Schedule identifier.
            provider: ``pagerduty`` or ``opsgenie``.

        Returns:
            OnCallSchedule or None.
        """
        resolved_provider = provider or self._detect_provider()
        user = await self.get_current_oncall(schedule_id, resolved_provider)

        if user is None:
            return None

        return OnCallSchedule(
            schedule_id=schedule_id,
            name=f"{resolved_provider}:{schedule_id}",
            provider=resolved_provider,
            current_oncall=user,
        )

    async def list_schedules(
        self,
        provider: str = "",
    ) -> List[OnCallSchedule]:
        """List known on-call schedules.

        Currently returns schedules from cache only.

        Args:
            provider: Filter by provider.

        Returns:
            List of OnCallSchedule.
        """
        schedules: List[OnCallSchedule] = []
        for key, user in self._cache.items():
            prov, sched_id = key.split(":", 1)
            if provider and prov != provider:
                continue
            schedules.append(
                OnCallSchedule(
                    schedule_id=sched_id,
                    name=f"{prov}:{sched_id}",
                    provider=prov,
                    current_oncall=user,
                )
            )
        return schedules

    async def override_oncall(
        self,
        schedule_id: str,
        user_id: str,
        start: datetime,
        end: datetime,
        provider: str = "",
    ) -> Dict[str, Any]:
        """Create a temporary on-call override.

        Args:
            schedule_id: Schedule identifier.
            user_id: User to put on-call.
            start: Override start time.
            end: Override end time.
            provider: On-call provider.

        Returns:
            Provider response dict.
        """
        resolved_provider = provider or self._detect_provider()

        if not HTTPX_AVAILABLE:
            return {"error": "httpx not installed"}

        if resolved_provider == "pagerduty":
            return await self._pd_override(schedule_id, user_id, start, end)
        if resolved_provider == "opsgenie":
            return await self._og_override(schedule_id, user_id, start, end)

        return {"error": f"Unknown provider: {resolved_provider}"}

    # ------------------------------------------------------------------
    # PagerDuty
    # ------------------------------------------------------------------

    async def _fetch_pagerduty_oncall(
        self,
        schedule_id: str,
    ) -> Optional[OnCallUser]:
        """Fetch current on-call from PagerDuty REST API v2.

        Args:
            schedule_id: PD schedule ID.

        Returns:
            OnCallUser or None.
        """
        if not HTTPX_AVAILABLE or not self.config.pagerduty_api_key:
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{PD_API_BASE}/schedules/{schedule_id}/users",
                    headers={
                        "Authorization": f"Token token={self.config.pagerduty_api_key}",
                        "Content-Type": "application/json",
                    },
                )

            if resp.status_code != 200:
                logger.warning(
                    "PD on-call fetch failed: schedule=%s, status=%d",
                    schedule_id, resp.status_code,
                )
                return None

            users = resp.json().get("users", [])
            if not users:
                return None

            u = users[0]
            phone = ""
            contacts = u.get("contact_methods", [])
            for cm in contacts:
                if cm.get("type") == "phone_contact_method":
                    phone = cm.get("address", "")
                    break

            return OnCallUser(
                user_id=u.get("id", ""),
                name=u.get("name", ""),
                email=u.get("email", ""),
                phone=phone,
                provider="pagerduty",
                schedule_id=schedule_id,
            )

        except Exception as exc:
            logger.error("PD on-call fetch error: %s", exc)
            return None

    async def _pd_override(
        self,
        schedule_id: str,
        user_id: str,
        start: datetime,
        end: datetime,
    ) -> Dict[str, Any]:
        """Create a PagerDuty schedule override."""
        if not HTTPX_AVAILABLE or not self.config.pagerduty_api_key:
            return {"error": "PagerDuty not configured"}

        payload = {
            "override": {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "user": {"id": user_id, "type": "user_reference"},
            }
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{PD_API_BASE}/schedules/{schedule_id}/overrides",
                    json=payload,
                    headers={
                        "Authorization": f"Token token={self.config.pagerduty_api_key}",
                        "Content-Type": "application/json",
                    },
                )
            return resp.json()
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Opsgenie
    # ------------------------------------------------------------------

    async def _fetch_opsgenie_oncall(
        self,
        schedule_id: str,
    ) -> Optional[OnCallUser]:
        """Fetch current on-call from Opsgenie API v2.

        Args:
            schedule_id: OG schedule ID.

        Returns:
            OnCallUser or None.
        """
        if not HTTPX_AVAILABLE or not self.config.opsgenie_api_key:
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.config.opsgenie_api_url}/v2/schedules/{schedule_id}/on-calls",
                    headers={
                        "Authorization": f"GenieKey {self.config.opsgenie_api_key}",
                        "Content-Type": "application/json",
                    },
                )

            if resp.status_code != 200:
                return None

            data = resp.json().get("data", {})
            participants = data.get("onCallParticipants", [])
            if not participants:
                return None

            p = participants[0]
            return OnCallUser(
                user_id=p.get("id", ""),
                name=p.get("name", ""),
                email=p.get("name", ""),
                provider="opsgenie",
                schedule_id=schedule_id,
            )

        except Exception as exc:
            logger.error("OG on-call fetch error: %s", exc)
            return None

    async def _og_override(
        self,
        schedule_id: str,
        user_id: str,
        start: datetime,
        end: datetime,
    ) -> Dict[str, Any]:
        """Create an Opsgenie schedule override."""
        if not HTTPX_AVAILABLE or not self.config.opsgenie_api_key:
            return {"error": "Opsgenie not configured"}

        payload = {
            "user": {"type": "user", "id": user_id},
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.config.opsgenie_api_url}/v2/schedules/{schedule_id}/overrides",
                    json=payload,
                    headers={
                        "Authorization": f"GenieKey {self.config.opsgenie_api_key}",
                        "Content-Type": "application/json",
                    },
                )
            return resp.json()
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cached on-call entry is still fresh.

        Args:
            cache_key: Cache key (``provider:schedule_id``).

        Returns:
            True if entry exists and is within TTL.
        """
        ts = self._cache_timestamps.get(cache_key)
        if ts is None:
            return False
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        return age < CACHE_TTL_SECONDS

    def _detect_provider(self) -> str:
        """Auto-detect which provider is configured.

        Returns:
            ``pagerduty`` or ``opsgenie`` or empty string.
        """
        if self.config.pagerduty_enabled and self.config.pagerduty_api_key:
            return "pagerduty"
        if self.config.opsgenie_enabled and self.config.opsgenie_api_key:
            return "opsgenie"
        return ""
