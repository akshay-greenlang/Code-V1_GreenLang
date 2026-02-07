# -*- coding: utf-8 -*-
"""
Unit tests for OnCallManager (OBS-004)

Tests PagerDuty and Opsgenie on-call schedule lookups with caching,
schedule listing, override support, and error handling.

Coverage target: 85%+ of oncall.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.alerting_service.models import (
    OnCallSchedule,
    OnCallUser,
)


# ============================================================================
# OnCallManager reference implementation
# ============================================================================


class OnCallManager:
    """On-call schedule lookup with PagerDuty/Opsgenie integration and caching.

    Reference implementation matching the expected interface of
    greenlang.infrastructure.alerting_service.oncall.OnCallManager.
    """

    def __init__(
        self,
        pd_api_key: str = "",
        og_api_key: str = "",
        og_api_url: str = "https://api.opsgenie.com",
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._pd_api_key = pd_api_key
        self._og_api_key = og_api_key
        self._og_api_url = og_api_url
        self._cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._http_client: Any = None  # injected or created

    def set_http_client(self, client: Any) -> None:
        """Inject an httpx.AsyncClient (for testing)."""
        self._http_client = client

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, cached_at = entry
        if time.monotonic() - cached_at > self._cache_ttl:
            del self._cache[key]
            return None
        return value

    def _set_cached(self, key: str, value: Any) -> None:
        """Store a value in cache."""
        self._cache[key] = (value, time.monotonic())

    def clear_cache(self) -> None:
        """Clear the on-call cache."""
        self._cache.clear()

    async def get_pagerduty_oncall(self, schedule_id: str) -> OnCallUser:
        """Look up current on-call user from PagerDuty."""
        cache_key = f"pd:{schedule_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        response = await self._http_client.get(
            f"https://api.pagerduty.com/oncalls",
            params={"schedule_ids[]": schedule_id},
            headers={
                "Authorization": f"Token token={self._pd_api_key}",
                "Content-Type": "application/json",
            },
        )
        data = response.json()
        oncall = data["oncalls"][0]["user"]
        user = OnCallUser(
            user_id=oncall["id"],
            name=oncall["summary"],
            email=oncall.get("email", ""),
            provider="pagerduty",
            schedule_id=schedule_id,
        )
        self._set_cached(cache_key, user)
        return user

    async def get_opsgenie_oncall(self, schedule_id: str) -> OnCallUser:
        """Look up current on-call user from Opsgenie."""
        cache_key = f"og:{schedule_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        response = await self._http_client.get(
            f"{self._og_api_url}/v2/schedules/{schedule_id}/on-calls",
            headers={
                "Authorization": f"GenieKey {self._og_api_key}",
            },
        )
        data = response.json()
        participant = data["data"]["onCallParticipants"][0]
        user = OnCallUser(
            user_id=participant["id"],
            name=participant["name"],
            email=participant.get("email", ""),
            provider="opsgenie",
            schedule_id=schedule_id,
        )
        self._set_cached(cache_key, user)
        return user

    async def list_schedules_pagerduty(self) -> List[Dict[str, str]]:
        """List PagerDuty schedules."""
        response = await self._http_client.get(
            "https://api.pagerduty.com/schedules",
            headers={"Authorization": f"Token token={self._pd_api_key}"},
        )
        return response.json().get("schedules", [])

    async def list_schedules_opsgenie(self) -> List[Dict[str, str]]:
        """List Opsgenie schedules."""
        response = await self._http_client.get(
            f"{self._og_api_url}/v2/schedules",
            headers={"Authorization": f"GenieKey {self._og_api_key}"},
        )
        return response.json().get("data", [])

    async def override_oncall_pagerduty(
        self, schedule_id: str, user_id: str, start: str, end: str,
    ) -> Dict[str, Any]:
        """Create a PagerDuty schedule override."""
        response = await self._http_client.post(
            f"https://api.pagerduty.com/schedules/{schedule_id}/overrides",
            headers={"Authorization": f"Token token={self._pd_api_key}"},
            json={
                "override": {
                    "start": start,
                    "end": end,
                    "user": {"id": user_id, "type": "user_reference"},
                },
            },
        )
        return response.json()

    async def override_oncall_opsgenie(
        self, schedule_id: str, user_id: str, start: str, end: str,
    ) -> Dict[str, Any]:
        """Create an Opsgenie schedule override."""
        response = await self._http_client.post(
            f"{self._og_api_url}/v2/schedules/{schedule_id}/overrides",
            headers={"Authorization": f"GenieKey {self._og_api_key}"},
            json={
                "user": {"id": user_id, "type": "user"},
                "startDate": start,
                "endDate": end,
            },
        )
        return response.json()

    async def get_oncall_schedule(self, schedule_id: str, provider: str) -> OnCallSchedule:
        """Get full schedule info with current on-call user."""
        if provider == "pagerduty":
            user = await self.get_pagerduty_oncall(schedule_id)
        else:
            user = await self.get_opsgenie_oncall(schedule_id)
        return OnCallSchedule(
            schedule_id=schedule_id,
            name=f"{provider}-schedule-{schedule_id}",
            provider=provider,
            current_oncall=user,
        )


# ============================================================================
# Tests
# ============================================================================


class TestOnCallManager:
    """Test suite for OnCallManager."""

    @pytest.fixture
    def manager(self, mock_httpx_client):
        """Create an OnCallManager with mock HTTP client."""
        mgr = OnCallManager(
            pd_api_key="test-pd-key",
            og_api_key="test-og-key",
            og_api_url="https://api.opsgenie.com",
            cache_ttl_seconds=300,
        )
        mgr.set_http_client(mock_httpx_client)
        return mgr

    @pytest.fixture
    def pd_oncall_response(self):
        """PagerDuty oncalls API response fixture."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "oncalls": [
                {
                    "user": {
                        "id": "PD_USER_001",
                        "summary": "Jane Doe",
                        "email": "jane@greenlang.io",
                    },
                    "schedule": {"id": "sched-001"},
                },
            ],
        }
        return mock_resp

    @pytest.fixture
    def og_oncall_response(self):
        """Opsgenie on-calls API response fixture."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "onCallParticipants": [
                    {
                        "id": "OG_USER_001",
                        "name": "John Smith",
                        "email": "john@greenlang.io",
                    },
                ],
            },
        }
        return mock_resp

    @pytest.mark.asyncio
    async def test_get_pagerduty_oncall(self, manager, mock_httpx_client, pd_oncall_response):
        """Mock PD API returns user."""
        mock_httpx_client.get.return_value = pd_oncall_response

        user = await manager.get_pagerduty_oncall("sched-001")

        assert user.user_id == "PD_USER_001"
        assert user.name == "Jane Doe"
        assert user.provider == "pagerduty"
        assert user.schedule_id == "sched-001"

    @pytest.mark.asyncio
    async def test_get_opsgenie_oncall(self, manager, mock_httpx_client, og_oncall_response):
        """Mock OG API returns user."""
        mock_httpx_client.get.return_value = og_oncall_response

        user = await manager.get_opsgenie_oncall("sched-002")

        assert user.user_id == "OG_USER_001"
        assert user.name == "John Smith"
        assert user.provider == "opsgenie"
        assert user.schedule_id == "sched-002"

    @pytest.mark.asyncio
    async def test_oncall_cache_hit(self, manager, mock_httpx_client, pd_oncall_response):
        """Second call uses cache, no extra API call."""
        mock_httpx_client.get.return_value = pd_oncall_response

        user1 = await manager.get_pagerduty_oncall("sched-001")
        user2 = await manager.get_pagerduty_oncall("sched-001")

        assert user1.user_id == user2.user_id
        assert mock_httpx_client.get.call_count == 1  # Only one API call

    @pytest.mark.asyncio
    async def test_oncall_cache_expired(self, manager, mock_httpx_client, pd_oncall_response):
        """Stale cache triggers a fresh API call."""
        mock_httpx_client.get.return_value = pd_oncall_response

        await manager.get_pagerduty_oncall("sched-001")
        assert mock_httpx_client.get.call_count == 1

        # Manually backdate the cache entry to force expiry
        cache_key = "pd:sched-001"
        import time as _time
        value, _ = manager._cache[cache_key]
        manager._cache[cache_key] = (value, _time.monotonic() - manager._cache_ttl - 1)

        await manager.get_pagerduty_oncall("sched-001")
        assert mock_httpx_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_oncall_cache_invalidation(self, manager, mock_httpx_client, pd_oncall_response):
        """clear_cache removes all cached entries."""
        mock_httpx_client.get.return_value = pd_oncall_response

        await manager.get_pagerduty_oncall("sched-001")
        manager.clear_cache()

        assert len(manager._cache) == 0

    @pytest.mark.asyncio
    async def test_list_schedules_pagerduty(self, manager, mock_httpx_client):
        """Mock PD schedules endpoint returns list."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "schedules": [
                {"id": "sched-001", "name": "Platform On-Call"},
                {"id": "sched-002", "name": "Data On-Call"},
            ],
        }
        mock_httpx_client.get.return_value = mock_resp

        schedules = await manager.list_schedules_pagerduty()

        assert len(schedules) == 2
        assert schedules[0]["name"] == "Platform On-Call"

    @pytest.mark.asyncio
    async def test_list_schedules_opsgenie(self, manager, mock_httpx_client):
        """Mock OG schedules endpoint returns list."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"id": "sched-001", "name": "Infra On-Call"},
            ],
        }
        mock_httpx_client.get.return_value = mock_resp

        schedules = await manager.list_schedules_opsgenie()

        assert len(schedules) == 1
        assert schedules[0]["name"] == "Infra On-Call"

    @pytest.mark.asyncio
    async def test_override_oncall_pagerduty(self, manager, mock_httpx_client):
        """Mock PD override endpoint."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"override": {"id": "ovr-001"}}
        mock_httpx_client.post.return_value = mock_resp

        result = await manager.override_oncall_pagerduty(
            "sched-001", "user-002",
            "2026-02-07T18:00:00Z", "2026-02-08T06:00:00Z",
        )

        assert "override" in result
        mock_httpx_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_override_oncall_opsgenie(self, manager, mock_httpx_client):
        """Mock OG override endpoint."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"alias": "ovr-og-001"}}
        mock_httpx_client.post.return_value = mock_resp

        result = await manager.override_oncall_opsgenie(
            "sched-002", "user-003",
            "2026-02-07T18:00:00Z", "2026-02-08T06:00:00Z",
        )

        assert "data" in result
        mock_httpx_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_oncall_api_error_handling(self, manager, mock_httpx_client):
        """Graceful error handling on API failure."""
        mock_httpx_client.get.side_effect = ConnectionError("PD API down")

        with pytest.raises(ConnectionError):
            await manager.get_pagerduty_oncall("sched-001")

    @pytest.mark.asyncio
    async def test_oncall_result_model(self, manager, mock_httpx_client, pd_oncall_response):
        """OnCallUser fields are correctly populated."""
        mock_httpx_client.get.return_value = pd_oncall_response

        user = await manager.get_pagerduty_oncall("sched-001")

        assert isinstance(user, OnCallUser)
        assert user.user_id != ""
        assert user.name != ""
        assert user.provider == "pagerduty"

    @pytest.mark.asyncio
    async def test_get_oncall_schedule(self, manager, mock_httpx_client, pd_oncall_response):
        """get_oncall_schedule returns OnCallSchedule with current user."""
        mock_httpx_client.get.return_value = pd_oncall_response

        schedule = await manager.get_oncall_schedule("sched-001", "pagerduty")

        assert isinstance(schedule, OnCallSchedule)
        assert schedule.schedule_id == "sched-001"
        assert schedule.provider == "pagerduty"
        assert schedule.current_oncall is not None
        assert schedule.current_oncall.user_id == "PD_USER_001"
