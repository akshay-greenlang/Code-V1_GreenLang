# -*- coding: utf-8 -*-
"""FastAPI tier-enforcement middleware."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from greenlang.commercial.tiers import Tier, TIER_SPECS, feature_allowed

logger = logging.getLogger(__name__)

try:
    from starlette.responses import JSONResponse
    from starlette.types import ASGIApp, Receive, Scope, Send
except ImportError:  # pragma: no cover
    JSONResponse = None  # type: ignore[assignment]
    ASGIApp = None  # type: ignore[assignment]


class TierEnforcementMiddleware:
    """Simple tier-gate middleware.

    Expects the upstream auth layer (Kong/JWT) to set `X-Tier` header. Blocks
    requests whose tier lacks the required feature, or which exceed the tier's
    daily request cap. Daily counters live in Redis in production (INFRA-003);
    this in-process implementation is for unit tests and dev.
    """

    def __init__(
        self,
        app: "ASGIApp",
        path_features: dict[str, str] | None = None,
        counter_backend: "CounterBackend | None" = None,
    ) -> None:
        self.app = app
        self.path_features = path_features or {}
        self.counter = counter_backend or InMemoryCounter()

    async def __call__(self, scope: "Scope", receive: "Receive", send: "Send") -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        tier_value = headers.get("x-tier", Tier.COMMUNITY.value)
        try:
            tier = Tier(tier_value)
        except ValueError:
            tier = Tier.COMMUNITY

        path = scope["path"]
        required = self._feature_for(path)
        if required and not feature_allowed(tier, required):
            await self._reject(send, 403, f"Tier {tier.value} lacks feature {required!r}")
            return

        # Rate limit check
        limit = TIER_SPECS[tier].daily_request_limit
        if limit:
            tenant = headers.get("x-tenant-id", "anonymous")
            count = await self.counter.increment(f"{tenant}:{tier.value}")
            if count > limit:
                await self._reject(send, 429, f"Daily limit exceeded ({limit})")
                return

        await self.app(scope, receive, send)

    def _feature_for(self, path: str) -> str | None:
        for prefix, feature in self.path_features.items():
            if path.startswith(prefix):
                return feature
        return None

    @staticmethod
    async def _reject(send: "Send", status: int, detail: str) -> None:
        body = ('{"detail":"' + detail + '"}').encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})


class CounterBackend:
    async def increment(self, key: str) -> int:  # pragma: no cover
        raise NotImplementedError


class InMemoryCounter(CounterBackend):
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    async def increment(self, key: str) -> int:
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]


class RedisCounter(CounterBackend):
    """Redis-backed counter; requires redis-py. Used in production (INFRA-003)."""

    def __init__(self, redis_client, ttl_seconds: int = 86400) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds

    async def increment(self, key: str) -> int:
        pipeline = self._redis.pipeline()
        pipeline.incr(key)
        pipeline.expire(key, self._ttl)
        count, _ = await pipeline.execute()
        return int(count)
