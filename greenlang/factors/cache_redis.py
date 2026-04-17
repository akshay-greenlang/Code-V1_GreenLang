# -*- coding: utf-8 -*-
"""
Optional Redis cache for hot factor payloads (stub when redis is unavailable).
"""

from __future__ import annotations

from typing import Any, Optional


class FactorPayloadCache:
    """get/set JSON string payloads by key; no-op when Redis not configured."""

    def __init__(self, url: Optional[str] = None) -> None:
        self._url = url
        self._client = None
        if url:
            try:
                import redis  # type: ignore

                self._client = redis.Redis.from_url(url, decode_responses=True)
            except Exception:
                self._client = None

    def get(self, key: str) -> Optional[str]:
        if not self._client:
            return None
        try:
            return self._client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: str, ttl_seconds: int = 3600) -> None:
        if not self._client:
            return
        try:
            self._client.setex(key, ttl_seconds, value)
        except Exception:
            return
