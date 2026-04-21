# -*- coding: utf-8 -*-
"""HTTP transport layer for the Factors SDK.

Wraps :mod:`httpx` with:

    * Tenacity-based retry on 429 / 5xx / network errors
    * ``Retry-After`` header honoring
    * Transparent ETag response cache (``If-None-Match``)
    * Request-id + rate-limit header parsing
    * Structured error mapping via :func:`greenlang.factors.sdk.python.errors.error_from_response`

The design deliberately separates sync and async transports so callers
can pick a single code path without paying the coroutine cost.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    import httpx
except ImportError:  # pragma: no cover - handled at runtime with clear error
    httpx = None  # type: ignore[assignment]

try:
    from tenacity import (
        RetryError,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except ImportError:  # pragma: no cover - handled at runtime with clear error
    retry = None  # type: ignore[assignment]

from .auth import AuthProvider, compose_auth_headers
from .errors import FactorsAPIError, RateLimitError, error_from_response

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT: float = 30.0
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_USER_AGENT: str = "greenlang-factors-sdk-python/1.0.0"


@dataclass
class RateLimitInfo:
    """Rate-limit headers parsed from a response."""

    limit: Optional[int] = None
    remaining: Optional[int] = None
    reset: Optional[int] = None
    retry_after: Optional[float] = None

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> "RateLimitInfo":
        def _int(v: Optional[str]) -> Optional[int]:
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        def _float(v: Optional[str]) -> Optional[float]:
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        return cls(
            limit=_int(headers.get("X-RateLimit-Limit")),
            remaining=_int(headers.get("X-RateLimit-Remaining")),
            reset=_int(headers.get("X-RateLimit-Reset")),
            retry_after=_float(headers.get("Retry-After")),
        )


@dataclass
class TransportResponse:
    """Decoded response payload + relevant metadata."""

    status_code: int
    data: Any
    headers: Dict[str, str] = field(default_factory=dict)
    etag: Optional[str] = None
    from_cache: bool = False
    request_id: Optional[str] = None
    rate_limit: RateLimitInfo = field(default_factory=RateLimitInfo)
    edition: Optional[str] = None


# ---------------------------------------------------------------------------
# ETag cache
# ---------------------------------------------------------------------------


class ETagCache:
    """Simple thread-safe in-memory ETag/body cache.

    Keyed on ``(method, url, sorted(params))``.  Not LRU (size-bounded
    by ``max_entries``).  Sufficient for SDK read-mostly workloads; if
    a user needs more, they can plug their own via ``cache=`` in the
    client constructor.
    """

    def __init__(self, max_entries: int = 512) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, Tuple[str, Any, Dict[str, str]]] = {}
        self._max_entries = max_entries

    @staticmethod
    def key(method: str, url: str, params: Optional[Mapping[str, Any]] = None) -> str:
        if params:
            items = sorted((k, str(v)) for k, v in params.items() if v is not None)
            param_str = "&".join(f"{k}={v}" for k, v in items)
        else:
            param_str = ""
        return f"{method.upper()} {url}?{param_str}"

    def get(self, key: str) -> Optional[Tuple[str, Any, Dict[str, str]]]:
        with self._lock:
            return self._entries.get(key)

    def set(
        self,
        key: str,
        etag: str,
        data: Any,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        with self._lock:
            if len(self._entries) >= self._max_entries:
                # Drop oldest (insertion order preserved in dict).
                self._entries.pop(next(iter(self._entries)))
            self._entries[key] = (etag, data, dict(headers or {}))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_httpx() -> None:
    if httpx is None:  # pragma: no cover
        raise ImportError(
            "httpx is required for the Factors SDK. "
            "Install with: pip install 'httpx>=0.27'"
        )


def _decode_body(response: "httpx.Response") -> Any:
    ctype = (response.headers.get("content-type") or "").lower()
    try:
        if "application/json" in ctype or response.text.lstrip().startswith(("{", "[")):
            return response.json()
        return response.text
    except (json.JSONDecodeError, ValueError):
        return response.text


def _extract_response_meta(response: "httpx.Response") -> Dict[str, Any]:
    rate = RateLimitInfo.from_headers(response.headers)
    return {
        "etag": response.headers.get("ETag"),
        "request_id": response.headers.get("X-Request-ID"),
        "edition": response.headers.get("X-Factors-Edition")
        or response.headers.get("X-GreenLang-Edition"),
        "rate_limit": rate,
    }


def _should_retry(status: int) -> bool:
    return status == 429 or (500 <= status < 600)


# ---------------------------------------------------------------------------
# Sync transport
# ---------------------------------------------------------------------------


class Transport:
    """Synchronous HTTP transport backed by :class:`httpx.Client`."""

    def __init__(
        self,
        base_url: str,
        *,
        auth: Optional[AuthProvider] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        user_agent: str = DEFAULT_USER_AGENT,
        default_edition: Optional[str] = None,
        cache: Optional[ETagCache] = None,
        client: Optional[Any] = None,
        transport: Optional[Any] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        _check_httpx()
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.user_agent = user_agent
        self.default_edition = default_edition
        self.cache = cache if cache is not None else ETagCache()
        self._extra_headers = dict(extra_headers or {})
        self._owns_client = client is None
        if client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=timeout,
                transport=transport,
            )
        else:
            self._client = client

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "Transport":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _headers(
        self,
        *,
        method: str,
        path: str,
        body: Optional[bytes],
        extra: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, str]:
        base = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        base.update(self._extra_headers)
        if self.default_edition:
            base["X-Factors-Edition"] = self.default_edition
        if extra:
            base.update(extra)
        return compose_auth_headers(
            self.auth, base, method=method, path=path, body=body
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Optional[Any] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        use_cache: bool = True,
    ) -> TransportResponse:
        url_path = path if path.startswith("/") else "/" + path
        cache_key = ETagCache.key(method, url_path, params)

        body_bytes: Optional[bytes] = None
        if json_body is not None:
            body_bytes = json.dumps(json_body, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )

        # ETag pre-flight: attach If-None-Match when we have a cached entry.
        cached = self.cache.get(cache_key) if (use_cache and method.upper() == "GET") else None
        cond_extra: Dict[str, str] = dict(extra_headers or {})
        if cached is not None and "If-None-Match" not in cond_extra:
            cond_extra["If-None-Match"] = cached[0]

        headers = self._headers(
            method=method, path=url_path, body=body_bytes, extra=cond_extra
        )
        if body_bytes is not None:
            headers["Content-Type"] = "application/json"

        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.request(
                    method,
                    url_path,
                    params=dict(params) if params else None,
                    content=body_bytes,
                    headers=headers,
                )
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning(
                    "Network error on %s %s (attempt %d/%d): %s",
                    method, url_path, attempt, self.max_retries, exc,
                )
                if attempt >= self.max_retries:
                    raise FactorsAPIError(
                        "Network error: %s" % exc,
                        context={"attempts": attempt, "url": url_path},
                    ) from exc
                self._sleep_backoff(attempt)
                continue

            meta = _extract_response_meta(response)
            rate: RateLimitInfo = meta["rate_limit"]

            if response.status_code == 304 and cached is not None:
                logger.debug("ETag cache hit (304) for %s", cache_key)
                return TransportResponse(
                    status_code=200,
                    data=cached[1],
                    headers=dict(response.headers),
                    etag=cached[0],
                    from_cache=True,
                    request_id=meta["request_id"],
                    rate_limit=rate,
                    edition=meta["edition"],
                )

            if _should_retry(response.status_code) and attempt < self.max_retries:
                wait = self._compute_wait(attempt, rate.retry_after)
                logger.warning(
                    "Retrying %s %s after HTTP %d in %.2fs (attempt %d/%d)",
                    method, url_path, response.status_code, wait, attempt, self.max_retries,
                )
                time.sleep(wait)
                continue

            body = _decode_body(response)

            if response.is_success:
                etag = meta["etag"]
                if etag and method.upper() == "GET" and use_cache:
                    self.cache.set(cache_key, etag, body, dict(response.headers))
                return TransportResponse(
                    status_code=response.status_code,
                    data=body,
                    headers=dict(response.headers),
                    etag=etag,
                    from_cache=False,
                    request_id=meta["request_id"],
                    rate_limit=rate,
                    edition=meta["edition"],
                )

            raise error_from_response(
                status_code=response.status_code,
                url=str(response.url),
                body=body,
                request_id=meta["request_id"],
                retry_after=rate.retry_after,
            )

        # Should be unreachable, but guard against it.
        if last_exc is not None:
            raise FactorsAPIError(
                "Request failed after %d attempts: %s" % (self.max_retries, last_exc)
            ) from last_exc
        raise FactorsAPIError(
            "Request failed after %d attempts" % self.max_retries
        )

    def _sleep_backoff(self, attempt: int) -> None:
        time.sleep(self._compute_wait(attempt, None))

    @staticmethod
    def _compute_wait(attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None and retry_after >= 0:
            return min(float(retry_after), 60.0)
        return min(2 ** (attempt - 1), 30.0)


# ---------------------------------------------------------------------------
# Async transport
# ---------------------------------------------------------------------------


class AsyncTransport:
    """Asynchronous HTTP transport backed by :class:`httpx.AsyncClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        auth: Optional[AuthProvider] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        user_agent: str = DEFAULT_USER_AGENT,
        default_edition: Optional[str] = None,
        cache: Optional[ETagCache] = None,
        client: Optional[Any] = None,
        transport: Optional[Any] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        _check_httpx()
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.user_agent = user_agent
        self.default_edition = default_edition
        self.cache = cache if cache is not None else ETagCache()
        self._extra_headers = dict(extra_headers or {})
        self._owns_client = client is None
        if client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeout,
                transport=transport,
            )
        else:
            self._client = client

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncTransport":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def _headers(
        self,
        *,
        method: str,
        path: str,
        body: Optional[bytes],
        extra: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, str]:
        base = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        base.update(self._extra_headers)
        if self.default_edition:
            base["X-Factors-Edition"] = self.default_edition
        if extra:
            base.update(extra)
        return compose_auth_headers(
            self.auth, base, method=method, path=path, body=body
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Optional[Any] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        use_cache: bool = True,
    ) -> TransportResponse:
        import asyncio

        url_path = path if path.startswith("/") else "/" + path
        cache_key = ETagCache.key(method, url_path, params)

        body_bytes: Optional[bytes] = None
        if json_body is not None:
            body_bytes = json.dumps(json_body, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )

        cached = self.cache.get(cache_key) if (use_cache and method.upper() == "GET") else None
        cond_extra: Dict[str, str] = dict(extra_headers or {})
        if cached is not None and "If-None-Match" not in cond_extra:
            cond_extra["If-None-Match"] = cached[0]

        headers = self._headers(
            method=method, path=url_path, body=body_bytes, extra=cond_extra
        )
        if body_bytes is not None:
            headers["Content-Type"] = "application/json"

        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._client.request(
                    method,
                    url_path,
                    params=dict(params) if params else None,
                    content=body_bytes,
                    headers=headers,
                )
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning(
                    "Network error on %s %s (attempt %d/%d): %s",
                    method, url_path, attempt, self.max_retries, exc,
                )
                if attempt >= self.max_retries:
                    raise FactorsAPIError(
                        "Network error: %s" % exc,
                        context={"attempts": attempt, "url": url_path},
                    ) from exc
                await asyncio.sleep(Transport._compute_wait(attempt, None))
                continue

            meta = _extract_response_meta(response)
            rate: RateLimitInfo = meta["rate_limit"]

            if response.status_code == 304 and cached is not None:
                return TransportResponse(
                    status_code=200,
                    data=cached[1],
                    headers=dict(response.headers),
                    etag=cached[0],
                    from_cache=True,
                    request_id=meta["request_id"],
                    rate_limit=rate,
                    edition=meta["edition"],
                )

            if _should_retry(response.status_code) and attempt < self.max_retries:
                wait = Transport._compute_wait(attempt, rate.retry_after)
                logger.warning(
                    "Retrying %s %s after HTTP %d in %.2fs (attempt %d/%d)",
                    method, url_path, response.status_code, wait, attempt, self.max_retries,
                )
                await asyncio.sleep(wait)
                continue

            body = _decode_body(response)

            if response.is_success:
                etag = meta["etag"]
                if etag and method.upper() == "GET" and use_cache:
                    self.cache.set(cache_key, etag, body, dict(response.headers))
                return TransportResponse(
                    status_code=response.status_code,
                    data=body,
                    headers=dict(response.headers),
                    etag=etag,
                    from_cache=False,
                    request_id=meta["request_id"],
                    rate_limit=rate,
                    edition=meta["edition"],
                )

            raise error_from_response(
                status_code=response.status_code,
                url=str(response.url),
                body=body,
                request_id=meta["request_id"],
                retry_after=rate.retry_after,
            )

        if last_exc is not None:
            raise FactorsAPIError(
                "Request failed after %d attempts: %s" % (self.max_retries, last_exc)
            ) from last_exc
        raise FactorsAPIError(
            "Request failed after %d attempts" % self.max_retries
        )


__all__ = [
    "Transport",
    "AsyncTransport",
    "TransportResponse",
    "RateLimitInfo",
    "ETagCache",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_USER_AGENT",
]
