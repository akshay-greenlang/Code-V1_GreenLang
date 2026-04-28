# -*- coding: utf-8 -*-
"""HTTP / local file fetchers (D2).

Phase 3 / Wave 2.5 additionally ships :class:`ApiFetcher` and
:class:`WebhookReplayFetcher` to cover the api_webhook source family.
The original :class:`HttpFetcher` / :class:`FileFetcher` shapes are
preserved verbatim — Wave 2.5 is purely additive — so every existing
caller (including the unified :class:`IngestionPipelineRunner`) keeps
running with no behavioural drift.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.request import Request, urlopen

from greenlang.factors.ingestion.artifacts import StoredArtifact

logger = logging.getLogger(__name__)


class BaseFetcher(ABC):
    @abstractmethod
    def fetch(self, url: str) -> bytes:
        raise NotImplementedError


class HttpFetcher(BaseFetcher):
    def __init__(self, timeout_s: float = 30.0, user_agent: str = "GreenLang-Factors/1.0"):
        self.timeout_s = timeout_s
        self.user_agent = user_agent

    def fetch(self, url: str) -> bytes:
        req = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(req, timeout=self.timeout_s) as resp:  # nosec B310 — controlled registry URLs
            return resp.read()


class FileFetcher(BaseFetcher):
    def fetch(self, url: str) -> bytes:
        p = Path(url)
        if not p.is_file():
            raise FileNotFoundError(url)
        return p.read_bytes()


def head_exists(url: str, timeout_s: float = 10.0) -> bool:
    """Best-effort GET reachability check for source watch (U1); False on failure."""
    try:
        req = Request(url, headers={"User-Agent": "GreenLang-Factors-Watch/1.0"})
        with urlopen(req, timeout=timeout_s) as resp:  # nosec B310
            code = getattr(resp, "status", resp.getcode())
            return 200 <= int(code) < 400
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Phase 3 / Wave 2.5 — API + webhook-replay fetchers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ApiFetcherResult:
    """Result of an :class:`ApiFetcher` call.

    Carries the raw aggregated body bytes plus the response headers we
    captured so the runner can persist them on the resulting
    ``raw_artifacts`` row (Phase 3 plan §"Artifact storage contract").
    """

    body_bytes: bytes
    sha256: str
    fetched_at: str
    source_url: str
    content_type: str
    response_headers: Dict[str, str]
    pages_followed: int

    def as_stored_artifact(self, artifact_id: Optional[str] = None) -> StoredArtifact:
        return StoredArtifact(
            artifact_id=artifact_id or str(uuid.uuid4()),
            sha256=self.sha256,
            storage_uri=self.source_url,
            bytes_size=len(self.body_bytes),
        )


@dataclasses.dataclass(frozen=True)
class _RetryPolicy:
    """Tiny retry policy primitive (test-friendly; no external deps)."""

    max_attempts: int = 3
    backoff_s: float = 0.0  # zero in tests; production wraps urllib3 retry.

    @classmethod
    def default(cls) -> "_RetryPolicy":
        return cls()


# Public alias so callers can ``from fetchers import RetryPolicy``.
RetryPolicy = _RetryPolicy


_AUTH_METHODS: Tuple[str, ...] = ("bearer", "api_key", "oauth2_client_credentials")


class ApiFetcher(BaseFetcher):
    """REST API fetcher with pluggable auth + cursor-based pagination.

    Usage:

        fetcher = ApiFetcher(
            base_url="https://api.partner.example/factors",
            auth_method="bearer",
            credentials_resolver=lambda: {"token": os.environ["PARTNER_TOKEN"]},
            transport=httpx.MockTransport(handler),
        )
        result = fetcher.fetch_paginated()

    The fetcher is designed to remain network-free in tests by accepting
    an injected ``transport`` callable that mimics the
    :class:`httpx.MockTransport` shape: ``transport(method, url,
    headers, params) -> (status_code, headers, body_bytes)``. That keeps
    Wave 2.5's e2e tests hermetic per the Wave 1.0 acceptance constraint.
    """

    #: Conform to the :class:`BaseFetcher` ABC. Most callers will use
    #: :meth:`fetch_paginated` instead, which returns a richer result.
    def fetch(self, url: str) -> bytes:  # type: ignore[override]
        result = self.fetch_paginated(url=url)
        return result.body_bytes

    def __init__(
        self,
        base_url: str,
        *,
        auth_method: str = "bearer",
        credentials_resolver: Optional[Callable[[], Mapping[str, str]]] = None,
        retry_policy: Optional[RetryPolicy] = None,
        pagination_cursor_param: str = "cursor",
        transport: Optional[Callable[..., Tuple[int, Dict[str, str], bytes]]] = None,
        max_pages: int = 100,
        user_agent: str = "GreenLang-Factors-API/1.0",
    ) -> None:
        if auth_method not in _AUTH_METHODS:
            raise ValueError(
                "unknown auth_method=%r; expected one of %r" % (auth_method, _AUTH_METHODS),
            )
        self.base_url = base_url.rstrip("/")
        self.auth_method = auth_method
        self._credentials_resolver = credentials_resolver
        self.retry_policy = retry_policy or RetryPolicy.default()
        self.pagination_cursor_param = pagination_cursor_param
        self._transport = transport
        self.max_pages = max_pages
        self.user_agent = user_agent

    # -- public -------------------------------------------------------------

    def fetch_paginated(
        self,
        *,
        url: Optional[str] = None,
        params: Optional[Mapping[str, str]] = None,
    ) -> ApiFetcherResult:
        """Drive cursor-based pagination + return an aggregated body.

        Each page's body is concatenated with newline separators into a
        single byte stream. The aggregate sha256 covers every page in
        order, so checksum drift across pagination is detectable.
        """
        request_url = url or self.base_url
        merged_params: Dict[str, str] = dict(params or {})
        pages: List[bytes] = []
        last_headers: Dict[str, str] = {}
        cursor: Optional[str] = None
        pages_followed = 0
        for _ in range(self.max_pages):
            page_params = dict(merged_params)
            if cursor is not None:
                page_params[self.pagination_cursor_param] = cursor
            status_code, response_headers, body = self._send(
                method="GET", url=request_url, params=page_params,
            )
            if status_code < 200 or status_code >= 300:
                raise RuntimeError(
                    "ApiFetcher: non-2xx response status=%d url=%s"
                    % (status_code, request_url),
                )
            pages.append(body)
            last_headers = dict(response_headers)
            pages_followed += 1
            cursor = self._extract_next_cursor(body, response_headers)
            if cursor is None:
                break
        else:
            logger.warning(
                "ApiFetcher: hit max_pages=%d for url=%s; pagination truncated",
                self.max_pages, request_url,
            )

        aggregate = b"\n".join(pages)
        sha256 = hashlib.sha256(aggregate).hexdigest()
        return ApiFetcherResult(
            body_bytes=aggregate,
            sha256=sha256,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            source_url=request_url,
            content_type=last_headers.get("content-type", "application/json"),
            response_headers=last_headers,
            pages_followed=pages_followed,
        )

    # -- internals ----------------------------------------------------------

    def _send(
        self,
        *,
        method: str,
        url: str,
        params: Mapping[str, str],
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Drive the injected transport with retry semantics.

        The production deployment is expected to inject an ``httpx``-
        compatible transport via the constructor; if none is given, the
        fetcher refuses to make a network call (Wave 2.5 acceptance
        forbids implicit egress).
        """
        if self._transport is None:
            raise RuntimeError(
                "ApiFetcher requires an explicit transport (Wave 2.5 acceptance "
                "forbids implicit network calls). Inject httpx.MockTransport "
                "or a real transport for production.",
            )
        headers = self._build_headers()
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt < max(1, self.retry_policy.max_attempts):
            attempt += 1
            try:
                return self._transport(  # type: ignore[no-any-return]
                    method=method, url=url, headers=headers, params=dict(params),
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.retry_policy.max_attempts:
                    break
                if self.retry_policy.backoff_s > 0:
                    time.sleep(self.retry_policy.backoff_s)
        raise RuntimeError(
            "ApiFetcher: all retries exhausted url=%s attempts=%d last_error=%s"
            % (url, attempt, last_exc),
        )

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"User-Agent": self.user_agent}
        creds: Mapping[str, str] = (
            self._credentials_resolver() if self._credentials_resolver else {}
        )
        if self.auth_method == "bearer":
            token = creds.get("token", "")
            if token:
                headers["Authorization"] = "Bearer %s" % token
        elif self.auth_method == "api_key":
            api_key = creds.get("api_key", "")
            header_name = creds.get("header", "X-API-Key")
            if api_key:
                headers[header_name] = api_key
        elif self.auth_method == "oauth2_client_credentials":
            # Production callers will pre-resolve the access token and
            # pass it through ``credentials_resolver`` -> ``token``;
            # implementing the OAuth2 dance is out of scope for the
            # hermetic e2e suite.
            token = creds.get("token", "")
            if token:
                headers["Authorization"] = "Bearer %s" % token
        return headers

    def _extract_next_cursor(
        self, body: bytes, response_headers: Mapping[str, str],
    ) -> Optional[str]:
        """Return the next pagination cursor from a JSON body or header.

        Recognised body shapes:
        * ``{"pagination": {"next_cursor": "..."}}``
        * ``{"next_cursor": "..."}``
        * ``{"pagination": {"cursor": "..."}}``

        Header form: ``X-Next-Cursor: <value>``.
        """
        cursor = response_headers.get("x-next-cursor") or response_headers.get(
            "X-Next-Cursor",
        )
        if cursor:
            return str(cursor)
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(decoded, dict):
            return None
        pagination = decoded.get("pagination")
        if isinstance(pagination, dict):
            for key in ("next_cursor", "cursor", "next"):
                val = pagination.get(key)
                if isinstance(val, str) and val:
                    return val
        for key in ("next_cursor", "next"):
            val = decoded.get(key)
            if isinstance(val, str) and val:
                return val
        return None


class WebhookReplayFetcher(BaseFetcher):
    """Replay a stored webhook artifact through the pipeline as raw bytes.

    Constructed with a :class:`WebhookArtifactStore` (from
    :mod:`greenlang.factors.ingestion.webhook`). The ``url`` argument
    is interpreted as a webhook artifact id; the fetcher returns the
    stored body bytes verbatim so the parser sees exactly what was
    received over the wire.
    """

    def __init__(self, artifact_store: Any) -> None:
        # Avoid an import-time cycle by typing the store as Any.
        self._store = artifact_store

    def fetch(self, url: str) -> bytes:
        artifact = self._store.get(url)
        if artifact is None:
            raise FileNotFoundError("webhook artifact_id=%s not found" % url)
        return bytes(artifact.body_bytes)

    def fetch_with_metadata(self, artifact_id: str) -> Tuple[bytes, Dict[str, Any]]:
        """Return both the bytes and the captured metadata (for the runner)."""
        artifact = self._store.get(artifact_id)
        if artifact is None:
            raise FileNotFoundError("webhook artifact_id=%s not found" % artifact_id)
        return bytes(artifact.body_bytes), artifact.metadata()
