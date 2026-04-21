# -*- coding: utf-8 -*-
"""High-level Factors SDK clients (sync + async).

:class:`FactorsClient` and :class:`AsyncFactorsClient` are the primary
entry points.  Each method wraps one server route from
``/api/v1/factors`` or ``/api/v1/editions`` and returns a typed Pydantic
model (see :mod:`.models`).

Every network call goes through :class:`.transport.Transport` (or its
async cousin), which already handles authentication, retries with
exponential backoff, ETag caching, and HTTP-error to exception mapping.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

from .auth import APIKeyAuth, AuthProvider, JWTAuth
from .errors import FactorsAPIError, RateLimitError
from .models import (
    AuditBundle,
    BatchJobHandle,
    CoverageReport,
    Edition,
    Factor,
    FactorDiff,
    FactorMatch,
    MethodPack,
    Override,
    ResolutionRequest,
    ResolvedFactor,
    SearchResponse,
    Source,
)
from .pagination import (
    AsyncOffsetPaginator,
    OffsetPaginator,
    extract_items,
)
from .transport import (
    AsyncTransport,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    ETagCache,
    Transport,
    TransportResponse,
)

logger = logging.getLogger(__name__)

_BATCH_TERMINAL_STATES = {"completed", "failed", "cancelled"}


def _normalize_auth(
    auth: Optional[AuthProvider],
    api_key: Optional[str],
    jwt_token: Optional[str],
) -> Optional[AuthProvider]:
    """Collapse constructor auth shortcuts into a single ``AuthProvider``."""
    if auth is not None:
        return auth
    if api_key:
        return APIKeyAuth(api_key=api_key)
    if jwt_token:
        return JWTAuth(token=jwt_token)
    return None


def _bool_param(value: Optional[bool]) -> Optional[str]:
    if value is None:
        return None
    return "true" if value else "false"


def _build_search_response(payload: Any) -> SearchResponse:
    """Build a SearchResponse from any of the server's search shapes."""
    if isinstance(payload, dict):
        return SearchResponse.model_validate(payload)
    if isinstance(payload, list):
        factors = [
            Factor.model_validate(f) if isinstance(f, dict) else f for f in payload
        ]
        return SearchResponse(factors=factors, count=len(factors))
    return SearchResponse()


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class FactorsClient:
    """Synchronous client for the GreenLang Factors REST API.

    Args:
        base_url: API host (e.g. ``"https://api.greenlang.io"``).  The
            SDK prepends ``/api/v1`` automatically — do NOT include it
            here unless you want to override the default prefix.
        auth: Explicit auth provider (takes precedence over shortcuts).
        api_key: Shortcut for ``APIKeyAuth(api_key=...)``.
        jwt_token: Shortcut for ``JWTAuth(token=...)``.
        default_edition: Sent as ``X-Factors-Edition`` on every request.
        timeout: Per-request timeout in seconds.
        max_retries: Retry budget for 429/5xx/network errors.
        user_agent: Overridable UA string.
        cache: Optional shared :class:`ETagCache` for cross-client reuse.
        transport: Pass an ``httpx`` transport (e.g. ``MockTransport``)
            for testing without touching the network.
        api_prefix: Override the ``/api/v1`` prefix.

    Example::

        with FactorsClient(base_url="https://api.greenlang.io", api_key="gl_...") as c:
            hits = c.search("natural gas US Scope 1", limit=10)
            for f in hits.factors:
                print(f.factor_id, f.co2e_per_unit)
    """

    DEFAULT_API_PREFIX: str = "/api/v1"

    def __init__(
        self,
        base_url: str,
        *,
        auth: Optional[AuthProvider] = None,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        default_edition: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        user_agent: str = DEFAULT_USER_AGENT,
        cache: Optional[ETagCache] = None,
        transport: Optional[Any] = None,
        api_prefix: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._api_prefix = (api_prefix or self.DEFAULT_API_PREFIX).rstrip("/") or ""
        self._transport = Transport(
            base_url=base_url,
            auth=_normalize_auth(auth, api_key, jwt_token),
            timeout=timeout,
            max_retries=max_retries,
            user_agent=user_agent,
            default_edition=default_edition,
            cache=cache,
            transport=transport,
            extra_headers=extra_headers,
        )

    # ---- Lifecycle -------------------------------------------------------

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "FactorsClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- Transport accessors --------------------------------------------

    @property
    def cache(self) -> ETagCache:
        return self._transport.cache

    def _path(self, suffix: str) -> str:
        s = suffix if suffix.startswith("/") else "/" + suffix
        return self._api_prefix + s

    def _get(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> TransportResponse:
        return self._transport.request(
            "GET", self._path(path), params=params, use_cache=use_cache
        )

    def _post(
        self,
        path: str,
        *,
        json_body: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> TransportResponse:
        return self._transport.request(
            "POST",
            self._path(path),
            params=params,
            json_body=json_body,
            use_cache=False,
        )

    # =====================================================================
    # Search / listing
    # =====================================================================

    def search(
        self,
        query: str,
        *,
        geography: Optional[str] = None,
        limit: int = 20,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> SearchResponse:
        """GET /factors/search — full-text search."""
        params: Dict[str, Any] = {"q": query, "limit": limit}
        if geography:
            params["geography"] = geography
        if edition:
            params["edition"] = edition
        ip = _bool_param(include_preview)
        ic = _bool_param(include_connector)
        if ip is not None:
            params["include_preview"] = ip
        if ic is not None:
            params["include_connector"] = ic
        resp = self._get("/factors/search", params=params)
        return _build_search_response(resp.data)

    def search_v2(
        self,
        query: str,
        *,
        geography: Optional[str] = None,
        fuel_type: Optional[str] = None,
        scope: Optional[str] = None,
        source_id: Optional[str] = None,
        factor_status: Optional[str] = None,
        license_class: Optional[str] = None,
        dqs_min: Optional[float] = None,
        valid_on_date: Optional[str] = None,
        sector_tags: Optional[List[str]] = None,
        activity_tags: Optional[List[str]] = None,
        sort_by: str = "relevance",
        sort_order: str = "desc",
        offset: int = 0,
        limit: int = 20,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> SearchResponse:
        """POST /factors/search/v2 — advanced search with sort + pagination."""
        body: Dict[str, Any] = {
            "query": query,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "offset": offset,
            "limit": limit,
        }
        for key, value in (
            ("geography", geography),
            ("fuel_type", fuel_type),
            ("scope", scope),
            ("source_id", source_id),
            ("factor_status", factor_status),
            ("license_class", license_class),
            ("dqs_min", dqs_min),
            ("valid_on_date", valid_on_date),
            ("sector_tags", sector_tags),
            ("activity_tags", activity_tags),
        ):
            if value is not None:
                body[key] = value
        if include_preview is not None:
            body["include_preview"] = include_preview
        if include_connector is not None:
            body["include_connector"] = include_connector
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = self._post("/factors/search/v2", json_body=body, params=params or None)
        return _build_search_response(resp.data)

    def list_factors(
        self,
        *,
        fuel_type: Optional[str] = None,
        geography: Optional[str] = None,
        scope: Optional[str] = None,
        boundary: Optional[str] = None,
        edition: Optional[str] = None,
        page: int = 1,
        limit: int = 100,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> SearchResponse:
        """GET /factors — paginated list with filters."""
        params: Dict[str, Any] = {"page": page, "limit": limit}
        for key, value in (
            ("fuel_type", fuel_type),
            ("geography", geography),
            ("scope", scope),
            ("boundary", boundary),
            ("edition", edition),
        ):
            if value is not None:
                params[key] = value
        ip = _bool_param(include_preview)
        ic = _bool_param(include_connector)
        if ip is not None:
            params["include_preview"] = ip
        if ic is not None:
            params["include_connector"] = ic
        resp = self._get("/factors", params=params)
        return _build_search_response(resp.data)

    def paginate_search(
        self,
        query: str,
        *,
        page_size: int = 100,
        max_items: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Factor]:
        """Iterate across all pages of /search/v2 results."""

        def _fetch(offset: int, limit: int) -> "tuple[List[Factor], Optional[int]]":
            resp = self.search_v2(query, offset=offset, limit=limit, **kwargs)
            return list(resp.factors), resp.total_count

        return OffsetPaginator(
            _fetch, page_size=page_size, max_items=max_items
        )

    # =====================================================================
    # Factors
    # =====================================================================

    def get_factor(
        self,
        factor_id: str,
        *,
        edition: Optional[str] = None,
    ) -> Factor:
        """GET /factors/{factor_id} — fetch a factor by id."""
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = self._get(f"/factors/{factor_id}", params=params or None)
        return Factor.model_validate(resp.data)

    def match(
        self,
        activity_description: str,
        *,
        geography: Optional[str] = None,
        fuel_type: Optional[str] = None,
        scope: Optional[str] = None,
        limit: int = 10,
        edition: Optional[str] = None,
    ) -> List[FactorMatch]:
        """POST /factors/match — NL-to-factor matching."""
        body: Dict[str, Any] = {
            "activity_description": activity_description,
            "limit": limit,
        }
        for key, value in (
            ("geography", geography),
            ("fuel_type", fuel_type),
            ("scope", scope),
        ):
            if value is not None:
                body[key] = value
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = self._post("/factors/match", json_body=body, params=params or None)
        data = resp.data or {}
        candidates = (
            data.get("candidates", [])
            if isinstance(data, dict)
            else []
        )
        return [FactorMatch.model_validate(c) for c in candidates]

    def coverage(self, *, edition: Optional[str] = None) -> CoverageReport:
        """GET /factors/coverage — coverage statistics."""
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = self._get("/factors/coverage", params=params or None)
        return CoverageReport.model_validate(resp.data)

    # =====================================================================
    # Resolution (Pro+ tier)
    # =====================================================================

    def resolve_explain(
        self,
        factor_id: str,
        *,
        method_profile: Optional[str] = None,
        alternates: Optional[int] = None,
        edition: Optional[str] = None,
    ) -> ResolvedFactor:
        """GET /factors/{id}/explain — Pro+ explain payload."""
        params: Dict[str, Any] = {}
        if method_profile:
            params["method_profile"] = method_profile
        if alternates is not None:
            params["limit"] = alternates
        if edition:
            params["edition"] = edition
        resp = self._get(f"/factors/{factor_id}/explain", params=params or None)
        return ResolvedFactor.model_validate(resp.data)

    def resolve(
        self,
        request: Union[ResolutionRequest, Dict[str, Any]],
        *,
        alternates: Optional[int] = None,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> ResolvedFactor:
        """POST /factors/resolve-explain — Pro+ full cascade resolve."""
        if isinstance(request, ResolutionRequest):
            body = request.model_dump(exclude_none=True)
        else:
            body = dict(request)
        params: Dict[str, Any] = {}
        if alternates is not None:
            params["limit"] = alternates
        if edition:
            params["edition"] = edition
        if include_preview is not None:
            params["include_preview"] = _bool_param(include_preview)
        if include_connector is not None:
            params["include_connector"] = _bool_param(include_connector)
        resp = self._post(
            "/factors/resolve-explain",
            json_body=body,
            params=params or None,
        )
        return ResolvedFactor.model_validate(resp.data)

    def alternates(
        self,
        factor_id: str,
        *,
        method_profile: Optional[str] = None,
        limit: Optional[int] = None,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """GET /factors/{id}/alternates — Pro+ alternate candidates."""
        params: Dict[str, Any] = {}
        if method_profile:
            params["method_profile"] = method_profile
        if limit is not None:
            params["limit"] = limit
        if edition:
            params["edition"] = edition
        if include_preview is not None:
            params["include_preview"] = _bool_param(include_preview)
        if include_connector is not None:
            params["include_connector"] = _bool_param(include_connector)
        resp = self._get(
            f"/factors/{factor_id}/alternates", params=params or None
        )
        return resp.data if isinstance(resp.data, dict) else {"raw": resp.data}

    # =====================================================================
    # Batch resolution
    # =====================================================================

    def resolve_batch(
        self,
        requests: List[Union[ResolutionRequest, Dict[str, Any]]],
        *,
        edition: Optional[str] = None,
    ) -> BatchJobHandle:
        """POST /factors/resolve/batch — submit a batch resolution job."""
        items: List[Dict[str, Any]] = []
        for r in requests:
            if isinstance(r, ResolutionRequest):
                items.append(r.model_dump(exclude_none=True))
            else:
                items.append(dict(r))
        body: Dict[str, Any] = {"requests": items}
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = self._post(
            "/factors/resolve/batch", json_body=body, params=params or None
        )
        return BatchJobHandle.model_validate(resp.data)

    def get_batch_job(self, job_id: str) -> BatchJobHandle:
        """GET /factors/jobs/{job_id} — check batch job status."""
        resp = self._get(f"/factors/jobs/{job_id}", use_cache=False)
        return BatchJobHandle.model_validate(resp.data)

    def wait_for_batch(
        self,
        job: Union[BatchJobHandle, str],
        *,
        poll_interval: float = 2.0,
        timeout: Optional[float] = 600.0,
    ) -> BatchJobHandle:
        """Poll ``get_batch_job`` until the job reaches a terminal state.

        Args:
            job: Handle returned by :meth:`resolve_batch`, or a raw job id.
            poll_interval: Seconds between polls.
            timeout: Maximum wall-clock seconds to wait before raising.

        Raises:
            RateLimitError: propagated from the underlying HTTP call.
            FactorsAPIError: if the poll times out or the job fails.
        """
        job_id = job.job_id if isinstance(job, BatchJobHandle) else str(job)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            current = self.get_batch_job(job_id)
            if current.status in _BATCH_TERMINAL_STATES:
                if current.status == "failed":
                    raise FactorsAPIError(
                        "Batch job %s failed: %s"
                        % (job_id, current.error_message or "unknown error"),
                        context={"job_id": job_id, "status": current.status},
                    )
                return current
            if deadline is not None and time.monotonic() > deadline:
                raise FactorsAPIError(
                    "Timeout waiting for batch job %s (status=%s)"
                    % (job_id, current.status),
                    context={"job_id": job_id, "timeout": timeout},
                )
            time.sleep(poll_interval)

    # =====================================================================
    # Editions
    # =====================================================================

    def list_editions(
        self,
        *,
        include_pending: bool = True,
    ) -> List[Edition]:
        """GET /editions — list all editions."""
        params = {"include_pending": _bool_param(include_pending)}
        resp = self._get("/editions", params=params)
        data = resp.data if isinstance(resp.data, dict) else {}
        editions = data.get("editions", []) if isinstance(data, dict) else []
        return [Edition.model_validate(e) for e in editions]

    def get_edition(self, edition_id: str) -> Dict[str, Any]:
        """GET /editions/{edition_id}/changelog — edition details + changelog."""
        resp = self._get(f"/editions/{edition_id}/changelog")
        return resp.data if isinstance(resp.data, dict) else {"raw": resp.data}

    def diff(
        self,
        factor_id: str,
        left_edition: str,
        right_edition: str,
    ) -> FactorDiff:
        """GET /factors/{id}/diff — field-level diff between editions."""
        params = {"left_edition": left_edition, "right_edition": right_edition}
        resp = self._get(f"/factors/{factor_id}/diff", params=params)
        return FactorDiff.model_validate(resp.data)

    def audit_bundle(
        self,
        factor_id: str,
        *,
        edition: Optional[str] = None,
    ) -> AuditBundle:
        """GET /factors/{id}/audit-bundle — Enterprise-only audit trail."""
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = self._get(
            f"/factors/{factor_id}/audit-bundle", params=params or None
        )
        return AuditBundle.model_validate(resp.data)

    # =====================================================================
    # Sources / method packs (stubs — forwards to the catalog endpoints)
    # =====================================================================

    def list_sources(
        self,
        *,
        edition: Optional[str] = None,
    ) -> List[Source]:
        """GET /factors/source-registry — list sources."""
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = self._get("/factors/source-registry", params=params or None)
        data = resp.data
        if isinstance(data, dict):
            rows = data.get("sources") or data.get("items") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return [Source.model_validate(r) for r in rows]

    def get_source(self, source_id: str) -> Source:
        """GET /factors/sources/{source_id} — fetch a source descriptor."""
        resp = self._get(f"/factors/sources/{source_id}")
        return Source.model_validate(resp.data)

    def list_method_packs(self) -> List[MethodPack]:
        """GET /method-packs — list method packs."""
        resp = self._get("/method-packs")
        data = resp.data
        if isinstance(data, dict):
            rows = data.get("method_packs") or data.get("items") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return [MethodPack.model_validate(r) for r in rows]

    def get_method_pack(self, method_pack_id: str) -> MethodPack:
        """GET /method-packs/{id} — fetch a method pack descriptor."""
        resp = self._get(f"/method-packs/{method_pack_id}")
        return MethodPack.model_validate(resp.data)

    # =====================================================================
    # Tenant overrides (Consulting/Platform tier)
    # =====================================================================

    def set_override(
        self,
        override: Union[Override, Dict[str, Any]],
    ) -> Override:
        """POST /factors/overrides — create or update a tenant override."""
        body = (
            override.model_dump(exclude_none=True)
            if isinstance(override, Override)
            else dict(override)
        )
        resp = self._post("/factors/overrides", json_body=body)
        return Override.model_validate(resp.data)

    def list_overrides(
        self,
        *,
        tenant_id: Optional[str] = None,
    ) -> List[Override]:
        """GET /factors/overrides — list tenant overrides."""
        params: Dict[str, Any] = {}
        if tenant_id:
            params["tenant_id"] = tenant_id
        resp = self._get("/factors/overrides", params=params or None)
        data = resp.data
        if isinstance(data, dict):
            rows = data.get("overrides") or data.get("items") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return [Override.model_validate(r) for r in rows]


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class AsyncFactorsClient:
    """Asynchronous mirror of :class:`FactorsClient`.

    Example::

        async with AsyncFactorsClient(base_url="...", api_key="...") as c:
            hits = await c.search("diesel")
    """

    DEFAULT_API_PREFIX: str = "/api/v1"

    def __init__(
        self,
        base_url: str,
        *,
        auth: Optional[AuthProvider] = None,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        default_edition: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        user_agent: str = DEFAULT_USER_AGENT,
        cache: Optional[ETagCache] = None,
        transport: Optional[Any] = None,
        api_prefix: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._api_prefix = (api_prefix or self.DEFAULT_API_PREFIX).rstrip("/") or ""
        self._transport = AsyncTransport(
            base_url=base_url,
            auth=_normalize_auth(auth, api_key, jwt_token),
            timeout=timeout,
            max_retries=max_retries,
            user_agent=user_agent,
            default_edition=default_edition,
            cache=cache,
            transport=transport,
            extra_headers=extra_headers,
        )

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> "AsyncFactorsClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    @property
    def cache(self) -> ETagCache:
        return self._transport.cache

    def _path(self, suffix: str) -> str:
        s = suffix if suffix.startswith("/") else "/" + suffix
        return self._api_prefix + s

    async def _get(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> TransportResponse:
        return await self._transport.request(
            "GET", self._path(path), params=params, use_cache=use_cache
        )

    async def _post(
        self,
        path: str,
        *,
        json_body: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> TransportResponse:
        return await self._transport.request(
            "POST",
            self._path(path),
            params=params,
            json_body=json_body,
            use_cache=False,
        )

    # ---- Search / list ---------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        geography: Optional[str] = None,
        limit: int = 20,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> SearchResponse:
        params: Dict[str, Any] = {"q": query, "limit": limit}
        if geography:
            params["geography"] = geography
        if edition:
            params["edition"] = edition
        ip = _bool_param(include_preview)
        ic = _bool_param(include_connector)
        if ip is not None:
            params["include_preview"] = ip
        if ic is not None:
            params["include_connector"] = ic
        resp = await self._get("/factors/search", params=params)
        return _build_search_response(resp.data)

    async def search_v2(
        self,
        query: str,
        *,
        geography: Optional[str] = None,
        fuel_type: Optional[str] = None,
        scope: Optional[str] = None,
        source_id: Optional[str] = None,
        factor_status: Optional[str] = None,
        license_class: Optional[str] = None,
        dqs_min: Optional[float] = None,
        valid_on_date: Optional[str] = None,
        sector_tags: Optional[List[str]] = None,
        activity_tags: Optional[List[str]] = None,
        sort_by: str = "relevance",
        sort_order: str = "desc",
        offset: int = 0,
        limit: int = 20,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> SearchResponse:
        body: Dict[str, Any] = {
            "query": query,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "offset": offset,
            "limit": limit,
        }
        for key, value in (
            ("geography", geography),
            ("fuel_type", fuel_type),
            ("scope", scope),
            ("source_id", source_id),
            ("factor_status", factor_status),
            ("license_class", license_class),
            ("dqs_min", dqs_min),
            ("valid_on_date", valid_on_date),
            ("sector_tags", sector_tags),
            ("activity_tags", activity_tags),
        ):
            if value is not None:
                body[key] = value
        if include_preview is not None:
            body["include_preview"] = include_preview
        if include_connector is not None:
            body["include_connector"] = include_connector
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = await self._post(
            "/factors/search/v2", json_body=body, params=params or None
        )
        return _build_search_response(resp.data)

    async def list_factors(
        self,
        *,
        fuel_type: Optional[str] = None,
        geography: Optional[str] = None,
        scope: Optional[str] = None,
        boundary: Optional[str] = None,
        edition: Optional[str] = None,
        page: int = 1,
        limit: int = 100,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> SearchResponse:
        params: Dict[str, Any] = {"page": page, "limit": limit}
        for key, value in (
            ("fuel_type", fuel_type),
            ("geography", geography),
            ("scope", scope),
            ("boundary", boundary),
            ("edition", edition),
        ):
            if value is not None:
                params[key] = value
        ip = _bool_param(include_preview)
        ic = _bool_param(include_connector)
        if ip is not None:
            params["include_preview"] = ip
        if ic is not None:
            params["include_connector"] = ic
        resp = await self._get("/factors", params=params)
        return _build_search_response(resp.data)

    def paginate_search(
        self,
        query: str,
        *,
        page_size: int = 100,
        max_items: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncOffsetPaginator:
        async def _fetch(offset: int, limit: int):
            resp = await self.search_v2(query, offset=offset, limit=limit, **kwargs)
            return list(resp.factors), resp.total_count

        return AsyncOffsetPaginator(
            _fetch, page_size=page_size, max_items=max_items
        )

    # ---- Factors ---------------------------------------------------------

    async def get_factor(
        self,
        factor_id: str,
        *,
        edition: Optional[str] = None,
    ) -> Factor:
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = await self._get(f"/factors/{factor_id}", params=params or None)
        return Factor.model_validate(resp.data)

    async def match(
        self,
        activity_description: str,
        *,
        geography: Optional[str] = None,
        fuel_type: Optional[str] = None,
        scope: Optional[str] = None,
        limit: int = 10,
        edition: Optional[str] = None,
    ) -> List[FactorMatch]:
        body: Dict[str, Any] = {
            "activity_description": activity_description,
            "limit": limit,
        }
        for key, value in (
            ("geography", geography),
            ("fuel_type", fuel_type),
            ("scope", scope),
        ):
            if value is not None:
                body[key] = value
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = await self._post(
            "/factors/match", json_body=body, params=params or None
        )
        data = resp.data or {}
        candidates = (
            data.get("candidates", []) if isinstance(data, dict) else []
        )
        return [FactorMatch.model_validate(c) for c in candidates]

    async def coverage(self, *, edition: Optional[str] = None) -> CoverageReport:
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = await self._get("/factors/coverage", params=params or None)
        return CoverageReport.model_validate(resp.data)

    # ---- Resolution ------------------------------------------------------

    async def resolve_explain(
        self,
        factor_id: str,
        *,
        method_profile: Optional[str] = None,
        alternates: Optional[int] = None,
        edition: Optional[str] = None,
    ) -> ResolvedFactor:
        params: Dict[str, Any] = {}
        if method_profile:
            params["method_profile"] = method_profile
        if alternates is not None:
            params["limit"] = alternates
        if edition:
            params["edition"] = edition
        resp = await self._get(
            f"/factors/{factor_id}/explain", params=params or None
        )
        return ResolvedFactor.model_validate(resp.data)

    async def resolve(
        self,
        request: Union[ResolutionRequest, Dict[str, Any]],
        *,
        alternates: Optional[int] = None,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> ResolvedFactor:
        if isinstance(request, ResolutionRequest):
            body = request.model_dump(exclude_none=True)
        else:
            body = dict(request)
        params: Dict[str, Any] = {}
        if alternates is not None:
            params["limit"] = alternates
        if edition:
            params["edition"] = edition
        if include_preview is not None:
            params["include_preview"] = _bool_param(include_preview)
        if include_connector is not None:
            params["include_connector"] = _bool_param(include_connector)
        resp = await self._post(
            "/factors/resolve-explain",
            json_body=body,
            params=params or None,
        )
        return ResolvedFactor.model_validate(resp.data)

    async def alternates(
        self,
        factor_id: str,
        *,
        method_profile: Optional[str] = None,
        limit: Optional[int] = None,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if method_profile:
            params["method_profile"] = method_profile
        if limit is not None:
            params["limit"] = limit
        if edition:
            params["edition"] = edition
        if include_preview is not None:
            params["include_preview"] = _bool_param(include_preview)
        if include_connector is not None:
            params["include_connector"] = _bool_param(include_connector)
        resp = await self._get(
            f"/factors/{factor_id}/alternates", params=params or None
        )
        return resp.data if isinstance(resp.data, dict) else {"raw": resp.data}

    async def resolve_batch(
        self,
        requests: List[Union[ResolutionRequest, Dict[str, Any]]],
        *,
        edition: Optional[str] = None,
    ) -> BatchJobHandle:
        items: List[Dict[str, Any]] = []
        for r in requests:
            if isinstance(r, ResolutionRequest):
                items.append(r.model_dump(exclude_none=True))
            else:
                items.append(dict(r))
        body: Dict[str, Any] = {"requests": items}
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = await self._post(
            "/factors/resolve/batch", json_body=body, params=params or None
        )
        return BatchJobHandle.model_validate(resp.data)

    async def get_batch_job(self, job_id: str) -> BatchJobHandle:
        resp = await self._get(f"/factors/jobs/{job_id}", use_cache=False)
        return BatchJobHandle.model_validate(resp.data)

    async def wait_for_batch(
        self,
        job: Union[BatchJobHandle, str],
        *,
        poll_interval: float = 2.0,
        timeout: Optional[float] = 600.0,
    ) -> BatchJobHandle:
        job_id = job.job_id if isinstance(job, BatchJobHandle) else str(job)
        loop = asyncio.get_event_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while True:
            current = await self.get_batch_job(job_id)
            if current.status in _BATCH_TERMINAL_STATES:
                if current.status == "failed":
                    raise FactorsAPIError(
                        "Batch job %s failed: %s"
                        % (job_id, current.error_message or "unknown error"),
                        context={"job_id": job_id, "status": current.status},
                    )
                return current
            if deadline is not None and loop.time() > deadline:
                raise FactorsAPIError(
                    "Timeout waiting for batch job %s (status=%s)"
                    % (job_id, current.status),
                    context={"job_id": job_id, "timeout": timeout},
                )
            await asyncio.sleep(poll_interval)

    # ---- Editions --------------------------------------------------------

    async def list_editions(
        self,
        *,
        include_pending: bool = True,
    ) -> List[Edition]:
        params = {"include_pending": _bool_param(include_pending)}
        resp = await self._get("/editions", params=params)
        data = resp.data if isinstance(resp.data, dict) else {}
        editions = data.get("editions", []) if isinstance(data, dict) else []
        return [Edition.model_validate(e) for e in editions]

    async def get_edition(self, edition_id: str) -> Dict[str, Any]:
        resp = await self._get(f"/editions/{edition_id}/changelog")
        return resp.data if isinstance(resp.data, dict) else {"raw": resp.data}

    async def diff(
        self,
        factor_id: str,
        left_edition: str,
        right_edition: str,
    ) -> FactorDiff:
        params = {"left_edition": left_edition, "right_edition": right_edition}
        resp = await self._get(f"/factors/{factor_id}/diff", params=params)
        return FactorDiff.model_validate(resp.data)

    async def audit_bundle(
        self,
        factor_id: str,
        *,
        edition: Optional[str] = None,
    ) -> AuditBundle:
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = await self._get(
            f"/factors/{factor_id}/audit-bundle", params=params or None
        )
        return AuditBundle.model_validate(resp.data)

    # ---- Sources / method packs -----------------------------------------

    async def list_sources(
        self,
        *,
        edition: Optional[str] = None,
    ) -> List[Source]:
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = await self._get(
            "/factors/source-registry", params=params or None
        )
        data = resp.data
        if isinstance(data, dict):
            rows = data.get("sources") or data.get("items") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return [Source.model_validate(r) for r in rows]

    async def get_source(self, source_id: str) -> Source:
        resp = await self._get(f"/factors/sources/{source_id}")
        return Source.model_validate(resp.data)

    async def list_method_packs(self) -> List[MethodPack]:
        resp = await self._get("/method-packs")
        data = resp.data
        if isinstance(data, dict):
            rows = data.get("method_packs") or data.get("items") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return [MethodPack.model_validate(r) for r in rows]

    async def get_method_pack(self, method_pack_id: str) -> MethodPack:
        resp = await self._get(f"/method-packs/{method_pack_id}")
        return MethodPack.model_validate(resp.data)

    # ---- Overrides -------------------------------------------------------

    async def set_override(
        self,
        override: Union[Override, Dict[str, Any]],
    ) -> Override:
        body = (
            override.model_dump(exclude_none=True)
            if isinstance(override, Override)
            else dict(override)
        )
        resp = await self._post("/factors/overrides", json_body=body)
        return Override.model_validate(resp.data)

    async def list_overrides(
        self,
        *,
        tenant_id: Optional[str] = None,
    ) -> List[Override]:
        params: Dict[str, Any] = {}
        if tenant_id:
            params["tenant_id"] = tenant_id
        resp = await self._get("/factors/overrides", params=params or None)
        data = resp.data
        if isinstance(data, dict):
            rows = data.get("overrides") or data.get("items") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return [Override.model_validate(r) for r in rows]


__all__ = ["FactorsClient", "AsyncFactorsClient"]
