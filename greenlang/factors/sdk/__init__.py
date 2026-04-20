# -*- coding: utf-8 -*-
"""
GreenLang Factors Python SDK (F037).

Full typed HTTP client for the Factors API. Stdlib-only (no requests dependency).
Supports: editions, search, match, calculate, export, audit-bundle, diff, provenance.

Usage:
    from greenlang.factors.sdk import FactorsClient, FactorsConfig

    client = FactorsClient(FactorsConfig(
        base_url="https://api.greenlang.io/api/v1",
        api_key="gl_...",
    ))
    results = client.search("diesel US Scope 1")
    for f in results["factors"]:
        print(f["factor_id"], f["co2e_per_unit"])
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

SDK_VERSION = "1.1.0"


class FactorsApiError(Exception):
    """Raised when the API returns a non-2xx status code."""

    def __init__(self, status_code: int, message: str, body: Optional[str] = None):
        self.status_code = status_code
        self.message = message
        self.body = body
        super().__init__(f"HTTP {status_code}: {message}")


class FactorsConnectionError(Exception):
    """Raised when the client cannot connect to the API."""


@dataclass
class FactorsConfig:
    """Configuration for the Factors SDK client."""

    base_url: str
    api_key: Optional[str] = None
    edition: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    retry_backoff: float = 1.0
    user_agent: str = field(default_factory=lambda: f"greenlang-factors-sdk/{SDK_VERSION}")


# Backward compat alias
FactorsSdkConfig = FactorsConfig


class FactorsClient:
    """
    Full-featured HTTP client for the GreenLang Factors API.

    All methods return parsed JSON dicts. Raises FactorsApiError on
    non-2xx responses and FactorsConnectionError on network failures.
    """

    def __init__(self, config: FactorsConfig):
        self._cfg = config
        self._base = config.base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": self._cfg.user_agent,
        }
        if self._cfg.api_key:
            h["Authorization"] = f"Bearer {self._cfg.api_key}"
        if self._cfg.edition:
            h["X-Factors-Edition"] = self._cfg.edition
        return h

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base}{path}"
        if params:
            q = urlencode({k: v for k, v in params.items() if v is not None})
            if q:
                url = f"{url}?{q}"

        headers = self._headers()
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        last_err = None
        for attempt in range(self._cfg.max_retries):
            try:
                req = urllib.request.Request(
                    url, data=data, headers=headers, method=method,
                )
                with urllib.request.urlopen(req, timeout=self._cfg.timeout) as resp:  # nosec B310
                    resp_body = resp.read().decode("utf-8")
                    return json.loads(resp_body) if resp_body else {}
            except urllib.error.HTTPError as e:
                resp_body = ""
                try:
                    resp_body = e.read().decode("utf-8")
                except Exception:
                    pass
                # Retry on 429 and 5xx
                if e.code in (429, 500, 502, 503, 504) and attempt < self._cfg.max_retries - 1:
                    wait = self._cfg.retry_backoff * (2 ** attempt)
                    logger.warning(
                        "Retrying %s %s (HTTP %d) in %.1fs (attempt %d/%d)",
                        method, path, e.code, wait, attempt + 1, self._cfg.max_retries,
                    )
                    time.sleep(wait)
                    last_err = FactorsApiError(e.code, str(e.reason), resp_body)
                    continue
                raise FactorsApiError(e.code, str(e.reason), resp_body) from e
            except urllib.error.URLError as e:
                if attempt < self._cfg.max_retries - 1:
                    wait = self._cfg.retry_backoff * (2 ** attempt)
                    logger.warning(
                        "Connection error %s %s: %s, retrying in %.1fs",
                        method, path, e, wait,
                    )
                    time.sleep(wait)
                    last_err = FactorsConnectionError(str(e))
                    continue
                raise FactorsConnectionError(str(e)) from e

        if last_err:
            raise last_err
        raise FactorsConnectionError("Max retries exceeded")

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("POST", path, params=params, body=body)

    # ---- Edition endpoints ----

    def list_editions(self, include_pending: bool = True) -> Dict[str, Any]:
        """List all catalog editions."""
        return self._get("/editions", {"include_pending": str(include_pending).lower()})

    def get_changelog(self, edition_id: str) -> Dict[str, Any]:
        """Get changelog for an edition."""
        return self._get(f"/editions/{edition_id}/changelog")

    def compare_editions(self, left: str, right: str) -> Dict[str, Any]:
        """Compare two editions (added/removed/changed factors)."""
        return self._get("/editions/compare", {"left": left, "right": right})

    # ---- Factor endpoints ----

    def list_factors(
        self,
        *,
        fuel_type: Optional[str] = None,
        geography: Optional[str] = None,
        scope: Optional[str] = None,
        page: int = 1,
        limit: int = 100,
        include_preview: bool = False,
        include_connector: bool = False,
    ) -> Dict[str, Any]:
        """List factors with optional filters and pagination."""
        params: Dict[str, Any] = {
            "page": page,
            "limit": limit,
            "include_preview": str(include_preview).lower(),
            "include_connector": str(include_connector).lower(),
        }
        if fuel_type:
            params["fuel_type"] = fuel_type
        if geography:
            params["geography"] = geography
        if scope:
            params["scope"] = scope
        return self._get("/factors", params)

    def get_factor(self, factor_id: str) -> Dict[str, Any]:
        """Get detailed factor by ID."""
        return self._get(f"/factors/{factor_id}")

    def get_provenance(self, factor_id: str) -> Dict[str, Any]:
        """Get provenance and license info for a factor."""
        return self._get(f"/factors/{factor_id}/provenance")

    def get_replacements(self, factor_id: str) -> Dict[str, Any]:
        """Get deprecation replacement chain for a factor."""
        return self._get(f"/factors/{factor_id}/replacements")

    def get_audit_bundle(self, factor_id: str) -> Dict[str, Any]:
        """Get audit bundle for a factor (enterprise tier only)."""
        return self._get(f"/factors/{factor_id}/audit-bundle")

    def diff_factor(self, factor_id: str, left_edition: str, right_edition: str) -> Dict[str, Any]:
        """Get field-by-field diff of a factor between two editions."""
        return self._get(
            f"/factors/{factor_id}/diff",
            {"left_edition": left_edition, "right_edition": right_edition},
        )

    # ---- Search endpoints ----

    def search(
        self,
        query: str,
        *,
        geography: Optional[str] = None,
        limit: int = 20,
        include_preview: bool = False,
    ) -> Dict[str, Any]:
        """Search factors by text query."""
        params: Dict[str, Any] = {
            "q": query,
            "limit": limit,
            "include_preview": str(include_preview).lower(),
        }
        if geography:
            params["geography"] = geography
        return self._get("/factors/search", params)

    def search_v2(
        self,
        query: str,
        *,
        geography: Optional[str] = None,
        fuel_type: Optional[str] = None,
        scope: Optional[str] = None,
        source_id: Optional[str] = None,
        dqs_min: Optional[float] = None,
        sort_by: str = "relevance",
        sort_order: str = "desc",
        offset: int = 0,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Advanced search with filters, sort, and pagination."""
        body: Dict[str, Any] = {"query": query, "sort_by": sort_by, "sort_order": sort_order, "offset": offset, "limit": limit}
        if geography:
            body["geography"] = geography
        if fuel_type:
            body["fuel_type"] = fuel_type
        if scope:
            body["scope"] = scope
        if source_id:
            body["source_id"] = source_id
        if dqs_min is not None:
            body["dqs_min"] = dqs_min
        return self._post("/factors/search/v2", body)

    def get_facets(self, include_preview: bool = False) -> Dict[str, Any]:
        """Get facet counts for filter UI."""
        return self._get(
            "/factors/search/facets",
            {"include_preview": str(include_preview).lower()},
        )

    # ---- Match endpoint ----

    def match(
        self,
        activity_description: str,
        *,
        geography: Optional[str] = None,
        fuel_type: Optional[str] = None,
        scope: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Match an activity description to emission factors."""
        body: Dict[str, Any] = {
            "activity_description": activity_description,
            "limit": limit,
        }
        if geography:
            body["geography"] = geography
        if fuel_type:
            body["fuel_type"] = fuel_type
        if scope:
            body["scope"] = scope
        return self._post("/factors/match", body)

    # ---- Calculation endpoints ----

    def calculate(
        self,
        fuel_type: str,
        activity_amount: float,
        activity_unit: str,
        *,
        geography: str = "US",
        scope: str = "1",
        boundary: str = "combustion",
    ) -> Dict[str, Any]:
        """Calculate emissions for a single activity."""
        return self._post("/calculate", {
            "fuel_type": fuel_type,
            "activity_amount": activity_amount,
            "activity_unit": activity_unit,
            "geography": geography,
            "scope": scope,
            "boundary": boundary,
        })

    def calculate_batch(self, calculations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate emissions for multiple activities."""
        return self._post("/calculate/batch", {"calculations": calculations})

    # ---- Export endpoint ----

    def export(
        self,
        *,
        status: Optional[str] = None,
        geography: Optional[str] = None,
        fuel_type: Optional[str] = None,
        scope: Optional[str] = None,
        source_id: Optional[str] = None,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Bulk export factors (pro/enterprise tier)."""
        params: Dict[str, Any] = {"format": format}
        if status:
            params["status"] = status
        if geography:
            params["geography"] = geography
        if fuel_type:
            params["fuel_type"] = fuel_type
        if scope:
            params["scope"] = scope
        if source_id:
            params["source_id"] = source_id
        return self._get("/factors/export", params)

    # ---- System endpoints ----

    def health(self) -> Dict[str, Any]:
        """Check API health."""
        return self._get("/health")

    def stats(self) -> Dict[str, Any]:
        """Get API statistics."""
        return self._get("/stats")

    def coverage(self) -> Dict[str, Any]:
        """Get coverage statistics."""
        return self._get("/stats/coverage")

    def source_registry(self) -> Dict[str, Any]:
        """List source registry entries."""
        return self._get("/factors/source-registry")


# Backward compat alias
FactorsSdk = FactorsClient


__all__ = [
    "FactorsClient",
    "FactorsConfig",
    "FactorsApiError",
    "FactorsConnectionError",
    "FactorsSdk",
    "FactorsSdkConfig",
    "SDK_VERSION",
]
