# -*- coding: utf-8 -*-
"""Workday connector — headcount + facility assignment feed.

Workday RaaS (Report as a Service) endpoints return JSON.  The connector
queries a customer-provided report URL with basic auth and flattens the
rows into canonical records used by Scope 3 Cat 7 (employee commuting).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from greenlang.connect.base import (
    BaseConnector,
    ConnectorExtractionError,
    SourceSpec,
)

logger = logging.getLogger(__name__)


class WorkdayConnector(BaseConnector):
    """Workday RaaS JSON report extractor."""

    connector_id = "workday"
    required_credentials = ("report_url", "username", "password")
    required_python_package = "httpx"

    async def _extract_records(self, spec: SourceSpec) -> Iterable[dict[str, Any]]:
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConnectorExtractionError(
                "httpx is required for Workday extraction"
            ) from exc

        url = spec.credentials["report_url"]
        auth = (spec.credentials["username"], spec.credentials["password"])
        params = dict(spec.filters.get("params") or {})
        params.setdefault("format", "json")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(url, params=params, auth=auth)
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            raise ConnectorExtractionError(
                f"Workday RaaS request failed: {exc}"
            ) from exc

        rows = payload.get("Report_Entry") or payload.get("rows") or []
        records: list[dict[str, Any]] = []
        for row in rows:
            records.append(
                {
                    "source_system": "workday",
                    "tenant_id": spec.tenant_id,
                    "raw": row,
                }
            )
        return records
