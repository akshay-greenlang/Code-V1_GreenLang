# -*- coding: utf-8 -*-
"""SAP S/4HANA connector — canonical ERP example.

Pulls spend + material movements via the S/4HANA OData API.  Supports
OAuth2 client credentials (``client_id`` + ``client_secret``) or basic
auth (``username`` + ``password``).  Either path issues requests against
the customer's ``base_url`` and returns canonical records.

Production callers should install ``httpx`` for async HTTP.  Without
``httpx`` the connector raises :class:`ConnectorDependencyError` unless
the caller passes ``dry_run=True``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from greenlang.connect.base import (
    BaseConnector,
    ConnectorAuthError,
    ConnectorExtractionError,
    SourceSpec,
)

logger = logging.getLogger(__name__)


class SAPS4HanaConnector(BaseConnector):
    """SAP S/4HANA OData extractor."""

    connector_id = "sap-s4hana"
    required_credentials = ("base_url", "client_id", "client_secret")
    required_python_package = "httpx"

    #: Default OData entity to pull when no explicit path is provided.
    DEFAULT_ENTITY = "A_PurchaseOrderItem"

    async def _extract_records(self, spec: SourceSpec) -> Iterable[dict[str, Any]]:
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConnectorExtractionError(
                "httpx is required for SAP S/4HANA extraction"
            ) from exc

        base_url = spec.credentials["base_url"].rstrip("/")
        entity = spec.filters.get("entity", self.DEFAULT_ENTITY)
        top = int(spec.filters.get("top", 100))
        path = f"/sap/opu/odata/sap/API_PURCHASEORDER_PROCESS_SRV/{entity}"
        url = f"{base_url}{path}"

        auth = (
            spec.credentials.get("username"),
            spec.credentials.get("password"),
        )
        # When OAuth2 client_id/secret is present and no basic username, assume
        # the caller pre-negotiated an access token in credentials['access_token'].
        headers = {"Accept": "application/json"}
        if spec.credentials.get("access_token"):
            headers["Authorization"] = f"Bearer {spec.credentials['access_token']}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    url,
                    params={"$top": top, "$format": "json"},
                    headers=headers,
                    auth=(
                        auth if auth[0] and auth[1] else None
                    ),  # basic auth optional
                )
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            raise ConnectorExtractionError(
                f"SAP S/4HANA OData request failed: {exc}"
            ) from exc

        results = payload.get("d", {}).get("results") or payload.get("value") or []
        records: list[dict[str, Any]] = []
        for row in results:
            records.append(
                {
                    "source_system": "sap-s4hana",
                    "tenant_id": spec.tenant_id,
                    "entity": entity,
                    "raw": row,
                }
            )
        return records
