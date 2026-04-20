# -*- coding: utf-8 -*-
"""Databricks connector — SQL warehouse query extractor.

Uses ``databricks-sql-connector``.  Supports personal access token (PAT)
or OAuth M2M.  Designed for Scope 1/2/3 activity data served from a
Lakehouse workspace.
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


class DatabricksConnector(BaseConnector):
    """Databricks SQL warehouse query extractor."""

    connector_id = "databricks"
    required_credentials = ("server_hostname", "http_path")
    required_python_package = "databricks.sql"

    async def _extract_records(self, spec: SourceSpec) -> Iterable[dict[str, Any]]:
        query = spec.filters.get("query")
        if not query:
            raise ConnectorExtractionError(
                "DatabricksConnector requires filters['query']"
            )

        access_token = spec.credentials.get("access_token")
        client_id = spec.credentials.get("client_id")
        client_secret = spec.credentials.get("client_secret")
        if not access_token and not (client_id and client_secret):
            raise ConnectorAuthError(
                "databricks: provide access_token OR client_id+client_secret"
            )

        try:
            from databricks import sql as dbsql  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConnectorExtractionError(
                "databricks-sql-connector is required for Databricks extraction"
            ) from exc

        connect_kwargs: dict[str, Any] = {
            "server_hostname": spec.credentials["server_hostname"],
            "http_path": spec.credentials["http_path"],
        }
        if access_token:
            connect_kwargs["access_token"] = access_token
        else:
            connect_kwargs["auth_type"] = "databricks-oauth"
            connect_kwargs["client_id"] = client_id
            connect_kwargs["client_secret"] = client_secret

        with dbsql.connect(**connect_kwargs) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

        records: list[dict[str, Any]] = []
        for row in rows:
            records.append(
                {
                    "source_system": "databricks",
                    "tenant_id": spec.tenant_id,
                    "query": query,
                    "raw": dict(zip(columns, row)),
                }
            )
        return records
