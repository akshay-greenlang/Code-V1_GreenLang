# -*- coding: utf-8 -*-
"""Snowflake connector — canonical data-warehouse example.

Requires ``snowflake-connector-python``.  Production deployments use
key-pair auth + PrivateLink; password auth is supported for dev.
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


class SnowflakeConnector(BaseConnector):
    """Snowflake SQL query extractor."""

    connector_id = "snowflake"
    required_credentials = ("account", "user", "warehouse")
    required_python_package = "snowflake.connector"

    async def _extract_records(self, spec: SourceSpec) -> Iterable[dict[str, Any]]:
        query = spec.filters.get("query")
        if not query:
            raise ConnectorExtractionError(
                "SnowflakeConnector requires filters['query']"
            )

        # Either 'password' or 'private_key' (or 'private_key_path') must be present.
        if not (
            spec.credentials.get("password")
            or spec.credentials.get("private_key")
            or spec.credentials.get("private_key_path")
        ):
            raise ConnectorAuthError(
                "snowflake: missing password or private_key"
            )

        try:
            import snowflake.connector as snow  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConnectorExtractionError(
                "snowflake-connector-python is required for Snowflake extraction"
            ) from exc

        connect_kwargs = {
            "account": spec.credentials["account"],
            "user": spec.credentials["user"],
            "warehouse": spec.credentials["warehouse"],
        }
        for optional in ("database", "schema", "role"):
            if spec.credentials.get(optional):
                connect_kwargs[optional] = spec.credentials[optional]
        if spec.credentials.get("password"):
            connect_kwargs["password"] = spec.credentials["password"]
        elif spec.credentials.get("private_key"):
            connect_kwargs["private_key"] = spec.credentials["private_key"]
        elif spec.credentials.get("private_key_path"):
            connect_kwargs["private_key_file"] = spec.credentials["private_key_path"]

        conn = snow.connect(**connect_kwargs)
        try:
            cursor = conn.cursor(snow.DictCursor)
            try:
                cursor.execute(query)
                rows = cursor.fetchall()
            finally:
                cursor.close()
        finally:
            conn.close()

        records = [
            {
                "source_system": "snowflake",
                "tenant_id": spec.tenant_id,
                "query": query,
                "raw": dict(row),
            }
            for row in rows
        ]
        return records
