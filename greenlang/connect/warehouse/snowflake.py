# -*- coding: utf-8 -*-
"""Snowflake connector — canonical data-warehouse example.

Requires `snowflake-connector-python`. Production uses key-pair auth + PrivateLink.
"""

from __future__ import annotations

import hashlib
import json
import logging

from greenlang.connect.base import BaseConnector, ConnectorResult, SourceSpec

logger = logging.getLogger(__name__)


class SnowflakeConnector(BaseConnector):
    connector_id = "snowflake"

    async def extract(self, spec: SourceSpec) -> ConnectorResult:
        query = spec.filters.get("query")
        if not query:
            raise ValueError("SnowflakeConnector requires filters['query']")
        # Production: open snowflake.connector.connect(...) with spec.credentials
        # and execute query, stream rows to canonical records.
        records: list[dict] = []
        logger.info("SnowflakeConnector.extract tenant=%s rows=%d", spec.tenant_id, len(records))
        checksum = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        return ConnectorResult(
            connector_id=self.connector_id,
            records=records,
            row_count=len(records),
            checksum=checksum,
            metadata={"source_system": "Snowflake", "query": query},
        )

    async def healthcheck(self, credentials: dict[str, str]) -> bool:
        return (
            bool(credentials.get("account"))
            and bool(credentials.get("user"))
            and (bool(credentials.get("password")) or bool(credentials.get("private_key")))
        )
