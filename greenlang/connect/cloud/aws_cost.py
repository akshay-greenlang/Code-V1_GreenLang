# -*- coding: utf-8 -*-
"""AWS Cost Explorer connector — compute spend-based emissions for Scope 3 Cat 1."""

from __future__ import annotations

import hashlib
import json
import logging

from greenlang.connect.base import BaseConnector, ConnectorResult, SourceSpec

logger = logging.getLogger(__name__)


class AWSCostExplorerConnector(BaseConnector):
    connector_id = "aws-cost-explorer"

    async def extract(self, spec: SourceSpec) -> ConnectorResult:
        # Production: boto3 CostExplorer client
        #   ce = boto3.client('ce', aws_access_key_id=..., aws_secret_access_key=...)
        #   resp = ce.get_cost_and_usage(TimePeriod={...}, Granularity='MONTHLY', Metrics=['UnblendedCost'])
        records: list[dict] = []
        logger.info("AWSCostExplorer.extract tenant=%s rows=%d", spec.tenant_id, len(records))
        checksum = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        return ConnectorResult(
            connector_id=self.connector_id,
            records=records,
            row_count=len(records),
            checksum=checksum,
            metadata={"source_system": "AWS Cost Explorer"},
        )

    async def healthcheck(self, credentials: dict[str, str]) -> bool:
        return bool(credentials.get("aws_access_key_id")) and bool(
            credentials.get("aws_secret_access_key")
        )
