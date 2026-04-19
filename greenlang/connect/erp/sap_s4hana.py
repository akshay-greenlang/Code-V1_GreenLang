# -*- coding: utf-8 -*-
"""SAP S/4HANA connector — canonical ERP example.

Pulls spend + material movements via the OData API. Real auth uses OAuth2
client credentials or SAML assertions. This skeleton shows the expected
data-shape; production implementation wires to the existing ERP agents under
`greenlang/agents/data/erp/`.
"""

from __future__ import annotations

import hashlib
import json
import logging

from greenlang.connect.base import BaseConnector, ConnectorResult, SourceSpec

logger = logging.getLogger(__name__)


class SAPS4HanaConnector(BaseConnector):
    connector_id = "sap-s4hana"

    async def extract(self, spec: SourceSpec) -> ConnectorResult:
        # In production, call the S/4HANA OData endpoints (e.g.
        # `/sap/opu/odata/sap/API_PURCHASEORDER_PROCESS_SRV/A_PurchaseOrderItem`)
        # with spec.credentials (baseUrl, client_id, client_secret, x-csrf-token).
        # Here we emit an empty canonical payload so the path is runnable.
        records: list[dict] = []
        logger.info(
            "SAPS4HanaConnector.extract tenant=%s filters=%s",
            spec.tenant_id, spec.filters,
        )
        checksum = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        return ConnectorResult(
            connector_id=self.connector_id,
            records=records,
            row_count=len(records),
            checksum=checksum,
            metadata={"source_system": "SAP S/4HANA", "endpoint": "odata"},
        )

    async def healthcheck(self, credentials: dict[str, str]) -> bool:
        return bool(credentials.get("base_url")) and bool(credentials.get("client_id"))
