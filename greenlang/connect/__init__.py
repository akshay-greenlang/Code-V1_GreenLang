# -*- coding: utf-8 -*-
"""GreenLang Connect — customer-side integration hub.

Bridges customer source systems (SAP/Oracle/Workday/Snowflake/BigQuery/Azure/
Salesforce/AWS Cost Explorer) into GreenLang's ingestion pipeline.

Usage:
    from greenlang.connect import ConnectorRegistry, SourceSpec
    registry = ConnectorRegistry()
    connector = registry.get("sap-s4hana")
    data = await connector.extract(SourceSpec(tenant_id="...", credentials={...}))
"""

from greenlang.connect.base import (  # noqa: F401
    BaseConnector,
    ConnectorRegistry,
    ConnectorResult,
    SourceSpec,
    default_registry,
)
from greenlang.connect.erp.sap_s4hana import SAPS4HanaConnector  # noqa: F401
from greenlang.connect.warehouse.snowflake import SnowflakeConnector  # noqa: F401
from greenlang.connect.cloud.aws_cost import AWSCostExplorerConnector  # noqa: F401


# Auto-register built-in connectors
_registry = default_registry()
_registry.register("sap-s4hana", SAPS4HanaConnector)
_registry.register("snowflake", SnowflakeConnector)
_registry.register("aws-cost-explorer", AWSCostExplorerConnector)


__all__ = [
    "BaseConnector",
    "ConnectorRegistry",
    "ConnectorResult",
    "SourceSpec",
    "default_registry",
    "SAPS4HanaConnector",
    "SnowflakeConnector",
    "AWSCostExplorerConnector",
]
