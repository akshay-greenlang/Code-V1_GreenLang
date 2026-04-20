# -*- coding: utf-8 -*-
"""GreenLang Connect — customer-side integration hub.

Bridges customer source systems (SAP/Workday/Snowflake/Databricks/AWS Cost
Explorer) into GreenLang's ingestion pipeline for Scope 1/2/3 accounting.

Usage::

    from greenlang.connect import ConnectorRegistry, SourceSpec
    registry = ConnectorRegistry()
    connector = registry.get("sap-s4hana")
    result = await connector.extract(
        SourceSpec(tenant_id="t1", connector_id="sap-s4hana",
                   credentials={...}, dry_run=True),
    )
"""

from greenlang.connect.base import (  # noqa: F401
    BaseConnector,
    ConnectorAuthError,
    ConnectorDependencyError,
    ConnectorError,
    ConnectorExtractionError,
    ConnectorRegistry,
    ConnectorResult,
    HealthCheckResult,
    SourceSpec,
    default_registry,
)
from greenlang.connect.erp.sap_s4hana import SAPS4HanaConnector  # noqa: F401
from greenlang.connect.hris.workday import WorkdayConnector  # noqa: F401
from greenlang.connect.warehouse.snowflake import SnowflakeConnector  # noqa: F401
from greenlang.connect.warehouse.databricks import DatabricksConnector  # noqa: F401
from greenlang.connect.cloud.aws_cost import AWSCostExplorerConnector  # noqa: F401


# Auto-register built-in connectors.
_registry = default_registry()
_registry.register("sap-s4hana", SAPS4HanaConnector)
_registry.register("workday", WorkdayConnector)
_registry.register("snowflake", SnowflakeConnector)
_registry.register("databricks", DatabricksConnector)
_registry.register("aws-cost-explorer", AWSCostExplorerConnector)


__all__ = [
    "BaseConnector",
    "ConnectorAuthError",
    "ConnectorDependencyError",
    "ConnectorError",
    "ConnectorExtractionError",
    "ConnectorRegistry",
    "ConnectorResult",
    "HealthCheckResult",
    "SourceSpec",
    "default_registry",
    "SAPS4HanaConnector",
    "WorkdayConnector",
    "SnowflakeConnector",
    "DatabricksConnector",
    "AWSCostExplorerConnector",
]
