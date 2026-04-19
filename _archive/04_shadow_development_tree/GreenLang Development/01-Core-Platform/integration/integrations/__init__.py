"""
GreenLang Integrations - External System Integration Framework
===============================================================

Production-grade integration framework for external systems with:
- Retry logic and circuit breaker patterns
- Health monitoring and metrics
- Type-safe generic interfaces
- Mock implementations for testing
- Connector registry and discovery

Available Connectors:
- SCADAConnector: SCADA systems (OPC UA, Modbus, DNP3, BACnet)
- ERPConnector: ERP systems (SAP, Oracle, Dynamics, NetSuite)
- CEMSConnector: Emissions monitoring systems
- HistorianConnector: Time-series historians (PI, PHD, GE)
- CMMSConnector: Maintenance management systems

Example:
    >>> from greenlang.integrations import IntegrationRegistry
    >>> from greenlang.integrations import SCADAConfig, SCADAConnector
    >>>
    >>> # Using registry
    >>> registry = IntegrationRegistry()
    >>> config = SCADAConfig(...)
    >>> connector = registry.create_connector("scada-connector", config)
    >>>
    >>> # Direct usage
    >>> connector = SCADAConnector(config)
    >>> async with connector:
    ...     data, prov = await connector.fetch_data(query)

Author: GreenLang Backend Team
Date: 2025-12-01
"""

from greenlang.integrations.base_connector import (
    BaseConnector,
    MockConnector,
    ConnectorConfig,
    ConnectorMetrics,
    ConnectorProvenance,
    HealthStatus,
    ConnectionState,
)

from greenlang.integrations.registry import (
    IntegrationRegistry,
    ConnectorRegistration,
    get_registry,
)

from greenlang.integrations.health_monitor import (
    HealthMonitor,
    HealthCheckResult,
    AggregatedHealth,
)

# Import concrete connectors
try:
    from greenlang.integrations.scada_connector import (
        SCADAConnector,
        SCADAConfig,
        SCADAQuery,
        SCADAPayload,
        SCADATag,
    )
except ImportError:
    pass

try:
    from greenlang.integrations.erp_connector import (
        ERPConnector,
        ERPConfig,
        ERPQuery,
        ERPPayload,
        ERPRecord,
    )
except ImportError:
    pass

try:
    from greenlang.integrations.cems_connector import (
        CEMSConnector,
        CEMSConfig,
        CEMSQuery,
        CEMSPayload,
        CEMSDataPoint,
    )
except ImportError:
    pass

try:
    from greenlang.integrations.historian_connector import (
        HistorianConnector,
        HistorianConfig,
        HistorianQuery,
        HistorianPayload,
        HistorianDataPoint,
    )
except ImportError:
    pass

try:
    from greenlang.integrations.cmms_connector import (
        CMMSConnector,
        CMMSConfig,
        CMMSQuery,
        CMMSPayload,
        CMMSWorkOrder,
    )
except ImportError:
    pass

__all__ = [
    # Base classes
    "BaseConnector",
    "MockConnector",
    "ConnectorConfig",
    "ConnectorMetrics",
    "ConnectorProvenance",
    "HealthStatus",
    "ConnectionState",
    # Registry
    "IntegrationRegistry",
    "ConnectorRegistration",
    "get_registry",
    # Health monitoring
    "HealthMonitor",
    "HealthCheckResult",
    "AggregatedHealth",
    # SCADA
    "SCADAConnector",
    "SCADAConfig",
    "SCADAQuery",
    "SCADAPayload",
    "SCADATag",
    # ERP
    "ERPConnector",
    "ERPConfig",
    "ERPQuery",
    "ERPPayload",
    "ERPRecord",
    # CEMS
    "CEMSConnector",
    "CEMSConfig",
    "CEMSQuery",
    "CEMSPayload",
    "CEMSDataPoint",
    # Historian
    "HistorianConnector",
    "HistorianConfig",
    "HistorianQuery",
    "HistorianPayload",
    "HistorianDataPoint",
    # CMMS
    "CMMSConnector",
    "CMMSConfig",
    "CMMSQuery",
    "CMMSPayload",
    "CMMSWorkOrder",
]

__version__ = "1.0.0"
