"""
GL-016 Waterguard OPC-UA Integration Package

Provides OPC-UA connectivity for industrial control systems, including
tag subscription, setpoint writing, and connection health monitoring.
"""

from integrations.opcua.opcua_connector import (
    WaterguardOPCUAConnector,
    OPCUAConfig,
    ConnectionState,
)
from integrations.opcua.opcua_schemas import (
    TagValue,
    TagSubscription,
    OPCUAQuality,
)
from integrations.opcua.opcua_write_handler import (
    WriteHandler,
    WriteCommand,
    WriteResult,
)
from integrations.opcua.tag_mapping import (
    TagMapping,
    TagDefinition,
    load_tag_mappings,
)

__all__ = [
    # Connector
    "WaterguardOPCUAConnector",
    "OPCUAConfig",
    "ConnectionState",
    # Schemas
    "TagValue",
    "TagSubscription",
    "OPCUAQuality",
    # Write handler
    "WriteHandler",
    "WriteCommand",
    "WriteResult",
    # Tag mapping
    "TagMapping",
    "TagDefinition",
    "load_tag_mappings",
]

__version__ = "1.0.0"
