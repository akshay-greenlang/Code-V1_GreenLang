"""
OPC-UA Connector Package for GreenLang Process Heat Agents.

This package provides comprehensive OPC-UA connectivity for industrial systems,
enabling real-time data acquisition from DCS, PLC, and SCADA systems, as well
as exposing agent data to external clients.

Modules:
    client: OPC-UA client for reading/writing industrial data
    server: OPC-UA server for exposing agent outputs
    subscription: Real-time data subscription handler
    security: Certificate and authentication management
    types: Pydantic models for OPC-UA data structures
    utils: Helper functions for OPC-UA operations

Features:
    - Secure connections with X.509 certificates
    - Multiple authentication methods (Anonymous, Username/Password, Certificate)
    - Async operations for non-blocking I/O
    - Real-time data subscriptions with configurable intervals
    - Historical data access (HDA)
    - Data quality handling with provenance tracking
    - Address space browsing
    - Both client and server modes

Quick Start:
    # Client mode - Read from industrial systems
    from connectors.opcua import OPCUAClient, SecurityPolicy

    async with OPCUAClient("opc.tcp://192.168.1.100:4840") as client:
        # Read a temperature value
        value = await client.read_node("ns=2;s=Furnace1.Temperature.PV")
        print(f"Temperature: {value.value} (quality: {value.quality_level})")

        # Subscribe to real-time updates
        async def handle_change(notification):
            for val in notification.values:
                print(f"{val.node_id} = {val.value}")

        await client.subscribe(
            ["ns=2;s=Temperature", "ns=2;s=Pressure"],
            callback=handle_change,
        )

    # Server mode - Expose agent data
    from connectors.opcua import OPCUAServer

    async with OPCUAServer(port=4840, name="GreenLang-Server") as server:
        # Register agent namespace
        ns = await server.register_namespace("urn:greenlang:agents")

        # Expose agent outputs
        nodes = await server.expose_agent_outputs(
            agent_id="furnace_optimizer",
            outputs={
                "efficiency_score": 85.5,
                "energy_savings": 12.3,
                "recommendations": "Reduce excess air",
            }
        )

        # Keep server running
        await asyncio.sleep(3600)

Dependencies:
    - asyncua (optional, for production use)
    - cryptography (optional, for certificate generation)
    - pydantic (required, for data models)

Installation:
    pip install asyncua cryptography

Security Recommendations:
    - Always use Basic256Sha256 or stronger security policies in production
    - Use certificate-based authentication for critical systems
    - Store certificates securely using a certificate manager
    - Enable audit logging for compliance

Version: 1.0.0
Author: GreenLang Team
License: MIT
"""

# =============================================================================
# Client Exports
# =============================================================================

from .client import (
    OPCUAClient,
    OPCUAClientConfig,
    ClientStatistics,
    create_opcua_client,
)

# =============================================================================
# Server Exports
# =============================================================================

from .server import (
    OPCUAServer,
    OPCUAServerConfig,
    ServerNode,
    ServerStatistics,
    create_opcua_server,
)

# =============================================================================
# Subscription Exports
# =============================================================================

from .subscription import (
    OPCUASubscription,
    SubscriptionManager,
    MonitoredItem,
    DataChangeHandler,
    SubscriptionStatistics,
    DataChangeCallback,
    StatusChangeCallback,
)

# =============================================================================
# Security Exports
# =============================================================================

from .security import (
    CertificateManager,
    SecurityManager,
    CredentialManager,
    CertificateConfig,
    CertificateInfo,
    CertificateType,
    KeyType,
)

# =============================================================================
# Type Exports
# =============================================================================

from .types import (
    # Quality codes
    OPCUAQuality,
    QualityLevel,
    # Data types
    OPCUADataType,
    NodeClass,
    AccessLevel,
    # Node values
    DataProvenance,
    NodeValue,
    NodeValueBatch,
    # Node information
    NodeInfo,
    BrowseResult,
    # Subscriptions
    MonitoringMode,
    DataChangeTrigger,
    MonitoredItemConfig,
    SubscriptionConfig,
    DataChangeNotification,
    # Historical data
    AggregateType,
    HistoricalReadConfig,
    HistoricalValue,
    HistoricalDataResult,
    # Connection
    SecurityPolicy,
    MessageSecurityMode,
    AuthenticationType,
    ConnectionConfig,
    SessionInfo,
    # Server info
    ServerInfo,
    EndpointInfo,
    # Errors
    OPCUAError,
)

# =============================================================================
# Utility Exports
# =============================================================================

from .utils import (
    # Node ID utilities
    NodeIdType,
    parse_node_id,
    format_node_id,
    is_valid_node_id,
    get_node_id_namespace,
    # Data type conversions
    convert_to_opcua_type,
    convert_from_opcua_type,
    get_python_type_for_opcua,
    # Quality code utilities
    get_quality_description,
    get_quality_level,
    is_quality_good,
    is_quality_usable,
    is_quality_bad,
    parse_status_code,
    # Endpoint utilities
    parse_endpoint_url,
    build_endpoint_url,
    get_security_level,
    # Time utilities
    datetime_to_opcua,
    opcua_to_datetime,
    get_opcua_now,
    # Validation utilities
    validate_sampling_interval,
    validate_queue_size,
    validate_node_id_list,
    # Browse path utilities
    parse_browse_path,
    format_browse_path,
    # Well-known nodes
    WellKnownNodes,
)


# =============================================================================
# Package Metadata
# =============================================================================

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__all__ = [
    # Client
    "OPCUAClient",
    "OPCUAClientConfig",
    "ClientStatistics",
    "create_opcua_client",
    # Server
    "OPCUAServer",
    "OPCUAServerConfig",
    "ServerNode",
    "ServerStatistics",
    "create_opcua_server",
    # Subscription
    "OPCUASubscription",
    "SubscriptionManager",
    "MonitoredItem",
    "DataChangeHandler",
    "SubscriptionStatistics",
    "DataChangeCallback",
    "StatusChangeCallback",
    # Security
    "CertificateManager",
    "SecurityManager",
    "CredentialManager",
    "CertificateConfig",
    "CertificateInfo",
    "CertificateType",
    "KeyType",
    # Types - Quality
    "OPCUAQuality",
    "QualityLevel",
    # Types - Data
    "OPCUADataType",
    "NodeClass",
    "AccessLevel",
    # Types - Values
    "DataProvenance",
    "NodeValue",
    "NodeValueBatch",
    "NodeInfo",
    "BrowseResult",
    # Types - Subscriptions
    "MonitoringMode",
    "DataChangeTrigger",
    "MonitoredItemConfig",
    "SubscriptionConfig",
    "DataChangeNotification",
    # Types - Historical
    "AggregateType",
    "HistoricalReadConfig",
    "HistoricalValue",
    "HistoricalDataResult",
    # Types - Security
    "SecurityPolicy",
    "MessageSecurityMode",
    "AuthenticationType",
    "ConnectionConfig",
    "SessionInfo",
    # Types - Server
    "ServerInfo",
    "EndpointInfo",
    "OPCUAError",
    # Utils - Node ID
    "NodeIdType",
    "parse_node_id",
    "format_node_id",
    "is_valid_node_id",
    "get_node_id_namespace",
    # Utils - Conversion
    "convert_to_opcua_type",
    "convert_from_opcua_type",
    "get_python_type_for_opcua",
    # Utils - Quality
    "get_quality_description",
    "get_quality_level",
    "is_quality_good",
    "is_quality_usable",
    "is_quality_bad",
    "parse_status_code",
    # Utils - Endpoint
    "parse_endpoint_url",
    "build_endpoint_url",
    "get_security_level",
    # Utils - Time
    "datetime_to_opcua",
    "opcua_to_datetime",
    "get_opcua_now",
    # Utils - Validation
    "validate_sampling_interval",
    "validate_queue_size",
    "validate_node_id_list",
    # Utils - Browse
    "parse_browse_path",
    "format_browse_path",
    "WellKnownNodes",
]
