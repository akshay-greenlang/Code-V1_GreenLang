"""
GL-001 ThermalCommand - Integrations Package

This package provides integration modules for external systems including:
- Webhooks for event notification
- OPC-UA for industrial communication (comprehensive module)
- Kafka for event streaming
- ERP system integrations
- CMMS integration

OPC-UA Module Components:
- opcua_schemas: Pydantic data models for OPC-UA operations
- tag_mapping: Canonical tag naming, unit conversions, and governance
- opcua_connector: OPC-UA client with certificate-based authentication
- opcua_write_handler: Write path with safety gates and two-step confirmation

OPC-UA Features:
- Certificate-based mTLS authentication
- Tag subscription with configurable sampling intervals (1-5s)
- Timestamping and quality code enforcement
- Whitelisted supervisory tag writes only
- Two-step write confirmation (recommendation + apply)
- Safety boundary validation
- Rate limiting on writes
- SHA-256 provenance tracking

Security:
- Network segmentation mandatory
- OT cybersecurity standards compliance

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

__version__ = "1.0.0"

# OPC-UA Module Exports
from integrations.opcua_schemas import (
    OPCUATagConfig,
    OPCUASubscription,
    OPCUASubscriptionConfig,
    OPCUAWriteRequest,
    OPCUAWriteResponse,
    OPCUADataPoint,
    OPCUAQualityCode,
    OPCUAConnectionConfig,
    OPCUASecurityConfig,
    TagMetadata,
    EngineeringUnit,
    SafetyBoundary,
    WriteConfirmationStatus,
)
from integrations.tag_mapping import (
    TagMapper,
    TagGovernance,
    UnitConverter,
    TagMappingConfig,
    CanonicalTagName,
    TagMappingEntry,
)
from integrations.opcua_connector import (
    OPCUAConnector,
    OPCUASubscriptionManager,
    ConnectionPool,
)
from integrations.opcua_write_handler import (
    OPCUAWriteHandler,
    WriteConfirmation,
    SafetyGate,
    WriteRateLimiter,
    WriteRecommendation,
)

__all__ = [
    # Schemas
    "OPCUATagConfig",
    "OPCUASubscription",
    "OPCUASubscriptionConfig",
    "OPCUAWriteRequest",
    "OPCUAWriteResponse",
    "OPCUADataPoint",
    "OPCUAQualityCode",
    "OPCUAConnectionConfig",
    "OPCUASecurityConfig",
    "TagMetadata",
    "EngineeringUnit",
    "SafetyBoundary",
    "WriteConfirmationStatus",
    # Tag Mapping
    "TagMapper",
    "TagGovernance",
    "UnitConverter",
    "TagMappingConfig",
    "CanonicalTagName",
    "TagMappingEntry",
    # Connector
    "OPCUAConnector",
    "OPCUASubscriptionManager",
    "ConnectionPool",
    # Write Handler
    "OPCUAWriteHandler",
    "WriteConfirmation",
    "SafetyGate",
    "WriteRateLimiter",
    "WriteRecommendation",
]
