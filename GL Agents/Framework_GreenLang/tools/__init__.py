"""
GreenLang Framework - Development Tools

Tools for creating, validating, and maintaining GreenLang agents.

Modules:
- scaffolding: Agent project scaffolding
- validator: Agent validation utilities
- mcp_calculators: MCP calculator tool definitions (ASME PTC 4, IAPWS-IF97, EPA)
- mcp_connectors: MCP connector tool definitions (SCADA, OPC-UA, Kafka, Database)
- tool_discovery: Tool discovery, health monitoring, and circuit breaker patterns
- tool_export: Export to OpenAI, Anthropic, Qwen, OpenAPI formats
"""

from .scaffolding import AgentScaffolder
from .validator import AgentValidator

# MCP Calculator Tools
from .mcp_calculators import (
    # Tool definitions
    COMBUSTION_EFFICIENCY_DEFINITION,
    HEAT_BALANCE_DEFINITION,
    STEAM_PROPERTIES_DEFINITION,
    EMISSION_RATE_DEFINITION,
    HEAT_EXCHANGER_DEFINITION,
    # Tool classes
    CombustionEfficiencyTool,
    HeatBalanceTool,
    SteamPropertiesTool,
    EmissionRateTool,
    HeatExchangerTool,
    # Registry
    CALCULATOR_REGISTRY,
    create_calculator_registry,
    get_calculator_tools,
    invoke_calculator,
)

# MCP Connector Tools
from .mcp_connectors import (
    # Security
    AccessLevel,
    SecurityContext,
    AuditRecord,
    AUDIT_LOGGER,
    # Tool definitions
    SCADA_READ_DEFINITION,
    SCADA_WRITE_DEFINITION,
    OPC_UA_READ_DEFINITION,
    OPC_UA_WRITE_DEFINITION,
    KAFKA_PRODUCE_DEFINITION,
    KAFKA_CONSUME_DEFINITION,
    DATABASE_QUERY_DEFINITION,
    # Tool classes
    ScadaReadTool,
    ScadaWriteTool,
    OpcUaReadTool,
    OpcUaWriteTool,
    KafkaProduceTool,
    KafkaConsumeTool,
    DatabaseQueryTool,
    # Registry
    CONNECTOR_REGISTRY,
    create_connector_registry,
    get_connector_tools,
    invoke_connector,
    get_audit_records,
)

# Tool Discovery Service
from .tool_discovery import (
    # Enums
    ToolHealthStatus,
    CircuitState,
    CapabilityType,
    # Data models
    ToolCapability,
    ToolMetrics,
    ToolHealthCheck,
    CircuitBreaker,
    ToolRegistration,
    # Main service
    ToolDiscoveryService,
    # Decorators and builders
    discoverable_tool,
    CapabilityBuilder,
    capability,
    # Global functions
    get_discovery_service,
    register_tool_globally,
)

# Tool Export Utilities
from .tool_export import (
    # Enums
    ExportFormat,
    # Data models
    ExportResult,
    OpenAPISchema,
    # Exporter classes
    ToolExporter,
    BatchExporter,
    # Convenience functions
    export_to_openai,
    export_to_anthropic,
    export_to_qwen,
    export_to_openapi,
    export_to_json_schema,
    export_registry,
)

__all__ = [
    # Original exports
    "AgentScaffolder",
    "AgentValidator",
    # Calculator tool definitions
    "COMBUSTION_EFFICIENCY_DEFINITION",
    "HEAT_BALANCE_DEFINITION",
    "STEAM_PROPERTIES_DEFINITION",
    "EMISSION_RATE_DEFINITION",
    "HEAT_EXCHANGER_DEFINITION",
    # Calculator tool classes
    "CombustionEfficiencyTool",
    "HeatBalanceTool",
    "SteamPropertiesTool",
    "EmissionRateTool",
    "HeatExchangerTool",
    # Calculator registry
    "CALCULATOR_REGISTRY",
    "create_calculator_registry",
    "get_calculator_tools",
    "invoke_calculator",
    # Security
    "AccessLevel",
    "SecurityContext",
    "AuditRecord",
    "AUDIT_LOGGER",
    # Connector tool definitions
    "SCADA_READ_DEFINITION",
    "SCADA_WRITE_DEFINITION",
    "OPC_UA_READ_DEFINITION",
    "OPC_UA_WRITE_DEFINITION",
    "KAFKA_PRODUCE_DEFINITION",
    "KAFKA_CONSUME_DEFINITION",
    "DATABASE_QUERY_DEFINITION",
    # Connector tool classes
    "ScadaReadTool",
    "ScadaWriteTool",
    "OpcUaReadTool",
    "OpcUaWriteTool",
    "KafkaProduceTool",
    "KafkaConsumeTool",
    "DatabaseQueryTool",
    # Connector registry
    "CONNECTOR_REGISTRY",
    "create_connector_registry",
    "get_connector_tools",
    "invoke_connector",
    "get_audit_records",
    # Discovery enums
    "ToolHealthStatus",
    "CircuitState",
    "CapabilityType",
    # Discovery data models
    "ToolCapability",
    "ToolMetrics",
    "ToolHealthCheck",
    "CircuitBreaker",
    "ToolRegistration",
    # Discovery service
    "ToolDiscoveryService",
    "discoverable_tool",
    "CapabilityBuilder",
    "capability",
    "get_discovery_service",
    "register_tool_globally",
    # Export enums
    "ExportFormat",
    # Export data models
    "ExportResult",
    "OpenAPISchema",
    # Export classes
    "ToolExporter",
    "BatchExporter",
    # Export functions
    "export_to_openai",
    "export_to_anthropic",
    "export_to_qwen",
    "export_to_openapi",
    "export_to_json_schema",
    "export_registry",
]
