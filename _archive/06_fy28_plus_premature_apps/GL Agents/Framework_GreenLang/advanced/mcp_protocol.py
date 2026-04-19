"""
GreenLang Framework - Model Context Protocol (MCP) v2025
Universal Tool Interface (Like USB-C for AI Agents)

Based on:
- Anthropic MCP Specification v2025-06-18
- OpenAI Function Calling
- Qwen-Agent Tool Protocol
- Google Gemini Function Calling

This module provides a standardized interface for AI agents to connect
to external tools, data sources, and services across platforms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
import hashlib
import json
import logging
from functools import wraps


logger = logging.getLogger(__name__)

T = TypeVar('T')
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class MCPVersion(Enum):
    """Supported MCP versions."""
    V2024_11_05 = "2024-11-05"
    V2025_06_18 = "2025-06-18"  # Latest with security improvements


class ToolCategory(Enum):
    """Tool categories for organization."""
    CALCULATOR = "calculator"
    CONNECTOR = "connector"
    ANALYZER = "analyzer"
    OPTIMIZER = "optimizer"
    MONITOR = "monitor"
    VALIDATOR = "validator"
    TRANSFORMER = "transformer"


class SecurityLevel(Enum):
    """Tool security levels."""
    READ_ONLY = "read_only"
    ADVISORY = "advisory"
    CONTROLLED_WRITE = "controlled_write"
    FULL_ACCESS = "full_access"


class ExecutionMode(Enum):
    """Tool execution modes."""
    SYNC = "sync"
    ASYNC = "async"
    STREAMING = "streaming"
    BATCH = "batch"


@dataclass
class ToolParameter:
    """Parameter definition for a tool."""
    name: str
    type: str  # JSON Schema type
    description: str
    required: bool = True
    default: Any = None
    enum: List[Any] = field(default_factory=list)
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    pattern: Optional[str] = None  # Regex pattern for strings

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema = {
            "type": self.type,
            "description": self.description
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.pattern:
            schema["pattern"] = self.pattern
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """Complete tool definition following MCP specification."""
    name: str
    description: str
    parameters: List[ToolParameter]
    category: ToolCategory = ToolCategory.CALCULATOR
    security_level: SecurityLevel = SecurityLevel.READ_ONLY
    execution_mode: ExecutionMode = ExecutionMode.SYNC
    timeout_seconds: int = 30
    retry_count: int = 3
    rate_limit_per_minute: int = 100
    requires_confirmation: bool = False
    audit_level: str = "full"  # none, basic, full
    version: str = "1.0.0"

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI Function Calling format."""
        required = [p.name for p in self.parameters if p.required]
        properties = {p.name: p.to_json_schema() for p in self.parameters}

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic Claude tool format."""
        required = [p.name for p in self.parameters if p.required]
        properties = {p.name: p.to_json_schema() for p in self.parameters}

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def to_qwen_tool(self) -> Dict[str, Any]:
        """Convert to Qwen-Agent tool format."""
        return {
            "name": self.name,
            "name_for_human": self.name.replace("_", " ").title(),
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required
                }
                for p in self.parameters
            ]
        }


@dataclass
class ToolCallRequest:
    """Request to invoke a tool."""
    tool_name: str
    arguments: Dict[str, Any]
    request_id: str = ""
    caller_agent_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.request_id:
            data = f"{self.tool_name}:{json.dumps(self.arguments, sort_keys=True)}:{self.timestamp.isoformat()}"
            self.request_id = hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class ToolCallResponse:
    """Response from a tool invocation."""
    request_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    provenance_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.provenance_hash and self.result is not None:
            data = f"{self.request_id}:{json.dumps(self.result, sort_keys=True, default=str)}"
            self.provenance_hash = hashlib.sha256(data.encode()).hexdigest()


class MCPTool(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for MCP-compliant tools.

    All GreenLang tools should inherit from this class to ensure
    compatibility with the Model Context Protocol.
    """

    def __init__(self, definition: ToolDefinition):
        self.definition = definition
        self._call_count = 0
        self._error_count = 0
        self._total_execution_time_ms = 0.0

    @abstractmethod
    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute the tool with the given request."""
        pass

    def validate_arguments(self, arguments: Dict[str, Any]) -> List[str]:
        """Validate arguments against parameter definitions."""
        errors = []
        for param in self.definition.parameters:
            if param.required and param.name not in arguments:
                errors.append(f"Missing required parameter: {param.name}")
            elif param.name in arguments:
                value = arguments[param.name]
                # Type validation
                if param.type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Parameter {param.name} must be a number")
                elif param.type == "string" and not isinstance(value, str):
                    errors.append(f"Parameter {param.name} must be a string")
                elif param.type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Parameter {param.name} must be a boolean")
                # Range validation
                if param.minimum is not None and isinstance(value, (int, float)):
                    if value < param.minimum:
                        errors.append(f"Parameter {param.name} must be >= {param.minimum}")
                if param.maximum is not None and isinstance(value, (int, float)):
                    if value > param.maximum:
                        errors.append(f"Parameter {param.name} must be <= {param.maximum}")
                # Enum validation
                if param.enum and value not in param.enum:
                    errors.append(f"Parameter {param.name} must be one of: {param.enum}")
        return errors

    def get_metrics(self) -> Dict[str, Any]:
        """Get tool execution metrics."""
        return {
            "call_count": self._call_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / self._call_count if self._call_count > 0 else 0,
            "avg_execution_time_ms": self._total_execution_time_ms / self._call_count if self._call_count > 0 else 0
        }


class MCPToolRegistry:
    """
    Central registry for MCP-compliant tools.

    Provides tool discovery, versioning, and lifecycle management
    following the MCP server specification.
    """

    def __init__(self, server_name: str = "GreenLang MCP Server"):
        self.server_name = server_name
        self.version = MCPVersion.V2025_06_18
        self._tools: Dict[str, MCPTool] = {}
        self._tool_versions: Dict[str, List[str]] = {}

    def register(self, tool: MCPTool) -> None:
        """Register a tool with the registry."""
        name = tool.definition.name
        version = tool.definition.version

        if name in self._tools:
            logger.warning(f"Overwriting existing tool: {name}")

        self._tools[name] = tool

        if name not in self._tool_versions:
            self._tool_versions[name] = []
        self._tool_versions[name].append(version)

        logger.info(f"Registered tool: {name} v{version}")

    def unregister(self, name: str) -> bool:
        """Unregister a tool from the registry."""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """List all registered tools."""
        return [tool.definition for tool in self._tools.values()]

    def invoke(self, request: ToolCallRequest) -> ToolCallResponse:
        """Invoke a tool by name with arguments."""
        tool = self.get_tool(request.tool_name)
        if not tool:
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=f"Tool not found: {request.tool_name}"
            )

        # Validate arguments
        errors = tool.validate_arguments(request.arguments)
        if errors:
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=f"Validation errors: {'; '.join(errors)}"
            )

        # Check security level
        if tool.definition.requires_confirmation:
            logger.warning(f"Tool {request.tool_name} requires confirmation before execution")

        # Execute tool
        import time
        start_time = time.time()
        try:
            response = tool.execute(request)
            execution_time = (time.time() - start_time) * 1000
            response.execution_time_ms = execution_time
            tool._call_count += 1
            tool._total_execution_time_ms += execution_time
            if not response.success:
                tool._error_count += 1
            return response
        except Exception as e:
            tool._call_count += 1
            tool._error_count += 1
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=str(e)
            )

    def to_openai_functions(self) -> List[Dict[str, Any]]:
        """Export all tools as OpenAI functions."""
        return [tool.definition.to_openai_function() for tool in self._tools.values()]

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Export all tools as Anthropic Claude tools."""
        return [tool.definition.to_anthropic_tool() for tool in self._tools.values()]

    def to_qwen_tools(self) -> List[Dict[str, Any]]:
        """Export all tools as Qwen-Agent tools."""
        return [tool.definition.to_qwen_tool() for tool in self._tools.values()]

    def get_server_info(self) -> Dict[str, Any]:
        """Get MCP server information."""
        return {
            "name": self.server_name,
            "version": self.version.value,
            "protocolVersion": self.version.value,
            "capabilities": {
                "tools": True,
                "resources": False,
                "prompts": False,
                "sampling": False
            },
            "tool_count": len(self._tools)
        }


# ============================================================================
# DECORATOR FOR EASY TOOL CREATION
# ============================================================================

def mcp_tool(
    name: str,
    description: str,
    category: ToolCategory = ToolCategory.CALCULATOR,
    security_level: SecurityLevel = SecurityLevel.READ_ONLY,
    requires_confirmation: bool = False
):
    """
    Decorator to convert a function into an MCP-compliant tool.

    Usage:
        @mcp_tool(
            name="calculate_efficiency",
            description="Calculate boiler efficiency per ASME PTC 4",
            category=ToolCategory.CALCULATOR
        )
        def calculate_efficiency(fuel_flow: float, steam_flow: float) -> float:
            ...
    """
    def decorator(func: Callable) -> MCPTool:
        import inspect
        sig = inspect.signature(func)

        # Extract parameters from function signature
        parameters = []
        for param_name, param in sig.parameters.items():
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation in (int, float):
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == str:
                    param_type = "string"

            has_default = param.default != inspect.Parameter.empty
            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=f"Parameter: {param_name}",
                required=not has_default,
                default=param.default if has_default else None
            ))

        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            category=category,
            security_level=security_level,
            requires_confirmation=requires_confirmation
        )

        class FunctionTool(MCPTool):
            def execute(self, request: ToolCallRequest) -> ToolCallResponse:
                try:
                    result = func(**request.arguments)
                    return ToolCallResponse(
                        request_id=request.request_id,
                        tool_name=request.tool_name,
                        success=True,
                        result=result
                    )
                except Exception as e:
                    return ToolCallResponse(
                        request_id=request.request_id,
                        tool_name=request.tool_name,
                        success=False,
                        error=str(e)
                    )

        return FunctionTool(definition)

    return decorator


# ============================================================================
# GLOBAL REGISTRY INSTANCE
# ============================================================================

GREENLANG_MCP_REGISTRY = MCPToolRegistry()


# ============================================================================
# EXAMPLE TOOL IMPLEMENTATIONS
# ============================================================================

@mcp_tool(
    name="calculate_combustion_efficiency",
    description="Calculate combustion efficiency using ASME PTC 4 method",
    category=ToolCategory.CALCULATOR,
    security_level=SecurityLevel.READ_ONLY
)
def calculate_combustion_efficiency(
    fuel_higher_heating_value: float,
    flue_gas_temperature: float,
    ambient_temperature: float,
    excess_air_percentage: float
) -> Dict[str, float]:
    """Calculate combustion efficiency per ASME PTC 4."""
    # Simplified calculation for demonstration
    stack_loss = (flue_gas_temperature - ambient_temperature) * 0.024
    radiation_loss = 1.5
    unburned_loss = excess_air_percentage * 0.05
    total_loss = stack_loss + radiation_loss + unburned_loss
    efficiency = 100 - total_loss

    return {
        "efficiency_percent": round(efficiency, 2),
        "stack_loss_percent": round(stack_loss, 2),
        "radiation_loss_percent": round(radiation_loss, 2),
        "unburned_loss_percent": round(unburned_loss, 2),
        "total_loss_percent": round(total_loss, 2)
    }


# Register example tool
GREENLANG_MCP_REGISTRY.register(calculate_combustion_efficiency)
