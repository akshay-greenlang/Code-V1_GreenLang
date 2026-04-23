"""
GreenLang Framework - Tool Export Utilities

This module provides utilities to export MCP tool definitions to various
AI platform formats for interoperability.

Supported export formats:
- OpenAI Function Calling format
- Anthropic Claude tool format
- Qwen-Agent tool format
- OpenAPI 3.0 schema format
- JSON Schema format

All exports include full provenance and versioning information.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import hashlib
import json
import logging
import re

# Import from MCP protocol module
import sys
from pathlib import Path

# Add parent path for imports
_framework_path = Path(__file__).parent.parent
if str(_framework_path) not in sys.path:
    sys.path.insert(0, str(_framework_path))

from advanced.mcp_protocol import (
    MCPToolRegistry,
    ToolDefinition,
    ToolParameter,
    ToolCategory,
    SecurityLevel,
    ExecutionMode,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EXPORT FORMAT ENUMS
# =============================================================================

class ExportFormat(Enum):
    """Supported export formats."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    QWEN = "qwen"
    OPENAPI = "openapi"
    JSON_SCHEMA = "json_schema"
    MCP_NATIVE = "mcp_native"


# =============================================================================
# EXPORT RESULT MODELS
# =============================================================================

@dataclass
class ExportResult:
    """Result of a tool export operation."""
    format: ExportFormat
    tools: List[Dict[str, Any]]
    tool_count: int
    exported_at: datetime
    version: str
    provenance_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format": self.format.value,
            "tools": self.tools,
            "tool_count": self.tool_count,
            "exported_at": self.exported_at.isoformat(),
            "version": self.version,
            "provenance_hash": self.provenance_hash,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class OpenAPISchema:
    """OpenAPI 3.0 schema export."""
    openapi: str = "3.0.3"
    info: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, Any] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "openapi": self.openapi,
            "info": self.info,
            "paths": self.paths,
            "components": self.components,
        }

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        try:
            import yaml
            return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        except ImportError:
            # Fallback to JSON if yaml not available
            return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# TOOL EXPORTER
# =============================================================================

class ToolExporter:
    """
    Exporter for converting MCP tools to various AI platform formats.

    Supports:
    - OpenAI Function Calling (GPT-4, GPT-3.5-turbo)
    - Anthropic Claude Tools (Claude 3, Claude 2.1)
    - Qwen-Agent Tool Protocol
    - OpenAPI 3.0 Schema
    - JSON Schema

    Example:
        >>> exporter = ToolExporter()
        >>> openai_tools = exporter.to_openai([tool1, tool2])
        >>> anthropic_tools = exporter.to_anthropic([tool1, tool2])
        >>> openapi_spec = exporter.to_openapi([tool1, tool2], title="GreenLang API")
    """

    VERSION = "1.0.0"

    def __init__(self, include_metadata: bool = True):
        """
        Initialize the tool exporter.

        Args:
            include_metadata: Whether to include extra metadata in exports
        """
        self.include_metadata = include_metadata

    def export(
        self,
        tools: Union[List[ToolDefinition], MCPToolRegistry],
        format: ExportFormat,
        **options: Any,
    ) -> ExportResult:
        """
        Export tools to the specified format.

        Args:
            tools: List of tool definitions or registry
            format: Target export format
            **options: Format-specific options

        Returns:
            ExportResult with exported tools
        """
        if isinstance(tools, MCPToolRegistry):
            tool_list = tools.list_tools()
        else:
            tool_list = tools

        if format == ExportFormat.OPENAI:
            exported = self.to_openai(tool_list, **options)
        elif format == ExportFormat.ANTHROPIC:
            exported = self.to_anthropic(tool_list, **options)
        elif format == ExportFormat.QWEN:
            exported = self.to_qwen(tool_list, **options)
        elif format == ExportFormat.OPENAPI:
            exported = self.to_openapi(tool_list, **options)
        elif format == ExportFormat.JSON_SCHEMA:
            exported = self.to_json_schema(tool_list, **options)
        elif format == ExportFormat.MCP_NATIVE:
            exported = self.to_mcp_native(tool_list, **options)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        # Calculate provenance hash
        content = json.dumps(exported, sort_keys=True, default=str)
        provenance_hash = hashlib.sha256(content.encode()).hexdigest()

        return ExportResult(
            format=format,
            tools=exported,
            tool_count=len(tool_list),
            exported_at=datetime.now(timezone.utc),
            version=self.VERSION,
            provenance_hash=provenance_hash,
            metadata={
                "exporter_version": self.VERSION,
                "include_metadata": self.include_metadata,
                "options": options,
            },
        )

    def to_openai(
        self,
        tools: List[ToolDefinition],
        strict: bool = True,
        include_descriptions: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Convert tools to OpenAI Function Calling format.

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "Tool description",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }

        Args:
            tools: List of tool definitions
            strict: Enable strict mode for function calling
            include_descriptions: Include parameter descriptions

        Returns:
            List of OpenAI function definitions
        """
        result = []

        for tool in tools:
            properties = {}
            required = []

            for param in tool.parameters:
                prop = self._param_to_json_schema(param, include_descriptions)
                properties[param.name] = prop
                if param.required:
                    required.append(param.name)

            function_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description[:1024] if tool.description else "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }

            if strict:
                function_def["function"]["strict"] = True
                function_def["function"]["parameters"]["additionalProperties"] = False

            if self.include_metadata:
                function_def["function"]["_greenlang_metadata"] = {
                    "category": tool.category.value,
                    "security_level": tool.security_level.value,
                    "version": tool.version,
                }

            result.append(function_def)

        return result

    def to_anthropic(
        self,
        tools: List[ToolDefinition],
        include_examples: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Convert tools to Anthropic Claude tool format.

        Anthropic format:
        {
            "name": "tool_name",
            "description": "Tool description",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }

        Args:
            tools: List of tool definitions
            include_examples: Include example values in descriptions

        Returns:
            List of Anthropic tool definitions
        """
        result = []

        for tool in tools:
            properties = {}
            required = []

            for param in tool.parameters:
                prop = self._param_to_json_schema(param, include_descriptions=True)
                if include_examples and param.default is not None:
                    prop["description"] = f"{prop.get('description', '')} (example: {param.default})"
                properties[param.name] = prop
                if param.required:
                    required.append(param.name)

            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }

            if self.include_metadata:
                tool_def["cache_control"] = {"type": "ephemeral"}
                tool_def["_greenlang_metadata"] = {
                    "category": tool.category.value,
                    "security_level": tool.security_level.value,
                    "version": tool.version,
                    "requires_confirmation": tool.requires_confirmation,
                }

            result.append(tool_def)

        return result

    def to_qwen(
        self,
        tools: List[ToolDefinition],
        include_human_names: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Convert tools to Qwen-Agent tool format.

        Qwen format:
        {
            "name": "tool_name",
            "name_for_human": "Tool Name",
            "description": "Tool description",
            "parameters": [
                {"name": "param", "type": "string", "description": "...", "required": true}
            ]
        }

        Args:
            tools: List of tool definitions
            include_human_names: Include human-readable names

        Returns:
            List of Qwen-Agent tool definitions
        """
        result = []

        for tool in tools:
            parameters = []
            for param in tool.parameters:
                param_def = {
                    "name": param.name,
                    "type": param.type,
                    "description": param.description,
                    "required": param.required,
                }
                if param.default is not None:
                    param_def["default"] = param.default
                if param.enum:
                    param_def["enum"] = param.enum
                parameters.append(param_def)

            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            }

            if include_human_names:
                # Convert snake_case to Title Case
                human_name = tool.name.replace("_", " ").title()
                tool_def["name_for_human"] = human_name

            if self.include_metadata:
                tool_def["category"] = tool.category.value
                tool_def["security_level"] = tool.security_level.value
                tool_def["version"] = tool.version

            result.append(tool_def)

        return result

    def to_openapi(
        self,
        tools: List[ToolDefinition],
        title: str = "GreenLang Tool API",
        version: str = "1.0.0",
        description: str = "Auto-generated API from GreenLang MCP tools",
        server_url: str = "http://localhost:8000",
    ) -> Dict[str, Any]:
        """
        Convert tools to OpenAPI 3.0 specification.

        Creates a full OpenAPI spec with:
        - POST endpoint for each tool
        - Request/response schemas
        - Security definitions
        - Tag grouping by category

        Args:
            tools: List of tool definitions
            title: API title
            version: API version
            description: API description
            server_url: Base server URL

        Returns:
            OpenAPI 3.0 specification dictionary
        """
        paths = {}
        schemas = {}
        tags = set()

        for tool in tools:
            # Create path
            path = f"/tools/{tool.name}"
            tag = tool.category.value

            tags.add(tag)

            # Create request schema
            request_schema_name = f"{tool.name}_request"
            request_schema = {
                "type": "object",
                "properties": {},
                "required": [],
            }

            for param in tool.parameters:
                request_schema["properties"][param.name] = self._param_to_json_schema(param)
                if param.required:
                    request_schema["required"].append(param.name)

            schemas[request_schema_name] = request_schema

            # Create response schema
            response_schema_name = f"{tool.name}_response"
            response_schema = {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "result": {"type": "object"},
                    "error": {"type": "string", "nullable": True},
                    "execution_time_ms": {"type": "number"},
                    "provenance_hash": {"type": "string"},
                },
                "required": ["success"],
            }
            schemas[response_schema_name] = response_schema

            # Create path item
            paths[path] = {
                "post": {
                    "operationId": tool.name,
                    "summary": tool.description[:100] if tool.description else tool.name,
                    "description": tool.description,
                    "tags": [tag],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": f"#/components/schemas/{request_schema_name}"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": f"#/components/schemas/{response_schema_name}"}
                                }
                            },
                        },
                        "400": {
                            "description": "Validation error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/error_response"}
                                }
                            },
                        },
                        "500": {
                            "description": "Internal error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/error_response"}
                                }
                            },
                        },
                    },
                    "security": self._get_security_for_level(tool.security_level),
                },
            }

        # Add common schemas
        schemas["error_response"] = {
            "type": "object",
            "properties": {
                "error": {"type": "string"},
                "code": {"type": "string"},
                "details": {"type": "object"},
            },
            "required": ["error"],
        }

        # Build OpenAPI spec
        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": title,
                "version": version,
                "description": description,
                "contact": {
                    "name": "GreenLang Framework",
                    "url": "https://github.com/greenlang",
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT",
                },
            },
            "servers": [
                {"url": server_url, "description": "Tool API Server"}
            ],
            "tags": [
                {"name": tag, "description": f"Tools in {tag} category"}
                for tag in sorted(tags)
            ],
            "paths": paths,
            "components": {
                "schemas": schemas,
                "securitySchemes": {
                    "apiKey": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key",
                    },
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT",
                    },
                },
            },
        }

        return spec

    def to_json_schema(
        self,
        tools: List[ToolDefinition],
        include_definitions: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Convert tools to JSON Schema format.

        Args:
            tools: List of tool definitions
            include_definitions: Include reusable schema definitions

        Returns:
            List of JSON Schema definitions
        """
        result = []

        for tool in tools:
            properties = {}
            required = []

            for param in tool.parameters:
                properties[param.name] = self._param_to_json_schema(param)
                if param.required:
                    required.append(param.name)

            schema = {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": f"urn:greenlang:tool:{tool.name}",
                "title": tool.name,
                "description": tool.description,
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }

            if self.include_metadata:
                schema["$comment"] = json.dumps({
                    "category": tool.category.value,
                    "security_level": tool.security_level.value,
                    "version": tool.version,
                    "execution_mode": tool.execution_mode.value,
                })

            result.append(schema)

        return result

    def to_mcp_native(
        self,
        tools: List[ToolDefinition],
        include_server_info: bool = True,
    ) -> Dict[str, Any]:
        """
        Export to MCP native format (for MCP server implementation).

        Args:
            tools: List of tool definitions
            include_server_info: Include server capabilities

        Returns:
            MCP server tool list format
        """
        tool_list = []

        for tool in tools:
            properties = {}
            required = []

            for param in tool.parameters:
                properties[param.name] = self._param_to_json_schema(param)
                if param.required:
                    required.append(param.name)

            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
            tool_list.append(tool_def)

        result = {"tools": tool_list}

        if include_server_info:
            result["_meta"] = {
                "serverInfo": {
                    "name": "GreenLang MCP Server",
                    "version": self.VERSION,
                },
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "tools": {"listChanged": True},
                },
            }

        return result

    def _param_to_json_schema(
        self,
        param: ToolParameter,
        include_descriptions: bool = True,
    ) -> Dict[str, Any]:
        """Convert a ToolParameter to JSON Schema property."""
        schema: Dict[str, Any] = {"type": param.type}

        if include_descriptions and param.description:
            schema["description"] = param.description

        if param.default is not None:
            schema["default"] = param.default

        if param.enum:
            schema["enum"] = param.enum

        if param.minimum is not None:
            schema["minimum"] = param.minimum

        if param.maximum is not None:
            schema["maximum"] = param.maximum

        if param.pattern:
            schema["pattern"] = param.pattern

        # Handle array type
        if param.type == "array":
            schema["items"] = {"type": "string"}  # Default to string items

        # Handle object type
        if param.type == "object":
            schema["additionalProperties"] = True

        return schema

    def _get_security_for_level(self, level: SecurityLevel) -> List[Dict[str, List[str]]]:
        """Get OpenAPI security requirement based on security level."""
        if level == SecurityLevel.READ_ONLY:
            return []  # No auth required for read-only
        elif level == SecurityLevel.ADVISORY:
            return [{"apiKey": []}]
        elif level == SecurityLevel.CONTROLLED_WRITE:
            return [{"bearerAuth": []}]
        else:  # FULL_ACCESS
            return [{"bearerAuth": []}, {"apiKey": []}]


# =============================================================================
# BATCH EXPORT UTILITIES
# =============================================================================

class BatchExporter:
    """
    Batch exporter for exporting tools to multiple formats at once.

    Example:
        >>> batch = BatchExporter()
        >>> results = batch.export_all(tools, formats=[ExportFormat.OPENAI, ExportFormat.ANTHROPIC])
    """

    def __init__(self):
        """Initialize batch exporter."""
        self._exporter = ToolExporter()

    def export_all(
        self,
        tools: Union[List[ToolDefinition], MCPToolRegistry],
        formats: Optional[List[ExportFormat]] = None,
        **options: Any,
    ) -> Dict[ExportFormat, ExportResult]:
        """
        Export tools to multiple formats.

        Args:
            tools: List of tool definitions or registry
            formats: List of formats to export (default: all formats)
            **options: Export options

        Returns:
            Dictionary mapping format to export result
        """
        if formats is None:
            formats = list(ExportFormat)

        results = {}
        for fmt in formats:
            try:
                result = self._exporter.export(tools, fmt, **options)
                results[fmt] = result
            except Exception as e:
                logger.error(f"Export to {fmt.value} failed: {e}")

        return results

    def export_to_files(
        self,
        tools: Union[List[ToolDefinition], MCPToolRegistry],
        output_dir: Path,
        formats: Optional[List[ExportFormat]] = None,
        **options: Any,
    ) -> Dict[ExportFormat, Path]:
        """
        Export tools to files.

        Args:
            tools: List of tool definitions or registry
            output_dir: Directory for output files
            formats: List of formats to export
            **options: Export options

        Returns:
            Dictionary mapping format to output file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.export_all(tools, formats, **options)
        file_paths = {}

        for fmt, result in results.items():
            if fmt == ExportFormat.OPENAPI:
                filename = "openapi.yaml"
                content = OpenAPISchema(**result.tools).to_yaml() if hasattr(result.tools, '__iter__') and not isinstance(result.tools, dict) else json.dumps(result.tools, indent=2)
            else:
                filename = f"tools_{fmt.value}.json"
                content = result.to_json()

            file_path = output_dir / filename
            file_path.write_text(content)
            file_paths[fmt] = file_path
            logger.info(f"Exported {fmt.value} to {file_path}")

        return file_paths


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def export_to_openai(tools: List[ToolDefinition], **options: Any) -> List[Dict[str, Any]]:
    """Export tools to OpenAI format."""
    return ToolExporter().to_openai(tools, **options)


def export_to_anthropic(tools: List[ToolDefinition], **options: Any) -> List[Dict[str, Any]]:
    """Export tools to Anthropic format."""
    return ToolExporter().to_anthropic(tools, **options)


def export_to_qwen(tools: List[ToolDefinition], **options: Any) -> List[Dict[str, Any]]:
    """Export tools to Qwen-Agent format."""
    return ToolExporter().to_qwen(tools, **options)


def export_to_openapi(tools: List[ToolDefinition], **options: Any) -> Dict[str, Any]:
    """Export tools to OpenAPI format."""
    return ToolExporter().to_openapi(tools, **options)


def export_to_json_schema(tools: List[ToolDefinition], **options: Any) -> List[Dict[str, Any]]:
    """Export tools to JSON Schema format."""
    return ToolExporter().to_json_schema(tools, **options)


def export_registry(
    registry: MCPToolRegistry,
    format: ExportFormat,
    **options: Any,
) -> ExportResult:
    """Export an entire registry to the specified format."""
    return ToolExporter().export(registry, format, **options)


# Export list
__all__ = [
    # Enums
    "ExportFormat",
    # Data models
    "ExportResult",
    "OpenAPISchema",
    # Exporter classes
    "ToolExporter",
    "BatchExporter",
    # Convenience functions
    "export_to_openai",
    "export_to_anthropic",
    "export_to_qwen",
    "export_to_openapi",
    "export_to_json_schema",
    "export_registry",
]
