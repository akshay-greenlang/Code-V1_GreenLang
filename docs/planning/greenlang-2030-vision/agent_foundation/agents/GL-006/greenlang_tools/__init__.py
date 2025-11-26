# -*- coding: utf-8 -*-
"""
GreenLang Tools Framework for GL-006 HeatRecoveryMaximizer.

This module provides tool definitions and utilities for building agent tools
that can be used by LLM-based orchestration systems.
"""

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import functools
import inspect
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ToolCategory(str, Enum):
    """Categories of tools."""
    ANALYSIS = "analysis"
    CALCULATION = "calculation"
    OPTIMIZATION = "optimization"
    DATA_ACCESS = "data_access"
    INTEGRATION = "integration"
    VALIDATION = "validation"
    REPORTING = "reporting"
    UTILITY = "utility"


class ParameterType(str, Enum):
    """Types of tool parameters."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ParameterDefinition:
    """Definition of a tool parameter."""
    name: str
    description: str
    param_type: ParameterType
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    items_type: Optional[ParameterType] = None  # For arrays
    properties: Optional[Dict[str, "ParameterDefinition"]] = None  # For objects

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: Dict[str, Any] = {
            "type": self.param_type.value,
            "description": self.description,
        }

        if self.enum:
            schema["enum"] = self.enum

        if self.minimum is not None:
            schema["minimum"] = self.minimum

        if self.maximum is not None:
            schema["maximum"] = self.maximum

        if self.param_type == ParameterType.ARRAY and self.items_type:
            schema["items"] = {"type": self.items_type.value}

        if self.param_type == ParameterType.OBJECT and self.properties:
            schema["properties"] = {
                name: prop.to_json_schema()
                for name, prop in self.properties.items()
            }

        if self.default is not None:
            schema["default"] = self.default

        return schema


@dataclass
class ToolDefinition:
    """
    Definition of a tool.

    Attributes:
        name: Tool name (function name)
        description: Human-readable description
        category: Tool category
        parameters: List of parameter definitions
        return_type: Return type description
        examples: Usage examples
        tags: Tags for searchability
    """
    name: str
    description: str
    category: ToolCategory = ToolCategory.UTILITY
    parameters: List[ParameterDefinition] = field(default_factory=list)
    return_type: str = "object"
    return_description: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    deterministic: bool = True
    idempotent: bool = True

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
        }

    @classmethod
    def success_result(cls, data: Any, metadata: Optional[Dict[str, Any]] = None) -> "ToolResult":
        """Create a success result."""
        return cls(success=True, data=data, metadata=metadata or {})

    @classmethod
    def error_result(cls, error: str, metadata: Optional[Dict[str, Any]] = None) -> "ToolResult":
        """Create an error result."""
        return cls(success=False, error=error, metadata=metadata or {})


class BaseTool(ABC):
    """
    Abstract base class for tools.

    Example:
        >>> class MyTool(BaseTool):
        ...     @property
        ...     def definition(self) -> ToolDefinition:
        ...         return ToolDefinition(
        ...             name="my_tool",
        ...             description="Does something useful",
        ...         )
        ...
        ...     async def execute(self, **kwargs) -> ToolResult:
        ...         return ToolResult.success_result({"result": "done"})
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Get the tool definition."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass

    def validate_parameters(self, **kwargs) -> Optional[str]:
        """
        Validate input parameters.

        Returns:
            Error message if validation fails, None otherwise
        """
        for param in self.definition.parameters:
            if param.required and param.name not in kwargs:
                return f"Missing required parameter: {param.name}"

            if param.name in kwargs:
                value = kwargs[param.name]

                if param.param_type == ParameterType.NUMBER:
                    if not isinstance(value, (int, float)):
                        return f"Parameter {param.name} must be a number"
                    if param.minimum is not None and value < param.minimum:
                        return f"Parameter {param.name} must be >= {param.minimum}"
                    if param.maximum is not None and value > param.maximum:
                        return f"Parameter {param.name} must be <= {param.maximum}"

                if param.param_type == ParameterType.INTEGER:
                    if not isinstance(value, int):
                        return f"Parameter {param.name} must be an integer"

                if param.param_type == ParameterType.STRING:
                    if not isinstance(value, str):
                        return f"Parameter {param.name} must be a string"

                if param.param_type == ParameterType.BOOLEAN:
                    if not isinstance(value, bool):
                        return f"Parameter {param.name} must be a boolean"

                if param.param_type == ParameterType.ARRAY:
                    if not isinstance(value, list):
                        return f"Parameter {param.name} must be an array"

                if param.enum and value not in param.enum:
                    return f"Parameter {param.name} must be one of {param.enum}"

        return None


class ToolRegistry:
    """
    Registry for managing tools.

    Provides registration, discovery, and execution of tools.
    """

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._functions: Dict[str, Callable] = {}
        self._definitions: Dict[str, ToolDefinition] = {}

    def register(self, tool: BaseTool):
        """Register a tool."""
        definition = tool.definition
        self._tools[definition.name] = tool
        self._definitions[definition.name] = definition
        logger.debug(f"Registered tool: {definition.name}")

    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: ToolCategory = ToolCategory.UTILITY,
        **kwargs,
    ) -> ToolDefinition:
        """
        Register a function as a tool.

        Automatically extracts parameter information from function signature.
        """
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or ""

        # Extract parameters from function signature
        sig = inspect.signature(func)
        parameters = []

        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue

            # Determine parameter type from annotation
            param_type = ParameterType.STRING
            if param.annotation != inspect.Parameter.empty:
                ann = param.annotation
                if ann == int:
                    param_type = ParameterType.INTEGER
                elif ann == float:
                    param_type = ParameterType.NUMBER
                elif ann == bool:
                    param_type = ParameterType.BOOLEAN
                elif ann == list or (hasattr(ann, '__origin__') and ann.__origin__ == list):
                    param_type = ParameterType.ARRAY

            parameters.append(ParameterDefinition(
                name=param_name,
                description=f"Parameter: {param_name}",
                param_type=param_type,
                required=param.default == inspect.Parameter.empty,
                default=None if param.default == inspect.Parameter.empty else param.default,
            ))

        definition = ToolDefinition(
            name=tool_name,
            description=tool_description,
            category=category,
            parameters=parameters,
            **kwargs,
        )

        self._functions[tool_name] = func
        self._definitions[tool_name] = definition
        logger.debug(f"Registered function as tool: {tool_name}")
        return definition

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self._definitions.get(name)

    def get_all_definitions(self) -> List[ToolDefinition]:
        """Get all tool definitions."""
        return list(self._definitions.values())

    def get_tools_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        """Get tools by category."""
        return [d for d in self._definitions.values() if d.category == category]

    async def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        import time
        start_time = time.time()

        if name in self._tools:
            tool = self._tools[name]
            validation_error = tool.validate_parameters(**kwargs)
            if validation_error:
                return ToolResult.error_result(validation_error)

            try:
                result = await tool.execute(**kwargs)
                result.execution_time_ms = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                return ToolResult.error_result(str(e))

        elif name in self._functions:
            func = self._functions[name]
            try:
                import asyncio
                if asyncio.iscoroutinefunction(func):
                    data = await func(**kwargs)
                else:
                    data = func(**kwargs)

                result = ToolResult.success_result(data)
                result.execution_time_ms = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                return ToolResult.error_result(str(e))

        return ToolResult.error_result(f"Tool not found: {name}")

    def to_openai_functions(self) -> List[Dict[str, Any]]:
        """Export all tools as OpenAI function definitions."""
        return [d.to_openai_function() for d in self._definitions.values()]

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Export all tools as Anthropic tool definitions."""
        return [d.to_anthropic_tool() for d in self._definitions.values()]


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: ToolCategory = ToolCategory.UTILITY,
    **kwargs,
):
    """
    Decorator to register a function as a tool.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        category: Tool category
        **kwargs: Additional tool definition parameters

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        # Store tool metadata on the function
        func._tool_name = name or func.__name__
        func._tool_description = description or func.__doc__ or ""
        func._tool_category = category
        func._tool_kwargs = kwargs
        func._is_tool = True
        return func

    return decorator


# Global tool registry
tool_registry = ToolRegistry()


__all__ = [
    'ToolCategory',
    'ParameterType',
    'ParameterDefinition',
    'ToolDefinition',
    'ToolResult',
    'BaseTool',
    'ToolRegistry',
    'tool',
    'tool_registry',
]
