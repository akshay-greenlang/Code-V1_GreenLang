# -*- coding: utf-8 -*-
"""
GreenLang Tool Registry
=======================

Manages registration and discovery of tools/functions available to agents.
Provides a centralized registry for agent capabilities.

This is a stub implementation - TODO: Complete implementation.

Author: GreenLang Framework Team
"""

from typing import Dict, List, Optional, Any, Callable, Type
from dataclasses import dataclass, field
import inspect
import logging
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Defines a tool that can be used by agents."""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    returns: Optional[Type] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """
    Central registry for agent tools and capabilities.

    TODO: This is a stub implementation. Full implementation should include:
    - Tool validation
    - Permission management
    - Tool versioning
    - Dynamic loading
    - Tool composition
    """

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, ToolDefinition] = {}
        self._categories: Dict[str, List[str]] = {}
        logger.info("Initialized ToolRegistry")

    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **metadata
    ):
        """
        Decorator to register a function as a tool.

        Args:
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            tags: Optional tags for categorization
            **metadata: Additional metadata

        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or (inspect.getdoc(func) or "No description")

            # Extract parameter information
            sig = inspect.signature(func)
            parameters = {}
            for param_name, param in sig.parameters.items():
                param_info = {"type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"}
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                parameters[param_name] = param_info

            # Create tool definition
            tool = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                function=func,
                parameters=parameters,
                returns=sig.return_annotation if sig.return_annotation != inspect.Signature.empty else None,
                tags=tags or [],
                metadata=metadata
            )

            # Register the tool
            self._tools[tool_name] = tool

            # Update categories
            for tag in (tags or []):
                if tag not in self._categories:
                    self._categories[tag] = []
                self._categories[tag].append(tool_name)

            logger.info(f"Registered tool: {tool_name}")
            return func

        return decorator

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def execute_tool(self, name: str, **kwargs) -> Any:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            **kwargs: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")

        try:
            result = tool.function(**kwargs)
            logger.debug(f"Executed tool: {name}")
            return result
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            raise

    def list_tools(self, tag: Optional[str] = None) -> List[str]:
        """
        List available tools.

        Args:
            tag: Optional tag filter

        Returns:
            List of tool names
        """
        if tag:
            return self._categories.get(tag, [])
        return list(self._tools.keys())

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a tool.

        Args:
            name: Tool name

        Returns:
            Tool information dictionary or None
        """
        tool = self.get_tool(name)
        if not tool:
            return None

        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "returns": str(tool.returns) if tool.returns else None,
            "tags": tool.tags,
            "metadata": tool.metadata
        }

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._categories.clear()
        logger.info("Cleared all tools from registry")

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self._tools)}, categories={len(self._categories)})"


# Global registry instance
global_registry = ToolRegistry()