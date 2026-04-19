# -*- coding: utf-8 -*-
"""
GreenLang Tool Registry
=======================

Dynamic tool discovery, registration, and management system.

Features:
- Automatic tool discovery from modules
- Dynamic tool registration
- Tool versioning and dependencies
- Tool composition
- Tool cataloging and search

Author: GreenLang Framework Team
Date: October 2025
Status: Production Ready
"""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable

from .base import BaseTool, ToolDef, ToolSafety

logger = logging.getLogger(__name__)


# ==============================================================================
# Tool Registry
# ==============================================================================

class ToolRegistry:
    """
    Central registry for all shared tools.

    Provides:
    - Dynamic tool registration
    - Tool discovery from modules
    - Tool retrieval by name
    - Tool cataloging
    - Version management

    Usage:
        # Get global registry
        registry = ToolRegistry.get_instance()

        # Register a tool
        registry.register(my_tool)

        # Get a tool by name
        tool = registry.get("calculate_emissions")

        # List all tools
        tools = registry.list_tools()

        # Discover tools in a module
        registry.discover("greenlang.agents.tools.emissions")
    """

    _instance: Optional[ToolRegistry] = None

    def __init__(self):
        """Initialize empty registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
        self._categories: Dict[str, List[str]] = {}
        self.logger = logging.getLogger(__name__)

    @classmethod
    def get_instance(cls) -> ToolRegistry:
        """
        Get singleton instance of ToolRegistry.

        Returns:
            Global ToolRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton instance (useful for testing)."""
        cls._instance = None

    def register(
        self,
        tool: BaseTool,
        category: str = "general",
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool instance to register
            category: Tool category for organization
            version: Tool version
            metadata: Additional metadata (author, tags, etc.)

        Raises:
            ValueError: If tool name already registered
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                f"Use a different name or unregister the existing tool first."
            )

        self._tools[tool.name] = tool

        self._tool_metadata[tool.name] = {
            "category": category,
            "version": version,
            "safety": tool.safety.value,
            "class": tool.__class__.__name__,
            **(metadata or {})
        }

        # Add to category
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(tool.name)

        self.logger.info(f"Registered tool: {tool.name} (category={category}, version={version})")

    def unregister(self, tool_name: str) -> None:
        """
        Unregister a tool from the registry.

        Args:
            tool_name: Name of tool to unregister

        Raises:
            KeyError: If tool not found
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' not found in registry")

        # Remove from category
        metadata = self._tool_metadata[tool_name]
        category = metadata.get("category", "general")
        if category in self._categories:
            self._categories[category].remove(tool_name)

        # Remove tool and metadata
        del self._tools[tool_name]
        del self._tool_metadata[tool_name]

        self.logger.info(f"Unregistered tool: {tool_name}")

    def get(self, tool_name: str) -> BaseTool:
        """
        Get a tool by name.

        Args:
            tool_name: Name of tool to retrieve

        Returns:
            Tool instance

        Raises:
            KeyError: If tool not found
        """
        if tool_name not in self._tools:
            raise KeyError(
                f"Tool '{tool_name}' not found. "
                f"Available tools: {list(self._tools.keys())}"
            )

        return self._tools[tool_name]

    def has(self, tool_name: str) -> bool:
        """
        Check if tool is registered.

        Args:
            tool_name: Tool name to check

        Returns:
            True if registered, False otherwise
        """
        return tool_name in self._tools

    def list_tools(
        self,
        category: Optional[str] = None,
        safety: Optional[ToolSafety] = None,
    ) -> List[str]:
        """
        List all registered tools.

        Args:
            category: Filter by category (optional)
            safety: Filter by safety level (optional)

        Returns:
            List of tool names
        """
        tools = list(self._tools.keys())

        # Filter by category
        if category is not None:
            tools = [
                name for name in tools
                if self._tool_metadata[name].get("category") == category
            ]

        # Filter by safety
        if safety is not None:
            tools = [
                name for name in tools
                if self._tool_metadata[name].get("safety") == safety.value
            ]

        return sorted(tools)

    def get_tool_defs(
        self,
        category: Optional[str] = None,
        safety: Optional[ToolSafety] = None,
    ) -> List[ToolDef]:
        """
        Get ToolDef objects for all tools (for ChatSession).

        Args:
            category: Filter by category (optional)
            safety: Filter by safety level (optional)

        Returns:
            List of ToolDef objects
        """
        tool_names = self.list_tools(category=category, safety=safety)
        return [self._tools[name].get_tool_def() for name in tool_names]

    def get_metadata(self, tool_name: str) -> Dict[str, Any]:
        """
        Get metadata for a tool.

        Args:
            tool_name: Tool name

        Returns:
            Tool metadata dictionary

        Raises:
            KeyError: If tool not found
        """
        if tool_name not in self._tool_metadata:
            raise KeyError(f"Tool '{tool_name}' not found")

        return self._tool_metadata[tool_name].copy()

    def list_categories(self) -> List[str]:
        """
        List all tool categories.

        Returns:
            List of category names
        """
        return sorted(self._categories.keys())

    def discover(self, module_name: str) -> List[str]:
        """
        Discover and register tools from a module.

        This scans a module for BaseTool subclasses and registers them.

        Args:
            module_name: Python module name (e.g., "greenlang.agents.tools.emissions")

        Returns:
            List of discovered tool names

        Raises:
            ImportError: If module cannot be imported
        """
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"Cannot import module '{module_name}': {e}")

        discovered = []

        # Scan module for BaseTool subclasses
        for name, obj in inspect.getmembers(module):
            # Skip private members
            if name.startswith("_"):
                continue

            # Check if it's a BaseTool subclass (but not BaseTool itself)
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseTool)
                and obj is not BaseTool
            ):
                try:
                    # Instantiate tool (assumes no-arg constructor)
                    tool = obj()

                    # Register if not already registered
                    if not self.has(tool.name):
                        # Try to infer category from module name
                        parts = module_name.split(".")
                        category = parts[-1] if len(parts) > 0 else "general"

                        self.register(tool, category=category)
                        discovered.append(tool.name)
                        self.logger.info(f"Discovered tool: {tool.name} from {module_name}")

                except Exception as e:
                    self.logger.warning(
                        f"Cannot instantiate tool {name} from {module_name}: {e}"
                    )

        return discovered

    def discover_all(self, base_module: str = "greenlang.agents.tools") -> List[str]:
        """
        Discover all tools in base_module and its submodules.

        Args:
            base_module: Base module to scan

        Returns:
            List of all discovered tool names
        """
        discovered = []

        # Get base module path
        try:
            module = importlib.import_module(base_module)
            base_path = Path(module.__file__).parent
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Cannot discover tools in {base_module}: {e}")
            return discovered

        # Scan all Python files in the directory
        for py_file in base_path.glob("*.py"):
            if py_file.stem.startswith("_") or py_file.stem in ("base", "registry"):
                continue

            module_name = f"{base_module}.{py_file.stem}"

            try:
                tools = self.discover(module_name)
                discovered.extend(tools)
            except Exception as e:
                self.logger.warning(f"Cannot discover tools in {module_name}: {e}")

        return discovered

    def get_catalog(self) -> Dict[str, Any]:
        """
        Get complete tool catalog with metadata.

        Returns:
            Dictionary with catalog information
        """
        return {
            "total_tools": len(self._tools),
            "categories": {
                cat: len(tools) for cat, tools in self._categories.items()
            },
            "tools": {
                name: {
                    "metadata": self._tool_metadata[name],
                    "stats": self._tools[name].get_stats(),
                }
                for name in self._tools.keys()
            }
        }

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._tool_metadata.clear()
        self._categories.clear()
        self.logger.info("Cleared all tools from registry")

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self._tools)}, categories={len(self._categories)})"


# ==============================================================================
# Global Registry Instance
# ==============================================================================

# Create global registry instance
_global_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """
    Get the global tool registry.

    Returns:
        Global ToolRegistry instance
    """
    return _global_registry


def register_tool(
    tool: BaseTool,
    category: str = "general",
    version: str = "1.0.0",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Register a tool in the global registry.

    Args:
        tool: Tool to register
        category: Tool category
        version: Tool version
        metadata: Additional metadata
    """
    _global_registry.register(tool, category, version, metadata)


def get_tool(tool_name: str) -> BaseTool:
    """
    Get a tool from the global registry.

    Args:
        tool_name: Tool name

    Returns:
        Tool instance
    """
    return _global_registry.get(tool_name)
