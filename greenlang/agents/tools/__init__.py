"""
GreenLang Shared Tool Library
==============================

Centralized tool library for all GreenLang agents.

This library provides:
- Reusable tools for emission calculations
- Tool registry for dynamic discovery
- Standard tool interfaces
- Tool composition capabilities

Usage:
    from greenlang.agents.tools import get_registry, CalculateEmissionsTool

    # Get global registry
    registry = get_registry()

    # Register a tool
    tool = CalculateEmissionsTool()
    registry.register(tool, category="emissions")

    # Get a tool
    calc_tool = registry.get("calculate_emissions")

    # Execute tool
    result = calc_tool(
        fuel_type="natural_gas",
        amount=1000,
        unit="therms",
        emission_factor=53.06,
        emission_factor_unit="kgCO2e/therm"
    )

Author: GreenLang Framework Team
Date: October 2025
Status: Production Ready
"""

# Base classes
from .base import (
    BaseTool,
    ToolDef,
    ToolResult,
    ToolSafety,
    CompositeTool,
    tool,
)

# Registry
from .registry import (
    ToolRegistry,
    get_registry,
    register_tool,
    get_tool,
)

# Emissions tools
from .emissions import (
    CalculateEmissionsTool,
    AggregateEmissionsTool,
    CalculateBreakdownTool,
)

__all__ = [
    # Base
    "BaseTool",
    "ToolDef",
    "ToolResult",
    "ToolSafety",
    "CompositeTool",
    "tool",
    # Registry
    "ToolRegistry",
    "get_registry",
    "register_tool",
    "get_tool",
    # Emissions Tools
    "CalculateEmissionsTool",
    "AggregateEmissionsTool",
    "CalculateBreakdownTool",
]


# ==============================================================================
# Auto-register Standard Tools
# ==============================================================================

def _register_standard_tools():
    """Register all standard tools on module import."""
    registry = get_registry()

    # Register emissions tools
    try:
        registry.register(CalculateEmissionsTool(), category="emissions", version="1.0.0")
        registry.register(AggregateEmissionsTool(), category="emissions", version="1.0.0")
        registry.register(CalculateBreakdownTool(), category="emissions", version="1.0.0")
    except ValueError:
        # Tools already registered (e.g., in testing)
        pass


# Auto-register on import
_register_standard_tools()
