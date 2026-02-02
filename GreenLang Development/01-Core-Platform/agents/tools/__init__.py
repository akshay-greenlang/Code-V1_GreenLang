# -*- coding: utf-8 -*-
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
    CalculateScopeEmissionsTool,
    RegionalEmissionFactorTool,
)

# Financial tools
from .financial import (
    FinancialMetricsTool,
)

# Grid/Utility tools
from .grid import (
    GridIntegrationTool,
)

# Security components
from .validation import (
    ValidationRule,
    ValidationResult,
    RangeValidator,
    TypeValidator,
    EnumValidator,
    RegexValidator,
    CustomValidator,
    CompositeValidator,
)

from .rate_limiting import (
    RateLimiter,
    RateLimitExceeded,
    get_rate_limiter,
    configure_rate_limiter,
)

from .audit import (
    AuditLogger,
    AuditLogEntry,
    get_audit_logger,
    configure_audit_logger,
)

from .security_config import (
    SecurityConfig,
    get_security_config,
    configure_security,
    development_config,
    testing_config,
    production_config,
    high_security_config,
    SecurityContext,
)

# Telemetry
from .telemetry import (
    TelemetryCollector,
    ToolMetrics,
    get_telemetry,
    reset_global_telemetry,
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
    "CalculateScopeEmissionsTool",
    "RegionalEmissionFactorTool",
    # Financial Tools
    "FinancialMetricsTool",
    # Grid/Utility Tools
    "GridIntegrationTool",
    # Security - Validation
    "ValidationRule",
    "ValidationResult",
    "RangeValidator",
    "TypeValidator",
    "EnumValidator",
    "RegexValidator",
    "CustomValidator",
    "CompositeValidator",
    # Security - Rate Limiting
    "RateLimiter",
    "RateLimitExceeded",
    "get_rate_limiter",
    "configure_rate_limiter",
    # Security - Audit Logging
    "AuditLogger",
    "AuditLogEntry",
    "get_audit_logger",
    "configure_audit_logger",
    # Security - Configuration
    "SecurityConfig",
    "get_security_config",
    "configure_security",
    "development_config",
    "testing_config",
    "production_config",
    "high_security_config",
    "SecurityContext",
    # Telemetry
    "TelemetryCollector",
    "ToolMetrics",
    "get_telemetry",
    "reset_global_telemetry",
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
        registry.register(CalculateScopeEmissionsTool(), category="emissions", version="1.0.0")
        registry.register(RegionalEmissionFactorTool(), category="emissions", version="1.0.0")
    except ValueError:
        # Tools already registered (e.g., in testing)
        pass

    # Register financial tools
    try:
        registry.register(FinancialMetricsTool(), category="financial", version="1.0.0")
    except ValueError:
        pass

    # Register grid tools
    try:
        registry.register(GridIntegrationTool(), category="grid", version="1.0.0")
    except ValueError:
        pass


# Auto-register on import
_register_standard_tools()
