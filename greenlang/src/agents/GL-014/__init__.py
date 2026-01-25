# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Heat Exchanger Optimizer Agent

Industrial heat exchanger optimization agent for performance monitoring,
fouling prediction, and intelligent cleaning schedule generation.

Agent ID: GL-014
Codename: EXCHANGER-PRO
Version: 1.0.0
Category: Heat Exchangers
Type: Optimizer

This module provides the core foundation for the Heat Exchanger Optimizer agent,
which monitors heat exchanger performance, predicts fouling progression, and
generates cost-optimized cleaning schedules based on real-time process data.

Key Capabilities:
    - Real-time thermal performance monitoring
    - Fouling factor calculation and trend analysis
    - LMTD and effectiveness-NTU calculations
    - Predictive fouling progression modeling
    - Cost-benefit optimized cleaning schedules
    - Multi-framework compliance (TEMA, API, ASME, HTRI)
    - Integration with process historians and CMMS

Zero-Hallucination Guarantee:
    All thermal calculations use deterministic engineering formulas.
    No LLM inference for numeric values - only validated thermodynamic equations.

Example:
    >>> from agents.GL_014 import HeatExchangerOptimizerAgent
    >>> from agents.GL_014.config import HeatExchangerConfig
    >>>
    >>> config = HeatExchangerConfig()
    >>> agent = HeatExchangerOptimizerAgent(config)
    >>> result = agent.process(temperature_data, pressure_data, flow_data)

Author: GreenLang AI Agent Factory
License: Apache-2.0
Repository: https://github.com/greenlang/agents/gl-014
Documentation: https://docs.greenlang.io/agents/gl-014
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Any, List

__version__ = "1.0.0"
__agent_id__ = "GL-014"
__agent_codename__ = "EXCHANGER-PRO"
__agent_name__ = "HeatExchangerOptimizerAgent"
__author__ = "GreenLang AI Agent Factory"
__license__ = "Apache-2.0"

# Public API exports - these are lazily loaded for performance
__all__: List[str] = [
    # Agent metadata
    "__version__",
    "__agent_id__",
    "__agent_codename__",
    "__agent_name__",
    # Core enumerations
    "ExchangerType",
    "FoulingMechanism",
    "CleaningMethod",
    "FluidType",
    "FlowArrangement",
    "ShellType",
    "TubeLayout",
    "MaterialType",
    "PerformanceStatus",
    "FoulingState",
    "MaintenanceUrgency",
    "CleaningSide",
    # Input models
    "TemperatureData",
    "PressureData",
    "FlowData",
    "ExchangerParameters",
    "FluidProperties",
    "OperatingHistory",
    "HeatExchangerInput",
    # Output models
    "CleaningSchedule",
    "PerformanceMetrics",
    "FoulingAnalysis",
    "EfficiencyReport",
    "EconomicImpact",
    "HeatExchangerOutput",
    # Configuration
    "HeatExchangerConfig",
    "Settings",
    # Validation results
    "ValidationResult",
    "ValidationError",
]


# Lazy loading implementation for improved import performance
# This pattern defers module imports until they are actually needed
def __getattr__(name: str) -> Any:
    """
    Lazy loading of module components.

    This function implements PEP 562 lazy module loading to improve
    import performance by deferring the loading of submodules until
    they are actually accessed.

    Args:
        name: The name of the attribute being accessed.

    Returns:
        The requested attribute from the appropriate submodule.

    Raises:
        AttributeError: If the requested attribute does not exist.
    """
    # Mapping of public names to their source modules
    _lazy_imports = {
        # Enumerations from config module
        "ExchangerType": "config",
        "FoulingMechanism": "config",
        "CleaningMethod": "config",
        "FluidType": "config",
        "FlowArrangement": "config",
        "ShellType": "config",
        "TubeLayout": "config",
        "MaterialType": "config",
        "PerformanceStatus": "config",
        "FoulingState": "config",
        "MaintenanceUrgency": "config",
        "CleaningSide": "config",
        # Input models from config module
        "TemperatureData": "config",
        "PressureData": "config",
        "FlowData": "config",
        "ExchangerParameters": "config",
        "FluidProperties": "config",
        "OperatingHistory": "config",
        "HeatExchangerInput": "config",
        # Output models from config module
        "CleaningSchedule": "config",
        "PerformanceMetrics": "config",
        "FoulingAnalysis": "config",
        "EfficiencyReport": "config",
        "EconomicImpact": "config",
        "HeatExchangerOutput": "config",
        # Configuration from config module
        "HeatExchangerConfig": "config",
        "Settings": "config",
        # Validation from config module
        "ValidationResult": "config",
        "ValidationError": "config",
    }

    if name in _lazy_imports:
        module_name = _lazy_imports[name]
        # Import the submodule relative to this package
        module = importlib.import_module(f".{module_name}", __package__)
        # Get the attribute from the submodule
        value = getattr(module, name)
        # Cache the attribute in the module namespace for future access
        setattr(sys.modules[__name__], name, value)
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    """
    Provide accurate attribute listing for the module.

    This enables proper auto-completion and introspection tools
    to discover all available public attributes.

    Returns:
        List of all public attribute names.
    """
    # Combine module-level attributes with lazy-loaded attributes
    module_attrs = list(globals().keys())
    return sorted(set(module_attrs + __all__))


# Type checking imports - these are only used for static analysis
# and do not affect runtime performance
if TYPE_CHECKING:
    from .config import (
        # Enumerations
        ExchangerType,
        FoulingMechanism,
        CleaningMethod,
        FluidType,
        FlowArrangement,
        ShellType,
        TubeLayout,
        MaterialType,
        PerformanceStatus,
        FoulingState,
        MaintenanceUrgency,
        CleaningSide,
        # Input models
        TemperatureData,
        PressureData,
        FlowData,
        ExchangerParameters,
        FluidProperties,
        OperatingHistory,
        HeatExchangerInput,
        # Output models
        CleaningSchedule,
        PerformanceMetrics,
        FoulingAnalysis,
        EfficiencyReport,
        EconomicImpact,
        HeatExchangerOutput,
        # Configuration
        HeatExchangerConfig,
        Settings,
        # Validation
        ValidationResult,
        ValidationError,
    )
