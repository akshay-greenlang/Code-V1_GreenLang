# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL - SteamQualityController Agent.

Master controller for maintaining optimal steam quality across industrial
facilities, managing pressure, temperature, moisture content, and 
desuperheater control following zero-hallucination principles.

Agent Specifications:
    - Agent ID: GL-012
    - Codename: STEAMQUAL
    - Name: SteamQualityController
    - Domain: Steam Systems
    - Type: Controller
    - Priority: P2
    - Market: $4B
    - Timeline: Q2 2026

Capabilities:
    - Steam quality parameter monitoring (pressure, temperature, dryness)
    - Desuperheater injection control
    - Pressure control valve management
    - Moisture analysis and condensation prevention
    - Multi-header coordination
    - SCADA integration
    - Real-time KPI dashboard generation

Standards Compliance:
    - ASME PTC 19.11 - Steam and Water Sampling
    - ASME PTC 4.4 - Gas Turbine Heat Recovery Steam Generators
    - ASME PTC 6 - Steam Turbines
    - IAPWS-IF97 - Steam Properties
    - IEC 61511 - Functional Safety
    - ISA-88 - Batch Control

Zero-Hallucination Guarantee:
    All calculations are fully deterministic using fixed seed (42),
    temperature=0.0 for any AI operations, and complete SHA-256
    provenance tracking for audit compliance.

Example:
    >>> from GL012 import SteamQualityOrchestrator, SteamQualityConfig
    >>> config = SteamQualityConfig(agent_id="GL-012")
    >>> orchestrator = SteamQualityOrchestrator(config)
    >>> result = await orchestrator.execute({
    ...     "request_type": "analyze",
    ...     "steam_headers": ["header_1"],
    ...     "measurement_data": {"header_1": {"pressure_bar": 10.0}}
    ... })

Author: GreenLang Industrial Optimization Team
Date: December 2025
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GreenLang Industrial Optimization Team"
__agent_id__ = "GL-012"
__codename__ = "STEAMQUAL"
__agent_name__ = "SteamQualityController"
__domain__ = "Steam Systems"
__priority__ = "P2"
__market_size__ = "B"
__timeline__ = "Q2 2026"

# Package exports
__all__ = [
    # Version info
    "__version__",
    "__agent_id__",
    "__codename__",
    # Main classes (to be imported from submodules)
    "SteamQualityOrchestrator",
    "SteamQualityConfig",
    "SteamQualityControllerConfig",
    "SteamQualityTools",
    # Enums
    "SteamState",
    "QualityLevel",
    "ControlMode",
    "DesuperheaterMode",
    "AlertSeverity",
    # Result classes
    "SteamQualityResult",
    "DesuperheaterControlResult",
    "PressureControlResult",
    "MoistureAnalysisResult",
    "SteamQualityKPIResult",
    # Convenience functions
    "create_default_orchestrator",
    "create_default_config",
]


def create_default_config():
    """Create default SteamQualityController configuration.
    
    Returns:
        SteamQualityControllerConfig with standard settings for
        10 bar saturated steam operation.
    """
    from .config import create_default_config as _create_config
    return _create_config()


def create_default_orchestrator():
    """Create SteamQualityOrchestrator with default configuration.
    
    Returns:
        SteamQualityOrchestrator ready for steam quality control.
    
    Example:
        >>> orchestrator = create_default_orchestrator()
        >>> result = await orchestrator.execute({...})
    """
    from .steam_quality_orchestrator import SteamQualityOrchestrator
    from .config import create_default_config
    
    config = create_default_config()
    return SteamQualityOrchestrator(config)


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import of submodule classes."""
    if name == "SteamQualityOrchestrator":
        from .steam_quality_orchestrator import SteamQualityOrchestrator
        return SteamQualityOrchestrator
    elif name in ("SteamQualityConfig", "SteamQualityControllerConfig"):
        from .config import SteamQualityControllerConfig
        return SteamQualityControllerConfig
    elif name == "SteamQualityTools":
        from .tools import SteamQualityTools
        return SteamQualityTools
    elif name in ("SteamState", "QualityLevel", "ControlMode", "DesuperheaterMode", "AlertSeverity"):
        from .steam_quality_orchestrator import (
            SteamState, QualityLevel, ControlMode, DesuperheaterMode, AlertSeverity
        )
        return locals()[name]
    elif name in ("SteamQualityResult", "DesuperheaterControlResult", 
                  "PressureControlResult", "MoistureAnalysisResult", "SteamQualityKPIResult"):
        from .tools import (
            SteamQualityResult, DesuperheaterControlResult,
            PressureControlResult, MoistureAnalysisResult, SteamQualityKPIResult
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
