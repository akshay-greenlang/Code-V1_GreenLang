"""
GL-042: PressureMaster Agent (PRESSUREMASTER)

This package provides the PressureMaster Agent for steam header pressure
optimization in industrial process heat systems.

Key Features:
- Steam header pressure control optimization
- Pressure setpoint recommendations
- Valve position optimization
- Supply/demand balancing
- Complete SHA-256 provenance tracking

Standards Compliance:
- ISA-5.1: Instrumentation Symbols and Identification
- ASME B31.1: Power Piping
- ISA-75.01: Control Valve Sizing

Example Usage:
    >>> from backend.agents.gl_042_steam_pressure import (
    ...     PressureMasterAgent,
    ...     PressureMasterInput,
    ... )
    >>> agent = PressureMasterAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Stability Score: {result.overall_stability_score}")
"""

from .agent import (
    PressureMasterAgent,
    PressureMasterInput,
    PressureMasterOutput,
    HeaderPressure,
    BoilerStatus,
    ValvePosition,
    SteamDemand,
    SetpointRecommendation,
    ValveAdjustment,
    HeaderAnalysis,
    PACK_SPEC,
)

__all__ = [
    "PressureMasterAgent",
    "PressureMasterInput",
    "PressureMasterOutput",
    "HeaderPressure",
    "BoilerStatus",
    "ValvePosition",
    "SteamDemand",
    "SetpointRecommendation",
    "ValveAdjustment",
    "HeaderAnalysis",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-042"
__agent_name__ = "PRESSUREMASTER"
