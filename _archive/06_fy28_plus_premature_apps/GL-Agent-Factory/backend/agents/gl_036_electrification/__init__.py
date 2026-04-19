"""
GL-036: Electrification Analyzer Agent (ELECTRIFY-SCAN)

This package provides the ElectrificationAgent for analyzing process heat
electrification opportunities.

Key Features:
- Electrification feasibility assessment
- Heat pump and electric heating evaluation
- Economic analysis with grid carbon intensity
- Technology recommendation
- Complete SHA-256 provenance tracking

Standards Compliance:
- IEA Energy Technology Perspectives
- ASHRAE Electrification Guidelines
- IEEE Standards

Example Usage:
    >>> from backend.agents.gl_036_electrification import (
    ...     ElectrificationAgent,
    ...     ElectrificationInput,
    ... )
    >>> agent = ElectrificationAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Electrification Score: {result.feasibility_score}")
"""

from .agent import (
    ElectrificationAgent,
    ElectrificationInput,
    ElectrificationOutput,
    ProcessHeatLoad,
    ElectrificationOption,
    PACK_SPEC,
)

__all__ = [
    "ElectrificationAgent",
    "ElectrificationInput",
    "ElectrificationOutput",
    "ProcessHeatLoad",
    "ElectrificationOption",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-036"
__agent_name__ = "ELECTRIFY-SCAN"
