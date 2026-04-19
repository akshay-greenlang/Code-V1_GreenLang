"""
GL-033: District Heating Integrator Agent (DISTRICT-LINK)

This package provides the DistrictHeatingAgent for integrating industrial
waste heat into district heating networks.

Key Features:
- Waste heat source characterization
- District heating network integration analysis
- Temperature and pressure matching
- Economic feasibility assessment
- Complete SHA-256 provenance tracking

Standards Compliance:
- EN 13941: District Heating Networks Design
- ISO 50001: Energy Management Systems
- ASHRAE: District Energy Systems

Example Usage:
    >>> from backend.agents.gl_033_district_heating import (
    ...     DistrictHeatingAgent,
    ...     DistrictHeatingInput,
    ... )
    >>> agent = DistrictHeatingAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Integration Feasibility: {result.feasibility_score}")
"""

from .agent import (
    DistrictHeatingAgent,
    DistrictHeatingInput,
    DistrictHeatingOutput,
    WasteHeatSource,
    DistrictNetwork,
    IntegrationScenario,
    PACK_SPEC,
)

__all__ = [
    "DistrictHeatingAgent",
    "DistrictHeatingInput",
    "DistrictHeatingOutput",
    "WasteHeatSource",
    "DistrictNetwork",
    "IntegrationScenario",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-033"
__agent_name__ = "DISTRICT-LINK"
