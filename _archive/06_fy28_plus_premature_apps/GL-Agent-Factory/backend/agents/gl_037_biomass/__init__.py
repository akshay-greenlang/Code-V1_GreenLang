"""
GL-037: Biomass Combustion Optimizer Agent (BIOMASS-OPT)

This package provides the BiomassOptAgent for optimizing biomass
combustion systems.

Key Features:
- Biomass fuel characterization
- Combustion efficiency optimization
- Emissions control (NOx, PM, CO)
- Ash handling and slagging prediction
- Complete SHA-256 provenance tracking

Standards Compliance:
- EN 303-5: Biomass Boilers
- ISO 17225: Solid Biofuels
- EPA New Source Performance Standards

Example Usage:
    >>> from backend.agents.gl_037_biomass import (
    ...     BiomassOptAgent,
    ...     BiomassOptInput,
    ... )
    >>> agent = BiomassOptAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Combustion Efficiency: {result.thermal_efficiency_pct}%")
"""

from .agent import (
    BiomassOptAgent,
    BiomassOptInput,
    BiomassOptOutput,
    BiomassFuel,
    CombustionPerformance,
    PACK_SPEC,
)

__all__ = [
    "BiomassOptAgent",
    "BiomassOptInput",
    "BiomassOptOutput",
    "BiomassFuel",
    "CombustionPerformance",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-037"
__agent_name__ = "BIOMASS-OPT"
