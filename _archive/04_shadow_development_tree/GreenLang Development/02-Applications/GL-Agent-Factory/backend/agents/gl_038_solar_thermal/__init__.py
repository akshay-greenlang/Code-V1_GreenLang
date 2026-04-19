"""
GL-038: Solar Thermal Integrator Agent (SOLAR-THERMAL)

This package provides the SolarThermalAgent for integrating solar thermal
systems with industrial process heat.

Key Features:
- Solar resource assessment
- Collector sizing and selection
- Thermal storage integration
- Economic feasibility analysis
- Complete SHA-256 provenance tracking

Standards Compliance:
- ISO 9806: Solar Thermal Collectors
- ASHRAE 93: Solar Collector Testing
- EN 12975: Thermal Solar Systems

Example Usage:
    >>> from backend.agents.gl_038_solar_thermal import (
    ...     SolarThermalAgent,
    ...     SolarThermalInput,
    ... )
    >>> agent = SolarThermalAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Solar Fraction: {result.solar_fraction_pct}%")
"""

from .agent import (
    SolarThermalAgent,
    SolarThermalInput,
    SolarThermalOutput,
    SolarResource,
    CollectorSystem,
    PACK_SPEC,
)

__all__ = [
    "SolarThermalAgent",
    "SolarThermalInput",
    "SolarThermalOutput",
    "SolarResource",
    "CollectorSystem",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-038"
__agent_name__ = "SOLAR-THERMAL"
