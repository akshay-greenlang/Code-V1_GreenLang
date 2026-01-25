"""
GL-041: EnergyViz Agent (ENERGYVIZ)

This package provides the EnergyViz Agent for real-time energy management
dashboard visualization and analytics.

Key Features:
- Real-time energy consumption monitoring
- Peak demand tracking and forecasting
- Energy efficiency KPI calculation
- Cost analysis and budgeting
- Complete SHA-256 provenance tracking

Standards Compliance:
- ISO 50001: Energy Management Systems
- ASME PTC 19.1: Test Uncertainty
- ASHRAE 90.1: Energy Standard for Buildings

Example Usage:
    >>> from backend.agents.gl_041_energy_dashboard import (
    ...     EnergyVizAgent,
    ...     EnergyVizInput,
    ... )
    >>> agent = EnergyVizAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Total Energy: {result.total_energy_kwh} kWh")
"""

from .agent import (
    EnergyVizAgent,
    EnergyVizInput,
    EnergyVizOutput,
    EnergyMeter,
    CostRate,
    TargetKPI,
    EnergyBreakdown,
    PeakDemand,
    KPIResult,
    CostAnalysis,
    PACK_SPEC,
)

__all__ = [
    "EnergyVizAgent",
    "EnergyVizInput",
    "EnergyVizOutput",
    "EnergyMeter",
    "CostRate",
    "TargetKPI",
    "EnergyBreakdown",
    "PeakDemand",
    "KPIResult",
    "CostAnalysis",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-041"
__agent_name__ = "ENERGYVIZ"
