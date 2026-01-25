"""
GL-057: Induction Heating Optimizer Agent (INDUCTION-OPT)

This package provides the InductionHeatingAgent for optimizing induction heating
systems in metal processing, forging, and heat treatment applications.

Key Features:
- Power factor and efficiency optimization
- Coil design validation and performance analysis
- Frequency and power level optimization
- Thermal penetration depth calculations
- Complete SHA-256 provenance tracking

Standards Compliance:
- IEEE 1584: Arc Flash Hazard Calculation
- ASME SA-370: Standard Test Methods for Mechanical Testing of Steel Products
- IEC 60519: Safety in Electroheat Installations

Example Usage:
    >>> from backend.agents.gl_057_induction_heating import (
    ...     InductionHeatingAgent,
    ...     InductionHeatingInput,
    ... )
    >>> agent = InductionHeatingAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Efficiency: {result.efficiency.overall_efficiency_percent:.1f}%")
"""

from .agent import (
    InductionHeatingAgent,
    InductionHeatingInput,
    InductionHeatingOutput,
    WorkpieceData,
    CoilData,
    PowerSupplyData,
    ProcessData,
    ElectromagneticAnalysis,
    ThermalAnalysis,
    EfficiencyAnalysis,
    EnergyAnalysis,
    Recommendation,
    Warning,
    MaterialType,
    HeatingApplication,
    CoilConfiguration,
    PACK_SPEC,
)

__all__ = [
    "InductionHeatingAgent",
    "InductionHeatingInput",
    "InductionHeatingOutput",
    "WorkpieceData",
    "CoilData",
    "PowerSupplyData",
    "ProcessData",
    "ElectromagneticAnalysis",
    "ThermalAnalysis",
    "EfficiencyAnalysis",
    "EnergyAnalysis",
    "Recommendation",
    "Warning",
    "MaterialType",
    "HeatingApplication",
    "CoilConfiguration",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-057"
__agent_name__ = "INDUCTION-OPT"
