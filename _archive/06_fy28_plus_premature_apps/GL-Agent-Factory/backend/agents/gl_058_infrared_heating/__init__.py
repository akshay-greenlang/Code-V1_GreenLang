"""
GL-058: Infrared Heating Controller Agent (INFRARED-CTRL)

This package provides the InfraredHeatingAgent for optimizing infrared heating
systems in industrial drying, curing, and process heating applications.

Key Features:
- Wavelength and emitter optimization
- Energy transfer efficiency calculations
- Temperature uniformity analysis
- Complete SHA-256 provenance tracking

Standards Compliance:
- ASTM E1933: Standard Practice for Measuring Infrared Transmittance of Materials
- ISO 9288: Thermal Insulation - Heat Transfer by Radiation
- IEC 60519: Safety in Electroheat Installations

Example Usage:
    >>> from backend.agents.gl_058_infrared_heating import (
    ...     InfraredHeatingAgent,
    ...     InfraredHeatingInput,
    ... )
    >>> agent = InfraredHeatingAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Thermal Efficiency: {result.thermal_performance.thermal_efficiency_percent:.1f}%")
"""

from .agent import (
    InfraredHeatingAgent,
    InfraredHeatingInput,
    InfraredHeatingOutput,
    EmitterData,
    TargetData,
    ZoneConfiguration,
    EmitterAnalysis,
    ZoneAnalysis,
    ThermalPerformance,
    EnergyAnalysis,
    Recommendation,
    Warning,
    EmitterType,
    TargetMaterial,
    HeatingMode,
    PACK_SPEC,
)

__all__ = [
    "InfraredHeatingAgent",
    "InfraredHeatingInput",
    "InfraredHeatingOutput",
    "EmitterData",
    "TargetData",
    "ZoneConfiguration",
    "EmitterAnalysis",
    "ZoneAnalysis",
    "ThermalPerformance",
    "EnergyAnalysis",
    "Recommendation",
    "Warning",
    "EmitterType",
    "TargetMaterial",
    "HeatingMode",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-058"
__agent_name__ = "INFRARED-CTRL"
