"""
GreenLang ASME PTC Compliance Library

Zero-Hallucination Standards-Compliant Calculations

This module provides deterministic, ASME Performance Test Code compliant
calculations for industrial equipment performance assessment.

Modules:
    - ptc_4_1: Boiler efficiency (Steam Generating Units)
    - ptc_4_3: Air heater performance
    - ptc_4_4: HRSG performance (Heat Recovery Steam Generators)
    - ptc_19_10: Flue gas analysis
    - section_1: Pressure calculations (Power Boilers)
    - b31_1_pipe_stress: Power piping stress analysis (ASME B31.1)

All calculations provide:
    - Deterministic outputs (same input = same output)
    - Complete provenance tracking (SHA-256 hashes)
    - ASME standards compliance
    - Conservative design factors

Author: GreenLang Engineering Team
License: MIT
"""

# PTC 4.1 - Boiler Efficiency
from .ptc_4_1 import (
    PTC41BoilerEfficiency,
    BoilerEfficiencyResult,
    BoilerInputData,
    boiler_efficiency,
    excess_air_from_o2,
)

# PTC 4.3 - Air Heaters
from .ptc_4_3 import (
    PTC43AirHeater,
    AirHeaterResult,
    AirHeaterInputData,
    air_heater_performance,
    air_heater_leakage,
)

# PTC 4.4 - HRSG
from .ptc_4_4 import (
    PTC44HRSG,
    HRSGResult,
    HRSGInputData,
    HRSGSection,
    hrsg_performance,
    hrsg_pinch_point,
)

# PTC 19.10 - Flue Gas Analysis
from .ptc_19_10 import (
    PTC1910FlueGas,
    FlueGasAnalysisResult,
    FlueGasComposition,
    flue_gas_analysis,
    correct_emissions_to_o2,
)

# Section I - Pressure Calculations
from .section_1 import (
    ASMESectionI,
    PressureCalculationResult,
    MaterialProperties,
    TubeType,
    HeadType,
    shell_thickness,
    tube_wall_thickness,
)

# B31.1 - Power Piping Stress Analysis
from .b31_1_pipe_stress import (
    ASMEB311PipeStress,
    B311StressResult,
    MinimumThicknessResult,
    PipeGeometry,
    LoadData,
    PipeMaterial,
    PipeSchedule,
    LoadCategory,
    pipe_hoop_stress,
    pipe_sustained_stress,
    pipe_expansion_stress,
    pipe_allowable_stress_range,
    pipe_minimum_thickness,
    analyze_pipe_stress,
)

__version__ = "1.1.0"

__all__ = [
    # PTC 4.1
    "PTC41BoilerEfficiency",
    "BoilerEfficiencyResult",
    "BoilerInputData",
    "boiler_efficiency",
    "excess_air_from_o2",
    # PTC 4.3
    "PTC43AirHeater",
    "AirHeaterResult",
    "AirHeaterInputData",
    "air_heater_performance",
    "air_heater_leakage",
    # PTC 4.4
    "PTC44HRSG",
    "HRSGResult",
    "HRSGInputData",
    "HRSGSection",
    "hrsg_performance",
    "hrsg_pinch_point",
    # PTC 19.10
    "PTC1910FlueGas",
    "FlueGasAnalysisResult",
    "FlueGasComposition",
    "flue_gas_analysis",
    "correct_emissions_to_o2",
    # Section I
    "ASMESectionI",
    "PressureCalculationResult",
    "MaterialProperties",
    "TubeType",
    "HeadType",
    "shell_thickness",
    "tube_wall_thickness",
    # B31.1 - Power Piping
    "ASMEB311PipeStress",
    "B311StressResult",
    "MinimumThicknessResult",
    "PipeGeometry",
    "LoadData",
    "PipeMaterial",
    "PipeSchedule",
    "LoadCategory",
    "pipe_hoop_stress",
    "pipe_sustained_stress",
    "pipe_expansion_stress",
    "pipe_allowable_stress_range",
    "pipe_minimum_thickness",
    "analyze_pipe_stress",
]
