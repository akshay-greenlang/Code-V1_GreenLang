"""
GL-020: Economizer Performance Analysis Agent (ECONOPULSE)

This package provides the EconomizerPerformanceAgent for analyzing economizer
heat exchanger performance in industrial boiler and HRSG systems.

Key Features:
- Acid dew point calculation using Verhoff-Banchero correlation
- Heat exchanger effectiveness using NTU-epsilon method
- Steaming risk detection using IAPWS-IF97 saturation properties
- Cold-end corrosion risk assessment
- Complete SHA-256 provenance tracking

Thermodynamic Standards:
- Acid dew point: Verhoff-Banchero correlation (1975)
- Saturation temperature: IAPWS-IF97 Equation 31
- Effectiveness: Kays & London NTU method

Applications:
- Boiler economizer optimization
- HRSG (Heat Recovery Steam Generator) analysis
- Flue gas heat recovery systems
- Feed water heating performance monitoring

Example Usage:
    >>> from backend.agents.gl_020_economizer_performance import (
    ...     EconomizerPerformanceAgent,
    ...     EconomizerInput,
    ...     FlueGasComposition,
    ...     WaterSideConditions,
    ...     HeatExchangerGeometry,
    ... )
    >>>
    >>> agent = EconomizerPerformanceAgent()
    >>> input_data = EconomizerInput(
    ...     flue_gas=FlueGasComposition(
    ...         temperature_in_celsius=350.0,
    ...         temperature_out_celsius=150.0,
    ...         mass_flow_kg_s=50.0,
    ...         H2O_percent=8.0,
    ...         SO3_ppmv=15.0,
    ...         total_pressure_kPa=101.325,
    ...     ),
    ...     water_side=WaterSideConditions(
    ...         inlet_temperature_celsius=105.0,
    ...         outlet_temperature_celsius=180.0,
    ...         mass_flow_kg_s=20.0,
    ...         drum_pressure_MPa=4.0,
    ...     ),
    ...     heat_exchanger=HeatExchangerGeometry(
    ...         flow_arrangement="counter_flow",
    ...         tube_outer_diameter_mm=51.0,
    ...         tube_wall_thickness_mm=4.0,
    ...     ),
    ... )
    >>> result = agent.run(input_data)
    >>> print(f"Effectiveness: {result.effectiveness:.3f}")
    >>> print(f"Acid dew point: {result.acid_dew_point_celsius:.1f} C")
    >>> print(f"Steaming risk: {result.steaming_analysis.risk_level}")
    >>> print(f"Corrosion risk: {result.corrosion_analysis.risk_level}")
"""

from .agent import (
    # Main Agent
    EconomizerPerformanceAgent,

    # Input Models
    EconomizerInput,
    FlueGasComposition,
    WaterSideConditions,
    HeatExchangerGeometry,
    OperatingConditions,

    # Output Models
    EconomizerOutput,
    ThermalPerformance,
    AcidDewPointAnalysis,
    SteamingAnalysis,
    CorrosionAnalysis,
    ProvenanceRecord,

    # Enumerations
    FlowArrangement,
    RiskLevel,

    # Pack Specification
    PACK_SPEC,
)

from .calculators.acid_dew_point import (
    verhoff_banchero_acid_dew_point,
    calculate_partial_pressures,
)

from .calculators.effectiveness import (
    effectiveness_counter_flow,
    effectiveness_parallel_flow,
    effectiveness_cross_flow_both_unmixed,
    calculate_heat_transfer,
)

from .calculators.steaming import (
    saturation_temperature_IF97,
    detect_steaming_risk,
)

from .calculators.corrosion import (
    assess_corrosion_risk,
    estimate_tube_metal_temperature,
    CorrosionMechanism,
)

__all__ = [
    # Main Agent
    "EconomizerPerformanceAgent",

    # Input Models
    "EconomizerInput",
    "FlueGasComposition",
    "WaterSideConditions",
    "HeatExchangerGeometry",
    "OperatingConditions",

    # Output Models
    "EconomizerOutput",
    "ThermalPerformance",
    "AcidDewPointAnalysis",
    "SteamingAnalysis",
    "CorrosionAnalysis",
    "ProvenanceRecord",

    # Enumerations
    "FlowArrangement",
    "RiskLevel",
    "CorrosionMechanism",

    # Pack Specification
    "PACK_SPEC",

    # Calculators - Acid Dew Point
    "verhoff_banchero_acid_dew_point",
    "calculate_partial_pressures",

    # Calculators - Effectiveness
    "effectiveness_counter_flow",
    "effectiveness_parallel_flow",
    "effectiveness_cross_flow_both_unmixed",
    "calculate_heat_transfer",

    # Calculators - Steaming
    "saturation_temperature_IF97",
    "detect_steaming_risk",

    # Calculators - Corrosion
    "assess_corrosion_risk",
    "estimate_tube_metal_temperature",
]

# Agent metadata
__version__ = "1.0.0"
__agent_id__ = "GL-020"
__agent_name__ = "ECONOPULSE"
