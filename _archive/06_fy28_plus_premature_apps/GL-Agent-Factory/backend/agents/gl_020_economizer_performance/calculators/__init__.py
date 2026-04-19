"""
GL-020 Economizer Performance Calculators

This package contains deterministic thermodynamic calculators for
economizer performance analysis. All calculations follow zero-hallucination
principles with exact formulas from established standards.

Modules:
- acid_dew_point: Verhoff-Banchero correlation for sulfuric acid dew point
- effectiveness: NTU-epsilon heat exchanger effectiveness methods
- steaming: IAPWS-IF97 saturation properties and steaming risk detection
- corrosion: Cold-end corrosion risk assessment
"""

from .acid_dew_point import (
    verhoff_banchero_acid_dew_point,
    calculate_partial_pressures,
)

from .effectiveness import (
    effectiveness_counter_flow,
    effectiveness_parallel_flow,
    effectiveness_cross_flow_both_unmixed,
    calculate_heat_transfer,
)

from .steaming import (
    saturation_temperature_IF97,
    detect_steaming_risk,
)

from .corrosion import (
    assess_corrosion_risk,
    estimate_tube_metal_temperature,
)

__all__ = [
    # Acid Dew Point
    "verhoff_banchero_acid_dew_point",
    "calculate_partial_pressures",

    # Effectiveness
    "effectiveness_counter_flow",
    "effectiveness_parallel_flow",
    "effectiveness_cross_flow_both_unmixed",
    "calculate_heat_transfer",

    # Steaming
    "saturation_temperature_IF97",
    "detect_steaming_risk",

    # Corrosion
    "assess_corrosion_risk",
    "estimate_tube_metal_temperature",
]
