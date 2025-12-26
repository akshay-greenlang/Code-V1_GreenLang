"""
GL-014 EXCHANGERPRO - Deterministic Thermal Engine (DTE) Calculators

Zero-hallucination calculation engines for heat exchanger performance
monitoring, sizing, and optimization. All calculations are deterministic,
reproducible, and include SHA-256 provenance tracking.

Calculators:
- HeatDutyCalculator: Energy balance and duty calculations (Q = m*Cp*dT)
- LMTDCalculator: Log Mean Temperature Difference with F-factor correction
- EpsilonNTUCalculator: Effectiveness-NTU method for all flow configurations
- UACalculator: Overall heat transfer coefficient from LMTD and NTU methods
- PressureDropCalculator: Tube-side and shell-side pressure drop (Darcy-Weisbach)
- EffectivenessCalculator: Thermal effectiveness and approach temperatures

Standards Compliance:
- TEMA (Tubular Exchanger Manufacturers Association)
- ASME PTC 12.5 (Single Phase Heat Exchangers)
- HTRI/HTFS methodologies

Zero-Hallucination Principle:
    All thermodynamic and fluid mechanics calculations are performed by
    these deterministic calculators. The LLM NEVER computes Q, UA, LMTD,
    NTU, epsilon, or delta-P values. Same inputs ALWAYS produce same outputs.
"""

from .heat_duty import (
    HeatDutyCalculator,
    HeatDutyInputs,
    HeatDutyResult,
    EnergyBalanceResult,
)
from .lmtd_calculator import (
    LMTDCalculator,
    LMTDInputs,
    LMTDResult,
    FlowArrangement,
)
from .epsilon_ntu import (
    EpsilonNTUCalculator,
    EpsilonNTUInputs,
    EpsilonNTUResult,
    ExchangerConfiguration,
)
from .ua_calculator import (
    UACalculator,
    UAInputs,
    UAResult,
    FoulingResistance,
)
from .pressure_drop import (
    PressureDropCalculator,
    PressureDropInputs,
    PressureDropResult,
    FluidProperties,
)
from .effectiveness import (
    EffectivenessCalculator,
    EffectivenessInputs,
    EffectivenessResult,
    ApproachTemperatures,
)

__all__ = [
    # Heat Duty
    "HeatDutyCalculator",
    "HeatDutyInputs",
    "HeatDutyResult",
    "EnergyBalanceResult",
    # LMTD
    "LMTDCalculator",
    "LMTDInputs",
    "LMTDResult",
    "FlowArrangement",
    # Epsilon-NTU
    "EpsilonNTUCalculator",
    "EpsilonNTUInputs",
    "EpsilonNTUResult",
    "ExchangerConfiguration",
    # UA
    "UACalculator",
    "UAInputs",
    "UAResult",
    "FoulingResistance",
    # Pressure Drop
    "PressureDropCalculator",
    "PressureDropInputs",
    "PressureDropResult",
    "FluidProperties",
    # Effectiveness
    "EffectivenessCalculator",
    "EffectivenessInputs",
    "EffectivenessResult",
    "ApproachTemperatures",
]

# Module version
__version__ = "1.0.0"
