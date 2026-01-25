"""
GL-009_ThermalIQ Calculators Module
===================================

Zero-hallucination deterministic calculation engines for thermal fluid analysis.

All calculations are:
- Deterministic: Same input produces identical output (bit-perfect)
- Reproducible: Full provenance tracking with SHA-256 hashes
- Auditable: Complete calculation trails for regulatory compliance
- Standards-based: Formulas from ASME, IAPWS, thermodynamic textbooks

NO LLM in calculation path - guarantees zero hallucination risk.

References:
-----------
- ASME Steam Tables (2014)
- IAPWS-IF97 Industrial Formulation
- Moran & Shapiro, Fundamentals of Engineering Thermodynamics, 9th Ed.
- Bejan, Advanced Engineering Thermodynamics, 4th Ed.
- ISO 5167 Flow Measurement Standards
"""

from .thermal_efficiency import (
    ThermalEfficiencyCalculator,
    EfficiencyResult,
    LossBreakdown,
)

from .exergy_calculator import (
    ExergyCalculator,
    ExergyResult,
    DestructionResult,
)

from .heat_balance import (
    HeatBalanceCalculator,
    HeatBalanceResult,
    ClosureResult,
    LossSource,
)

from .uncertainty import (
    UncertaintyQuantifier,
    UncertaintyResult,
    SensitivityResult,
    ConfidenceInterval,
)

__all__ = [
    # Thermal Efficiency
    "ThermalEfficiencyCalculator",
    "EfficiencyResult",
    "LossBreakdown",
    # Exergy Analysis
    "ExergyCalculator",
    "ExergyResult",
    "DestructionResult",
    # Heat Balance
    "HeatBalanceCalculator",
    "HeatBalanceResult",
    "ClosureResult",
    "LossSource",
    # Uncertainty
    "UncertaintyQuantifier",
    "UncertaintyResult",
    "SensitivityResult",
    "ConfidenceInterval",
]

__version__ = "1.0.0"
__author__ = "GL-009_ThermalIQ"
