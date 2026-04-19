"""
GL-015 INSULSCAN Calculators Module

Zero-Hallucination Calculation Engine for Insulation Scanning & Thermal Assessment.

This module provides deterministic, bit-perfect, reproducible calculations
for thermal insulation analysis following ASTM C680 and related standards.

All calculations include SHA-256 provenance tracking for audit compliance.

Exports:
    - HeatLossCalculator: ASTM C680 compliant heat loss calculations
    - ThermalResistanceCalculator: R-value and thermal conductivity analysis
    - InsulationROICalculator: Financial analysis for insulation investments
    - InsulationConditionScorer: Deterministic condition assessment scoring
    - SurfaceEmissivityDatabase: Material emissivity values and corrections

Author: GL-CalculatorEngineer
Version: 1.0.0
Standard Compliance: ASTM C680, ISO 12241
"""

from .heat_loss import HeatLossCalculator, HeatLossResult, SurfaceType
from .thermal_resistance import (
    ThermalResistanceCalculator,
    ThermalResistanceResult,
    InsulationType,
    InsulationLayer,
    GeometryType,
)
from .roi_calculator import (
    InsulationROICalculator,
    ROIResult,
    LifecycleCostResult,
    EnergyType,
)
from .condition_scorer import (
    InsulationConditionScorer,
    ConditionScoreResult,
    SeverityLevel,
    VisualDefectType,
    ThermalAnomalyType,
    VisualInspectionData,
    ThermalImageData,
)
from .surface_emissivity import (
    SurfaceEmissivityDatabase,
    EmissivityResult,
    MaterialCategory,
    SurfaceCondition,
)

__all__ = [
    # Heat Loss Calculator
    "HeatLossCalculator",
    "HeatLossResult",
    "SurfaceType",
    # Thermal Resistance Calculator
    "ThermalResistanceCalculator",
    "ThermalResistanceResult",
    "InsulationType",
    "InsulationLayer",
    "GeometryType",
    # ROI Calculator
    "InsulationROICalculator",
    "ROIResult",
    "LifecycleCostResult",
    "EnergyType",
    # Condition Scorer
    "InsulationConditionScorer",
    "ConditionScoreResult",
    "SeverityLevel",
    "VisualDefectType",
    "ThermalAnomalyType",
    "VisualInspectionData",
    "ThermalImageData",
    # Surface Emissivity
    "SurfaceEmissivityDatabase",
    "EmissivityResult",
    "MaterialCategory",
    "SurfaceCondition",
]

__version__ = "1.0.0"
__standard__ = "ASTM C680"
