"""
GL-007 FurnacePulse - Calculators Module

This module provides deterministic calculators for furnace performance
monitoring and predictive maintenance. All calculators inherit from
DeterministicCalculator and include SHA-256 provenance tracking.

Calculators:
    - EfficiencyCalculator: Thermal efficiency and fuel consumption
    - HotspotDetector: TMT monitoring and spatial clustering
    - RULPredictor: Remaining Useful Life prediction
    - DraftAnalyzer: Draft pressure stability analysis

Example:
    >>> from calculators import EfficiencyCalculator, EfficiencyInputs
    >>> calc = EfficiencyCalculator(agent_id="GL-007")
    >>> result = calc.calculate(EfficiencyInputs(
    ...     fuel_mass_flow_kg_s=0.5,
    ...     fuel_lhv_kj_kg=42000,
    ...     useful_heat_output_kw=8000
    ... ))
    >>> print(f"Thermal efficiency: {result.result.thermal_efficiency_pct:.2f}%")
"""

from .efficiency_calculator import (
    EfficiencyCalculator,
    EfficiencyInputs,
    EfficiencyOutputs,
    ExcessAirInputs,
    ExcessAirOutputs,
    StackLossInputs,
    StackLossOutputs,
)
from .hotspot_detector import (
    HotspotDetector,
    TMTReadings,
    HotspotAnalysis,
    HotspotAlert,
    AlertLevel,
)
from .rul_predictor import (
    RULPredictor,
    RULInputs,
    RULOutputs,
    ComponentType,
    MaintenanceRecord,
    WeibullParameters,
)
from .draft_analyzer import (
    DraftAnalyzer,
    DraftInputs,
    DraftOutputs,
    DraftStabilityMetrics,
    DamperPerformance,
)


__all__ = [
    # Efficiency Calculator
    "EfficiencyCalculator",
    "EfficiencyInputs",
    "EfficiencyOutputs",
    "ExcessAirInputs",
    "ExcessAirOutputs",
    "StackLossInputs",
    "StackLossOutputs",
    # Hotspot Detector
    "HotspotDetector",
    "TMTReadings",
    "HotspotAnalysis",
    "HotspotAlert",
    "AlertLevel",
    # RUL Predictor
    "RULPredictor",
    "RULInputs",
    "RULOutputs",
    "ComponentType",
    "MaintenanceRecord",
    "WeibullParameters",
    # Draft Analyzer
    "DraftAnalyzer",
    "DraftInputs",
    "DraftOutputs",
    "DraftStabilityMetrics",
    "DamperPerformance",
]

__version__ = "1.0.0"
__agent__ = "GL-007_FurnacePulse"
