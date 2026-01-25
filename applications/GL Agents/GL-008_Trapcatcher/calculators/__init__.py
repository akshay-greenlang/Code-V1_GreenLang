# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER Calculators Package

This package provides deterministic calculators for steam trap monitoring,
diagnostics, and analysis following GreenLang's zero-hallucination principles.

Modules:
    steam_trap_energy_loss_calculator: Energy loss and ROI calculations
    trap_population_analyzer: Fleet-wide analysis and optimization
    acoustic_diagnostic_calculator: Ultrasonic signal analysis
    temperature_differential_calculator: Thermal differential analysis

Zero-Hallucination Guarantee:
All calculations use deterministic engineering formulas from published standards.
No LLM or AI inference in any calculation path.
Same inputs always produce identical outputs.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

# Steam Trap Energy Loss Calculator
from .steam_trap_energy_loss_calculator import (
    SteamTrapEnergyLossCalculator,
    EnergyLossConfig,
    FailureMode,
    TrapType,
    SeverityLevel,
    FuelType,
    TrapSpecifications,
    SteamConditions,
    SteamLossResult,
    EnergyLossMetrics,
    CarbonEmissionResult,
    ROIResult,
    EnergyLossAnalysisResult,
)

# Trap Population Analyzer
from .trap_population_analyzer import (
    TrapPopulationAnalyzer,
    PopulationAnalysisConfig,
    TrapStatus,
    PriorityLevel,
    TrendDirection,
    SurveyMethod,
    TrapRecord,
    FleetHealthMetrics,
    FailureRateTrend,
    PriorityRanking,
    SurveyFrequencyRecommendation,
    SparePartsRecommendation,
    TotalCostOfOwnership,
    PopulationAnalysisResult,
    create_trap_record,
)

# Acoustic Diagnostic Calculator
from .acoustic_diagnostic_calculator import (
    AcousticDiagnosticCalculator,
    AcousticConfig,
    AcousticSignature,
    SignatureType,
    FrequencyBand,
    NoiseSource,
    AcousticReading,
    SignatureAnalysisResult,
    LeakDetectionResult,
    SignalQualityMetrics,
    AcousticDiagnosticResult,
)

# Temperature Differential Calculator
from .temperature_differential_calculator import (
    TemperatureDifferentialCalculator,
    TemperatureConfig,
    TrapCondition,
    SeasonalFactor,
    TemperatureReading,
    DifferentialAnalysisResult,
    SubcoolingAnalysis,
    FailureDetectionResult,
    SaturationComparison,
    TemperatureDiagnosticResult,
)

# Shared Provenance Classes
from .steam_trap_energy_loss_calculator import (
    ProvenanceTracker,
    ProvenanceStep,
)

__version__ = "1.0.0"
__author__ = "GL-BackendDeveloper"

__all__ = [
    "__version__",
    "__author__",
    "SteamTrapEnergyLossCalculator",
    "EnergyLossConfig",
    "FailureMode",
    "TrapType",
    "SeverityLevel",
    "FuelType",
    "TrapSpecifications",
    "SteamConditions",
    "SteamLossResult",
    "EnergyLossMetrics",
    "CarbonEmissionResult",
    "ROIResult",
    "EnergyLossAnalysisResult",
    "TrapPopulationAnalyzer",
    "PopulationAnalysisConfig",
    "TrapStatus",
    "PriorityLevel",
    "TrendDirection",
    "SurveyMethod",
    "TrapRecord",
    "FleetHealthMetrics",
    "FailureRateTrend",
    "PriorityRanking",
    "SurveyFrequencyRecommendation",
    "SparePartsRecommendation",
    "TotalCostOfOwnership",
    "PopulationAnalysisResult",
    "create_trap_record",
    "AcousticDiagnosticCalculator",
    "AcousticConfig",
    "AcousticSignature",
    "SignatureType",
    "FrequencyBand",
    "NoiseSource",
    "AcousticReading",
    "SignatureAnalysisResult",
    "LeakDetectionResult",
    "SignalQualityMetrics",
    "AcousticDiagnosticResult",
    "TemperatureDifferentialCalculator",
    "TemperatureConfig",
    "TrapCondition",
    "SeasonalFactor",
    "TemperatureReading",
    "DifferentialAnalysisResult",
    "SubcoolingAnalysis",
    "FailureDetectionResult",
    "SaturationComparison",
    "TemperatureDiagnosticResult",
    "ProvenanceTracker",
    "ProvenanceStep",
]
