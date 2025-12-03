# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER Calculators Package

Provides deterministic calculators for steam trap inspection and diagnosis:
- AcousticSignatureAnalyzer: Ultrasonic leak detection (20-100 kHz)
- TemperatureDifferentialAnalyzer: Inlet/outlet thermal analysis
- SteamLossCostCalculator: Energy waste and ROI quantification
- SteamTrapEnergyLossCalculator: Advanced energy loss with ASME/ASTM compliance
- TrapPopulationAnalyzer: Fleet-wide statistics and optimization

All calculators follow zero-hallucination principles with deterministic
algorithms only. No LLM or AI inference in any calculation path.

Author: GreenLang Industrial Optimization Team
Date: December 2025
Version: 1.1.0
"""

from .acoustic_analyzer import (
    AcousticSignatureAnalyzer,
    AcousticAnalysisConfig,
    AcousticAnalysisResult,
    FrequencyBand,
    SignalPattern,
    AcousticDiagnosis,
)

from .thermal_analyzer import (
    TemperatureDifferentialAnalyzer,
    ThermalAnalysisConfig,
    ThermalAnalysisResult,
    ThermalPattern,
    ThermalDiagnosis,
)

from .cost_calculator import (
    SteamLossCostCalculator,
    CostCalculatorConfig,
    CostAnalysisResult,
    ROIAnalysis,
    EnergyMetrics,
)

from .steam_trap_energy_loss_calculator import (
    SteamTrapEnergyLossCalculator,
    EnergyLossConfig,
    FailureMode,
    TrapType as EnergyTrapType,
    SeverityLevel,
    FuelType,
    TrapSpecifications,
    SteamConditions,
    SteamLossResult,
    EnergyLossMetrics,
    CarbonEmissionResult,
    ROIResult,
    EnergyLossAnalysisResult,
    ProvenanceTracker as EnergyProvenanceTracker,
)

from .trap_population_analyzer import (
    TrapPopulationAnalyzer,
    PopulationAnalysisConfig,
    TrapStatus,
    TrapType as PopulationTrapType,
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
    ProvenanceTracker as PopulationProvenanceTracker,
    create_trap_record,
)

__all__ = [
    # Acoustic analyzer
    "AcousticSignatureAnalyzer",
    "AcousticAnalysisConfig",
    "AcousticAnalysisResult",
    "FrequencyBand",
    "SignalPattern",
    "AcousticDiagnosis",
    # Thermal analyzer
    "TemperatureDifferentialAnalyzer",
    "ThermalAnalysisConfig",
    "ThermalAnalysisResult",
    "ThermalPattern",
    "ThermalDiagnosis",
    # Cost calculator
    "SteamLossCostCalculator",
    "CostCalculatorConfig",
    "CostAnalysisResult",
    "ROIAnalysis",
    "EnergyMetrics",
    # Energy loss calculator (advanced)
    "SteamTrapEnergyLossCalculator",
    "EnergyLossConfig",
    "FailureMode",
    "EnergyTrapType",
    "SeverityLevel",
    "FuelType",
    "TrapSpecifications",
    "SteamConditions",
    "SteamLossResult",
    "EnergyLossMetrics",
    "CarbonEmissionResult",
    "ROIResult",
    "EnergyLossAnalysisResult",
    "EnergyProvenanceTracker",
    # Population analyzer
    "TrapPopulationAnalyzer",
    "PopulationAnalysisConfig",
    "TrapStatus",
    "PopulationTrapType",
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
    "PopulationProvenanceTracker",
    "create_trap_record",
]

__version__ = "1.1.0"
