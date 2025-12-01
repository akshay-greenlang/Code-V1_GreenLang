# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER Calculators Package

Provides deterministic calculators for steam trap inspection and diagnosis:
- AcousticSignatureAnalyzer: Ultrasonic leak detection (20-100 kHz)
- TemperatureDifferentialAnalyzer: Inlet/outlet thermal analysis
- SteamLossCostCalculator: Energy waste and ROI quantification

All calculators follow zero-hallucination principles with deterministic
algorithms only. No LLM or AI inference in any calculation path.

Author: GreenLang Industrial Optimization Team
Date: December 2025
Version: 1.0.0
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
]

__version__ = "1.0.0"
