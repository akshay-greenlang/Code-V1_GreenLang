# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC Diagnostics Module

Comprehensive diagnostics suite for condenser performance analysis including
fouling prediction, air in-leakage detection, tube leak detection, and
root cause analysis.

Key Components:
- FoulingPredictor: Weibull reliability-based fouling prediction with CF trend analysis
- AirInleakageDetector: Vacuum response and DO correlation for air leak detection
- TubeLeakDetector: Chemistry-based tube leak detection with severity classification
- RootCauseAnalyzer: Multi-variate analysis to distinguish between performance issues

Zero-Hallucination Guarantee:
All diagnostics use deterministic algorithms, threshold-based classification,
and documented industry correlations. No ML inference for critical calculations.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

# Fouling Predictor exports
from .fouling_predictor import (
    FoulingPredictor,
    FoulingPredictorConfig,
    FoulingPrediction,
    CFDataPoint,
    CondenserProfile,
    RollingStatistics,
    TimeToThreshold,
    FoulingAlert,
    PredictionInterval,
    DegradationTrend,
    FoulingState,
    FoulingMechanism,
    AlertSeverity,
)

# Air In-leakage Detector exports
from .air_inleakage_detector import (
    AirInleakageDetector,
    AirLeakageDetectorConfig,
    AirLeakageDetection,
    VacuumDataPoint,
    CondenserVacuumProfile,
    VacuumResponseAnalysis,
    EjectorPerformance,
    LeakRateEstimate,
    LeakLocationAssessment,
    AirLeakageAlert,
    LeakSeverity,
    LeakLocation,
    VacuumResponsePattern,
    DetectionConfidence,
)

# Tube Leak Detector exports
from .tube_leak_detector import (
    TubeLeakDetector,
    TubeLeakDetectorConfig,
    TubeLeakDetection,
    ChemistryDataPoint,
    CondenserChemistryProfile,
    ChemistryTrendAnalysis,
    LeakSignature,
    TubeLeakAlert,
    TurbineRiskAssessment,
    LeakType,
    CoolingWaterType,
    ChemistryTrend,
)

# Root Cause Analyzer exports
from .root_cause_analyzer import (
    RootCauseAnalyzer,
    RootCauseAnalyzerConfig,
    DiagnosticReport,
    PerformanceIndicators,
    CondenserDiagnosticProfile,
    IndicatorAssessment,
    EvidenceItem,
    RootCauseCandidate,
    RecommendedAction,
    RootCause,
    DiagnosticConfidence,
    ImpactSeverity,
    IndicatorStatus,
)

__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",

    # Fouling Predictor
    "FoulingPredictor",
    "FoulingPredictorConfig",
    "FoulingPrediction",
    "CFDataPoint",
    "CondenserProfile",
    "RollingStatistics",
    "TimeToThreshold",
    "FoulingAlert",
    "PredictionInterval",
    "DegradationTrend",
    "FoulingState",
    "FoulingMechanism",

    # Air In-leakage Detector
    "AirInleakageDetector",
    "AirLeakageDetectorConfig",
    "AirLeakageDetection",
    "VacuumDataPoint",
    "CondenserVacuumProfile",
    "VacuumResponseAnalysis",
    "EjectorPerformance",
    "LeakRateEstimate",
    "LeakLocationAssessment",
    "AirLeakageAlert",
    "LeakSeverity",
    "LeakLocation",
    "VacuumResponsePattern",
    "DetectionConfidence",

    # Tube Leak Detector
    "TubeLeakDetector",
    "TubeLeakDetectorConfig",
    "TubeLeakDetection",
    "ChemistryDataPoint",
    "CondenserChemistryProfile",
    "ChemistryTrendAnalysis",
    "LeakSignature",
    "TubeLeakAlert",
    "TurbineRiskAssessment",
    "LeakType",
    "CoolingWaterType",
    "ChemistryTrend",

    # Root Cause Analyzer
    "RootCauseAnalyzer",
    "RootCauseAnalyzerConfig",
    "DiagnosticReport",
    "PerformanceIndicators",
    "CondenserDiagnosticProfile",
    "IndicatorAssessment",
    "EvidenceItem",
    "RootCauseCandidate",
    "RecommendedAction",
    "RootCause",
    "DiagnosticConfidence",
    "ImpactSeverity",
    "IndicatorStatus",

    # Shared Types
    "AlertSeverity",
]
