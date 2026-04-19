"""
GL-004 BURNMASTER ML Models Package

This package contains ML models for combustion optimization:
    - PredictiveMaintenanceModel: Equipment failure prediction
    - CombustionAnomalyDetector: Real-time anomaly detection
    - EfficiencyPredictor: Combustion efficiency prediction

All models follow the zero-hallucination principle:
    - ML is ADVISORY ONLY
    - Control decisions use physics-based calculations
    - Predictions include uncertainty quantification
    - Complete provenance tracking for audit

Author: GreenLang ML Engineering Team
Version: 1.0.0
"""

from .predictive_maintenance import (
    PredictiveMaintenanceModel,
    MaintenancePrediction,
    FailureMode,
    MaintenanceFeatures,
)

from .anomaly_detector import (
    CombustionAnomalyDetector,
    AnomalyResult,
    AnomalyAlert,
    AnomalySeverity,
    CombustionFeatures,
)

from .efficiency_predictor import (
    EfficiencyPredictor,
    EfficiencyPrediction,
    EfficiencyFeatures,
    EfficiencyFactors,
)

__all__ = [
    # Predictive Maintenance
    "PredictiveMaintenanceModel",
    "MaintenancePrediction",
    "FailureMode",
    "MaintenanceFeatures",
    # Anomaly Detection
    "CombustionAnomalyDetector",
    "AnomalyResult",
    "AnomalyAlert",
    "AnomalySeverity",
    "CombustionFeatures",
    # Efficiency Prediction
    "EfficiencyPredictor",
    "EfficiencyPrediction",
    "EfficiencyFeatures",
    "EfficiencyFactors",
]

__version__ = "1.0.0"
