# -*- coding: utf-8 -*-
"""
Analytics Engine for GL-013 PredictiveMaintenance Agent.

Provides predictive analytics including:
- Weibull survival analysis
- Remaining Useful Life (RUL) estimation
- Anomaly detection
- Health index calculation
- Degradation modeling

All calculations are deterministic with zero-hallucination guarantees.
"""

from .weibull_survival import (
    WeibullDistribution,
    WeibullSurvivalAnalyzer,
    SurvivalCurve,
    HazardFunction,
    CensoringType,
)
from .rul_estimator import (
    RULEstimator,
    RULEstimatorConfig,
    RULResult,
    WeibullRULEstimator,
    EnsembleRULEstimator,
)
from .anomaly_detector import (
    AnomalyDetector,
    AnomalyDetectorConfig,
    AnomalyResult,
    IsolationForestDetector,
    AutoencoderDetector,
    StatisticalDetector,
)
from .health_calculator import (
    HealthCalculator,
    HealthCalculatorConfig,
    HealthResult,
    DegradationModel,
)

__all__ = [
    # Weibull survival
    "WeibullDistribution",
    "WeibullSurvivalAnalyzer",
    "SurvivalCurve",
    "HazardFunction",
    "CensoringType",
    # RUL estimation
    "RULEstimator",
    "RULEstimatorConfig",
    "RULResult",
    "WeibullRULEstimator",
    "EnsembleRULEstimator",
    # Anomaly detection
    "AnomalyDetector",
    "AnomalyDetectorConfig",
    "AnomalyResult",
    "IsolationForestDetector",
    "AutoencoderDetector",
    "StatisticalDetector",
    # Health calculation
    "HealthCalculator",
    "HealthCalculatorConfig",
    "HealthResult",
    "DegradationModel",
]
