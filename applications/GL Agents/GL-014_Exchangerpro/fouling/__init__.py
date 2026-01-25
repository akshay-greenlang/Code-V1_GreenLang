# -*- coding: utf-8 -*-
"""
Fouling Prediction ML Module for GL-014 Exchangerpro Agent.

Provides ML-based fouling prediction and monitoring:
- Feature engineering for heat exchanger fouling patterns
- UA degradation and fouling resistance (Rf) prediction
- Model versioning and registry management
- Uncertainty quantification with confidence intervals
- Feature and prediction drift detection

CRITICAL: All predictions MUST include confidence intervals.
Predictions are NEVER presented as certainties.

Author: GreenLang AI Team
Version: 1.0.0
"""

from .feature_engineering import (
    FoulingFeatureEngine,
    FoulingFeatureConfig,
    FoulingFeatures,
    RollingWindowConfig,
    LaggedFeatureConfig,
)
from .fouling_predictor import (
    FoulingPredictor,
    FoulingPredictorConfig,
    FoulingPrediction,
    UADegradationForecast,
    ConstraintRiskPrediction,
    RemainingUsefulPerformance,
    PredictionHorizon,
)
from .model_registry import (
    FoulingModelRegistry,
    ModelRegistryConfig,
    ModelArtifact,
    ModelVersion,
    FeatureSchema,
    TrainingSnapshot,
    PerformanceMetrics,
)
from .uncertainty_quantifier import (
    UncertaintyQuantifier,
    UncertaintyConfig,
    UncertaintyResult,
    PredictionInterval,
    ConformalPredictor,
    QuantileRegressor,
    EnsembleUncertainty,
)
from .drift_detector import (
    DriftDetector,
    DriftDetectorConfig,
    DriftResult,
    FeatureDrift,
    PredictionDrift,
    RetrainingTrigger,
    DriftSeverity,
)

__all__ = [
    # Feature engineering
    "FoulingFeatureEngine",
    "FoulingFeatureConfig",
    "FoulingFeatures",
    "RollingWindowConfig",
    "LaggedFeatureConfig",
    # Fouling predictor
    "FoulingPredictor",
    "FoulingPredictorConfig",
    "FoulingPrediction",
    "UADegradationForecast",
    "ConstraintRiskPrediction",
    "RemainingUsefulPerformance",
    "PredictionHorizon",
    # Model registry
    "FoulingModelRegistry",
    "ModelRegistryConfig",
    "ModelArtifact",
    "ModelVersion",
    "FeatureSchema",
    "TrainingSnapshot",
    "PerformanceMetrics",
    # Uncertainty quantification
    "UncertaintyQuantifier",
    "UncertaintyConfig",
    "UncertaintyResult",
    "PredictionInterval",
    "ConformalPredictor",
    "QuantileRegressor",
    "EnsembleUncertainty",
    # Drift detection
    "DriftDetector",
    "DriftDetectorConfig",
    "DriftResult",
    "FeatureDrift",
    "PredictionDrift",
    "RetrainingTrigger",
    "DriftSeverity",
]

__version__ = "1.0.0"
