# -*- coding: utf-8 -*-
"""
GL-004 Burnmaster - ML Models Module

This module provides AI/ML models for combustion prediction and optimization.
All models implement zero-hallucination principles for numeric calculations
and include comprehensive uncertainty quantification.

Key Components:
    - FlameStabilityPredictor: Predicts flame instability risk and operating regimes
    - EmissionsPredictor: Predicts NOx and CO emissions with uncertainty
    - SoftSensorModels: Infers unmeasured process variables
    - PerformanceSurrogateModel: Fast efficiency predictions for optimization
    - ModelRegistry: MLOps model lifecycle management
    - ModelDriftDetector: Detects data and concept drift
    - CombustionFeatureEngineer: Feature extraction for combustion models

Design Principles:
    - ZERO HALLUCINATION: All numeric predictions use deterministic models
    - Uncertainty Quantification: All predictions include confidence intervals
    - Provenance Tracking: SHA-256 hashes for complete audit trails
    - Interpretability: SHAP/LIME hooks for all models
    - MLOps Ready: Serializable models with versioning

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from .stability_predictor import (
    FlameStabilityPredictor,
    StabilityFeatures,
    InstabilityPrediction,
    BurnerState,
    OperatingRegime,
    Anomaly,
)

from .emissions_predictor import (
    EmissionsPredictor,
    EmissionsFeatures,
    NOxPrediction,
    COPrediction,
    EmissionPrediction,
    CalibrationResult,
)

from .soft_sensors import (
    SoftSensorModels,
    FuelQualityEstimate,
    ExcessAirEstimate,
    HeatDutyEstimate,
    BiasDetection,
)

from .performance_surrogate import (
    PerformanceSurrogateModel,
    EfficiencyPrediction,
    FuelRatePrediction,
    OptimalSetpoints,
    UpdateResult,
)

from .model_registry import (
    ModelRegistry,
    BaseModel,
    ModelMetadata,
)

from .drift_detector import (
    ModelDriftDetector,
    DriftResult,
    ConceptDriftResult,
    RetrainingRecommendation,
)

from .feature_engineering import (
    CombustionFeatureEngineer,
    FeatureMatrix,
)


__all__ = [
    # Stability Predictor
    "FlameStabilityPredictor",
    "StabilityFeatures",
    "InstabilityPrediction",
    "BurnerState",
    "OperatingRegime",
    "Anomaly",
    # Emissions Predictor
    "EmissionsPredictor",
    "EmissionsFeatures",
    "NOxPrediction",
    "COPrediction",
    "EmissionPrediction",
    "CalibrationResult",
    # Soft Sensors
    "SoftSensorModels",
    "FuelQualityEstimate",
    "ExcessAirEstimate",
    "HeatDutyEstimate",
    "BiasDetection",
    # Performance Surrogate
    "PerformanceSurrogateModel",
    "EfficiencyPrediction",
    "FuelRatePrediction",
    "OptimalSetpoints",
    "UpdateResult",
    # Model Registry
    "ModelRegistry",
    "BaseModel",
    "ModelMetadata",
    # Drift Detector
    "ModelDriftDetector",
    "DriftResult",
    "ConceptDriftResult",
    "RetrainingRecommendation",
    # Feature Engineering
    "CombustionFeatureEngineer",
    "FeatureMatrix",
]

__version__ = "1.0.0"
__author__ = "GreenLang Process Heat Team"

