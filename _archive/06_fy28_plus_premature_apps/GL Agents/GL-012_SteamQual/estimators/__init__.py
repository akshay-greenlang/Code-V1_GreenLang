# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL Estimators Module

This module provides soft sensors and ML components for steam quality estimation:
- Hybrid soft sensor for dryness fraction with physics + ML layers
- Extended Kalman Filter for state estimation
- Carryover event detection and early warning
- Separator/trap health analytics

All estimators are physics-guided with uncertainty quantification and safety gating.

Zero-Hallucination Guarantee:
- Physics constraints bound all ML predictions
- Uncertainty propagated through all calculations
- Graceful degradation when data quality is poor
- Complete audit trail with provenance hashing

Author: GL-BackendDeveloper
Date: December 2024
Version: 1.0.0
"""

from .kalman_filter import (
    ExtendedKalmanFilter,
    EKFConfig,
    EKFState,
    StateVector,
    MeasurementVector,
    ProcessModel,
    MeasurementModel,
    KalmanFilterResult,
)

from .soft_sensor import (
    SteamDrynessSoftSensor,
    SoftSensorConfig,
    SoftSensorResult,
    PhysicsLayerResult,
    DataDrivenCorrectionResult,
    SoftSensorInput,
    ConfidenceInterval,
)

from .carryover_detector import (
    CarryoverDetector,
    CarryoverDetectorConfig,
    CarryoverEvent,
    CarryoverWarning,
    CarryoverFeatures,
    DetectionResult,
    CarryoverRiskLevel,
)

from .separator_health_estimator import (
    SeparatorHealthEstimator,
    SeparatorHealthConfig,
    SeparatorHealthState,
    HealthAssessment,
    AnomalyEvent,
    EfficiencyTrend,
    FailureProbability,
    MaintenanceRecommendation,
)

__all__ = [
    # Kalman Filter
    "ExtendedKalmanFilter",
    "EKFConfig",
    "EKFState",
    "StateVector",
    "MeasurementVector",
    "ProcessModel",
    "MeasurementModel",
    "KalmanFilterResult",
    # Soft Sensor
    "SteamDrynessSoftSensor",
    "SoftSensorConfig",
    "SoftSensorResult",
    "PhysicsLayerResult",
    "DataDrivenCorrectionResult",
    "SoftSensorInput",
    "ConfidenceInterval",
    # Carryover Detector
    "CarryoverDetector",
    "CarryoverDetectorConfig",
    "CarryoverEvent",
    "CarryoverWarning",
    "CarryoverFeatures",
    "DetectionResult",
    "CarryoverRiskLevel",
    # Separator Health
    "SeparatorHealthEstimator",
    "SeparatorHealthConfig",
    "SeparatorHealthState",
    "HealthAssessment",
    "AnomalyEvent",
    "EfficiencyTrend",
    "FailureProbability",
    "MaintenanceRecommendation",
]

__version__ = "1.0.0"
__author__ = "GL-BackendDeveloper"
