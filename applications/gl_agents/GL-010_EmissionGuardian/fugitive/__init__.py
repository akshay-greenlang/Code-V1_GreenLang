# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Fugitive Emissions Detection Module

This module provides ML-based fugitive emissions detection with
SHAP/LIME explainability for regulatory compliance.

Components:
    - Feature Engineering: Deterministic feature computation
    - Anomaly Detection: Isolation Forest + Statistical ensemble
    - Classification: Supervised leak classification with explainability
    - Human Review: Workflow for confirmation and feedback

Zero-Hallucination Principle:
    - ML used for scoring/classification only
    - All thresholds are deterministic and configurable
    - Human review required for all alerts
    - Complete provenance tracking

Author: GreenLang GL-010 EmissionsGuardian
"""

from .feature_engineering import (
    WindStability,
    EquipmentType,
    SensorReading,
    MeteorologicalData,
    EquipmentContext,
    FeatureVector,
    FeatureEngineeringConfig,
    FeatureEngineer,
)

from .anomaly_detector import (
    AnomalyType,
    AnomalySeverity,
    DetectorType,
    AnomalyDetectorConfig,
    AnomalyScore,
    AnomalyDetection,
    IsolationForestDetector,
    StatisticalDetector,
    AnomalyDetector,
)

from .classifier import (
    LeakClassification,
    LeakCategory,
    ClassifierConfig,
    SHAPExplanation,
    LIMEExplanation,
    ClassificationResult,
    FugitiveClassifier,
)

__all__ = [
    # Feature Engineering
    "WindStability",
    "EquipmentType",
    "SensorReading",
    "MeteorologicalData",
    "EquipmentContext",
    "FeatureVector",
    "FeatureEngineeringConfig",
    "FeatureEngineer",
    # Anomaly Detection
    "AnomalyType",
    "AnomalySeverity",
    "DetectorType",
    "AnomalyDetectorConfig",
    "AnomalyScore",
    "AnomalyDetection",
    "IsolationForestDetector",
    "StatisticalDetector",
    "AnomalyDetector",
    # Classification
    "LeakClassification",
    "LeakCategory",
    "ClassifierConfig",
    "SHAPExplanation",
    "LIMEExplanation",
    "ClassificationResult",
    "FugitiveClassifier",
]

__version__ = "1.0.0"
__author__ = "GreenLang GL-010 EmissionsGuardian"
