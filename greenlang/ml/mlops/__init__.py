# -*- coding: utf-8 -*-
"""
GreenLang MLOps Pipeline

Provides production-ready MLOps capabilities including model registry,
experiment tracking, A/B testing, and drift detection.

Classes:
    ModelRegistry: MLflow-based model versioning and deployment
    ExperimentTracker: Experiment tracking and logging
    ABTesting: A/B testing framework for model comparison
    DriftDetector: Data and concept drift detection
    AutoRetrainer: Automatic retraining pipeline

Example:
    >>> from greenlang.ml.mlops import ModelRegistry, DriftDetector
    >>> registry = ModelRegistry()
    >>> registry.register_model(model, "emission_predictor", version="1.0.0")
    >>> detector = DriftDetector()
    >>> if detector.detect_drift(new_data):
    ...     trigger_retraining()
"""

from greenlang.ml.mlops.model_registry import ModelRegistry
from greenlang.ml.mlops.experiment_tracker import ExperimentTracker
from greenlang.ml.mlops.ab_testing import ABTesting
from greenlang.ml.mlops.drift_detector import DriftDetector
from greenlang.ml.mlops.auto_retrainer import AutoRetrainer

__all__ = [
    "ModelRegistry",
    "ExperimentTracker",
    "ABTesting",
    "DriftDetector",
    "AutoRetrainer",
]
