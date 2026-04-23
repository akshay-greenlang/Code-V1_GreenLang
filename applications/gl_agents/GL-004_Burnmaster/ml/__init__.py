"""
GL-004 BURNMASTER Machine Learning Module

Production ML inference pipeline for combustion optimization.
Provides predictive maintenance, anomaly detection, and efficiency prediction.

Key Features:
    - MLPipelineManager for model lifecycle management
    - Predictive maintenance for equipment failure prediction
    - Real-time anomaly detection using Isolation Forest
    - Combustion efficiency prediction
    - Physics-based fallback when ML unavailable
    - Zero-hallucination approach (ML is advisory only)

CRITICAL: ML predictions are ADVISORY ONLY.
Control decisions use deterministic physics-based calculations.

Author: GreenLang ML Engineering Team
Version: 1.0.0
"""

from .pipeline import (
    MLPipelineManager,
    MLPipelineConfig,
    ModelInfo,
    InferenceRequest,
    InferenceResponse,
    PredictionType,
    ModelStatus,
)

__all__ = [
    "MLPipelineManager",
    "MLPipelineConfig",
    "ModelInfo",
    "InferenceRequest",
    "InferenceResponse",
    "PredictionType",
    "ModelStatus",
]

__version__ = "1.0.0"
