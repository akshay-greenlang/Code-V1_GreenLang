# -*- coding: utf-8 -*-
"""
GreenLang ML Module

This module provides production-ready machine learning capabilities for GreenLang agents,
including explainability, self-learning, MLOps, uncertainty quantification, and robustness.

Modules:
    explainability: SHAP, LIME, and causal inference for model interpretability
    self_learning: Online learning, continual learning, and meta-learning
    mlops: Model registry, experiment tracking, A/B testing, and drift detection
    uncertainty: Ensemble prediction, conformal prediction, and calibration
    robustness: Adversarial testing, distribution shift detection, and fail-safes

Example:
    >>> from greenlang.ml.explainability import SHAPExplainer
    >>> from greenlang.ml.uncertainty import EnsemblePredictor
    >>> from greenlang.ml.mlops import ModelRegistry
"""

__version__ = "1.0.0"

# Lazy imports to avoid circular dependencies and improve startup time
__all__ = [
    "explainability",
    "self_learning",
    "mlops",
    "uncertainty",
    "robustness",
]
