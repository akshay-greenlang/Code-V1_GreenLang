# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER Explainability Module

SHAP-compatible explainability for steam trap diagnostics.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from .diagnostic_explainer import (
    DiagnosticExplainer,
    ExplainerConfig,
    ExplanationResult,
    FeatureContribution,
    CounterfactualExplanation,
)

__all__ = [
    "DiagnosticExplainer",
    "ExplainerConfig",
    "ExplanationResult",
    "FeatureContribution",
    "CounterfactualExplanation",
]
