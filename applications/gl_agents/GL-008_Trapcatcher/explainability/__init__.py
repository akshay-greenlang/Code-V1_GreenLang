# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER Explainability Module

SHAP and LIME compatible explainability for steam trap diagnostics.

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

from .lime_trap_explainer import (
    LimeTrapExplainer,
    LimeConfig,
    LimeExplanation,
    LimeFeatureWeight,
    LocalFidelity,
    KernelType,
    DiscretizationType,
    SamplingStrategy,
    TRAP_FEATURES,
)

__all__ = [
    # Diagnostic explainer
    "DiagnosticExplainer",
    "ExplainerConfig",
    "ExplanationResult",
    "FeatureContribution",
    "CounterfactualExplanation",
    # LIME explainer
    "LimeTrapExplainer",
    "LimeConfig",
    "LimeExplanation",
    "LimeFeatureWeight",
    "LocalFidelity",
    "KernelType",
    "DiscretizationType",
    "SamplingStrategy",
    "TRAP_FEATURES",
]
