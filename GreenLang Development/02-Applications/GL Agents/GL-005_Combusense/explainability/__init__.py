# -*- coding: utf-8 -*-
"""
GL-005 COMBUSENSE Explainability Module

This package provides explainability for combustion control decisions:
- SHAP (SHapley Additive exPlanations) for PID controller decisions
- LIME (Local Interpretable Model-agnostic Explanations) for combustion optimization
- Feature importance analysis
- Counterfactual explanations
- Combustion-specific interpretation

Author: GL-BackendDeveloper
"""

from .shap_explainer import (
    SHAPExplainer,
    SHAPExplanation,
    FeatureContribution as SHAPFeatureContribution,
    PIDExplainabilityConfig,
    ControlDecisionExplanation
)

from .lime_explainer import (
    CombustionLIMEExplainer,
    LIMEExplainerConfig,
    LIMEExplanation,
    FeatureContribution,
    CounterfactualExplanation,
    ImpactDirection,
    ConfidenceLevel,
    ExplanationType,
    CombustionParameter,
    FeatureStatistics,
    PredictorProtocol,
    create_default_explainer,
    create_high_fidelity_explainer,
    create_fast_explainer,
)

__all__ = [
    # SHAP Explainer
    "SHAPExplainer",
    "SHAPExplanation",
    "SHAPFeatureContribution",
    "PIDExplainabilityConfig",
    "ControlDecisionExplanation",
    # LIME Explainer
    "CombustionLIMEExplainer",
    "LIMEExplainerConfig",
    "LIMEExplanation",
    "FeatureContribution",
    "CounterfactualExplanation",
    "ImpactDirection",
    "ConfidenceLevel",
    "ExplanationType",
    "CombustionParameter",
    "FeatureStatistics",
    "PredictorProtocol",
    "create_default_explainer",
    "create_high_fidelity_explainer",
    "create_fast_explainer",
]
