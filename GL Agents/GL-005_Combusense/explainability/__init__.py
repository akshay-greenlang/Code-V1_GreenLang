# -*- coding: utf-8 -*-
"""
GL-005 COMBUSENSE Explainability Module

This package provides explainability for combustion control decisions:
- SHAP (SHapley Additive exPlanations) for PID controller decisions
- Feature importance analysis
- Counterfactual explanations

Author: GL-BackendDeveloper
"""

from .shap_explainer import (
    SHAPExplainer,
    SHAPExplanation,
    FeatureContribution,
    PIDExplainabilityConfig,
    ControlDecisionExplanation
)

__all__ = [
    "SHAPExplainer",
    "SHAPExplanation",
    "FeatureContribution",
    "PIDExplainabilityConfig",
    "ControlDecisionExplanation"
]
