"""
GL-006 HEATRECLAIM - Explainability Module

Provides two-layer explainability:
1. Deterministic engineering rationale (constraint satisfaction, pinch rules)
2. Statistical feature attribution (SHAP/LIME) on surrogate models

This module ensures all optimization decisions are traceable
and understandable by domain experts.
"""

from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from .causal_analyzer import CausalAnalyzer
from .engineering_rationale import EngineeringRationaleGenerator
from .report_generator import ExplainabilityReportGenerator

__all__ = [
    "SHAPExplainer",
    "LIMEExplainer",
    "CausalAnalyzer",
    "EngineeringRationaleGenerator",
    "ExplainabilityReportGenerator",
]
