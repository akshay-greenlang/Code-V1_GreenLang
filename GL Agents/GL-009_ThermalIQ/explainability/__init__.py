"""
ThermalIQ Explainability Module

Provides interpretable explanations for thermal system predictions
and recommendations using SHAP, LIME, and engineering rationale.
"""

from .shap_explainer import ThermalSHAPExplainer, SHAPExplanation
from .lime_explainer import ThermalLIMEExplainer, LIMEExplanation
from .engineering_rationale import EngineeringRationaleGenerator, Citation
from .report_generator import (
    ExplainabilityReportGenerator,
    ExplainabilityReport,
    Recommendation
)

__all__ = [
    # SHAP Explainer
    "ThermalSHAPExplainer",
    "SHAPExplanation",
    # LIME Explainer
    "ThermalLIMEExplainer",
    "LIMEExplanation",
    # Engineering Rationale
    "EngineeringRationaleGenerator",
    "Citation",
    # Report Generator
    "ExplainabilityReportGenerator",
    "ExplainabilityReport",
    "Recommendation",
]

__version__ = "1.0.0"
