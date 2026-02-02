"""
ThermalIQ Explainability Module

Provides interpretable explanations for thermal system predictions
and recommendations using SHAP, LIME, and engineering rationale.

The SHAP-Driven Recommendation Engine bridges ML explainability with
thermodynamic engineering principles for data-driven, auditable recommendations.
"""

from .shap_explainer import (
    ThermalSHAPExplainer,
    SHAPExplanation,
    FeatureContribution,
    ExplanationType,
    explain_thermal_prediction
)
from .lime_explainer import ThermalLIMEExplainer, LIMEExplanation
from .engineering_rationale import (
    EngineeringRationaleGenerator,
    EngineeringRationale,
    RationaleSection,
    RationaleCategory,
    Citation
)
from .shap_driven_recommendations import (
    SHAPDrivenRecommendationEngine,
    SHAPDrivenRationale,
    SHAPDrivenRecommendation,
    RecommendationPriority,
    RecommendationType,
    analyze_thermal_prediction
)
from .report_generator import (
    ExplainabilityReportGenerator,
    ExplainabilityReport,
    Recommendation
)

__all__ = [
    # SHAP Explainer
    "ThermalSHAPExplainer",
    "SHAPExplanation",
    "FeatureContribution",
    "ExplanationType",
    "explain_thermal_prediction",
    # LIME Explainer
    "ThermalLIMEExplainer",
    "LIMEExplanation",
    # Engineering Rationale
    "EngineeringRationaleGenerator",
    "EngineeringRationale",
    "RationaleSection",
    "RationaleCategory",
    "Citation",
    # SHAP-Driven Recommendations (NEW)
    "SHAPDrivenRecommendationEngine",
    "SHAPDrivenRationale",
    "SHAPDrivenRecommendation",
    "RecommendationPriority",
    "RecommendationType",
    "analyze_thermal_prediction",
    # Report Generator
    "ExplainabilityReportGenerator",
    "ExplainabilityReport",
    "Recommendation",
]

__version__ = "1.1.0"
