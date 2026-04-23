"""
GL-007 FurnacePulse - Explainability Module

Provides multi-layer explainability for furnace monitoring predictions:
1. SHAP-based feature attribution for local and global explanations
2. LIME-based local interpretable explanations for sensor readings
3. Engineering rationale linking predictions to physical phenomena
4. Model cards for documentation, versioning, and drift tracking
5. Comprehensive report generation for multiple stakeholders

This module ensures all predictions (hotspot detection, efficiency,
remaining useful life) are transparent, auditable, and actionable
for operators, engineers, and safety personnel.

Zero-hallucination principle: All explanations are derived from
deterministic model outputs and domain knowledge rules, never
from generative AI inference on numeric values.
"""

from .shap_explainer import (
    SHAPExplainer,
    SHAPResult,
    GlobalSHAPSummary,
    TopDriverInfo,
)
from .lime_explainer import (
    LIMEExplainer,
    LIMEResult,
    TabularExplanation,
)
from .engineering_rationale import (
    EngineeringRationale,
    RationaleItem,
    RootCauseAnalysis,
    CorrectiveAction,
    PhysicalPhenomenon,
)
from .model_cards import (
    ModelCardGenerator,
    ModelCard,
    ModelPerformanceMetrics,
    DriftSensitivityInfo,
)
from .report_generator import (
    ExplainabilityReportGenerator,
    ExplainabilityReport,
    AudienceType,
    ReportSection,
)

__all__ = [
    # SHAP
    "SHAPExplainer",
    "SHAPResult",
    "GlobalSHAPSummary",
    "TopDriverInfo",
    # LIME
    "LIMEExplainer",
    "LIMEResult",
    "TabularExplanation",
    # Engineering Rationale
    "EngineeringRationale",
    "RationaleItem",
    "RootCauseAnalysis",
    "CorrectiveAction",
    "PhysicalPhenomenon",
    # Model Cards
    "ModelCardGenerator",
    "ModelCard",
    "ModelPerformanceMetrics",
    "DriftSensitivityInfo",
    # Report Generator
    "ExplainabilityReportGenerator",
    "ExplainabilityReport",
    "AudienceType",
    "ReportSection",
]

__version__ = "1.0.0"
