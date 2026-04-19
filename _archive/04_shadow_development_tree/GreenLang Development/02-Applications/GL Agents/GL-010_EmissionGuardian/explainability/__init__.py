# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Explainability Module

Production-grade explainability capabilities for emissions compliance including:
- Deterministic template-based explanations (Zero-Hallucination)
- Step-by-step calculation traces
- Compliance decision explanations
- ML model explainability (SHAP/LIME)
- Natural language narratives from templates

Standards Compliance:
    - EPA 40 CFR Part 75 (Continuous Emissions Monitoring)
    - EPA 40 CFR Part 60 (NSPS)
    - EPA 40 CFR Part 63 (NESHAP)

Zero-Hallucination Principle:
    - All explanations use deterministic templates, NOT LLM generation
    - Complete provenance tracking via SHA-256 hashes
    - Version-controlled narrative templates

Example:
    >>> from explainability import CalculationExplainer, ComplianceExplainer
    >>> # Explain a NOx calculation
    >>> calc_explainer = CalculationExplainer()
    >>> steps = calc_explainer.explain_nox_emission_rate(50.0, 3.0)
    >>> # Explain a compliance exceedance
    >>> comp_explainer = ComplianceExplainer()
    >>> explanation = comp_explainer.explain_exceedance(120.0, 100.0, "ppm", "NOx", "R-001")

Author: GreenLang GL-010 EmissionsGuardian
Version: 1.0.0
"""

from .schemas import (
    # Enums
    AudienceLevel,
    ExplanationType,
    ConfidenceLevel,
    UnitType,
    # Version tracking
    TemplateVersion,
    # Core models
    FeatureContribution,
    ReasoningStep,
    DecisionTrace,
    Explanation,
    NarrativeSummary,
    # Calculation models
    CalculationStep,
    UnitConversionExplanation,
    UncertaintyExplanation,
    # Request/Response
    ExplanationRequest,
    ExplanationResponse,
    # ML models
    SimilarCase,
    MLExplanation,
    # Audit
    AuditTraceExplanation,
)

from .calculation_explainer import (
    CalculationExplainer,
)

from .compliance_explainer import (
    ComplianceExplainer,
)

from .ml_explainer import (
    MLExplainer,
)

__all__ = [
    # Enums
    "AudienceLevel",
    "ExplanationType",
    "ConfidenceLevel",
    "UnitType",
    # Version tracking
    "TemplateVersion",
    # Core models
    "FeatureContribution",
    "ReasoningStep",
    "DecisionTrace",
    "Explanation",
    "NarrativeSummary",
    # Calculation models
    "CalculationStep",
    "UnitConversionExplanation",
    "UncertaintyExplanation",
    # Request/Response
    "ExplanationRequest",
    "ExplanationResponse",
    # ML models
    "SimilarCase",
    "MLExplanation",
    # Audit
    "AuditTraceExplanation",
    # Explainers
    "CalculationExplainer",
    "ComplianceExplainer",
    "MLExplainer",
]

__version__ = "1.0.0"
__author__ = "GreenLang GL-010 EmissionsGuardian"
